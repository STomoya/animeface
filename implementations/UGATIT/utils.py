
from functools import partial
from itertools import chain
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image
import numpy as np
import cv2

from dataset import AnimeFaceCelebA, DanbooruPortraitCelebA, AAHQCelebA
from nnutils import get_device, loss
from utils import Status, add_args, save_args, make_image_grid

from implementations.UGATIT.model import Generator, MultiScaleD


def optim_step(loss, optimizer, scaler):
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
    else:
        loss.backward()
        optimizer.step()


@torch.no_grad()
def color_heatmap(tensor, size):
    b, dtype, device = tensor.size(0), tensor.dtype, tensor.device
    # numpy, cv2 image channel format
    arrays = tensor.detach().cpu().permute(0, 2, 3, 1).numpy()
    # -> [0., 1.], np.float32
    arrays -= arrays.min(axis=(1, 2, 3), keepdims=True)
    arrays /= arrays.max(axis=(1, 2, 3), keepdims=True)
    # -> [0, 255], np.uint8
    arrays = (arrays * 255).astype(np.uint8)
    # batched array with target size
    new_array = np.zeros((b, *size, 3), dtype=np.uint8)
    # resize + apply color map
    for i in range(len(arrays)):
        array = arrays[i].copy()
        array = cv2.resize(array, size)
        array = cv2.applyColorMap(array, cv2.COLORMAP_JET)
        new_array[i] = array
    # -> [0., 1.], np.float32, BGR2RGB
    new_array = (new_array.astype(np.float32) / 255)[..., ::-1].copy()
    # torch.Tensor, channel first
    tensor = torch.from_numpy(new_array).to(device=device, dtype=dtype)
    tensor = tensor.permute(0, 3, 1, 2)
    # normalize
    tensor = (tensor - 0.5) * 0.5
    return tensor


def train(args,
    max_iters, dataset,
    GA, GB, DA, DB, optim_G, optim_D,
    cycle_lambda, id_lambda, cam_lambda,
    device, amp, save, logfile, loginterval
):

    status = Status(max_iters, False, logfile, loginterval, __name__)
    scaler = GradScaler() if amp else None

    adv_fn = loss.LSGANLoss()
    l1_fn  = nn.L1Loss()
    cam_fn = nn.BCEWithLogitsLoss()

    status.log_training(args, GB, DA)

    while not status.is_end():
        for A, B in dataset:
            optim_D.zero_grad()
            optim_G.zero_grad()
            A = A.to(device)
            B = B.to(device)

            with autocast(amp):
                # G forward
                # translate
                AB, g_AB_cam_logit = GB(A)
                BA, g_BA_cam_logit = GA(B)
                # identity
                AA, g_AA_cam_logit = GA(A)
                BB, g_BB_cam_logit = GB(B)
                # cycle
                ABA, _ = GA(AB)
                BAB, _ = GB(BA)

                # D forward (SG)
                real_a_prob, real_a_cam_logit = DA(A)
                real_b_prob, real_b_cam_logit = DB(B)
                fake_a_prob, fake_a_cam_logit = DA(BA.detach())
                fake_b_prob, fake_b_cam_logit = DB(AB.detach())

                # loss
                adv_a_loss = adv_fn.d_loss(real_a_prob, fake_a_prob)
                adv_b_loss = adv_fn.d_loss(real_b_prob, fake_b_prob)
                adv_loss = adv_a_loss + adv_b_loss
                adv_a_cam_loss = adv_fn.d_loss(real_a_cam_logit, fake_a_cam_logit)
                adv_b_cam_loss = adv_fn.d_loss(real_b_cam_logit, fake_b_cam_logit)
                adv_cam_loss = adv_a_cam_loss + adv_b_cam_loss
                D_loss = adv_loss + adv_cam_loss

            optim_step(D_loss, optim_D, scaler)

            with autocast(amp):
                # D forward
                fake_a_prob, fake_a_cam_logit = DA(BA)
                fake_b_prob, fake_b_cam_logit = DB(AB)

                # loss
                adv_a_loss = adv_fn.g_loss(fake_a_prob)
                adv_b_loss = adv_fn.g_loss(fake_b_prob)
                adv_loss = adv_a_loss + adv_b_loss
                adv_a_cam_loss = adv_fn.g_loss(fake_a_cam_logit)
                adv_b_cam_loss = adv_fn.g_loss(fake_b_cam_logit)
                adv_cam_loss = adv_a_cam_loss + adv_b_cam_loss
                id_loss, cycle_loss, cam_loss = 0, 0, 0
                if id_lambda > 0:
                    id_a_loss = l1_fn(AA, A)
                    id_b_loss = l1_fn(BB, B)
                    id_loss = (id_a_loss + id_b_loss) * id_lambda
                if cycle_lambda > 0:
                    cycle_a_loss = l1_fn(ABA, A)
                    cycle_b_loss = l1_fn(BAB, B)
                    cycle_loss = (cycle_a_loss + cycle_b_loss) * cycle_lambda
                if cam_lambda > 0:
                    cam_a_loss = cam_fn(g_BA_cam_logit, torch.ones_like(g_BA_cam_logit)) \
                        + cam_fn(g_AA_cam_logit, torch.zeros_like(g_AA_cam_logit))
                    cam_b_loss = cam_fn(g_AB_cam_logit, torch.ones_like(g_AB_cam_logit)) \
                        + cam_fn(g_BB_cam_logit, torch.zeros_like(g_BB_cam_logit))
                    cam_loss = (cam_a_loss + cam_b_loss) * cam_lambda
                G_loss = adv_loss + adv_cam_loss + id_loss + cycle_loss + cam_loss

            optim_step(G_loss, optim_G, scaler)

            if status.batches_done % save == 0:
                size = A.size(-1)
                to_heatmap = partial(color_heatmap, size=(size, size))
                with torch.no_grad():
                    AB, _, AB_heatmap = GB(A, return_heatmap=True)
                    BA, _, BA_heatmap = GA(B, return_heatmap=True)
                save_image(make_image_grid(A, to_heatmap(AB_heatmap), AB, B, to_heatmap(BA_heatmap), BA),
                    os.path.join(f'implementations/UGATIT/result/{status.batches_done}.jpg'),
                    nrow=6, normalize=True, value_range=(-1, 1))
                torch.save(dict(GA=GA.state_dict(), GB=GB.state_dict()),
                    os.path.join(f'implementations/UGATIT/result/model_{status.batches_done}.pth'))
            save_image(make_image_grid(AB, BA), 'running.jpg', nrow=2*4, normalize=True, value_range=(-1, 1))

            if scaler is not None: scaler.update()
            status.update(G=G_loss.item(), D=D_loss.item())

            if status.is_end():
                break
    status.plot_loss()


def main(parser):

    parser = add_args(parser,
        dict(
            image_channels  = [3, 'image channels'],
            bottom          = [int, 'bottom size. if not specified, will be image_size // 4'],
            g_channels      = [64, 'minimum channel width'],
            g_max_channels  = [512, 'maximum channel width'],
            resblocks       = [6, 'number of residual blocks'],
            adalinresblocks = [6, 'number of adalin residual blocks'],
            g_act_name      = ['relu', 'activation function name'],
            norm_name       = ['in', 'normalization layer name'],
            light           = [False, 'light weight'],

            num_scale       = [2, 'number of scales for multi scale D'],
            num_layers      = [3, 'number of layers'],
            d_channels      = [64, 'minimum channel width'],
            d_max_channels  = [512, 'maximum channel width'],
            d_act_name      = ['relu', 'activation function name'],

            g_lr            = [0.0002, 'learning rate'],
            d_lr            = [0.0002, 'learning rate'],
            betas           = [[0.5, 0.999], 'betas'],
            cycle_lambda    = [10., 'lambda for cycle consistency loss'],
            identity_lambda = [10., 'lambda for identity loss'],
            cam_lambda      = [1000., 'lambda for CAM loss']))
    args = parser.parse_args()
    save_args(args)

    device = get_device(not args.disable_gpu)
    amp = not args.disable_amp and not args.disable_gpu

    # dataset
    if args.dataset == 'animeface':
        dataset = AnimeFaceCelebA.asloader(
            args.batch_size, (args.image_size, args.min_year),
            pin_memory=not args.disable_gpu)
    elif args.dataset == 'danbooru':
        dataset = DanbooruPortraitCelebA.asloader(
            args.batch_size, (args.image_size, args.num_images),
            pin_memory=not args.disable_gpu)
    elif args.dataset == 'aahq':
        dataset = AAHQCelebA.asloader(
            args.batch_size, (args.image_size, args.num_images),
            pin_memory=not args.disable_gpu)

    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs

    # models
    # B -> A
    GA = Generator(
        args.image_size, args.bottom, args.g_channels, args.g_max_channels,
        args.resblocks, args.adalinresblocks, args.g_act_name,
        args.norm_name, args.light, args.image_channels)
    # A -> B
    GB = Generator(
        args.image_size, args.bottom, args.g_channels, args.g_max_channels,
        args.resblocks, args.adalinresblocks, args.g_act_name,
        args.norm_name, args.light, args.image_channels)
    # D(A)
    DA = MultiScaleD(
        args.num_scale, args.num_layers, args.d_channels, args.d_max_channels,
        args.d_act_name, args.image_channels)
    # D(B)
    DB = MultiScaleD(
        args.num_scale, args.num_layers, args.d_channels, args.d_max_channels,
        args.d_act_name, args.image_channels)

    GA.to(device)
    GB.to(device)
    DA.to(device)
    DB.to(device)

    # optimizers
    optim_G = optim.Adam(chain(GA.parameters(), GB.parameters()), lr=args.g_lr, betas=args.betas)
    optim_D = optim.Adam(chain(DA.parameters(), DB.parameters()), lr=args.d_lr, betas=args.betas)

    train(args, args.max_iters, dataset,
        GA, GB, DA, DB, optim_G, optim_D,
        args.cycle_lambda, args.identity_lambda, args.cam_lambda,
        device, amp, args.save, args.log_file, args.log_interval)
