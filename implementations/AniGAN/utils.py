
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import random_split
from torchvision.utils import save_image

from dataset import AnimeFaceCelebA, DanbooruPortraitCelebA, to_loader
from utils import Status, save_args, add_args
from nnutils import get_device, init, update_ema
from nnutils.loss import HingeLoss
from nnutils.loss.penalty import calc_grad, Variable

from .model import Generator, Discriminator

l1 = nn.L1Loss()

class R1:
    def __init__(self) -> None:
        self._clac_grad=calc_grad
    def penalty(self, A, B, D, scaler):
        loc_real = Variable(A, requires_grad=True)
        a_prob, _ = D(loc_real, False)
        gradients = self._calc_grad(a_prob, loc_real, scaler)
        gradients = gradients.reshape(gradients.size(0), -1)
        penalty_a = gradients.norm(2, dim=1).pow(2).mean() / 2.

        loc_real = Variable(B, requires_grad=True)
        _, b_prob = D(loc_real, False)
        gradients = self._calc_grad(b_prob, loc_real, scaler)
        gradients = gradients.reshape(gradients.size(0), -1)
        penalty_b = gradients.norm(2, dim=1).pow(2).mean() / 2.

        return penalty_a + penalty_b

def feature_matching(featsA, featsB):
    loss = 0
    for a, b in zip(featsA, featsB):
        a = F.adaptive_avg_pool2d(a, 1)
        b = F.adaptive_avg_pool2d(b, 1)
        loss = loss + l1(a, b)
    return loss

def train(
    max_iters, dataset, test_batch,
    G, G_ema, D, optimizer_G, optimizer_D,
    rec_lambda, feature_lambda, gp_lambda,
    device, amp, save
):

    status = Status(max_iters)
    loss   = HingeLoss()
    gp     = R1()
    scaler = GradScaler() if amp else None

    while status.batches_done < max_iters:
        for A, B in dataset:
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()
            A = A.to(device)
            B = B.to(device)

            with autocast(amp):
                # generate images
                # G(x, y)
                fakeA = G(B, A)
                fakeB = G(A, B)
                # G(h, h)
                recB = G(B, B)

                # D(h)
                real_a_prob, _ = D(A, False)
                _, real_b_prob = D(B, False)
                # D(G(x, y))
                fake_a_prob, _ = D(fakeA.detach(), False)
                _, fake_b_prob = D(fakeB.detach(), False)

                # loss
                adv_a_loss = loss.d_loss(real_a_prob, fake_a_prob)
                adv_b_loss = loss.d_loss(real_b_prob, fake_b_prob)
                # gp_loss    = gp.penalty(A, B, D, scaler)
                D_loss = adv_a_loss + adv_b_loss #+ gp_loss * gp_lambda

            if scaler is not None:
                scaler.scale(D_loss).backward()
                scaler.step(optimizer_D)
            else:
                D_loss.backward()
                optimizer_D.step()

            with autocast(amp):
                # D(h)
                _, _, sh_real_a_feats, real_a_feats, _ = D(A)
                _, _, sh_real_b_feats, _, real_b_feats = D(B)
                # D(G(x, y))
                fake_a_prob, _, sh_fake_a_feats, fake_a_feats, _ = D(fakeA)
                _, fake_b_prob, sh_fake_b_feats, _, fake_b_feats = D(fakeB)

                # loss
                sh_a_fmloss = feature_matching(sh_fake_a_feats, sh_real_a_feats)
                sh_b_fmloss = feature_matching(sh_fake_b_feats, sh_real_b_feats)
                a_fmloss = feature_matching(fake_a_feats, real_a_feats)
                b_fmloss = feature_matching(fake_b_feats, real_b_feats)
                fm_loss = sh_a_fmloss + sh_b_fmloss + a_fmloss + b_fmloss
                adv_a_loss = loss.g_loss(fake_a_prob)
                adv_b_loss = loss.g_loss(fake_b_prob)
                adv_loss = adv_a_loss + adv_b_loss
                rec_loss = l1(recB, B)
                G_loss = adv_loss + fm_loss * feature_lambda + rec_loss * rec_lambda

            if scaler is not None:
                scaler.scale(G_loss).backward()
                scaler.step(optimizer_G)
            else:
                G_loss.backward()
                optimizer_G.step()

            update_ema(G, G_ema)

            # save
            if status.batches_done % save == 0:
                images = G_ema(*test_batch)
                image_grid = _image_grid(A, B, fakeA, fakeB)
                save_image(
                    image_grid, f'implementations/AniGAN/result/{status.batches_done}.jpg',
                    normalize=True, value_range=(-1, 1), nrow=4*2)
                torch.save(
                    G_ema.state_dict(),
                    f'implementations/AniGAN/result/G_{status.batches_done}.pt')
            save_image(fakeA, 'running_anigan_a.jpg', normalize=True, value_range=(-1, 1))

            # updates
            loss_dict = dict(
                g=G_loss.item() if not torch.any(torch.isnan(G_loss)) else 0,
                d=D_loss.item() if not torch.any(torch.isnan(D_loss)) else 0,
            )
            status.update(**loss_dict)
            if scaler is not None:
                scaler.update()
        random.shuffle(dataset.dataset.dataset.images1)

    status.plot_loss()

def _image_grid(*args):
    _split = lambda x: x.chunk(x.size(0), dim=0)
    image_lists = [_split(arg) for arg in args]
    out = []
    for images in zip(*image_lists):
        out.extend(list(images))
    return torch.cat(out, dim=0)

def main(parser):

    # parser = add_arguments(parser)
    parser = add_args(parser,
        dict(
            image_channels = [3, 'image channels'],
            num_test       = [4, 'number of test images'],
            bottom_width   = [16, 'bottom width'],
            g_channels     = [32, 'channel width multiplier'],
            affine         = [False, 'affine transform style code in PoLIN. experimental.'],
            style_dim      = [256, 'style dimension'],
            g_norm_name    = ['in', 'norm layer in encoder'],
            g_act_name     = ['lrelu', 'activation layer name'],
            branch_width   = [32, 'last feature size in shared layers'],
            d_channels     = [32, 'channel width multiplier'],
            max_channels   = [512, 'maximum channels width'],
            d_norm_name    = ['in', 'normalization layer name'],
            d_act_name     = ['lrelu', 'activation function name'],
            disable_bias   = [False, 'disable bias'],
            lr             = [0.00002, 'learning rate'],
            betas          = [[0., 0.999], 'betas'],
            rec_lambda     = [1.2, 'lambda for reconstruction loss'],
            feature_lambda = [1., 'lambda for feature mapping loss'],
            gp_lambda      = [1., 'lambda for r1 panalty'])
    )
    args = parser.parse_args()
    save_args(args)

    device = get_device(not args.disable_gpu)
    amp = not args.disable_gpu and not args.disable_amp

    # data
    if args.dataset == 'animeface':
        dataset = AnimeFaceCelebA(args.image_size, args.min_year)
    elif args.dataset == 'danbooru':
        dataset = DanbooruPortraitCelebA(args.image_size, args.num_images)
    dataset, test = random_split(dataset, [len(dataset)-args.num_test, args.num_test])
    dataset = to_loader(dataset, args.batch_size, pin_memory=not args.disable_gpu)
    # test
    test = to_loader(test, args.num_test, shuffle=False, pin_memory=False)
    test_batch = next(iter(test))
    test_batch = (test_batch[0].to(device), test_batch[1].to(device))

    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs

    # model
    G = Generator(
        args.image_size, args.image_channels, args.image_channels,
        args.bottom_width, args.g_channels, args.affine,
        args.style_dim, not args.disable_bias,
        args.g_norm_name, args.g_act_name
    )
    G_ema = Generator(
        args.image_size, args.image_channels, args.image_channels,
        args.bottom_width, args.g_channels, args.affine,
        args.style_dim, not args.disable_bias,
        args.g_norm_name, args.g_act_name
    )
    D = Discriminator(
        args.image_size, args.branch_width, args.image_channels,
        args.d_channels, args.max_channels, not args.disable_bias,
        args.d_norm_name, args.d_act_name
    )
    G.to(device)
    G_ema.to(device)
    G_ema.eval()
    for param in G_ema.parameters():
        param.requires_grad = False
    D.to(device)
    G.apply(init().N002)
    D.apply(init().N002)
    update_ema(G, G_ema, 0.)

    # optimizers
    optimizer_G = optim.Adam(G.parameters(), lr=args.lr, betas=args.betas)
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr, betas=args.betas)

    train(
        args.max_iters, dataset, test_batch,
        G, G_ema, D, optimizer_G, optimizer_D,
        args.rec_lambda, args.feature_lambda, args.gp_lambda,
        device, amp, args.save
    )
