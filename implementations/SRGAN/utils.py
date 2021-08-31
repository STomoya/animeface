
import functools

import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.utils.data import random_split

from dataset import DanbooruPortraitSR, to_loader
from utils import Status, save_args, add_args
from nnutils import get_device
from nnutils.loss import NonSaturatingLoss, VGGLoss

from .model import Generator, Discriminator, init_weight_N002, init_weight_xavier, init_weight_kaiming

def train(
    max_iters, dataset, test_batch,
    G, D, optimizer_G, optimizer_D,
    adv_lambda, vgg_lambda,
    amp, device, save
):

    status = Status(max_iters)
    loss   = NonSaturatingLoss()
    vgg    = VGGLoss(device)
    scaler = GradScaler() if amp else None

    while status.batches_done < max_iters:
        for lr, hr in dataset:
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()

            lr = lr.to(device)
            hr = hr.to(device)

            '''Discriminator'''
            with autocast(amp):
                # D(hr)
                real_outs = D(hr)
                # D(G(lr))
                fake = G(lr)
                fake_outs = D(fake.detach())

                # loss
                D_loss = 0
                for real_out, fake_out in zip(real_outs, fake_outs):
                    D_loss = D_loss \
                        + loss.d_loss(real_out[0], fake_out[0])

            if scaler is not None:
                scaler.scale(D_loss).backward()
                scaler.step(optimizer_D)
            else:
                D_loss.backward()
                optimizer_D.step()

            '''Generator'''
            with autocast(amp):
                # D(G(lr))
                fake_outs = D(fake)

                # loss
                G_loss = vgg.content_loss(hr, fake) * vgg_lambda
                for fake_out in fake_outs:
                    G_loss = G_loss \
                        + loss.g_loss(fake_out[0]) * adv_lambda

            if scaler is not None:
                scaler.scale(G_loss).backward()
                scaler.step(optimizer_G)
            else:
                G_loss.backward()
                optimizer_G.step()

            # save
            if status.batches_done % save == 0:
                with torch.no_grad():
                    G.eval()
                    image = G(test_batch[0])
                    G.train()
                image_grid = _image_grid(test_batch[0], test_batch[1], image)
                save_image(
                    image_grid, f'implementations/SRGAN/result/{status.batches_done}.jpg',
                    nrow=3*3, normalize=True, value_range=(-1, 1))
                torch.save(G.state_dict(), f'implementations/SRGAN/result/G_{status.batches_done}.pt')
            save_image(fake, 'running_srgan.jpg', normalize=True, value_range=(-1, 1))

            # updates
            loss_dict = dict(
                D=D_loss.item() if not torch.any(torch.isnan(D_loss)) else 0.,
                G=G_loss.item() if not torch.any(torch.isnan(G_loss)) else 0.
            )
            status.update(**loss_dict)
            if scaler is not None:
                scaler.update()

            if status.batches_done == max_iters:
                break

    status.plot_loss()

def _image_grid(src, dst, gen, upsample=False):
    _split = lambda x: x.chunk(x.size(0), dim=0)
    srcs = _split(src)
    dsts = _split(dst)
    gens = _split(gen)

    images = []
    for src, dst, gen in zip(srcs, dsts, gens):
        if upsample:
            src = TF.resize(src, dst.size(2))
        else:
            pad = (dst.size(2) - src.size(2)) // 2
            src = F.pad(src, (pad, pad, pad, pad))
        images.extend([src, dst, gen])
    return torch.cat(images, dim=0)

def main(parser):

    # parser = add_arguments(parser)
    parser = add_args(parser,
        dict(
            num_test       = [6, 'number of samples used in eval'],
            scale          = [2, 'upsample scale'],
            disable_sn     = [False, 'disable spectral norm'],
            disable_bias   = [False, 'disable bias'],
            image_channels = [3, 'input image channels'],
            g_channels     = [64, 'channel width multiplier for G'],
            num_blocks     = [5, 'number of residual blocks in G'],
            g_norm_name    = ['in', 'normalization layer name for G'],
            g_act_name     = ['prelu', 'activation function name for G'],
            num_scale      = [2, 'number of scales in D'],
            d_channels     = [32, 'channel width multiplier for D'],
            num_layers     = [3, 'number of layers in D'],
            d_norm_name    = ['in', 'normalization layer name for D'],
            d_act_name     = ['prelu', 'activation function name for D'],
            lr             = [0.0002, 'learning rate'],
            betas          = [[0.5, 0.999], 'betas'],
            adv_lambda     = [0.001, 'lambda for adversarial loss'],
            vgg_lambda     = [1., 'lambda for perceptual loss']))
    args = parser.parse_args()
    save_args(args)

    amp = not args.disable_amp and not args.disable_gpu
    device = get_device(not args.disable_gpu)

    # dataset
    dataset = DanbooruPortraitSR(args.image_size, scale=args.scale, num_images=args.num_images+args.num_test)
    dataset, test = random_split(dataset, [len(dataset)-args.num_test, args.num_test], generator=torch.Generator().manual_seed(1234))
    ## training dataset
    dataset = to_loader(dataset, args.batch_size)
    ## test batch
    test    = to_loader(test, args.num_test, shuffle=False, pin_memory=False)
    test_batch = next(iter(test))
    test_batch = (test_batch[0].to(device), test_batch[1].to(device))

    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs

    # model
    G = Generator(
        args.scale, args.image_channels, args.g_channels,
        args.num_blocks, not args.disable_sn, not args.disable_bias,
        args.g_norm_name, args.g_act_name
    )
    D = Discriminator(
        args.image_channels, args.num_scale, args.num_layers,
        args.d_channels, not args.disable_sn, not args.disable_bias,
        args.d_norm_name, args.d_act_name
    )
    G.apply(init_weight_xavier)
    D.apply(init_weight_xavier)
    G.to(device)
    D.to(device)

    # optimizer
    optimizer_G = optim.Adam(G.parameters(), lr=args.lr, betas=args.betas)
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr, betas=args.betas)

    train(
        args.max_iters, dataset, test_batch,
        G, D, optimizer_G, optimizer_D,
        args.adv_lambda, args.vgg_lambda,
        amp, device, args.save
    )
