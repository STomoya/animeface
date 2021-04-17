
import functools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.utils.data import random_split

from ..general import get_device, Status, save_args
from ..gan_utils.losses import GANLoss, VGGLoss
from ..general import DanbooruPortraitSRDataset, to_loader

from .model import Generator, Discriminator, init_weight_N002, init_weight_xavier, init_weight_kaiming

def train(
    max_iters, dataset, test_batch,
    G, D, optimizer_G, optimizer_D,
    adv_lambda, vgg_lambda,
    amp, device, save
):
    
    status = Status(max_iters)
    loss   = GANLoss()
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
            status.update(loss_dict)
            if scaler is not None:
                scaler.update()

            if status.batches_done == max_iters:
                break

    status.plot()

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

def add_arguments(parser):
    parser.add_argument('--num-test', default=6, type=int, help='number of samples used in eval')
    parser.add_argument('--scale', default=2, type=int, help='upsample scale')
    # model
    parser.add_argument('--disable-sn', default=False, action='store_true', help='disable spectral norm')
    parser.add_argument('--disable-bias', default=False, action='store_true', help='disable bias')
    # G
    parser.add_argument('--image-channels', default=3, type=int, help='input image channels')
    parser.add_argument('--g-channels', default=64, type=int, help='channel width multiplier for G')
    parser.add_argument('--num-blocks', default=5, type=int, help='number of residual blocks in G')
    parser.add_argument('--g-norm-name', default='in', choices=['bn', 'in'], help='normalization layer name for G')
    parser.add_argument('--g-act-name', default='prelu', choices=['relu', 'lrelu', 'prelu'], help='activation function name for G')
    # D
    parser.add_argument('--num-scale', default=2, type=int, help='number of scales in D')
    parser.add_argument('--d-channels', default=32, type=int, help='channel width multiplier for D')
    parser.add_argument('--num-layers', default=3, type=int, help='number of layers in D')
    parser.add_argument('--d-norm-name', default='in', choices=['bn', 'in'], help='normalization layer name for D')
    parser.add_argument('--d-act-name', default='prelu', choices=['relu', 'lrelu', 'prelu'], help='activation function name for D')

    # training
    parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
    parser.add_argument('--betas', default=[0.5, 0.999], type=float, nargs=2, help='betas')
    parser.add_argument('--adv-lambda', default=0.001, type=float, help='lambda for adversarial loss')
    parser.add_argument('--vgg-lambda', default=1., type=float, help='lambda for perceptual loss')
    return parser

def main(parser):

    parser = add_arguments(parser)
    args = parser.parse_args()
    save_args(args)

    amp = not args.disable_amp and not args.disable_gpu
    device = get_device(not args.disable_gpu)

    # dataset
    dataset = DanbooruPortraitSRDataset(args.image_size, scale=args.scale, num_images=args.num_images+args.num_test)
    dataset, test = random_split(dataset, [len(dataset)-args.num_test, args.num_test], generator=torch.Generator().manual_seed(1234))
    ## training dataset
    dataset = to_loader(dataset, args.batch_size)
    ## test batch
    test    = to_loader(test, args.num_test, shuffle=False, use_gpu=False)
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