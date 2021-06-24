
import functools
import itertools
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image
import numpy as np

from ..general import YearAnimeFaceDataset, to_loader
from ..general import get_device, Status, save_args
from ..gan_utils import sample_nnoise, update_ema, init, DiffAugment
from ..gan_utils.losses import GANLoss, GradPenalty

from .model import Generator, Discriminator, MultiAxisAttention

def train(
    max_iters, dataset, sampler, latent_dim, const_z,
    G, G_ema, D, optimizer_G, optimizer_D,
    gp_lambda, augment,
    device, amp, save
):

    status = Status(max_iters)
    loss   = GANLoss()
    gp     = GradPenalty()
    scaler = GradScaler() if amp else None

    while status.batches_done < max_iters:
        for data in dataset:
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()

            real = data.to(device)
            z = sampler((real.size(0), latent_dim))

            with autocast(amp):
                # generate
                fake = G(z)
                # augment
                real_aug = augment(real)
                fake_aug = augment(fake)

                '''Discriminator'''
                real_prob = D(real_aug)
                fake_prob = D(fake_aug.detach())

                D_loss = loss.d_loss(real_prob, fake_prob)
                if gp_lambda > 0:
                    D_loss = D_loss \
                        + gp.r1_regularizer(real, D, scaler) * gp_lambda

            if scaler is not None:
                scaler.scale(D_loss).backward()
                scaler.step(optimizer_D)
            else:
                D_loss.backward()
                optimizer_D.step()

            with autocast(amp):
                '''Generator'''
                fake_prob = D(fake_aug)

                G_loss = loss.g_loss(fake_prob)

            if scaler is not None:
                scaler.scale(G_loss).backward()
                scaler.step(optimizer_G)
            else:
                G_loss.backward()
                optimizer_G.step()

            if G_ema is not None:
                update_ema(G, G_ema)

            # save
            if status.batches_done % save == 0:
                if G_ema is not None:
                    image = G_ema(const_z)
                    state_dict = G_ema.state_dict()
                else:
                    G.eval()
                    image = G(const_z)
                    G.train()
                    state_dict = G.state_dict()
                save_image(
                    image, f'implementations/HiT/result/{status.batches_done}.jpg',
                    normalize=True, value_range=(-1, 1), nrow=4)
                torch.save(state_dict, f'implementations/HiT/result/G_{status.batches_done}.pt')
            save_image(fake, 'running.jpg', normalize=True, value_range=(-1, 1))

            # updates
            loss_dict = dict(
                G=G_loss.item() if not torch.isnan(G_loss).any() else 0,
                D=D_loss.item() if not torch.isnan(D_loss).any() else 0
            )
            status.update(loss_dict)
            if scaler is not None:
                scaler.update()

            if status.batches_done == max_iters:
                break

    status.plot()

def add_arguments(parser):
    parser.add_argument('--num-test', default=16, type=int)

    parser.add_argument('--arch', default=None, choices=['s', 'b', 'l'])
    parser.add_argument('--latent-dim', default=128, type=int)
    parser.add_argument('--dims', default=[512, 512, 256, 128, 64, 64], type=int, nargs='+')
    parser.add_argument('--bottom', default=8, type=int)
    parser.add_argument('--low-stages', default=4, type=int)
    parser.add_argument('--num-heads', default=[16, 8, 4, 4, 4, 4], type=int, nargs='+')
    parser.add_argument('--num-blocks', default=[2, 2, 2, 2, 2, 2], type=int, nargs='+')
    parser.add_argument('--patch-sizes', default=[4, 4, 8, 8], type=int, nargs='+')
    parser.add_argument('--channels', default=32, type=int)
    parser.add_argument('--max-channels', default=512, type=int)
    parser.add_argument('--act-name', default='lrelu', choices=['lrelu', 'relu'])
    parser.add_argument('--ema', default=False, action='store_true')

    parser.add_argument('--init-func', default='xavier', choices=['N01', 'N002', 'xavier', 'kaiming'])
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--betas', default=[0.5, 0.99], type=float, nargs=2)
    parser.add_argument('--gp-lambda', default=0., type=float)
    parser.add_argument('--policy', default='color,translation')
    return parser

def set_args_by_arch(args):
    # same params for all arch
    args.bottom = 8
    args.low_stages = 4
    args.num_heads = [16, 8, 4, 4, 4, 4]
    args.patch_sizes = [4, 4, 8, 8]

    # others
    if args.arch == 's':
        args.dims = [512, 256, 128, 64, 32, 32]
        args.num_blocks = [2, 2, 1, 1, 1, 1]
    if args.arch == 'b': # default
        args.dims = [512, 512, 256, 128, 64, 64]
        args.num_blocks = [2, 2, 2, 2, 2, 2]
    if args.arch == 'l':
        args.dims = [1024, 512, 256, 128, 128, 128]
        args.num_blocks = [2, 2, 2, 2, 2, 2]

    return args

def adjust_by_size(args):
    if args.image_size == 128:
        args.dims = args.dims[:-1]
        args.num_heads = args.num_heads[:-1]
        args.num_blocks = args.num_blocks[:-1]
    return args

def main(parser):
    parser = add_arguments(parser)
    args = parser.parse_args()
    if args.arch is not None:
        args = set_args_by_arch(args)
    args = adjust_by_size(args)
    save_args(args)

    device = get_device(not args.disable_gpu)
    amp = not args.disable_gpu and not args.disable_amp

    dataset = YearAnimeFaceDataset(args.image_size, args.min_year)
    dataset = to_loader(dataset, args.batch_size)

    sampler = functools.partial(sample_nnoise, device=device)
    const_z = sampler((args.num_test, args.latent_dim))

    # model
    G = Generator(
        args.latent_dim, args.dims, args.bottom, args.low_stages,
        args.num_heads, args.num_blocks, args.patch_sizes)
    if args.ema:
        G_ema = Generator(
            args.latent_dim, args.dims, args.bottom, args.low_stages,
            args.num_heads, args.num_blocks, args.patch_sizes)
    else: G_ema = None
    D = Discriminator(
        args.image_size, args.channels, args.max_channels,
        args.bottom, args.act_name)
    init_func = getattr(init([MultiAxisAttention], names=['q', 'k', 'v', 'o']), args.init_func)
    G.apply(init_func)
    D.apply(init_func)
    if G_ema is not None:
        update_ema(G, G_ema, 0)
        G.to(device)
    G.to(device)
    D.to(device)

    # optimizer
    optimizer_G = optim.Adam(G.parameters(), lr=args.lr/2, betas=args.betas)
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr*2, betas=args.betas)

    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs

    augment = functools.partial(DiffAugment, policy=args.policy)

    train(
        args.max_iters, dataset, sampler, args.latent_dim, const_z,
        G, G_ema, D, optimizer_G, optimizer_D,
        args.gp_lambda, augment,
        device, amp, args.save
    )
