
import functools

import torch
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image

from dataset import AnimeFace
from utils import Status, save_args, add_args
from nnutils import get_device, sample_nnoise
from nnutils.loss.gan import NonSaturatingLoss
from nnutils.loss.penalty import r1_regularizer
from thirdparty.diffaugment import DiffAugment

from ..StyleGAN2.model import Generator, Discriminator, init_weight_N01
from .AdaBelief import AdaBelief

def train(
    max_iters, dataset, sample_noise, style_dim,
    G, D, optimizer_G, optimizer_D,
    gp_lambda, policy,
    device, amp, save_interval=1000
):

    status = Status(max_iters)
    loss   = NonSaturatingLoss()
    r1_gp  = r1_regularizer()
    scaler = GradScaler() if amp else None
    const_z = sample_noise((16, style_dim))

    if policy != '':
        augment = functools.partial(DiffAugment, policy=policy)
    else:
        augment = lambda x: x

    while status.batches_done < max_iters:
        for index, real in enumerate(dataset):
            G.zero_grad()
            D.zero_grad()

            real = real.to(device)
            z = sample_noise((real.size(0), style_dim))

            '''Discriminator'''
            with autocast(amp):
                # D(x)
                real = augment(real)
                real_prob = D(real)
                # D(G(z))
                fake, _ = G(z)
                fake = augment(fake)
                fake_prob = D(fake.detach())
                # loss
                adv_loss = loss.d_loss(real_prob, fake_prob)
                if gp_lambda > 0:
                    r1_loss = r1_gp(real, D, scaler)
                else:
                    r1_loss = 0
                D_loss = adv_loss + r1_loss * gp_lambda

            if scaler is not None:
                scaler.scale(D_loss).backward()
                scaler.step(optimizer_D)
            else:
                D_loss.backward()
                optimizer_D.step()

            '''Generator'''
            with autocast(amp):
                # D(G(z))
                fake_prob = D(fake)
                # loss
                G_loss = loss.g_loss(fake_prob)

            if scaler is not None:
                scaler.scale(G_loss).backward()
                scaler.step(optimizer_G)
            else:
                G_loss.backward()
                optimizer_G.step()

            if status.batches_done % save_interval == 0:
                G.eval()
                with torch.no_grad():
                    image, _ = G(const_z)
                G.train()
                save_image(
                    image, f'implementations/AdaBelief/result/{status.batches_done}.png',
                    nrow=4, normalize=True, value_range=(-1, 1))
            save_image(fake, 'running.jpg', normalize=True, value_range=(-1, 1))

            status.update(g=G_loss.item(), d=D_loss.item())
            if scaler is not None:
                scaler.update()

            if status.batches_done == max_iters:
                break

def main(parser):

    parser = add_args(parser,
        dict(
            image_channels   = [3,     'number of channels for the generated image'],
            style_dim        = [512,   'style feature dimension'],
            channels         = [32,    'channel width multiplier'],
            max_channels     = [512,   'maximum channels'],
            block_num_conv   = [2,     'number of convolution layers in residual block'],
            map_num_layers   = [8,     'number of layers in mapping network'],
            map_lr           = [0.01,  'learning rate for mapping network'],
            disable_map_norm = [False, 'disable pixel normalization'],
            mbsd_groups      = [4,     'number of groups in mini-batch stddev'],
            lr               = [0.001, 'learning rate'],
            betas            = [[0.1, 0.99], 'betas'],
            gp_lambda        = [10.,   'lambda for r1'],
            policy           = ['color,translation', 'policy for DiffAugment']))
    args = parser.parse_args()
    save_args(args)

    normalize = not args.disable_map_norm

    amp = not args.disable_amp and not args.disable_gpu
    device = get_device(not args.disable_gpu)

    dataset = AnimeFace.asloader(
        args.batch_size, (args.image_size, args.min_year))

    sample_noise = functools.partial(sample_nnoise, device=device)

    G = Generator(
            args.image_size, args.image_channels, args.style_dim, args.channels, args.max_channels,
            args.block_num_conv, args.map_num_layers, normalize, args.map_lr)
    D = Discriminator(
            args.image_size, args.image_channels, args.channels, args.max_channels,
            args.block_num_conv, args.mbsd_groups)
    G.init_weight(
        map_init_func=functools.partial(init_weight_N01, lr=args.map_lr),
        syn_init_func=init_weight_N01)
    D.apply(init_weight_N01)
    G.to(device)
    D.to(device)

    assert args.betas[0] != 0 and args.betas[1] != 0
    optimizer_G = AdaBelief(G.parameters(), lr=args.lr, betas=args.betas)
    optimizer_D = AdaBelief(D.parameters(), lr=args.lr, betas=args.betas)

    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs

    train(
        args.max_iters, dataset, sample_noise, args.style_dim,
        G, D, optimizer_G, optimizer_D, args.gp_lambda, args.policy,
        device, amp, args.save
    )
