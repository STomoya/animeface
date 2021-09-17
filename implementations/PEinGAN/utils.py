
import random
import functools

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image

from dataset import AnimeFace, DanbooruPortrait
from utils import Status, add_args, save_args
from nnutils import get_device, sample_nnoise, update_ema
from nnutils.loss import NonSaturatingLoss, r1_regularizer
from thirdparty.diffaugment import DiffAugment

from .model import Generator, Discriminator, init_weight_N01

def train(
    max_iters, dataset, sampler, const_z, latent_dim,
    G, G_ema, D, optim_G, optim_D,
    gp_lambda, d_k, augment,
    scales, scale_probs, image_size, mix_prob,
    device, amp, save=1000
):

    status = Status(max_iters)
    loss   = NonSaturatingLoss()
    gp     = r1_regularizer()
    scaler = GradScaler() if amp else None

    while status.batches_done < max_iters:
        for data in dataset:
            optim_G.zero_grad()
            optim_D.zero_grad()

            real = data.to(device)
            z = sampler((real.size(0), 2, latent_dim))
            scale = chose_scale(scales, scale_probs)
            real_size = int(image_size*scale)

            with autocast(amp):
                G.scale_image(scale)
                real = F.interpolate(
                    real, size=(real_size, real_size),
                    mode='bilinear', align_corners=True)

                # generate
                fake = G(z, mix_prob=mix_prob)
                # augment
                real_aug = augment(real)
                fake_aug = augment(fake)
                # D(real)
                real_prob = D(real_aug)
                # D(G(z))
                fake_prob = D(fake_aug.detach())

                # loss
                D_loss = loss.d_loss(real_prob, fake_prob)
                if gp_lambda > 0 and status.batches_done % d_k == 0:
                    D_loss = D_loss \
                        + gp(real, D, scaler) * gp_lambda

            if scaler is not None:
                scaler.scale(D_loss).backward()
                scaler.step(optim_D)
            else:
                D_loss.backward()
                optim_D.step()

            with autocast(amp):
                # D(G(z))
                fake_prob = D(fake_aug)

                # loss
                G_loss = loss.g_loss(fake_prob)

            if scaler is not None:
                scaler.scale(G_loss).backward()
                scaler.step(optim_G)
            else:
                G_loss.backward()
                optim_G.step()

            if G_ema is not None:
                update_ema(G, G_ema)

            # save
            if status.batches_done % save == 0:
                with torch.no_grad():
                    images = G_ema(const_z, truncation_psi=0.7)
                save_image(
                    images, f'implementations/PEinGAN/result/{status.batches_done}.jpg',
                    nrow=4, normalize=True, value_range=(-1, 1))
                torch.save(G_ema.state_dict(), f'implementations/PEinGAN/result/G_{status.batches_done}.pt')
            save_image(fake, f'running.jpg', nrow=4, normalize=True, value_range=(-1, 1))

            # updates
            loss_dict = dict(
                G=G_loss.item() if not torch.isnan(G_loss).any() else 0,
                D=D_loss.item() if not torch.isnan(D_loss).any() else 0
            )
            status.update(**loss_dict)
            if scaler is not None:
                scaler.update()

            if status.batches_done == max_iters:
                break

    status.plot_loss()

def chose_scale(scales, scale_probs):
    assert len(scales) == len(scale_probs)
    r = random.random()
    for i, prob in enumerate(scale_probs):
        if r < prob:
            break
        r -= prob
    return scales[i]

def main(parser):

    parser = add_args(parser,
        dict(
            num_test       = [16, 'number of test images'],
            no_spe         = [False, 'no position encoding'],
            g_bottom       = [4, 'bottom width'],
            latent_dim     = [512, 'input latent dim'],
            in_channels    = [512, 'synthesis input channels'],
            style_dim      = [512, 'style code dimension'],
            out_channels   = [3, 'output image channels'],
            g_channels     = [32, 'channel_width multiplier'],
            g_max_channels = [512, 'maximum channel width'],
            pad            = [False, 'use zero padding'],
            map_num_layers = [8, 'number of layers in mapping network'],
            no_pixelnorm   = [False, 'disable pixel norm'],
            filter_size    = [4, 'size of binomial filter'],
            g_act_name     = ['lrelu', 'activation function name'],
            d_bottom       = [2, 'discriminator bottom before GAP'],
            d_channels     = [32, 'channel width multiplier'],
            d_max_channels = [512, 'maximum channel width'],
            mbsd_groups    = [4, 'mini batch stddev groups'],
            no_gap         = [False, 'no gap layer'],
            d_act_name     = ['lrelu', 'activation function name'],
            map_lr         = [0.01, 'mappinf layer learning rate'],
            lr             = [0.001, 'learning rate'],
            betas          = [[0., 0.99], 'betas'],
            gp_lambda      = [5., 'lambda for r1'],
            d_k            = [16, 'calc r1 every'],
            scales         = [[1., 1.5, 2.], 'image scales'],
            scale_probs    = [[1/3, 1/3, 1/3], 'image scale probability'],
            mix_prob       = [0.9, 'style mixing probability'],
            policy         = ['color,translation']))
    args = parser.parse_args()
    save_args(args)

    device = get_device(not args.disable_gpu)
    amp = not args.disable_amp and not args.disable_gpu

    # data
    if args.dataset == 'animeface':
        dataset = AnimeFace.asloader(
            args.batch_size, (args.image_size, args.min_year),
            pin_memory=not args.disable_gpu)
    elif args.dataset == 'danbooru':
        dataset = DanbooruPortrait.asloader(
            args.batch_size, (args.image_size, args.num_images),
            pin_memory=not args.disable_gpu)

    sampler = functools.partial(sample_nnoise, device=device)
    const_z = sampler((args.num_test, args.latent_dim))

    # model
    G = Generator(
        args.image_size, args.g_bottom, not args.no_spe,
        args.latent_dim, args.in_channels, args.style_dim,
        args.out_channels, args.g_channels,
        args.g_max_channels, not args.pad, args.map_num_layers, args.map_lr,
        not args.no_pixelnorm, args.filter_size, args.g_act_name)
    G_ema = Generator(
        args.image_size, args.g_bottom, not args.no_spe,
        args.latent_dim, args.in_channels, args.style_dim,
        args.out_channels, args.g_channels,
        args.g_max_channels, not args.pad, args.map_num_layers, args.map_lr,
        not args.no_pixelnorm, args.filter_size, args.g_act_name)
    D = Discriminator(
        args.image_size, args.d_bottom, args.out_channels,
        args.d_channels, args.d_max_channels, args.mbsd_groups,
        not args.no_gap, args.filter_size, args.d_act_name)

    G.to(device)
    G_ema.to(device)
    G_ema.eval()
    for param in G_ema.parameters():
        param.requires_grad = False
    D.to(device)
    G.init_weight(init_weight_N01, args.map_lr)
    D.init_weight(init_weight_N01)
    update_ema(G, G_ema, 0.)

    # optim
    optim_G = optim.Adam(G.parameters(), lr=args.lr, betas=args.betas)
    optim_D = optim.Adam(D.parameters(), lr=args.lr, betas=args.betas)

    augment = functools.partial(DiffAugment, policy=args.policy)

    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs

    train(
        args.max_iters, dataset, sampler, const_z, args.latent_dim,
        G, G_ema, D, optim_G, optim_D,
        args.gp_lambda, args.d_k, augment,
        args.scales, args.scale_probs, args.image_size, args.mix_prob,
        device, amp, args.save
    )
