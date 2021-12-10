
from functools import partial
from implementations.CIPS import model

import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image

from dataset import AnimeFace, DanbooruPortrait
from utils import Status, save_args, add_args
from nnutils import sample_nnoise, update_ema, get_device
from nnutils.loss import NonSaturatingLoss, r1_regularizer
from thirdparty.diffaugment import DiffAugment

from .model import Generator, Discriminator

def train(args,
    max_iters, dataset, const_z, latent_dim,
    G, G_ema, D, optimizer_G, optimizer_D,
    gp_lambda, augment,
    device, amp, save=1000, log_file='log.log', log_interval=10
):

    status = Status(
        max_iters, log_file is None,
        log_file, log_interval, __name__)
    adv_fn = NonSaturatingLoss()
    gp_fn  = r1_regularizer()
    scaler = GradScaler() if amp else None

    status.log_training(args, G, D)

    while not status.is_end():
        for data in dataset:
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            real = data.to(device)
            z = sample_nnoise((data.size(0), latent_dim), device)

            '''generate'''
            with autocast(amp):
                fake = G(z)
                real_aug = augment(real)
                fake_aug = augment(fake)

            '''D'''
            with autocast(amp):
                # D(real)
                real_prob = D(real_aug)
                # D(G(z))
                fake_prob = D(fake_aug.detach())

                # loss
                adv_loss = adv_fn.d_loss(real_prob, fake_prob)
                gp_loss = gp_fn(real, D, scaler) * gp_lambda
                D_loss = adv_loss + gp_loss

            if scaler is not None:
                scaler.scale(D_loss).backward()
                scaler.step(optimizer_D)
            else:
                D_loss.backward()
                optimizer_D.step()

            '''G'''
            with autocast(amp):
                # D(G(z))
                fake_prob = D(fake_aug)

                # loss
                G_loss = adv_fn.g_loss(fake_prob)

            if scaler is not None:
                scaler.scale(G_loss).backward()
                scaler.step(optimizer_G)
            else:
                G_loss.backward()
                optimizer_G.step()

            update_ema(G, G_ema, copy_buffers=True)

            if status.batches_done % save == 0:
                with torch.no_grad():
                    images = G_ema(const_z)
                save_image(
                    images, f'implementations/CIPS/result/{status.batches_done}.jpg',
                    nrow=int(images.size(0)**0.5), normalize=True, value_range=(-1, 1))
                torch.save(
                    G_ema.state_dict(),
                    f'implementations/CIPS/result/G_{status.batches_done}.pth')
            save_image(fake, 'running.jpg', normalize=True, value_range=(-1, 1))

            status.update(D=D_loss.item(), G=G_loss.item())
            if scaler is not None:
                scaler.update()

            if status.is_end():
                break

    status.plot_loss()

def main(parser):
    parser = add_args(parser,
        dict(
            num_test=[16, 'number of images for eval'],
            image_channels=[3, 'image channels'],

            latent_dim=[512, 'dimension of input latent'],
            style_dim=[512, 'dimension of style code'],
            num_layers=[14, 'number of style layers'],
            g_channels=[32, 'minimum channel width'],
            g_max_channels=[512, 'maximum channel width'],
            map_num_layers=[4, 'number of layers in mapping network'],
            no_pixel_norm=[False, 'disable pixel normalization'],

            d_channels=[64, 'minimum channel width'],
            d_max_channels=[512, 'maximum channel width'],
            mbsd_group_size=[4, 'mini-batch standard deviation group size'],
            mbsd_channels=[1, 'mini-batch standard deviation channels'],
            bottom=[4, 'bottom width'],
            filter_size=[4, 'filter size'],

            lr=[0.0025, 'learning rate'],
            map_lr_scale=[0.01, 'scale learning rate for mapping network with'],
            betas=[[0., 0.99], 'betas'],
            gp_lambda=[10., 'lambda for gradient penalty'],
            policy=['color,translation', 'policy for diffaugment']))
    args = parser.parse_args()
    save_args(args)

    device = get_device(not args.disable_gpu)
    amp    = not args.disable_amp and not args.disable_gpu

    # dataset
    if args.dataset == 'animeface':
        dataset = AnimeFace.asloader(
            args.batch_size, (args.image_size, args.min_year),
            pin_memory=not args.disable_gpu)
    elif args.dataset == 'danbooru':
        dataset = DanbooruPortrait.asloader(
            args.batch_size, (args.image_size, args.num_images),
            pin_memory=not args.disable_gpu)
    const_z = sample_nnoise((args.num_test, args.latent_dim), device=device)

    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs

    # model
    G = Generator(
        args.image_size, args.latent_dim, args.style_dim,
        args.num_layers, args.g_channels, args.g_max_channels,
        args.image_channels, args.map_num_layers, not args.no_pixel_norm)
    G_ema = Generator(
        args.image_size, args.latent_dim, args.style_dim,
        args.num_layers, args.g_channels, args.g_max_channels,
        args.image_channels, args.map_num_layers, not args.no_pixel_norm)
    update_ema(G, G_ema, 0., True)
    D = Discriminator(
        args.image_size, args.image_channels, args.d_channels, args.d_max_channels,
        args.mbsd_group_size, args.mbsd_channels, args.bottom,
        args.filter_size, 'lrelu', 1.)
    G.to(device)
    G_ema.to(device)
    D.to(device)

    # optimizers
    optimizer_G = optim.Adam(
        [{'params': G.synthesis.parameters()},
         {'params': G.mapping.parameters(), 'lr': args.lr*args.map_lr_scale}],
        lr=args.lr, betas=args.betas)
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr, betas=args.betas)

    train(args,
        args.max_iters, dataset, const_z, args.latent_dim,
        G, G_ema, D, optimizer_G, optimizer_D,
        args.gp_lambda, partial(DiffAugment, policy=args.policy),
        device, amp, args.save, args.log_file, args.log_interval)
