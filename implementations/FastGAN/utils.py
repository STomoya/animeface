
import functools
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image

from dataset import AnimeFace, DanbooruPortrait
from utils import Status, save_args, add_args, make_image_grid
from nnutils import sample_nnoise, get_device, update_ema, init, freeze
from nnutils.loss import HingeLoss
from thirdparty.diffaugment import DiffAugment

from .model import Generator, Discriminator

def train(args,
    max_iters, dataset, const_z, latent_dim,
    G, G_ema, D, optimizer_G, optimizer_D,
    augment,
    device, amp, save=1000,
    log_file='log.log', log_interval=10
):

    status = Status(
        max_iters, log_file is None,
        log_file, log_interval, __name__)
    adv_fn = HingeLoss()
    scaler = GradScaler() if amp else None
    G_test = G_ema if G_ema is not None else G

    status.log_training(args, G, D)

    while not status.is_end():
        for data in dataset:
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            real = data.to(device)
            z = sample_nnoise((real.size(0), latent_dim), device)

            # generate
            with autocast(amp):
                # G(z)
                fake = G(z)
                # augmentation
                real_aug = augment(real)
                fake_aug = augment(fake)

            '''Discriminator'''
            with autocast(amp):
                # D(real)
                real_prob, recon_loss, recon_images = D(real_aug)
                # D(G(z))
                fake_prob, _ = D(fake_aug.detach(), False)

                # loss
                adv_loss = adv_fn.d_loss(real_prob, fake_prob)
                D_loss = adv_loss + recon_loss

            if scaler is not None:
                scaler.scale(D_loss).backward()
                scaler.step(optimizer_D)
            else:
                D_loss.backward()
                optimizer_D.step()

            with autocast(amp):
                # D(G(z))
                fake_prob, _ = D(fake_aug, False)

                # loss
                G_loss = adv_fn.g_loss(fake_prob)

            if scaler is not None:
                scaler.scale(G_loss).backward()
                scaler.step(optimizer_G)
            else:
                G_loss.backward()
                optimizer_G.step()

            if G_ema is not None:
                update_ema(G, G_ema, copy_buffers=True)

            if status.batches_done % save == 0:
                with torch.no_grad():
                    G.eval()
                    images = G_test(const_z)
                    G.train()
                save_image(
                    images, f'implementations/FastGAN/result/{status.batches_done}.jpg',
                    nrow=int(images.size(0)**0.5), normalize=True, value_range=(-1, 1))
                torch.save(
                    G.state_dict(),
                    f'implementations/FastGAN/result/G_{status.batches_done}.pth')
            save_image(
                fake, 'running.jpg', nrow=int(fake.size(0)**0.5),
                normalize=True, value_range=(-1, 1))
            save_image(
                make_image_grid(*recon_images), 'recon.jpg', nrow=8,
                normalize=True, value_range=(-1, 1))

            status.update(G=G_loss.item(), D=D_loss.item())
            if scaler is not None:
                scaler.update()

            if status.is_end():
                break

    status.plot_loss()

def main(parser):
    parser = add_args(parser,
        dict(
            num_test           = [16, 'number of images for test'],
            image_channels     = [3, 'image_channels'],

            latent_dim         = [128, 'dimension for input latent'],
            g_channels         = [32, 'minimum channel width'],
            g_max_channels     = [512, 'maximum channel width'],
            interp_size        = [4, 'interpolation size in SLE'],
            g_bottom           = [4, 'bottom width'],
            norm_name          = ['bn', 'normlization name'],
            transposed         = [False, 'use ConvTransposed2d instead of Upsample'],
            num_sle            = [int, 'number of sle blocks'],

            d_channels         = [32, 'minimum channel width'],
            d_max_channels     = [512, 'maximum channel width'],
            init_down_size     = [256, 'initial size to downsample to'],
            d_bottom           = [8, 'bottom width'],
            decoder_image_size = [128, 'image size to work with for decoders'],

            lr                 = [0.0002, 'learning rate'],
            betas              = [[0.5, 0.999], 'betas'],
            policy             = ['color,translation', 'policy for diffaugment'],
            ema                = [False, 'Moving average.']))
    args = parser.parse_args()
    save_args(args)

    device = get_device(not args.disable_gpu)
    amp    = not args.disable_gpu and not args.disable_amp

    # dataset
    if args.dataset == 'animeface':
        warnings.warn(f'AnimeFace original images sizes are around 100~120 pixels.')
        dataset = AnimeFace.asloader(
            args.batch_size, (args.image_size, args.min_year),
            pin_memory=not args.disable_gpu)
    elif args.dataset == 'danbooru':
        dataset = DanbooruPortrait.asloader(
            args.batch_size, (args.image_size, args.num_images),
            pin_memory=not args.disable_gpu)
    const_z = sample_nnoise((args.num_test, args.latent_dim), device)
    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs

    # model
    G = Generator(
        args.latent_dim, args.image_size, args.g_channels, args.g_max_channels,
        args.interp_size, args.image_channels, args.g_bottom, args.norm_name,
        args.transposed, args.num_sle)
    D = Discriminator(
        args.image_size, args.init_down_size, args.image_channels,
        args.d_channels, args.d_max_channels, args.norm_name,
        args.d_bottom, args.decoder_image_size)
    G.apply(init().N002)
    D.apply(init().N002)
    G.to(device)
    D.to(device)
    if args.ema:
        G_ema = Generator(
            args.latent_dim, args.image_size, args.g_channels, args.g_max_channels,
            args.interp_size, args.image_channels, args.g_bottom, args.norm_name,
            args.transposed, args.num_sle)
        G_ema.to(device)
        freeze(G_ema)
        update_ema(G, G_ema, 0., True)
    else: G_ema = None

    # optimizers
    optimizer_G = optim.Adam(G.parameters(), lr=args.lr, betas=args.betas)
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr, betas=args.betas)

    train(args,
        args.max_iters, dataset, const_z, args.latent_dim,
        G, G_ema, D, optimizer_G, optimizer_D,
        functools.partial(DiffAugment, policy=args.policy),
        device, amp, args.save)
