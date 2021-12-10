
import warnings
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from dataset import AnimeFace, DanbooruPortrait
from utils import Status, save_args, add_args
from nnutils import MiniAccelerator, get_device, freeze, update_ema, sample_nnoise
from nnutils.loss import NonSaturatingLoss, r1_regularizer
from thirdparty.diffaugment import DiffAugment

from ..StyleGAN3.model import Generator, Discriminator
from .model import APA

def train(args,
    max_iters, dataset, latent_dim, const_z,
    G, G_ema, D, optimizer_G, optimizer_D,
    gp_lambda, gp_every, augment, apa_kwargs,
    device, amp, save=1000,
    log_file=None, log_interval=10
):

    status = Status(
        max_iters, log_file is None,
        log_file, log_interval, __name__)
    adv_fn = NonSaturatingLoss()
    gp_fn  = r1_regularizer()

    apa_augment = APA(**apa_kwargs)

    accelerator = MiniAccelerator(amp, True, device)
    G, G_ema, D, optimizer_G, optimizer_D, dataset, apa_augment = accelerator.prepare(
        G, G_ema, D, optimizer_G, optimizer_D, dataset, apa_augment)

    status.log_training(args, G, D)

    while not status.is_end():
        for real in dataset:
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()

            z = sample_nnoise((real.size(0), latent_dim), device=accelerator.device)

            # generate
            with accelerator.autocast():
                fake = G(z)
                real_aug = augment(real)
                fake_aug = augment(fake)

                real_aug_apa = apa_augment(real_aug, fake_aug.detach())

            '''D'''
            with accelerator.autocast():
                # D(x)
                real_prob = D(real_aug_apa)
                # D(G(z))
                fake_prob = D(fake_aug.detach())

                # loss
                adv_loss = adv_fn.d_loss(real_prob, fake_prob)
                gp_loss = 0
                if status.batches_done % gp_every == 0:
                    gp_loss = gp_fn(real_aug_apa.detach(), D, accelerator.scaler) \
                        + gp_lambda
                D_loss = adv_loss + gp_loss

            '''APA'''
            apa_augment.update_p(real_prob=real_prob)

            accelerator.backward(D_loss)
            optimizer_D.step()

            '''G'''
            with accelerator.autocast():
                # D(G(z))
                fake_prob = D(fake_aug)

                # loss
                G_loss = adv_fn.g_loss(fake_prob)

            accelerator.backward(G_loss)
            optimizer_G.step()

            update_ema(G, G_ema, copy_buffers=True)

            if status.batches_done % save == 0:
                with torch.no_grad():
                    images = G_ema(const_z)
                save_image(
                    images, f'implementations/APA/result/{status.batches_done}.jpg',
                    nrow=int(images.size(0)**0.5), normalize=True, value_range=(-1, 1))
                torch.save(
                    G_ema.state_dict(),
                    f'implementations/APA/result/G_{status.batches_done}.pth')
            save_image(
                fake, 'running.jpg', nrow=int(fake.size(0)**0.5),
                normalize=True, value_range=(-1, 1))

            status.print(str(apa_augment.p) + f', {apa_augment._p_delta}')
            status.update(G=G_loss.item(), D=D_loss.item())
            accelerator.update()

            if status.is_end():
                break

    status.plot_loss()

def main(parser):

    parser = add_args(parser,
        dict(
            num_test        = [16, 'number of images for eval'],

            latent_dim      = [512, 'input latent dimension'],

            apa_interval    = [4, 'interval to update p.'],
            apa_target_kimg = [500, 'number of k images to expect apa probability to reach 1.'],
            apa_threshold   = [0.6, 'threshold to add or subtract from p.'],
            disable_apa     = [False, 'disable APA. For comparing results.'],

            lr              = [0.0025, 'learning rate'],
            map_lr_scale    = [0.01, 'scale learning rate for mapping network'],
            betas           = [[0., 0.99], 'betas'],
            gp_lambda       = [10., 'lambda for gradient penalty'],
            gp_every        = [16, 'calc gradient penalty every'],
            policy          = ['color,translation', 'policy for diffaugment']))
    args = parser.parse_args()
    save_args(args)

    device = get_device(not args.disable_gpu)
    amp    = not args.disable_amp and not args.disable_gpu

    # dataset
    if args.dataset == 'animeface':
        dataset = AnimeFace.asloader(
            args.batch_size, (args.image_size, args.min_year),
            pin_memory=not args.disable_gpu)
    if args.dataset == 'danbooru':
        dataset = DanbooruPortrait.asloader(
            args.batch_size, (args.image_size, args.num_images),
            pin_memory=not args.disable_gpu)

    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs

    const_z = sample_nnoise((args.num_test, args.latent_dim), device=device)

    # model
    # minimal controllable arguments
    G = Generator(args.image_size, args.latent_dim)
    G_ema = Generator(args.image_size, args.latent_dim)
    D = Discriminator(args.image_size)
    freeze(G_ema)
    update_ema(G, G_ema, 0., True)

    # optimizers
    optimizer_G = optim.Adam(
        [{'params': G.synthesis.parameters()},
         {'params': G.map.parameters(), 'lr': args.lr*args.map_lr_scale}],
        lr=args.lr, betas=args.betas)
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr, betas=args.betas)

    apa_kwargs = dict(
        interval=args.apa_interval, target_kimg=args.apa_target_kimg,
        threshold=args.apa_threshold, batch_size=args.batch_size,
        disable=args.disable_apa)

    augment = partial(DiffAugment, policy=args.policy)

    train(args,
        args.max_iters, dataset, args.latent_dim, const_z,
        G, G_ema, D, optimizer_G, optimizer_D,
        args.gp_lambda, args.gp_every, augment, apa_kwargs,
        device, amp, args.save,
        args.log_file, args.log_interval)
