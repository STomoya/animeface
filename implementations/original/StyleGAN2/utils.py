
from functools import partial
import random

import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image

from dataset import AnimeFace, DanbooruPortrait
from utils import Status, add_args, save_args
from nnutils import get_device, sample_nnoise, update_ema, init, freeze
from nnutils.loss import NonSaturatingLoss, r1_regularizer
from thirdparty.diffaugment import DiffAugment

from .model import Generator, Discriminator
from .model import (
    ModulatedConv2d,
    EqualizedLinearAct,
    EqualizedConv2dAct)
from .dataset import (
    XDoG,
    AnimePhoto,
    AnimePhotoSeparate)

def train(
    max_iters, dataset, latent_dim, const_input,
    G, G_ema, D, optim_G, optim_D,
    gp_lambda, gp_every, mix_prob, mix_max_z,
    augment,
    device, amp, save
):
    if mix_prob > 0:
        assert mix_max_z >= 2

    status = Status(max_iters)
    loss   = NonSaturatingLoss()
    r1     = r1_regularizer()
    scaler = GradScaler() if amp else None

    while status.batches_done < max_iters:
        for image in dataset:
            # reset grad
            optim_G.zero_grad()
            optim_D.zero_grad()

            # real
            real = image.to(device)
            # sample z
            # mixing regularization
            if random.random() < mix_prob:
                z_size = random.randint(2, mix_max_z+1)
                size = (real.size(0), z_size, latent_dim)
            else:
                size = (real.size(0), latent_dim)
            z = sample_nnoise(size, device)

            with autocast(amp):
                # generate
                fake, _ = G(z)
                # augment
                real_aug = augment(real)
                fake_aug = augment(fake)

                # D(real)
                real_prob = D(real_aug)
                # D(G(z))
                fake_prob = D(fake_aug.detach())

                # loss
                D_loss = loss.d_loss(real_prob, fake_prob)
                if status.batches_done % gp_every == 0 \
                    and gp_lambda > 0:
                    D_loss = D_loss + r1(real, D, scaler) * gp_lambda

            # optimize D
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

            # optimize G
            if scaler is not None:
                scaler.scale(G_loss).backward()
                scaler.step(optim_G)
            else:
                D_loss.backward()
                optim_D.step()

            # update ema model
            update_ema(G, G_ema)

            # save
            if status.batches_done % save == 0:
                with torch.no_grad():
                    images, _ = G_ema(*const_input)
                save_image(
                    images, f'implementations/original/StyleGAN2/result/{status.batches_done}.jpg',
                    normalize=True, value_range=(-1, 1))
                torch.save(
                    G_ema.state_dict(),
                    f'implementations/original/StyleGAN2/result/G_{status.batches_done}.pt')
            save_image(fake, 'running.jpg', normalize=True, value_range=(-1, 1))

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


def main(parser):

    # defualt
    parser = add_args(parser,
        dict(
            num_test        = [16, 'number of test samples'],
            const_noise     = [False, 'const noise when test'],
            image_channels  = [3, 'image channels'],
            # G
            style_dim       = [512, 'style code dimension'],
            const_channels  = [512, 'const input channels'],
            g_channels      = [32, 'channel width mutiplier'],
            g_max_channels  = [512, 'maximum channel width'],
            g_bottom        = [4, 'bottom width'],
            latent_dim      = [512, 'input latent dim'],
            map_num_layers  = [8, 'number of layers in mapping'],
            no_pixel_norm   = [False, 'disable pixel norm'],
            g_act_name      = ['lrelu', 'activation function name'],
            # D
            d_channels      = [32, 'channel width multiplier'],
            d_max_channels  = [512, 'maximum channel width'],
            mbsd_group_size = [4, 'mini batch stddev group size'],
            mbsd_channels   = [1, 'mini batch stddev channels'],
            d_bottom        = [4, 'bottom width'],
            d_act_name      = ['lrelu', 'activation function name'],
            # shared
            filter_size     = [4, 'binomial filter size'],
            kernel_size     = [3, 'kernel size'],
            gain            = [1., 'gain for ELR coefficient'],

            # training
            lr              = [0.001, 'learning rate'],
            map_lr_scale    = [0.01, 'learning rate scale for mapping'],
            betas           = [[0., 0.99], 'betas'],
            gp_lambda       = [10., 'lambda of r1 regularization'],
            gp_every        = [16, 'calc r1 every'],
            mix_prob        = [0.9, 'mixing regularization probability'],
            mix_max_z       = [2, 'maximum input z'],
            policy          = ['color,translation', 'policy for DiffAugment']))

    # experiments
    parser = add_args(parser,
        dict(
            xdog            = [False, 'XDoG'],
            with_photo      = [False, 'train model with anime+photos'],
            separate        = [False, 'train mode with anime+photo separately']))

    args = parser.parse_args()
    args.name = args.name.replace('.', '/')
    save_args(args)

    device = get_device(not args.disable_gpu)
    amp = not args.disable_amp and not args.disable_gpu

    # dataset
    if args.with_photo and args.separate:
        dataset = AnimePhotoSeparate.asloader(
            args.batch_size, (args.image_size, args.dataset),
            pin_memory=not args.disable_gpu)
        raise Exception('Training for anime+photo+separate is not available right now')
    elif args.with_photo:
        dataset = AnimePhoto.asloader(
            args.batch_size, (args.image_size, args.dataset),
            pin_memory=not args.disable_gpu)
    elif args.xdog:
        dataset = XDoG.asloader(
            args.batch_size, (args.image_size, args.dataset, args.num_images),
            pin_memory=not args.disable_gpu)
        args.image_channels = 1
    elif args.dataset == 'animeface':
        dataset = AnimeFace.asloader(
            args.batch_size, (args.image_size, args.min_year),
            pin_memory=not args.disable_gpu)
    elif args.dataset == 'danbooru':
        dataset = DanbooruPortrait.asloader(
            args.batch_size, (args.image_size, args.num_images),
            pin_memory=not args.disable_gpu)

    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs

    # model
    # G
    G = Generator(
        args.image_size, args.style_dim, args.const_channels,
        args.image_channels, args.g_channels, args.g_max_channels,
        args.g_bottom, args.kernel_size, args.filter_size,
        args.latent_dim, args.map_num_layers, not args.no_pixel_norm,
        args.g_act_name, args.gain)
    # clone G model
    G_ema = Generator(
        args.image_size, args.style_dim, args.const_channels,
        args.image_channels, args.g_channels, args.g_max_channels,
        args.g_bottom, args.kernel_size, args.filter_size,
        args.latent_dim, args.map_num_layers, not args.no_pixel_norm,
        args.g_act_name, args.gain)
    # freeze clone
    freeze(G_ema)
    # D
    D = Discriminator(
        args.image_size, args.image_channels, args.d_channels, args.d_max_channels,
        args.kernel_size, args.mbsd_group_size, args.mbsd_channels,
        args.d_bottom, args.filter_size, args.d_act_name, args.gain)
    # init
    initobj = init((ModulatedConv2d, EqualizedConv2dAct, EqualizedLinearAct), ('weight', ))
    G.apply(initobj.N01)
    D.apply(initobj.N01)
    update_ema(G, G_ema, 0.)
    # send to device
    G.to(device)
    G_ema.to(device)
    D.to(device)
    # build plugin before training with dummy input
    G(sample_nnoise((2, args.latent_dim), device))

    # test
    const_z = sample_nnoise((args.num_test, args.latent_dim), device)
    const_input = (const_z, 'random')
    if args.const_noise:
        const_input = (const_z, 'const')

    # optimizer
    optim_G = optim.Adam(
        [{'params': G.synthesis.parameters()},
         {'params': G.mapping.parameters(), 'lr': args.lr*args.map_lr_scale}],
        lr=args.lr, betas=args.betas)
    optim_D = optim.Adam(D.parameters(), lr=args.lr, betas=args.betas)

    augment = partial(DiffAugment, policy=args.policy)

    train(
        args.max_iters, dataset, args.latent_dim, const_input,
        G, G_ema, D, optim_G, optim_D,
        args.gp_lambda, args.gp_every,
        args.mix_prob, args.mix_max_z, augment,
        device, amp, args.save)
