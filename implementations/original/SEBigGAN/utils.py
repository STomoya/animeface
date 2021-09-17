
import functools

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torch.cuda.amp import autocast, GradScaler

from .model import Generator, Discriminator, init_weight_ortho

from dataset import AnimeFace, DanbooruPortrait
from utils import Status, add_args, save_args
from nnutils import sample_nnoise, update_ema, get_device
from nnutils.loss import HingeLoss
from thirdparty.diffaugment import DiffAugment

consistency = nn.MSELoss()

def train(
    max_iters, dataset, sampler, noise, z_dim, const_z,
    G, G_ema, D, optimizer_G, optimizer_D,
    policy, real_lambda, fake_lambda, latent_d_lambda, latent_g_lambda,
    device, amp, save=1000
):
    status = Status(max_iters)
    scaler = GradScaler() if amp else None
    loss = HingeLoss()

    if not policy == '':
        augment = functools.partial(DiffAugment, policy=policy)
    else:
        augment = lambda x: x

    while status.batches_done < max_iters:
        for image in dataset:
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()

            real = image.to(device)
            z = sampler((real.size(0), z_dim))
            n = noise((real.size(0), z_dim))

            '''Discriminator'''
            with autocast(amp):
                # D(x), D(T(x))
                real_aug = augment(real)
                real_prob = D(real)
                real_aug_prob = D(real_aug)
                # D(G(z)), D(T(G(z))), D(G(T(z)))
                fake = G(z)
                fake_n = G(z + n)
                fake_aug = augment(fake)
                fake_prob = D(fake.detach())
                fake_aug_prob = D(fake_aug.detach())
                fake_n_prob = D(fake_n.detach())
                # loss
                D_loss = loss.d_loss(real_prob, fake_prob)
                if real_lambda > 0:
                    D_loss = D_loss + consistency(real_aug_prob, real_prob) * real_lambda
                if fake_lambda > 0:
                    D_loss = D_loss + consistency(fake_aug_prob, fake_prob) * fake_lambda
                if latent_d_lambda > 0:
                    D_loss = D_loss + consistency(fake_n_prob, fake_prob) * latent_d_lambda

            if scaler is not None:
                scaler.scale(D_loss).backward()
                scaler.step(optimizer_D)
            else:
                D_loss.backward()
                optimizer_D.step()

            '''Generator'''
            with autocast(amp):
                # D(G(z))
                fake_prob = D(fake_aug)
                # loss
                G_loss = loss.g_loss(fake_prob)
                if latent_g_lambda > 0:
                    G_loss = G_loss - consistency(fake, fake_n) * latent_g_lambda

            if scaler is not None:
                scaler.scale(G_loss).backward()
                scaler.step(optimizer_G)
            else:
                G_loss.backward()
                optimizer_G.step()

            update_ema(G, G_ema)

            # save
            if status.batches_done % save == 0:
                with torch.no_grad():
                    images = G_ema(const_z)
                save_image(images, f'implementations/original/SEBigGAN/result/{status.batches_done}.jpg', nrow=4, normalize=True, value_range=(-1, 1))
                torch.save(G_ema.state_dict(), f'implementations/original/SEBigGAN/result/G_{status.batches_done}.pt')
            save_image(fake, './running.jpg', nrow=8, normalize=True, value_range=(-1, 1))

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

def add_arguments(parser):
    parser.add_argument('--channels', default=64, type=int, help='channel width multiplier')
    parser.add_argument('--deep', default=False, action='store_true', help='use deep model')
    parser.add_argument('--z-dim', default=120, type=int, help='dimension of latent')
    parser.add_argument('--g-disable-sn', default=False, action='store_true', help='do not use spectral normalization in G')
    parser.add_argument('--g-att-name', default='se', choices=['se'], help='attention type name for G')
    parser.add_argument('--g-act-name', default='relu', choices=['relu', 'lrelu'], help='activation function name for G')
    parser.add_argument('--g-norm-name', default='bn', choices=['bn', 'in'], help='normalization layer name for G')
    parser.add_argument('--d-disable-sn', default=False, action='store_true', help='do not use spectral normalization in D')
    parser.add_argument('--d-att-name', default='se', choices=['se'], help='attention type name for D')
    parser.add_argument('--d-act-name', default='relu', choices=['relu', 'lrelu'], help='activation function name for D')
    parser.add_argument('--g-lr', default=0.00005, type=float, help='learning rate for G')
    parser.add_argument('--d-lr', default=0.0002, type=float, help='learning rate for D')
    parser.add_argument('--beta1', default=0., type=float, help='beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta2')
    parser.add_argument('--real-lambda', default=10., type=float, help='lambda for consistency regularization on real images')
    parser.add_argument('--fake-lambda', default=10., type=float, help='lambda for consistency regularization on fake images')
    parser.add_argument('--latent-d-lambda', default=5., type=float, help='lambda for latent consistency regularization on D')
    parser.add_argument('--latent-g-lambda', default=0.5, type=float, help='lambda for latent consistency regularization on G')
    parser.add_argument('--noise-sigma', default=0.03, type=float, help='sigma for added noise in latent consistency regularization')
    parser.add_argument('--policy', default='translation', type=str, help='policy for DiffAugment')
    return parser

def main(parser):

    # parser = add_arguments(parser)
    parser = add_args(parser,
        dict(
            channels        = [64, 'channel_width, multiplier'],
            deep            = [False, 'deep model'],
            z_dim           = [120, 'input latent dim'],
            g_disable_sn    = [False, 'disable spectral norm'],
            g_att_name      = ['se', 'attention name'],
            g_act_name      = ['relu', 'activation function name'],
            g_norm_name     = ['bn', 'normalization layer name'],
            d_disable_sn    = [False, 'disable spectral norm'],
            d_att_name      = ['se', 'attention name'],
            d_act_name      = ['relu', 'activation function name'],

            g_lr            = [0.00005, 'learning rate for G'],
            d_lr            = [0.0002, 'learning rate for D'],
            betas           = [[0., 0.999], 'betas'],

            # when real_lambda >  0 and fake_lambda <= 0 and latent* <= 0 : CR
            # when real_lambda >  0 and fake_lambda >  0 and latent* <= 0 : bCR
            # when real_lambda <= 0 and fake_lambda <= 0 and latent* >  0 : zCR
            # when real_lambda >  0 and fake_lambda >  0 and latent* >  0 : ICR
            real_lambda     = [10., 'lambda for consistency regularization on real'],
            fake_lambda     = [10., 'lambda for consistency regularization on fake'],
            latent_d_lambda = [5., 'lambda for latent consistency regularization on D'],
            latent_g_lambda = [0.5, 'lambda for latent consistency regularization on G'],
            noise_sigma     = [0.03, 'sigma for added noise in latent consistency regularization'],

            policy          = ['color,translation', 'policy for diffaugmnet']))
    args = parser.parse_args()
    args.name = args.name.replace('.', '/')
    save_args(args)

    g_use_sn = not args.g_disable_sn
    d_use_sn = not args.d_disable_sn

    amp = not args.disable_amp
    device = get_device(not args.disable_gpu)

    # models
    G = Generator(
        args.image_size, args.z_dim, args.deep, args.channels,
        g_use_sn, args.g_att_name, args.g_act_name, args.g_norm_name)
    G_ema = Generator(
        args.image_size, args.z_dim, args.deep, args.channels,
        g_use_sn, args.g_att_name, args.g_act_name, args.g_norm_name)
    D = Discriminator(
        args.image_size, args.deep, args.channels,
        d_use_sn, args.d_att_name, args.d_act_name)
    G.apply(init_weight_ortho)
    D.apply(init_weight_ortho)

    update_ema(G, G_ema, 0)
    G.to(device)
    G_ema.to(device)
    D.to(device)

    # dataset
    if args.dataset == 'animeface':
        dataset = AnimeFace.asloader(
            args.batch_size, (args.image_size, args.min_year),
            pin_memory=not args.disable_gpu)
    elif args.dataset == 'danbooru':
        dataset = DanbooruPortrait.asloader(
            args.batch_size, (args.image_size, args.num_images),
            pin_memory=not args.disable_gpu)

    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs

    sampler = functools.partial(sample_nnoise, device=device)
    noise   = functools.partial(sample_nnoise, std=args.noise_sigma, device=device)
    const_z = sample_nnoise((16, args.z_dim), device=device)

    # optimizers
    optimizer_G = optim.Adam(G.parameters(), lr=args.g_lr, betas=args.betas)
    optimizer_D = optim.Adam(D.parameters(), lr=args.d_lr, betas=args.betas)

    train(
        args.max_iters, dataset, sampler, noise, args.z_dim, const_z,
        G, G_ema, D, optimizer_G, optimizer_D,
        args.policy, args.real_lambda, args.fake_lambda, args.latent_d_lambda, args.latent_g_lambda,
        device, amp, args.save
    )
