
import random
import functools

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image

from ..general import YearAnimeFaceDataset, DanbooruPortraitDataset, to_loader
from ..gan_utils import sample_nnoise, DiffAugment, update_ema
from ..general import get_device, Status, save_args
from ..gan_utils.losses import GANLoss, GradPenalty

from .model import Generator, Discriminator, init_weight_N01

def train(
    max_iters, dataset, sampler, const_z, latent_dim,
    G, G_ema, D, optim_G, optim_D,
    gp_lambda, d_k, augment,
    scales, scale_probs, image_size, mix_prob,
    device, amp, save=1000
):

    status = Status(max_iters)
    loss   = GANLoss()
    gp     = GradPenalty()
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
                        + gp.r1_regularizer(real, D, scaler) * gp_lambda

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
            status.update(loss_dict)
            if scaler is not None:
                scaler.update()

            if status.batches_done == max_iters:
                break

    status.plot()

def chose_scale(scales, scale_probs):
    assert len(scales) == len(scale_probs)
    r = random.random()
    for i, prob in enumerate(scale_probs):
        if r < prob:
            break
        r -= prob
    return scales[i]

def add_arguments(parser):
    parser.add_argument('--num-test', default=16, type=int)
    # G
    parser.add_argument('--g-bottom', default=4, type=int)
    parser.add_argument('--no-spe', default=False, action='store_true')
    parser.add_argument('--latent-dim', default=512, type=int)
    parser.add_argument('--in-channels', default=512, type=int)
    parser.add_argument('--style-dim', default=512, type=int)
    parser.add_argument('--out-channels', default=3, type=int)
    parser.add_argument('--g-channels', default=32, type=int)
    parser.add_argument('--g-max-channels', default=512, type=int)
    parser.add_argument('--pad', default=False, action='store_true')
    parser.add_argument('--map-num-layers', default=8, type=int)
    parser.add_argument('--no-pixelnorm', default=False, action='store_true')
    parser.add_argument('--filter-size', default=4, type=int)
    parser.add_argument('--g-act-name', default='lrelu')
    # D
    parser.add_argument('--d-bottom', default=2, type=int)
    parser.add_argument('--d-channels', default=32, type=int)
    parser.add_argument('--d-max-channels', default=512, type=int)
    parser.add_argument('--mbsd-groups', default=4, type=int)
    parser.add_argument('--no-gap', default=False, action='store_true')
    parser.add_argument('--d-act-name', default='lrelu')

    # training
    parser.add_argument('--map-lr', default=0.01, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--betas', default=[0., 0.99], type=float, nargs=2)
    parser.add_argument('--gp-lambda', default=5., type=float)
    parser.add_argument('--d-k', default=16, type=int)
    parser.add_argument('--scales', default=[1., 1.5, 2.])
    parser.add_argument('--scale-probs', default=[0.5, 0.25, 0.25])
    parser.add_argument('--mix-prob', default=0.5, type=float)
    parser.add_argument('--policy', default='color,translation')
    return parser

def main(parser):

    parser = add_arguments(parser)
    args = parser.parse_args()
    save_args(args)

    device = get_device(not args.disable_gpu)
    amp = not args.disable_amp and not args.disable_gpu

    # data
    if args.dataset == 'animeface':
        dataset = YearAnimeFaceDataset(args.image_size, args.min_year)
    elif args.dataset == 'danbooru':
        dataset = DanbooruPortraitDataset(args.image_size, num_images=args.num_images)
    dataset = to_loader(dataset, args.batch_size)

    sampler = functools.partial(sample_nnoise, device=device)
    const_z = sampler((args.num_test, args.latent_dim))

    # model
    G = Generator(
        args.image_size, args.g_bottom, not args.no_spe,
        args.latent_dim, args.in_channels, args.style_dim,
        args.out_channels, args.g_channels,
        args.g_max_channels, args.pad, args.map_num_layers, args.map_lr,
        not args.no_pixelnorm, args.filter_size, args.g_act_name)
    G_ema = Generator(
        args.image_size, args.g_bottom, not args.no_spe,
        args.latent_dim, args.in_channels, args.style_dim,
        args.out_channels, args.g_channels,
        args.g_max_channels, args.pad, args.map_num_layers, args.map_lr,
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
