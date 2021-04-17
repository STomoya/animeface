
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image
from torch.utils.data import random_split
import numpy as np

from ..general import XDoGAnimeFaceDataset, XDoGDanbooruPortraitDataset, to_loader
from ..general import Status, get_device, save_args
from ..gan_utils import DiffAugment
from ..gan_utils.losses import LSGANLoss, VGGLoss

from .tps import tps_transform
from .model import Generator, Discriminator, init_weight_kaiming, init_weight_xavier, init_weight_N002


l1_loss = nn.L1Loss()
def triplet_loss(anchor, negative, positive, margin):
    scale = np.sqrt(anchor[0].numel())
    B = anchor.size(0)
    an = torch.bmm(anchor.reshape(B, 1, -1), negative.reshape(B, -1, 1)) / scale
    ap = torch.bmm(anchor.reshape(B, 1, -1), positive.reshape(B, -1, 1)) / scale
    loss = F.relu(-ap+an+margin).mean()
    return loss

def train(
    max_iters, dataset, test_batch,
    G, D, optimizer_G, optimizer_D,
    color_augment, spatial_augment,
    recon_lambda, style_lambda, perc_lambda,
    triplet_lambda, margin,
    amp, device, save
):

    status = Status(max_iters)
    loss   = LSGANLoss()    
    vgg    = VGGLoss(device, p=1)
    scaler = GradScaler() if amp else None

    while status.batches_done < max_iters:
        for real, sketch in dataset:
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()

            real = real.to(device)
            sketch = sketch.to(device)

            '''Discriminator'''
            with autocast(amp):
                # augment
                real = color_augment(real)
                real_s = spatial_augment(real)
                # D(real)
                real_prob, _ = D(torch.cat([sketch, real], dim=1))
                # D(G(sketch, Is))
                fake, qk_p = G(sketch, real_s, True)
                fake_prob, _ = D(torch.cat([sketch, fake.detach()], dim=1))
                # D(G(sketch, Ir))
                _, qk_n = G(sketch, real, True)

                # loss
                # adv
                D_loss = loss.d_loss(real_prob, fake_prob)
            
            if scaler is not None:
                scaler.scale(D_loss).backward()
                scaler.step(optimizer_D)
            else:
                D_loss.backward()
                optimizer_D.step()

            '''Generator'''
            with autocast(amp):
                # D(G(sketch, Is))
                fake_prob, _ = D(torch.cat([sketch, fake], dim=1))

                # loss
                # adv
                G_loss = loss.g_loss(fake_prob)
                # reconstruction
                if recon_lambda > 0:
                    G_loss = G_loss \
                        + l1_loss(fake, real) * recon_lambda
                # style
                if style_lambda > 0:
                    G_loss = G_loss \
                        + vgg.style_loss(real, fake) * style_lambda
                # perceptual
                if perc_lambda > 0:
                    G_loss = G_loss \
                        + vgg.vgg_loss(real, fake, [0, 1, 2, 3]) * perc_lambda
                # triplet
                if triplet_lambda > 0:
                    G_loss = G_loss \
                        + triplet_loss(qk_p[0], qk_n[1], qk_p[1], margin) * triplet_lambda

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
                    image = G(test_batch[1], test_batch[0])
                    G.train()
                image_grid = _image_grid(test_batch[1], test_batch[0], image)
                save_image(
                    image_grid, f'implementations/SCFT/result/{status.batches_done}.jpg',
                    nrow=3*3, normalize=True, value_range=(-1, 1)
                )
                torch.save(G.state_dict(), f'implementations/SCFT/result/G_{status.batches_done}.pt')
            
            save_image(fake, 'running.jpg', nrow=5, normalize=True, value_range=(-1, 1))

            # updates
            loss_dict = dict(
                G=G_loss.item() if not torch.any(torch.isnan(G_loss)) else 0,
                D=D_loss.item() if not torch.any(torch.isnan(D_loss)) else 0
            )
            status.update(loss_dict)
            if scaler is not None:
                scaler.update()
            
            if status.batches_done == max_iters:
                break

    status.plot()

def _image_grid(src, ref, gen):
    srcs = src.expand(src.size(0), 3, *src.size()[2:]).chunk(src.size(0), dim=0)
    refs = ref.chunk(ref.size(0), dim=0)
    gens = gen.chunk(gen.size(0), dim=0)

    images = []
    for src, ref, gen in zip(srcs, refs, gens):
        images.extend([src, ref, gen])
    return torch.cat(images, dim=0)

def add_arguments(parser):
    parser.add_argument('--num-test', default=9, type=int, help='number of images for eval')

    parser.add_argument('--sketch-channels', default=1, type=int, help='number of channels in sketch images')
    parser.add_argument('--ref-channels', default=3, type=int, help='number of channels in reference images')
    parser.add_argument('--bottom-width', default=8, type=int, help='bottom width in model')
    parser.add_argument('--enc-channels', default=16, type=int, help='channel width multiplier for encoder/decoder')
    parser.add_argument('--layer-per-resl', default=2, type=int, help='number of layers per resolution')
    parser.add_argument('--num-res-blocks', default=7, type=int, help='number of residual blocks in G')
    parser.add_argument('--disable-sn', default=False, action='store_true', help='disable spectral norm')
    parser.add_argument('--disable-bias', default=False, action='store_true', help='disable bias')
    parser.add_argument('--enable-scft-bias', default=False, action='store_true', help='enable bias in SCFT (attention)')
    parser.add_argument('--norm-name', default='in', choices=['in', 'bn'], help='normalization layer name')
    parser.add_argument('--act-name', default='lrelu', choices=['lrelu', 'relu'], help='activation function name')
    parser.add_argument('--num-layers', default=3, type=int, help='number of layers in D')
    parser.add_argument('--d-channels', default=32, type=int, help='channel width multiplier for D')

    parser.add_argument('--d-lr', default=0.0002, type=float, help='learning rate for D')
    parser.add_argument('--g-lr', default=0.0001, type=float, help='learning rate for G')
    parser.add_argument('--betas', default=[0.5, 0.999], type=float, nargs=2, help='betas')
    parser.add_argument('--recon-lambda', default=30., type=float, help='lambda for reconstruction loss')
    parser.add_argument('--triplet-lambda', default=1., type=float, help='lambda for triplet loss')
    parser.add_argument('--margin', default=12., type=float, help='margin for triplet loss')
    parser.add_argument('--perc-lambda', default=0.01, type=float, help='lambda for perceptual loss')
    parser.add_argument('--style-lambda', default=50., type=float, help='lambda for style loss')
    return parser

def main(parser):
    
    parser = add_arguments(parser)
    args = parser.parse_args()
    save_args(args)

    amp = not args.disable_gpu and not args.disable_amp
    device = get_device(not args.disable_gpu)

    # data
    if args.dataset == 'animeface':
        dataset = XDoGAnimeFaceDataset(args.image_size, args.min_year)
    elif args.dataset == 'danbooru':
        dataset = XDoGDanbooruPortraitDataset(args.image_size, num_images=args.num_images+args.num_test)
    dataset, test = random_split(dataset, [len(dataset)-args.num_test, args.num_test])
    # train
    dataset = to_loader(dataset, args.batch_size, use_gpu=not args.disable_gpu)
    # test
    test = to_loader(test, args.num_test, shuffle=False, use_gpu=False)
    test_batch = next(iter(test))
    test_batch = (test_batch[0].to(device), test_batch[1].to(device))

    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs

    # model
    G = Generator(
        args.image_size, args.sketch_channels, args.ref_channels,
        args.bottom_width, args.enc_channels, args.layer_per_resl, args.num_res_blocks,
        not args.disable_sn, not args.disable_bias, args.enable_scft_bias, args.norm_name, args.act_name
    )
    D = Discriminator(
        args.image_size, args.sketch_channels+args.ref_channels, args.num_layers, args.d_channels,
        not args.disable_sn, not args.disable_bias, args.norm_name, args.act_name
    )
    G.apply(init_weight_N002)
    D.apply(init_weight_N002)
    G.to(device)
    D.to(device)

    # optimizers
    optimizer_G = optim.Adam(G.parameters(), lr=args.g_lr, betas=args.betas)
    optimizer_D = optim.Adam(D.parameters(), lr=args.d_lr, betas=args.betas)

    # augmentations
    color_augment = functools.partial(
        DiffAugment, policy='color'
    )
    spatial_augment = tps_transform


    train(
        args.max_iters, dataset, test_batch,
        G, D, optimizer_G, optimizer_D,
        color_augment, spatial_augment,
        args.recon_lambda, args.style_lambda, args.perc_lambda,
        args.triplet_lambda, args.margin,
        amp, device, args.save
    )
