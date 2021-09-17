
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image
from torch.utils.data import random_split
import numpy as np

from dataset import AnimeFaceXDoG, DanbooruPortraitXDoG, to_loader
from utils import Status, add_args, save_args
from nnutils.loss import LSGANLoss, VGGLoss
from nnutils import get_device
from thirdparty.diffaugment import DiffAugment

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
            status.update(**loss_dict)
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

def main(parser):

    parser = add_args(parser,
        dict(
            num_test         = [9, 'number of image for eval'],
            sketch_channels  = [1, 'number of channels for sketch images'],
            ref_channels     = [3, 'number of channels for reference images'],
            bottom_width     = [8, 'bottom width'],
            enc_channels     = [16, 'channel width multiplier for encoder/decoder'],
            layer_per_resl   = [2, 'number of layers per resolution'],
            num_res_blocks   = [7, 'number of residual blocks in G'],
            disable_sn       = [False, 'disable spectral norm'],
            disable_bias     = [False, 'disable bias'],
            enable_scft_bias = [False, 'enable bias in scft'],
            norm_name        = ['in', 'normalization layer name'],
            act_name         = ['lrelu', 'activation function name'],
            num_layers       = [3, 'number of layers in D'],
            d_channels       = [32, 'channels_width multiplier'],
            d_lr             = [0.0002, 'learning rate for D'],
            g_lr             = [0.0001, 'learning rate for G'],
            betas            = [[0.5, 0.999], 'betas'],
            recon_lambda     = [30., 'lambda for reconstruction loss'],
            triplet_lambda   = [1., 'lambda for triplet loss'],
            margin           = [12., 'margin for triplet loss'],
            perc_lambda      = [0.01, 'lambda for percrptual loss'],
            style_lambda     = [50., 'lambda for style loss']))
    args = parser.parse_args()
    save_args(args)

    amp = not args.disable_gpu and not args.disable_amp
    device = get_device(not args.disable_gpu)

    # data
    if args.dataset == 'animeface':
        dataset = AnimeFaceXDoG(args.image_size, args.min_year)
    elif args.dataset == 'danbooru':
        dataset = DanbooruPortraitXDoG(args.image_size, args.num_images+args.num_test)
    dataset, test = random_split(dataset, [len(dataset)-args.num_test, args.num_test])
    # train
    dataset = to_loader(dataset, args.batch_size, pin_memory=not args.disable_gpu)
    # test
    test = to_loader(test, args.num_test, shuffle=False, pin_memory=False)
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
