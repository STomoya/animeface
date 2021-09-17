
import functools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image

from utils import Status, save_args, add_args
from nnutils import get_device
from nnutils.loss import LSGANLoss, VGGLoss
from thirdparty.diffaugment import DiffAugment
from dataset import to_loader

from .dataset import AnimeFace, Danbooru
from .model import Generator, Discriminator, init_weight_N002, init_weight_kaiming, init_weight_xavier

l1_loss = nn.L1Loss()

def train(
    max_iters, dataset, test_batch,
    G, D, optimizer_G, optimizer_D,
    recon_lambda, style_lambda, vgg_lambda, content_lambda,
    color_augment,
    amp, device, save
):

    status = Status(max_iters)
    loss   = LSGANLoss()
    vgg    = VGGLoss(device, p=1)
    scaler = GradScaler() if amp else None

    while status.batches_done < max_iters:
        for rgb, gray in dataset:
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()

            rgb = rgb.to(device)
            gray = gray.to(device)

            '''Discriminator'''
            with autocast(amp):
                real = color_augment(rgb)
                # D(real)
                real_prob, _ = D(torch.cat([gray, real], dim=1))
                # D(G(gray, rgb))
                fake = G(gray, real)
                fake_prob, _ = D(torch.cat([gray, fake.detach()], dim=1))

                # loss
                D_loss = loss.d_loss(real_prob, fake_prob)

            if scaler is not None:
                scaler.scale(D_loss).backward()
                scaler.step(optimizer_D)
            else:
                D_loss.backward()
                optimizer_D.step()

            '''Generator'''
            with autocast(amp):
                # D(G(gray, rgb))
                fake_prob, _ = D(torch.cat([gray, fake], dim=1))

                # loss
                G_loss = loss.g_loss(fake_prob)
                if recon_lambda > 0:
                    G_loss += l1_loss(fake, real) * recon_lambda
                if style_lambda > 0:
                    G_loss += vgg.style_loss(real, fake) * style_lambda
                if vgg_lambda > 0:
                    G_loss += vgg.vgg_loss(real, fake) * vgg_lambda
                if content_lambda > 0:
                    G_loss += vgg.content_loss(gray.repeat(1, 3, 1, 1), fake) * vgg_lambda

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
                    image_grid, f'implementations/original/EDCNN/result/{status.batches_done}.jpg',
                    nrow=3*3, normalize=True, value_range=(-1, 1))
                torch.save(G.state_dict(), f'implementations/original/EDCNN/result/G_{status.batches_done}.pt')
            save_image(fake, 'running.jpg', nrow=4, normalize=True, value_range=(-1, 1))

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

    status.plot_loss()

def _image_grid(src, ref, gen):
    _split = lambda x: x.chunk(x.size(0), dim=0)
    srcs = _split(src.expand(src.size(0), 3, src.size(2), src.size(3)))
    refs = _split(ref)
    gens = _split(gen)

    images = []
    for src, ref, gen in zip(srcs, refs, gens):
        images.extend([src, ref, gen])
    return torch.cat(images, dim=0)

def main(parser):

    parser = add_args(parser,
        dict(
            num_test                = [6, 'number of test samples'],
            gray_channels           = [1, 'number of channels in gray images'],
            ref_channels            = [3, 'number of channels in reference images'],
            channels                = [32, 'channel width mutiplier'],
            style_dim               = [128, 'dimension of style code'],
            se_blocks_per_resl      = [1, 'resblocks per resolution in style encoder'],
            num_res_blocks          = [5, 'number of resblocks'],
            disable_sobel           = [False, 'disable sobel conv2d'],
            disable_learnable_sobel = ['disable learnability'],
            e_conv_per_resl         = [2, 'number of conv per resolution in en/decoder'],
            disable_sn              = [False, 'disable spectral norm'],
            disable_bias            = [False, 'disable bias'],
            norm_name               = ['in', 'normalization layer name'],
            act_name                = ['lrelu', 'activation function name'],
            bottom_width            = [8, 'bottom width'],
            num_layers              = [3, 'number of layer in D'],
            init_func               = ['N002', 'one of "N002", "xavier", "kaiming"'],
            lr                      = [0.0002, 'learning rate'],
            betas                   = [[0.5, 0.999], 'betas'],
            recon_lambda            = [10., 'lambda for reconstruction loss'],
            style_lambda            = [50., 'lambda for style loss'],
            vgg_lambda              = [10., 'lambda for vgg loss'],
            content_lambda          = [0., 'lambda for content loss'],
            ttur                    = [False, 'use TTUR']))
    args = parser.parse_args()
    args.name = args.name.replace('.', '/')
    save_args(args)

    amp = not args.disable_amp and not args.disable_gpu
    device = get_device(not args.disable_gpu)

    # dataset
    if args.dataset == 'animeface':
        dataset = AnimeFace(args.image_size)
    elif args.dataset == 'danbooru':
        dataset = Danbooru(args.image_size, num_images=args.num_images+args.num_test)
    dataset, test = random_split(dataset, [len(dataset)-args.num_test, args.num_test], generator=torch.Generator().manual_seed(1234))
    ## training dataset
    dataset = to_loader(dataset, args.batch_size)
    ## test batch
    test    = to_loader(test, args.num_test, shuffle=False, pin_memory=False)
    test_batch = next(iter(test))
    test_batch = (test_batch[0].to(device), test_batch[1].to(device))

    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs

    # model
    G = Generator(
        args.image_size, args.gray_channels, args.ref_channels,
        args.channels, args.style_dim, args.bottom_width,
        args.se_blocks_per_resl, args.num_res_blocks,
        not args.disable_sobel, not args.disable_learnable_sobel,
        args.e_conv_per_resl, not args.disable_sn, not args.disable_bias,
        args.norm_name, args.act_name
    )
    D = Discriminator(
        args.image_size, args.ref_channels+args.gray_channels, args.num_layers,
        args.channels, not args.disable_sn, not args.disable_bias,
        args.norm_name, args.act_name
    )
    if args.init_func == 'N002':
        G.apply(init_weight_N002)
        D.apply(init_weight_N002)
    elif args.init_func == 'xavier':
        G.apply(init_weight_xavier)
        D.apply(init_weight_xavier)
    elif args.init_func == 'kaiming':
        G.apply(init_weight_kaiming)
        D.apply(init_weight_kaiming)
    else:
        raise Exception('No such init function')
    G.to(device)
    D.to(device)

    # optimizers
    if args.ttur:
        g_lr = args.lr / 2
        d_lr = args.lr * 2
    else: g_lr, d_lr = args.lr, args.lr
    optimizer_G = optim.Adam(G.parameters(), lr=g_lr, betas=args.betas)
    optimizer_D = optim.Adam(D.parameters(), lr=d_lr, betas=args.betas)

    # augment
    color_augment = functools.partial(
        DiffAugment, policy='color'
    )

    train(
        args.max_iters, dataset, test_batch,
        G, D, optimizer_G, optimizer_D,
        args.recon_lambda, args.style_lambda, args.vgg_lambda, args.content_lambda,
        color_augment,
        amp, device, args.save
    )
