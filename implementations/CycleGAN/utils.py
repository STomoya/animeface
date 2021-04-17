
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import random_split
from torchvision.utils import save_image

from ..general import XDoGAnimeFaceDataset, XDoGDanbooruPortraitDataset, to_loader
from ..general import get_device, Status, save_args
from ..gan_utils import AdaBelief
from ..gan_utils.losses import LSGANLoss, HingeLoss

from .model import Generator, Discriminator, init_weight_normal

consistency = nn.L1Loss()

def update_lr(optimizer, delta):
    for param_group in optimizer.param_groups:
        param_group['lr'] -= delta

def train(
    dataset, max_iter, decay_from, delta,
    GC, GL, DC, DL,
    optimizer_G, optimizer_D,
    cycle_lambda, test,
    device, amp, save=1000
):
    
    status = Status(max_iter)
    loss = LSGANLoss()
    scaler = GradScaler() if amp else None
    D_input = lambda x, y: torch.cat([x, y], dim=1)

    while status.batches_done < max_iter:
        for rgb, line in dataset:
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            rgb = rgb.to(device)
            line = line.to(device)

            '''Discriminator'''
            with autocast(amp):
                # D(x)
                real_rgb_prob = DC(D_input(line, rgb))
                real_line_prob = DL(D_input(line, rgb))
                # generate images
                line2rgb = GC(line)
                rgb2line = GL(rgb)
                # D(G(y))
                fake_rgb_prob = DC(D_input(line, line2rgb.detach()))
                fake_line_prob = DL(D_input(rgb2line.detach(), rgb))
                # loss
                adv_loss_line = loss.d_loss(real_line_prob, fake_line_prob)
                adv_loss_rgb  = loss.d_loss(real_rgb_prob, fake_rgb_prob)
                D_loss = adv_loss_line + adv_loss_rgb
            
            if scaler is not None:
                scaler.scale(D_loss).backward()
                scaler.step(optimizer_D)
            else:
                D_loss.backward()
                optimizer_D.step()

            '''Generator'''
            with autocast(amp):
                # D(G(y))
                fake_rgb_prob = DC(D_input(line, line2rgb))
                fake_line_prob = DL(D_input(rgb2line, rgb))
                # G(G(x))
                line2rgb2line = GL(line2rgb)
                rgb2line2rgb  = GC(rgb2line)
                # loss
                adv_loss_line = loss.g_loss(fake_line_prob)
                adv_loss_rgb  = loss.g_loss(fake_rgb_prob)
                consistency_loss_line = consistency(line2rgb2line, line)
                consistency_loss_rgb  = consistency(rgb2line2rgb, rgb)
                G_loss = adv_loss_line + adv_loss_rgb \
                         + (consistency_loss_line + consistency_loss_rgb) * cycle_lambda
            
            if scaler is not None:
                scaler.scale(G_loss).backward()
                scaler.step(optimizer_G)
            else:
                G_loss.backward()
                optimizer_G.step()

            # save
            if status.batches_done % save == 0:
                image_grid = _image_grid(line, line2rgb)
                save_image(image_grid, f'implementations/CycleGAN/result/recons_{status.batches_done}.jpg', nrow=3*2, normalize=True, value_range=(-1, 1))
                # eval
                GC.eval()
                with torch.no_grad():
                    line2rgb = GC(test[1])
                GC.train()
                image_grid = _image_grid(test[1], line2rgb)
                save_image(image_grid, f'implementations/CycleGAN/result/test_{status.batches_done}.jpg', nrow=3*2, normalize=True, value_range=(-1, 1))
                # model
                torch.save(GC.state_dict(), f'implementations/CycleGAN/result/colorize_{status.batches_done}.pt')
                # torch.save(GL.state_dict(), f'implementations/CycleGAN/result/sketch_{status.batches_done}.pt')

            # updates
            loss_dict = dict(
                G=G_loss.item() if not torch.isnan(G_loss).any() else 0,
                D=D_loss.item() if not torch.isnan(D_loss).any() else 0
            )
            status.update(loss_dict)
            if scaler is not None:
                scaler.update()
            
            if status.batches_done == max_iter:
                break
        if status.batches_done > decay_from:
            update_lr(optimizer_D, delta)
            update_lr(optimizer_G, delta)

    status.plot()

def _image_grid(line, gen, num_images=6):
    lines = line.repeat(1, 3, 1, 1).chunk(line.size(0), dim=0) # convert to RGB.
    gens = gen.chunk(gen.size(0), dim=0)

    images = []
    for index, (line, gen) in enumerate(zip(lines, gens)):
        images.extend([line, gen])
        if index == num_images-1:
            break

    return torch.cat(images, dim=0)

def add_arguments(parser):
    parser.add_argument('--line-channels', default=1, type=int, help='number of channels of line art images')
    parser.add_argument('--rgb-channels', default=3, type=int, help='number of channels of the generated images')
    parser.add_argument('--test-images', default=6, type=int, help='numbers of images for test')

    parser.add_argument('--channels', default=32, type=int, help='channel width multiplier')
    parser.add_argument('--max-channels', default=1024, type=int, help='maximum number of channels')
    parser.add_argument('--downsample-to', default=32, type=int, help='the resolution to downsample to in G')
    parser.add_argument('--num-blocks', default=6, type=int, help='number of residual blocks in G')
    parser.add_argument('--block-num-conv', default=2, type=int, help='number of convolution layers per block')
    parser.add_argument('--g-disable-sn', default=False, action='store_true', help='do not use spectral normalization in G')
    parser.add_argument('--g-disable-bias', default=False, action='store_true', help='do not use bias in G')
    parser.add_argument('--g-norm-name', default='in', choices=['bn', 'in'], help='normalization layer name in G')
    parser.add_argument('--g-act-name', default='relu', choices=['relu', 'lrelu'], help='activation function in G')
    parser.add_argument('--num-layers', default=3, type=int, help='number of layers in D')
    parser.add_argument('--d-disable-sn', default=False, action='store_true', help='do not use spectral normalization in D')
    parser.add_argument('--d-disable-bias', default=False, action='store_true', help='do not use bias in D')
    parser.add_argument('--d-norm-name', default='in', choices=['bn', 'in'], help='normalization layer name in D')
    parser.add_argument('--d-act-name', default='relu', choices=['relu', 'lrelu'], help='activation function in D')

    parser.add_argument('--epochs', default=100, type=int, help='epochs to train with constant learning rate')
    parser.add_argument('--decay-epochs', default=100, type=int, help='epochs to train with a linearly decaying learning rate')
    parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.5, type=float, help='beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta2')
    parser.add_argument('--cycle-lambda', default=10., type=float, help='lambda for cycle consistency loss')
    return parser

def main(parser):
    
    parser = add_arguments(parser)
    args = parser.parse_args()
    save_args(args)

    # # generator
    g_use_sn   = not args.g_disable_sn
    g_use_bias = not args.g_disable_bias
    # discriminator
    d_use_sn   = not args.d_disable_sn
    d_use_bias = not args.d_disable_bias
    
    amp = not args.disable_amp
    device = get_device(not args.disable_gpu)

    # dataset
    # train
    dataset = XDoGAnimeFaceDataset(args.image_size, args.min_year)
    dataset = XDoGDanbooruPortraitDataset(args.image_size, num_images=args.num_images+args.test_images)
    dataset, test = random_split(dataset, [len(dataset)-args.test_images, args.test_images])
    dataset.dataset.shuffle_xdog()
    dataset = to_loader(dataset, args.batch_size)
    # test
    test = to_loader(test, args.test_images, shuffle=False, use_gpu=False)
    test_batch = next(iter(test))
    test_batch = (test_batch[0].to(device), test_batch[1].to(device))

    max_iters = len(dataset) * (args.epochs + args.decay_epochs)
    decay_from = len(dataset) * args.decay_epochs - 1 # -1 to avoid lr=0.
    delta = args.lr / args.decay_epochs

    # models
    # - line2color G
    GC = Generator(
        args.image_size, args.line_channels, args.rgb_channels, args.downsample_to,
        args.channels, args.max_channels, args.num_blocks, args.block_num_conv,
        g_use_sn, g_use_bias, args.g_norm_name, args.g_act_name
    )
    # - color2line G
    GL = Generator(
        args.image_size, args.rgb_channels, args.line_channels, args.downsample_to,
        args.channels, args.max_channels, args.num_blocks, args.block_num_conv,
        g_use_sn, g_use_bias, args.g_norm_name, args.g_act_name
    )
    # - line2color D
    DC = Discriminator(
        args.image_size, args.line_channels+args.rgb_channels, args.num_layers,
        args.channels, args.d_norm_name, args.d_act_name, d_use_sn, d_use_bias
    )
    # - color2line D
    DL = Discriminator(
        args.image_size, args.line_channels+args.rgb_channels, args.num_layers,
        args.channels, args.d_norm_name, args.d_act_name, d_use_sn, d_use_bias
    )
    # init
    GC.apply(init_weight_normal)
    GL.apply(init_weight_normal)
    DC.apply(init_weight_normal)
    DL.apply(init_weight_normal)
    # device
    GC.to(device)
    GL.to(device)
    DC.to(device)
    DL.to(device)

    # optimizers
    betas = (args.beta1, args.beta2)
    optimizer_G = AdaBelief(
        itertools.chain(GC.parameters(), GL.parameters()),
        lr=args.lr, betas=betas
    )
    optimizer_D = AdaBelief(
        itertools.chain(DC.parameters(), DL.parameters()),
        lr=args.lr, betas=betas
    )

    train(
        dataset, max_iters, decay_from, delta,
        GC, GL, DC, DL, optimizer_G, optimizer_D,
        args.cycle_lambda, test_batch,
        device, amp, 1000
    )