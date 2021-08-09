
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import random_split
from torchvision.utils import save_image

from dataset import AnimeFaceXDoG, DanbooruPortraitXDoG, to_loader
from utils import Status, add_args, save_args
from nnutils import get_device
from nnutils.loss import LSGANLoss

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

    while status.batches_done < max_iter:
        for rgb, line in dataset:
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            rgb = rgb.to(device)
            line = line.to(device)

            '''Discriminator'''
            with autocast(amp):
                # D(x)
                real_rgb_prob = DC(rgb)
                real_line_prob = DL(line)
                # generate images
                line2rgb = GC(line)
                rgb2line = GL(rgb)
                # D(G(y))
                fake_rgb_prob = DC(line2rgb.detach())
                fake_line_prob = DL(rgb2line.detach())
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
                fake_rgb_prob = DC(line2rgb)
                fake_line_prob = DL(rgb2line)
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
                save_image(
                    image_grid, f'implementations/CycleGAN/result/recons_{status.batches_done}.jpg',
                    nrow=3*2, normalize=True, value_range=(-1, 1))
                # eval
                GC.eval()
                with torch.no_grad():
                    line2rgb = GC(test[1])
                GC.train()
                image_grid = _image_grid(test[1], line2rgb)
                save_image(
                    image_grid, f'implementations/CycleGAN/result/test_{status.batches_done}.jpg',
                    nrow=3*2, normalize=True, value_range=(-1, 1))
                # model
                torch.save(
                    GC.state_dict(),
                    f'implementations/CycleGAN/result/colorize_{status.batches_done}.pt')
                # torch.save(GL.state_dict(), f'implementations/CycleGAN/result/sketch_{status.batches_done}.pt')
            save_image(line2rgb, 'running_color.jpg', normalize=True, value_range=(-1, 1))

            # updates
            loss_dict = dict(
                G=G_loss.item() if not torch.isnan(G_loss).any() else 0,
                D=D_loss.item() if not torch.isnan(D_loss).any() else 0
            )
            status.update(**loss_dict)
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


def main(parser):

    parser = add_args(parser,
        dict(
            line_channels  = [1, 'number of channels of line art images'],
            rgb_channels   = [3, 'number of channels of the generated images'],
            test_images    = [6, 'number of images for test'],
            channels       = [32, 'channel width multiplier'],
            max_channels   = [1024, 'maximum channels width'],
            downsample_to  = [32, 'bottom width'],
            num_blocks     = [6, 'number of residual blocks'],
            block_num_conv = [2, 'number of conv in resblock'],
            g_disable_sn   = [False, 'disable spectral norm'],
            g_disable_bias = [False, 'disable bias'],
            g_norm_name    = ['in', 'normalization layer name'],
            g_act_name     = ['relu', 'activation function name'],
            num_layers     = [3, 'number of layers in PatchGAN D'],
            d_disable_sn   = [False, 'disable spectral norm'],
            d_disable_bias = [False, 'disable bias'],
            d_norm_name    = ['in', 'normalization layer name'],
            d_act_name     = ['relu', 'activation function name'],
            epochs         = [100, 'epochs to train with const lr'],
            decay_epochs   = [1000, 'epochs to train with linearly decaying lr'],
            lr             = [0.0002, 'learning rate'],
            betas          = [[0.5, 0.999], 'betas'],
            cycle_lambda   = [10., 'lambda for cycle consistency loss'])
    )
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
    if args.dataset == 'animeface':
        dataset = AnimeFaceXDoG(args.image_size, args.min_year)
    elif args.dataset == 'danbooru':
        dataset = DanbooruPortraitXDoG(args.image_size, args.num_images+args.test_images)
    dataset, test = random_split(dataset, [len(dataset)-args.test_images, args.test_images])
    dataset.dataset.shuffle_xdog()
    dataset = to_loader(dataset, args.batch_size)
    # test
    test = to_loader(test, args.test_images, shuffle=False, pin_memory=False)
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
        args.image_size, args.rgb_channels, args.num_layers,
        args.channels, args.d_norm_name, args.d_act_name, d_use_sn, d_use_bias
    )
    # - color2line D
    DL = Discriminator(
        args.image_size, args.line_channels, args.num_layers,
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
    optimizer_G = optim.Adam(
        itertools.chain(GC.parameters(), GL.parameters()),
        lr=args.lr, betas=args.betas
    )
    optimizer_D = optim.Adam(
        itertools.chain(DC.parameters(), DL.parameters()),
        lr=args.lr, betas=args.betas
    )

    train(
        dataset, max_iters, decay_from, delta,
        GC, GL, DC, DL, optimizer_G, optimizer_D,
        args.cycle_lambda, test_batch,
        device, amp, 1000
    )
