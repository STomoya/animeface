
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import random_split
from torchvision.utils import save_image

from ..general import XDoGAnimeFaceDataset, XDoGDanbooruPortraitDataset, to_loader
from ..general import get_device, Status
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
                save_image(image_grid, f'implementations/CycleGAN/result/recons_{status.batches_done}.jpg', nrow=3*2, normalize=True, range=(-1, 1))
                # eval
                GC.eval()
                with torch.no_grad():
                    line2rgb = GC(test[1])
                GC.train()
                image_grid = _image_grid(test[1], line2rgb)
                save_image(image_grid, f'implementations/CycleGAN/result/test_{status.batches_done}.jpg', nrow=3*2, normalize=True, range=(-1, 1))
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

def main(parser):
    
    # params
    # data
    image_size = 256
    min_year   = 2005
    num_images = 60000
    batch_size = 24
    test_images = 6
    line_channels = 1
    rgb_channels  = 3

    # model
    channels = 32
    # generator
    max_channels = 256
    downsample_to = 32
    num_blocks = 6
    block_num_conv = 2
    g_use_sn = False
    g_use_bias = True
    g_norm_name = 'in'
    g_act_name = 'relu'
    # discriminator
    num_layers = 3
    d_use_sn = True
    d_use_bias = True
    d_norm_name = 'in'
    d_act_name = 'lrelu'

    # training
    epochs = 100
    decay_epochs = 100
    lr = 0.0002
    betas = (0.5, 0.999)
    cycle_lambda = 10.

    amp = True
    device = get_device()

    # dataset
    # train
    dataset = XDoGAnimeFaceDataset(image_size, min_year)
    dataset = XDoGDanbooruPortraitDataset(image_size, num_images=num_images+test_images)
    dataset, test = random_split(dataset, [len(dataset)-test_images, test_images])
    dataset.dataset.shuffle_xdog()
    dataset = to_loader(dataset, batch_size)
    # test
    test = to_loader(test, test_images, shuffle=False, use_gpu=False)
    test_batch = next(iter(test))
    test_batch = (test_batch[0].to(device), test_batch[1].to(device))

    max_iter = len(dataset) * (epochs + decay_epochs)
    decay_from = len(dataset) * decay_epochs - 1 # -1 to avoid lr=0.
    delta = lr / decay_epochs

    # models
    # - line2color G
    GC = Generator(
        image_size, line_channels, rgb_channels, downsample_to,
        channels, max_channels, num_blocks, block_num_conv,
        g_use_sn, g_use_bias, g_norm_name, g_act_name
    )
    # - color2line G
    GL = Generator(
        image_size, rgb_channels, line_channels, downsample_to,
        channels, max_channels, num_blocks, block_num_conv,
        g_use_sn, g_use_bias, g_norm_name, g_act_name
    )
    # - line2color D
    DC = Discriminator(
        image_size, line_channels+rgb_channels, num_layers,
        channels, d_norm_name, d_act_name, d_use_sn, d_use_bias
    )
    # - color2line D
    DL = Discriminator(
        image_size, line_channels+rgb_channels, num_layers,
        channels, d_norm_name, d_act_name, d_use_sn, d_use_bias
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
    optimizer_G = AdaBelief(
        itertools.chain(GC.parameters(), GL.parameters()),
        lr=lr, betas=betas
    )
    optimizer_D = AdaBelief(
        itertools.chain(DC.parameters(), DL.parameters()),
        lr=lr, betas=betas
    )

    train(
        dataset, max_iter, decay_from, delta,
        GC, GL, DC, DL, optimizer_G, optimizer_D,
        cycle_lambda, test_batch,
        device, amp, 1000
    )