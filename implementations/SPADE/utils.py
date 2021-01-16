
import itertools
import functools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image

from ..general import XDoGAnimeFaceDataset, XDoGDanbooruPortraitDataset, to_loader
from ..general import get_device, Status
from ..gan_utils import sample_nnoise, AdaBelief
from ..gan_utils.losses import HingeLoss, GANLoss

from .model import Generator, Discriminator, Encoder, init_weight_xavier

l1 = nn.L1Loss()

def KL_divergence(mu, logvar):
    return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def feature_matching(real_feats, fake_feats):
    feat_loss = 0
    layer_weight = 4 / len(real_feats) # 1.
    for real_feat, fake_feat in zip(real_feats, fake_feats):
        feat_loss += l1(real_feat, fake_feat) * layer_weight
    return feat_loss

def train(
    dataset, max_iters, sampler, z_dim, test,
    G, D, E, optimizer_G, optimizer_D,
    kld_lambda, feat_lambda, d_num_scale,
    device, amp, save=1000
):

    status = Status(max_iters)
    scaler = GradScaler() if amp else None
    loss = HingeLoss()
    D_input = lambda x, y: torch.cat([x, y], dim=1)

    while status.batches_done < max_iters:
        for index, (rgb, line) in enumerate(dataset):
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            rgb = rgb.to(device)
            line = line.to(device)

            '''Discriminator'''
            with autocast(amp):
                if E is not None:
                    # E(rgb)
                    z, mu, logvar = E(rgb)
                else:
                    # z : N(0, 1)
                    z = sampler((rgb.size(0), z_dim))
                # D(line, rgb)
                real_outs = D(D_input(line, rgb))
                # D(line, G(z, line))
                fake = G(z, line)
                fake_outs = D(D_input(line, fake.detach()))
                # loss
                D_loss = 0
                for real_out, fake_out in zip(real_outs, fake_outs):
                    D_loss = D_loss + loss.d_loss(real_out[0], fake_out[0]) / d_num_scale
            
            if scaler is not None:
                scaler.scale(D_loss).backward()
                scaler.step(optimizer_D)
            else:
                D_loss.backward()
                optimizer_D.step()

            '''Generator (+ Encoder)'''
            with autocast(amp):
                # D(line, rgb)
                real_outs = D(D_input(line, rgb))
                # D(line, G(z, line))
                fake_outs = D(D_input(line, fake))
                # loss
                G_loss = 0
                for real_out, fake_out in zip(real_outs, fake_outs):
                    # gan loss
                    G_loss = G_loss + loss.g_loss(fake_out[0]) / d_num_scale
                    # feature matching loss
                    if feat_lambda > 0:
                        G_loss = G_loss + feature_matching(real_out[1], fake_out[1]) / d_num_scale * feat_lambda
                # KLD loss if E exists
                if E is not None and kld_lambda > 0:
                    G_loss = G_loss + KL_divergence(mu, logvar) * kld_lambda
            
            if scaler is not None:
                scaler.scale(G_loss).backward()
                scaler.step(optimizer_G)
            else:
                G_loss.backward()
                optimizer_G.step()

            # save
            if status.batches_done % save == 0:
                # save running samples
                image_grid = _image_grid(line, fake, rgb)
                save_image(image_grid, f'implementations/SPADE/result/recons_{status.batches_done}.jpg', nrow=3*3, normalize=True, range=(-1, 1))
                # test
                if E is not None: E.eval()
                G.eval()
                with torch.no_grad():
                    if E is not None: z, _, _ = E(test[0])
                    else: z = sampler((test[0].size(0), z_dim))
                    fake = G(z, test[1])
                if E is not None: E.train()
                G.train()
                # save test samples
                image_grid = _image_grid(test[1], fake, test[0])
                save_image(image_grid, f'implementations/SPADE/result/test_{status.batches_done}.jpg', nrow=3*3, normalize=True, range=(-1, 1))
                # save models
                torch.save(G.state_dict(), f'implementations/SPADE/result/G_{status.batches_done}.pt')
                if E is not None:
                    torch.save(E.state_dict(), f'implementations/SPADE/result/E_{status.batches_done}.pt')
            
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

def _image_grid(line, gen, rgb, num_images=6):
    lines = line.repeat(1, 3, 1, 1).chunk(line.size(0), dim=0) # convert to RGB.
    gens = gen.chunk(gen.size(0), dim=0)
    rgbs = rgb.chunk(rgb.size(0), dim=0)

    images = []
    for index, (line, gen, rgb) in enumerate(zip(lines, gens, rgbs)):
        images.extend([line, gen, rgb])
        if index == num_images-1:
            break

    return torch.cat(images, dim=0)

def main():

    # params
    # data
    image_size = 256
    min_year   = 2005  # for animeface
    num_images = 60000 # for danbooru
    line_channels = 1
    rgb_channels  = 3
    batch_size = 32
    test_images = 6

    # model
    z_dim = 256
    channels = 32
    max_channels = 2**10
    # generator
    block_num_conv = 2
    spade_hidden_channels = 128
    g_norm_name = 'in'
    g_act_name  = 'lrelu'
    g_use_sn    = True
    g_use_bias  = True
    # discriminator
    num_scale = 2
    num_layers = 3
    d_norm_name = 'in'
    d_act_name  = 'lrelu'
    d_use_sn    = True
    d_use_bias  = True
    # encoder
    image_guided = True # if False, will not use Encoder
    target_resl = 4
    e_norm_name = 'in'
    e_act_name  = 'relu'
    e_use_sn    = True
    e_use_bias  = True

    # training
    max_iters = -1
    lr = 0.0002
    betas = (0.5, 0.999)
    kld_lambda = 0.05
    feat_lambda = 10.

    amp = True
    device = get_device()

    # dataset
    # dataset = XDoGAnimeFaceDataset(image_size, min_year)
    dataset = XDoGDanbooruPortraitDataset(image_size, num_images=num_images+test_images)
    dataset, test = random_split(dataset, [len(dataset)-test_images, test_images])
    ## training dataset
    dataset = to_loader(dataset, batch_size)
    ## test batch
    test    = to_loader(test, test_images, shuffle=False, use_gpu=False)
    test_batch = next(iter(test))
    test_batch = (test_batch[0].to(device), test_batch[1].to(device))
    if max_iters < 0:
        max_iters = len(dataset) * 100
    ## noise sampler (ignored when E exists)
    sampler = functools.partial(sample_nnoise, device=device)

    # models
    G = Generator(
        image_size, z_dim, line_channels, rgb_channels,
        channels, max_channels,
        block_num_conv, spade_hidden_channels,
        g_norm_name, g_act_name, g_use_sn, g_use_bias
    )
    D = Discriminator(
        image_size, line_channels+rgb_channels,
        num_scale, num_layers, channels,
        d_norm_name, d_act_name, d_use_sn, d_use_bias
    )
    G.apply(init_weight_xavier)
    D.apply(init_weight_xavier)
    G.to(device)
    D.to(device)
    ## create encoder (optional)
    if image_guided:
        E = Encoder(
            image_size, z_dim, rgb_channels, target_resl,
            channels, max_channels,
            e_use_sn, e_use_bias, e_norm_name, e_act_name
        )
        E.apply(init_weight_xavier)
        E.to(device)
    else:
        E = None
        kld_lambda = 0
    
    # optimizers
    optimizer_D = AdaBelief(D.parameters(), lr=lr, betas=betas)
    if E is None:
        g_params = G.parameters()
    else:
        g_params = itertools.chain(
            G.parameters(), E.parameters()
        )
    optimizer_G = AdaBelief(g_params, lr=lr, betas=betas)

    train(
        dataset, max_iters, sampler, z_dim, test_batch,
        G, D, E, optimizer_G, optimizer_D,
        kld_lambda, feat_lambda, num_scale,
        device, amp, 1000
    )