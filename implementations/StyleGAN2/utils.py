
import os
import functools
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad, Variable
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image
import numpy as np

from ..general import YearAnimeFaceDataset, DanbooruPortraitDataset, to_loader
from ..gan_utils import sample_nnoise, DiffAugment, update_ema
from ..general import get_device, Status
from ..gan_utils.losses import GANLoss, HingeLoss, LSGANLoss

from .model import Generator, Discriminator, init_weight_N01

def clac_grad(outputs, inputs, scaler=None):
    with autocast(False):
        if scaler is not None:
            outputs = scaler.scale(outputs)
        ones = torch.ones(outputs.size(), device=inputs.device)
        gradients = grad(
            outputs=outputs, inputs=inputs, grad_outputs=ones,
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        if scaler is not None:
            gradients = gradients / scaler.get_scale()
    return gradients

def r1_penalty(real, D, scaler=None):
    '''R1 regularizer'''
    real_loc = Variable(real, requires_grad=True)
    real_prob = D(real_loc)
    real_prob = real_prob.sum()

    gradients = clac_grad(real_prob, real_loc, scaler)

    # calc penalty
    gradients = gradients.reshape(gradients.size(0), -1)
    penalty = gradients.norm(2, dim=1).pow(2).mean()

    return penalty / 2

def pl_penalty(styles, images, pl_mean, scaler=None):
    '''path length regularizer'''
    # mul noise
    num_pixels = images.size()[2:].numel()
    noise = torch.randn(images.size(), device=images.device) / np.sqrt(num_pixels)
    outputs = (images * noise).sum()

    gradients = clac_grad(outputs, styles, scaler)

    # calc penalty
    gradients = gradients.pow(2).sum(dim=1).sqrt()
    return (gradients - pl_mean).pow(2).mean()

def update_pl_mean(old, new, decay=0.99):
    '''exponential moving average'''
    return decay * old + (1 - decay) * new

def train(
    max_iter, dataset, sampler, const_z,
    G, G_ema, D, optimizer_G, optimizer_D,
    r1_lambda, pl_lambda, d_k, g_k, policy,
    device, amp,
    save=1000
):
    
    status  = Status(max_iter)
    pl_mean = 0.
    loss    = GANLoss()
    scaler  = GradScaler() if amp else None
    augment = functools.partial(DiffAugment, policy=policy)

    if G_ema is not None:
        G_ema.eval()

    while status.batches_done < max_iter:
        for index, real in enumerate(dataset):
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            real = real.to(device)

            '''discriminator'''
            z = sampler()
            with autocast(amp):
                # D(x)
                real_aug = augment(real)
                real_prob = D(real_aug)
                # D(G(z))
                fake, _ = G(z)
                fake_aug = augment(fake)
                fake_prob = D(fake_aug.detach())
                # loss
                if status.batches_done % d_k == 0 \
                    and r1_lambda > 0 \
                    and status.batches_done is not 0:
                    # lazy regularization
                    r1 = r1_penalty(real, D, scaler)
                    D_loss = r1 * r1_lambda * d_k
                else:
                    # gan loss on other iter
                    D_loss = loss.d_loss(real_prob, fake_prob)
            
            if scaler is not None:
                scaler.scale(D_loss).backward()
                scaler.step(optimizer_D)
            else:
                D_loss.backward()
                optimizer_D.step()

            '''generator'''
            z = sampler()
            with autocast(amp):
                # D(G(z))
                fake, style = G(z)
                fake_aug = augment(fake)
                fake_prob = D(fake_aug)
                # loss
                if status.batches_done % g_k == 0 \
                    and pl_lambda > 0 \
                    and status.batches_done is not 0:
                    # lazy regularization
                    pl = pl_penalty(style, fake, pl_mean, scaler)
                    G_loss = pl * pl_lambda * g_k
                    avg_pl = np.mean(pl.detach().cpu().numpy())
                    pl_mean = update_pl_mean(pl_mean, avg_pl)
                else:
                    # gan loss on other iter
                    G_loss = loss.g_loss(fake_prob)
            
            if scaler is not None:
                scaler.scale(G_loss).backward()
                scaler.step(optimizer_G)
            else:
                G_loss.backward()
                optimizer_G.step()

            if G_ema is not None:
                update_ema(G, G_ema)

            # save
            if status.batches_done % save == 0:
                with torch.no_grad():
                    images, _ = G_ema(const_z)
                save_image(images, f'implementations/StyleGAN2/result/{status.batches_done}.jpg', nrow=4, normalize=True, range=(-1, 1))
                torch.save(G_ema.state_dict(), f'implementations/StyleGAN2/result/G_{status.batches_done}.pt')
            save_image(fake, f'running.jpg', nrow=4, normalize=True, range=(-1, 1))

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
    
    status.plot()

def main(parser):

    # params
    # data
    image_size = 128
    min_year = 2010
    num_images = 60000
    image_channels = 3
    batch_size = 12

    # model
    # G
    style_dim = 1024    # increase latent dim
    channels = 32
    max_channels = 1024 # increase model size
    block_num_conv = 2
    map_num_layers = 4  # decrease layers
    map_lr = 0.01
    # D
    normalize = False # disable normalizing latent
    mbsd_groups = 4

    # traning
    max_iter = -1 # if minus 100 epochs
    lr = 0.001
    betas = (0., 0.99)
    g_k = 8  # calc pl every g_k iter
    d_k = 16 # calc gp every d_k iter
    r1_lambda = 5. # decrease r1
    pl_lambda = 0. # disable pl
    policy = 'color,translation'
    amp = True

    device = get_device()

    # dataset
    dataset = YearAnimeFaceDataset(image_size, min_year)
    dataset = DanbooruPortraitDataset(image_size, num_images=num_images)
    dataset = to_loader(
                dataset, batch_size, shuffle=True,
                num_workers=os.cpu_count(), use_gpu=torch.cuda.is_available())

    # random noise sampler
    sampler = functools.partial(sample_nnoise, (batch_size, style_dim), device=device)
    # const input for eval
    const_z = sample_nnoise((16, style_dim), device=device)

    # models
    G = Generator(
            image_size, image_channels, style_dim, channels, max_channels,
            block_num_conv, map_num_layers, normalize, map_lr)
    G_ema = Generator(
            image_size, image_channels, style_dim, channels, max_channels,
            block_num_conv, map_num_layers, normalize, map_lr)
    D = Discriminator(
            image_size, image_channels, channels, max_channels, 
            block_num_conv, mbsd_groups)
    ## init
    G.init_weight(
        map_init_func=functools.partial(init_weight_N01, lr=map_lr),
        syn_init_func=init_weight_N01)
    G_ema.eval()
    update_ema(G, G_ema, decay=0)
    D.apply(init_weight_N01)

    G.to(device)
    G_ema.to(device)
    D.to(device)

    # optimizer
    if pl_lambda > 0:
        g_ratio = g_k / (g_k + 1)
        g_lr = lr * g_ratio
        g_betas = (betas[0]**g_ratio, betas[1]**g_ratio)
    else: g_lr, g_betas = lr, betas

    if r1_lambda > 0:
        d_ratio = d_k / (d_k + 1)
        d_lr = lr * d_ratio
        d_betas = (betas[0]**d_ratio, betas[1]**d_ratio)
    else: d_lr, d_betas = lr, betas
    
    optimizer_G = optim.Adam(G.parameters(), lr=g_lr, betas=g_betas)
    optimizer_D = optim.Adam(D.parameters(), lr=d_lr, betas=d_betas)

    if max_iter < 0:
        max_iter = len(dataset) * 100

    train(
        max_iter, dataset, sampler, const_z,
        G, G_ema, D, optimizer_G, optimizer_D,
        r1_lambda, pl_lambda, d_k, g_k,
        policy, device, amp, 1000
    )

    
