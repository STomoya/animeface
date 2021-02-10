
import functools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad, Variable
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from ..StyleGAN2.model import Generator, Discriminator
from ..general import YearAnimeFaceDataset, to_loader
from ..gan_utils import get_device, GANTrainingStatus, sample_nnoise, DiffAugment, AdaAugment
from ..gan_utils.losses import GANLoss
from .AdaBelief import AdaBelief

def r1_gp(D, real, scaler=None):
    inputs = Variable(real, requires_grad=True)
    outputs = D(inputs).sum()
    ones = torch.ones(outputs.size(), device=real.device)
    with autocast(False):
        if scaler is not None:
            outputs = scaler.scale(outputs)[0]
        gradients = grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        if scaler is not None:
            gradients = gradients / scaler.get_scale()
    gradients = gradients.view(gradients.size(0), -1)
    penalty = gradients.norm(2, dim=1).pow(2).mean()
    return penalty / 2

def train(
    max_iter, dataset, sample_noise, style_dim,
    G, D, optimizer_G, optimizer_D,
    gp_lambda, policy,
    device, amp, save_interval=1000
):
    
    status = GANTrainingStatus()
    loss = GANLoss()
    scaler = GradScaler() if amp else None
    const_z = sample_nnoise((16, style_dim))
    bar = tqdm(total=max_iter)

    if policy == 'ada':
        augment = AdaAugment(0.6, 500000, device)
    elif policy is not '':
        augment = functools.partial(DiffAugment, policy=policy)
    else:
        augment = lambda x: x
    
    while status.batches_done < max_iter:
        for index, real in enumerate(dataset):
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            real = real.to(device)
            z = sample_noise()

            '''Discriminator'''
            with autocast(amp):
                # D(x)
                real = augment(real)
                real_prob = D(real)
                # D(G(z))
                fake = G(z)
                fake = augment(fake)
                fake_prob = D(fake.detach())
                # loss
                adv_loss = loss.d_loss(real_prob, fake_prob)
                if gp_lambda > 0:
                    r1_loss = r1_gp(D, real, scaler)
                else:
                    r1_loss = 0
                D_loss = adv_loss + r1_loss * gp_lambda

            if scaler is not None:
                scaler.scale(D_loss).backward()
                scaler.step(optimizer_D)
            else:
                D_loss.backward()
                optimizer_D.step()
            
            '''Generator'''
            z = sample_noise()
            with autocast(amp):
                # D(G(z))
                fake = G(z)
                fake = augment(fake)
                fake_prob = D(fake)
                # loss
                G_loss = loss.g_loss(fake_prob)
            
            if scaler is not None:
                scaler.scale(G_loss).backward()
                scaler.step(optimizer_G)
            else:
                G_loss.backward()
                optimizer_G.step()

            if status.batches_done % save_interval == 0:
                G.eval()
                with torch.no_grad():
                    image = G(const_z)
                G.train()
                status.save_image('implementations/AdaBelief/result', image, nrow=4)

            G_loss_pyfloat, D_loss_pyfloat = G_loss.item(), D_loss.item()
            status.append(G_loss_pyfloat, D_loss_pyfloat)
            if scaler is not None:
                scaler.update()
            if policy == 'ada':
                augment.update_p(real_prob)
            bar.set_postfix_str('G {:.5f} D {:.5f}'.format(G_loss_pyfloat, D_loss_pyfloat))
            bar.update(1)

            if status.batches_done == max_iter:
                break


def main(parser):

    # data
    image_size = 128
    min_year = 2010
    batch_size = 32

    # model
    mapping_layers = 8
    mapping_lr = 0.01
    style_dim = 512

    # training
    max_iters = -1
    lr = 0.0002
    betas = (0.5, 0.999)
    gp_lambda = 0
    policy = 'color,translation'
    # policy = 'ada'

    amp = True

    device = get_device()

    dataset = YearAnimeFaceDataset(image_size, min_year)
    dataset = to_loader(dataset, batch_size)

    sample_noise = functools.partial(sample_nnoise, size=(batch_size, style_dim), device=device)

    G = Generator(mapping_layers, style_dim, mapping_lr)
    D = Discriminator()
    G.to(device)
    D.to(device)

    optimizer_G = AdaBelief(G.parameters(), lr=lr, betas=betas)
    optimizer_D = AdaBelief(D.parameters(), lr=lr, betas=betas)

    if max_iters < 0:
        max_iters = len(dataset) * 100

    train(
        max_iters, dataset, sample_noise, style_dim,
        G, D, optimizer_G, optimizer_D, gp_lambda, policy,
        device, amp, 1000
    )
