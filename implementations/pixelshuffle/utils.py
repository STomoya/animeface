
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import numpy as np

from .model import Generator, Discriminator, weights_init_normal

# from ..general import LabeledAnimeFaceDataset, to_loader
from dataset import AnimeFaceLabel
from utils import Status, add_args, save_args
from nnutils import sample_nnoise, get_device
from nnutils.loss import LSGANLoss

def train(
    max_iters, dataset, num_label, latent_dim,
    G, optimizer_G,
    D, optimizer_D,
    label_criterion,
    device, save_interval
):

    status = Status(max_iters)
    loss   = LSGANLoss()

    while status.batches_done < max_iters:
        for index, (image, label) in enumerate(dataset, 1):
            # generate class label
            gen_label = torch.randint(0, num_label, (image.size(0), ), device=device)
            # real image/label
            real = image.to(device)
            label = label.long().to(device)
            # noise
            z = sample_nnoise((real.size(0), latent_dim), device)

            # generate image
            fake = G(z, gen_label)

            '''
            Train Discriminator
            '''

            # D(real)
            real_prob, real_label_prob = D(real)
            # D(G(z))
            fake_prob, fake_label_prob = D(fake.detach())

            # loss
            adv_loss = loss.d_loss(real_prob, fake_prob)
            aux_real_loss = label_criterion(real_label_prob, label)
            aux_fake_loss = label_criterion(fake_label_prob, gen_label)
            D_loss = adv_loss + (aux_real_loss + aux_fake_loss) / 2

            # optimize
            optimizer_D.zero_grad()
            D_loss.backward()
            optimizer_D.step()

            '''
            Train Generator
            '''

            # train to fool D
            fake_prob, label_prob = D(fake)
            adv_loss = loss.g_loss(fake_prob)
            label_loss = label_criterion(label_prob, gen_label)
            G_loss = adv_loss + label_loss

            # optimize
            optimizer_G.zero_grad()
            G_loss.backward()
            optimizer_G.step()

            if status.batches_done % save_interval == 0:
                save_image(
                    fake[:25], f'implementations/pixelshuffle/result/{status.batches_done}.jpg',
                    nrow=5, normalize=True, value_range=(-1, 1))

            status.update(d=D_loss.item(), g=G_loss.item())
            if status.batches_done == max_iters:
                break

    status.plot_loss()

def main(parser):

    parser = add_args(parser,
        dict(
            latent_dim=[200]))
    args = parser.parse_args()
    save_args(args)

    dataset = AnimeFaceLabel.asloader(
        args.batch_size, (args.image_size, ),
        pin_memory=not args.disable_gpu)

    label_dim = dataset.dataset.num_classes

    G = Generator(latent_dim=args.latent_dim, label_dim=label_dim)
    D = Discriminator(label_dim=label_dim)

    G.apply(weights_init_normal)
    D.apply(weights_init_normal)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    G.to(device)
    D.to(device)

    optimizer_G = optim.Adam(G.parameters())
    optimizer_D = optim.Adam(D.parameters())

    label_criterion = nn.CrossEntropyLoss()

    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs

    train(
        args.max_iters, dataset, label_dim, args.latent_dim,
        G, optimizer_G, D, optimizer_D,
        label_criterion, device, args.save)
