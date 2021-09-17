
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import numpy as np

from .model import Generator, Discriminator

from utils import Status, add_args, save_args
from nnutils import get_device, sample_nnoise
from nnutils.loss.gan import GANLoss
from dataset import AnimeFace

def train(
    max_iters,
    dataset,
    latent_dim,
    G,
    optimizer_G,
    D,
    optimizer_D,
    device,
    save_interval
):

    status = Status(max_iters)
    loss   = GANLoss()
    const_z = sample_nnoise((16, latent_dim), device)

    while status.batches_done < max_iters:
        for index, image in enumerate(dataset, 1):
            image = image.to(device)

            z = sample_nnoise((image.size(0), latent_dim), device)

            # generate image
            fake = G(z)
            # D(real)
            real_prob = D(image)
            # D(fake)
            fake_prob = D(fake.detach())

            # detect real
            d_loss = loss.d_loss(real_prob, fake_prob)

            # optimize
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            # D(fake)
            fake_prob = D(fake)
            # train to fool D
            g_loss = loss.g_loss(fake_prob)

            # optimize
            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            if status.batches_done % save_interval == 0:
                with torch.no_grad():
                    G.eval()
                    images = G(const_z)
                    G.train()
                save_image(
                    images, f'implementations/GAN/result/{status.batches_done}.png',
                    nrow=4, normalize=True, value_range=(-1, 1))
                torch.save(G.state_dict(), f'implementations/GAN/result/G_{status.batches_done}.pt')
            save_image(fake, 'running.jpg', normalize=True, value_range=(-1, 1))

            status.update(d=d_loss.item(), g=g_loss.item())
            if status.batches_done == max_iters:
                break

    status.plot_loss()

def main(parser):

    parser = add_args(parser, {'latent-dim': [100, 'input latent dim']})
    args = parser.parse_args()
    save_args(args)

    dataset = AnimeFace.asloader(
        args.batch_size, (args.image_size, args.min_year))

    G = Generator(latent_dim=args.latent_dim, image_shape=(3, args.image_size, args.image_size))
    D = Discriminator(image_shape=(3, args.image_size, args.image_size))

    device = get_device(not args.disable_gpu)
    G.to(device)
    D.to(device)

    optimizer_G = optim.Adam(G.parameters())
    optimizer_D = optim.Adam(D.parameters())

    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs

    train(
        args.max_iters,
        dataset=dataset,
        latent_dim=args.latent_dim,
        G=G,
        optimizer_G=optimizer_G,
        D=D,
        optimizer_D=optimizer_D,
        device=device,
        save_interval=args.save
    )
