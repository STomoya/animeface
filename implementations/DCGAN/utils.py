
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from .model import Generator, Discriminator

from dataset import AnimeFace
from utils import save_args, add_args, Status
from nnutils import get_device, sample_nnoise, init
from nnutils.loss import GANLoss

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
    loss = GANLoss()

    while status.batches_done < max_iters:
        for index, image in enumerate(dataset, 1):

            real = image.to(device)
            z = sample_nnoise((real.size(0), latent_dim), device)

            # generate image
            fake = G(z)

            # D(real)
            real_prob = D(real)
            # D(G(z))
            fake_prob = D(fake.detach())

            # loss
            D_loss = loss.d_loss(real_prob, fake_prob)

            # optimize
            optimizer_D.zero_grad()
            D_loss.backward()
            optimizer_D.step()

            # D(G(z))
            fake_prob = D(fake)
            # train to fool D
            G_loss = loss.g_loss(fake_prob)

            # optimize
            optimizer_G.zero_grad()
            G_loss.backward()
            optimizer_G.step()

            if status.batches_done % save_interval == 0:
                save_image(
                    fake, f'implementations/DCGAN/result/{status.batches_done}.png',
                    normalize=True, value_range=(-1, 1))

            status.update(d=D_loss.item(), g=G_loss.item())
            if status.batches_done == max_iters:
                break

    status.plot_loss()

def main(parser):

    parser = add_args(parser, dict(latent_dim=[100, 'latent dimension']))
    args = parser.parse_args()
    save_args(args)

    device = get_device(not args.disable_gpu)

    dataset = AnimeFace.asloader(
        args.batch_size, (args.image_size, args.min_year),
        pin_memory=not args.disable_gpu)

    G = Generator(latent_dim=args.latent_dim)
    D = Discriminator()
    G.apply(init().N002)
    D.apply(init().N002)
    G.to(device)
    D.to(device)

    optimizer_G = optim.Adam(G.parameters())
    optimizer_D = optim.Adam(D.parameters())

    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs

    train(
        args.max_iters, dataset, args.latent_dim,
        G, optimizer_G,
        D, optimizer_D,
        device, args.save
    )
