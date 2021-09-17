
import torch
import torch.optim as optim
from torchvision.utils import save_image

from .model import Generator, Discriminator, weights_init_normal

from dataset import AnimeFace
from utils import Status, save_args, add_args
from nnutils import get_device, sample_nnoise

def train(
    epochs, n_critic, clip_value,
    dataset, latent_dim,
    G, optimizer_G,
    D, optimizer_D,
    device, save_interval
):

    status = Status(len(dataset) * epochs)

    for epoch in range(epochs):
        for index, image in enumerate(dataset, 1):
            real = image.to(device)

            z = sample_nnoise((real.size(0), latent_dim), device)

            # generate image
            fake = G(z)
            # D(real)
            real_prob = D(real)
            # D(G(z))
            fake_prob = D(fake.detach())

            # discriminator loss
            D_loss = - real_prob.mean() + fake_prob.mean()

            # optimize
            optimizer_D.zero_grad()
            D_loss.backward()
            optimizer_D.step()

            # clip weights
            for param in D.parameters():
                param.data.clamp_(-clip_value, clip_value)

            G_loss = torch.tensor([0.])
            if index % n_critic == 0:
                # D(G(z))
                fake_prob = D(fake)

                # train to fool D
                G_loss = - fake_prob.mean()

                # optimize
                optimizer_G.zero_grad()
                G_loss.backward()
                optimizer_G.step()

            if status.batches_done % save_interval == 0:
                save_image(
                    fake[:25], f'implementations/WGAN/result/{status.batches_done}.png',
                    nrow=5, normalize=True)

            status.update(
                g=G_loss.item(), d=D_loss.item())

    status.plot_loss()


def main(parser):

    parser = add_args(parser,
        dict(
            epochs     = [150, 'epochs to train'],
            latent_dim = [200, 'dimension of input latent'],
            lr         = [0.00005, 'learning rate'],
            n_critic   = [5, 'update G only n_critic step'],
            clip_value = [0.01, 'clip weight value to [-clip_value,clip_value]']))
    args = parser.parse_args()
    save_args(args)

    device = get_device(not args.disable_gpu)

    dataset = AnimeFace.asloader(
        args.batch_size, (args.image_size, args.min_year),
        pin_memory=not args.disable_gpu)

    G = Generator(latent_dim=args.latent_dim)
    D = Discriminator()

    G.apply(weights_init_normal)
    D.apply(weights_init_normal)

    G.to(device)
    D.to(device)

    optimizer_G = optim.RMSprop(G.parameters(), lr=args.lr)
    optimizer_D = optim.RMSprop(D.parameters(), lr=args.lr)

    train(
        args.epochs, args.n_critic, args.clip_value,
        dataset, args.latent_dim,
        G, optimizer_G, D, optimizer_D,
        device, args.save)
