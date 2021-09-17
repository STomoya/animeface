
import torch
import torch.optim as optim
import torch.autograd as autograd
from torchvision.utils import save_image

from .model import Generator, Discriminator, weights_init_normal

from dataset import AnimeFace
from utils import Status, save_args, add_args
from nnutils import get_device, sample_nnoise

def train(
    epochs, n_critic,
    gp_gamma, dataset, latent_dim,
    G, optimizer_G,
    D, optimizer_D,
    device, save_interval
):

    status = Status(len(dataset)*epochs)

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
            gp = gradient_penalty(D, real.data, fake.data, device)
            adv_loss = - real_prob.mean() + fake_prob.mean()
            D_loss = adv_loss + gp * gp_gamma

            # optimize
            optimizer_D.zero_grad()
            D_loss.backward()
            optimizer_D.step()

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
                    fake.data[:25], f"implementations/WGAN_gp/result/{status.batches_done}.png",
                    nrow=5, normalize=True)

            status.update(
                d=D_loss.item(), g=G_loss.item())

    status.plot_loss()

def gradient_penalty(D, real_image, fake_image, device):
    alpha = torch.rand(real_image.size(0), 1, 1, 1, device=device)

    interpolates = (alpha * real_image + ((1 - alpha) * fake_image)).requires_grad_(True)
    d_interpolates = D(interpolates)

    fake = torch.Tensor(real_image.size(0), 1).fill_(1.).requires_grad_(False)
    fake = fake.type(torch.FloatTensor).to(device)

    gradients = autograd.grad(
        outputs=d_interpolates.view(d_interpolates.size(0), 1),
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return penalty

def main(parser):

    parser = add_args(parser,
        dict(
            epochs     = [150, 'epochs to train'],
            latent_dim = [200, 'dimension of input latent'],
            lr         = [5e-5, 'learning rate for both generator and discriminator'],
            beta1      = [0.5, 'beta1'],
            beta2      = [0.999, 'beta2'],
            n_critic   = [5, 'train G only each "--n-critic" step'],
            gp_gamma   = [10., 'gamma for gradient penalty']))
    args = parser.parse_args()
    save_args(args)

    betas = (args.beta1, args.beta2)
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

    optimizer_G = optim.Adam(G.parameters(), lr=args.lr, betas=betas)
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr, betas=betas)

    train(
        args.epochs, args.n_critic, args.gp_gamma,
        dataset, args.latent_dim,
        G, optimizer_G, D, optimizer_D,
        device, args.save)
