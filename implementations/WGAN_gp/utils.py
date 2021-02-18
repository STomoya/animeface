
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torchvision.utils import save_image
import numpy as np

from .model import Generator, Discriminator, weights_init_normal

from ..general import AnimeFaceDataset, to_loader, save_args

def train(
    epochs,
    n_critic,
    gp_gamma,
    dataset,
    latent_dim,
    G,
    optimizer_G,
    D,
    optimizer_D,
    device,
    verbose_interval,
    save_interval
):
    
    for epoch in range(epochs):
        for index, image in enumerate(dataset, 1):
            g_loss = None

            image = image.to(device)

            z = torch.from_numpy(np.random.normal(0, 1, (image.size(0), latent_dim)))
            z = z.type(torch.FloatTensor).to(device)

            # generate image
            fake_image = G(z)

            # discriminator loss
            grad_penalty = gradient_penalty(D, image.data, fake_image.data, device)
            d_loss = -torch.mean(D(image)) + torch.mean(D(fake_image.detach())) + gp_gamma * grad_penalty

            # optimize
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            if index % n_critic == 0:

                # train to fool D
                g_loss = -torch.mean(D(fake_image))

                # optimize
                optimizer_G.zero_grad()
                g_loss.backward()
                optimizer_G.step()

            batches_done = epoch * len(dataset) + index

            if batches_done % verbose_interval == 0 and not g_loss == None:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, epochs, index, len(dataset), d_loss.item(), g_loss.item())
                )
            
            fake_image.view(fake_image.size(0), -1)
            if batches_done % save_interval == 0 or batches_done == 1:
                save_image(fake_image.data[:25], "implementations/WGAN_gp/result/%d.png" % batches_done, nrow=5, normalize=True)

def gradient_penalty(D, real_image, fake_image, device):
    alpha = torch.from_numpy(np.random.random((real_image.size(0), 1, 1, 1)))
    alpha = alpha.type(torch.FloatTensor).to(device)

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

def add_arguments(parser):
    parser.add_argument('--epochs', default=150, type=int, help='epochs to train')
    parser.add_argument('--latent-dim', default=200, type=int, help='dimension of input latent')
    parser.add_argument('--lr', default=5e-5, type=float, help='learning rate for both generator and discriminator')
    parser.add_argument('--beta1', default=0.5, type=float, help='beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta2')
    parser.add_argument('--n-critic', default=5, type=int, help='train G only each "--n-critic" step')
    parser.add_argument('--gp-gamma', default=10., type=float, help='gamma for gradient penalty')
    return parser

def main(parser):

    parser = add_arguments(parser)
    args = parser.parse_args()
    save_args(args)

    betas = (args.beta1, args.beta2)

    dataset = AnimeFaceDataset(args.image_size)
    dataset = to_loader(dataset, args.batch_size)

    G = Generator(latent_dim=args.latent_dim)
    D = Discriminator()

    G.apply(weights_init_normal)
    D.apply(weights_init_normal)

    if not args.disable_gpu:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    G.to(device)
    D.to(device)

    optimizer_G = optim.Adam(G.parameters(), lr=args.lr, betas=betas)
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr, betas=betas)

    train(
        epochs=args.epochs,
        n_critic=args.n_critic,
        gp_gamma=args.gp_gamma,
        dataset=dataset,
        latent_dim=args.latent_dim,
        G=G,
        optimizer_G=optimizer_G,
        D=D,
        optimizer_D=optimizer_D,
        device=device,
        verbose_interval=100,
        save_interval=500
    )