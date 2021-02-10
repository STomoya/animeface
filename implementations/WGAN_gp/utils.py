
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torchvision.utils import save_image
import numpy as np

from .model import Generator, Discriminator, weights_init_normal

from ..general import AnimeFaceDataset, to_loader

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

def main(parser):
    batch_size = 32
    image_size = 128
    epochs = 150
    latent_dim = 200
    lr = 2.e-4
    betas = (0.5, 0.999)
    n_critic = 5
    gp_gamma = 10

    dataset = AnimeFaceDataset(image_size)
    dataset = to_loader(dataset, batch_size)

    G = Generator(latent_dim=latent_dim)
    D = Discriminator()

    G.apply(weights_init_normal)
    D.apply(weights_init_normal)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    G.to(device)
    D.to(device)

    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=betas)
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=betas)

    train(
        epochs=epochs,
        n_critic=n_critic,
        gp_gamma=gp_gamma,
        dataset=dataset,
        latent_dim=latent_dim,
        G=G,
        optimizer_G=optimizer_G,
        D=D,
        optimizer_D=optimizer_D,
        device=device,
        verbose_interval=100,
        save_interval=500
    )