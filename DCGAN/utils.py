
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import numpy as np

from .model import Generator, Discriminator, weights_init_normal

def train(
    epochs,
    dataset,
    latent_dim,
    G,
    optimizer_G,
    D,
    optimizer_D,
    criterion,
    device,
    verbose_interval,
    save_interval
):
    
    for epoch in range(epochs):
        for index, image in enumerate(dataset, 1):
            # label smoothing
            gt   = torch.from_numpy((1.0 - 0.7) * np.random.randn(image.size(0), 1) + 0.7)
            gt   = gt.type(torch.FloatTensor).to(device)
            fake = torch.from_numpy((0.3 - 0.0) * np.random.randn(image.size(0), 1) + 0.0)
            fake = fake.type(torch.FloatTensor).to(device)

            image = image.to(device)

            z = torch.from_numpy(np.random.normal(0, 1, (image.size(0), latent_dim)))
            z = z.type(torch.FloatTensor).to(device)

            # generate image
            fake_image = G(z)

            # detect real
            real_loss = criterion(D(image).view(gt.size(0), 1), gt)
            # detect fake
            fake_loss = criterion(D(fake_image.detach()).view(fake.size(0), 1), fake)
            # total loss
            d_loss = (real_loss + fake_loss) / 2

            # optimize
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            # train to fool D
            g_loss = criterion(D(fake_image).view(gt.size(0), 1), gt)

            # optimize
            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            batches_done = epoch * len(dataset) + index

            if batches_done % verbose_interval == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, epochs, index, len(dataset), d_loss.item(), g_loss.item())
                )
            
            fake_image.view(fake_image.size(0), -1)
            if batches_done % save_interval == 0:
                save_image(fake_image.data[:25], "DCGAN/result/%d.png" % batches_done, nrow=5, normalize=True)

def main(
    dataset,
    image_size,
):
    epochs = 150
    latent_dim = 200

    G = Generator(latent_dim=latent_dim)
    D = Discriminator()

    G.apply(weights_init_normal)
    D.apply(weights_init_normal)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    G.to(device)
    D.to(device)

    optimizer_G = optim.Adam(G.parameters())
    optimizer_D = optim.Adam(D.parameters())

    criterion = nn.BCELoss()

    train(
        epochs=epochs,
        dataset=dataset,
        latent_dim=latent_dim,
        G=G,
        optimizer_G=optimizer_G,
        D=D,
        optimizer_D=optimizer_D,
        criterion=criterion,
        device=device,
        verbose_interval=100,
        save_interval=500
    )