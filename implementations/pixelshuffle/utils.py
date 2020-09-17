
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import numpy as np

from .model import Generator, Discriminator, weights_init_normal

from ..general import LabeledAnimeFaceDataset, to_loader

def train(
    epochs,
    dataset,
    label_type,
    latent_dim,
    G,
    optimizer_G,
    D,
    optimizer_D,
    validity_criterion,
    label_criterion,
    device,
    verbose_interval,
    save_interval
):
    
    for epoch in range(epochs):
        for index, (image, label) in enumerate(dataset, 1):
            # label smoothing
            gt   = torch.from_numpy((1.0 - 0.7) * np.random.randn(image.size(0), 1) + 0.7)
            gt   = gt.type(torch.FloatTensor).to(device)
            fake = torch.from_numpy((0.3 - 0.0) * np.random.randn(image.size(0), 1) + 0.0)
            fake = fake.type(torch.FloatTensor).to(device)

            # class label
            index_label = label[label_type].max(1)[1].view(image.size(0))
            index_label = index_label.type(torch.LongTensor).to(device)
            # generate class label
            gen_label = torch.from_numpy(np.random.randint(0, 28, (image.size(0), )))
            gen_label = gen_label.type(torch.LongTensor).to(device)

            # real image
            image = image.to(device)

            # noise
            noise = torch.from_numpy(np.random.normal(0, 1, (image.size(0), latent_dim)))
            noise = noise.type(torch.FloatTensor).to(device)



            # generate image
            fake_image = G(noise, gen_label)


            '''
            Train Discriminator
            '''

            # detect real
            validity, label_prob = D(image)
            real_validity_loss = validity_criterion(validity.view(gt.size(0), 1), gt)
            real_label_loss = label_criterion(label_prob, index_label)
            # detect fake
            validity, label_prob = D(fake_image.detach())
            fake_validity_loss = validity_criterion(validity.view(fake.size(0), 1), fake)
            fake_label_loss = label_criterion(label_prob, gen_label)
            # total loss
            d_validity_loss = (real_validity_loss + fake_validity_loss) / 2
            d_label_loss = (fake_validity_loss + fake_label_loss) / 2
            d_loss = (d_validity_loss + d_label_loss) / 2

            # optimize
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()


            '''
            Train Generator
            '''


            # train to fool D
            validity, label_prob = D(fake_image)
            g_validity_loss = validity_criterion(validity.view(gt.size(0), 1), gt)
            g_label_loss = label_criterion(label_prob, gen_label)
            g_loss = (g_validity_loss + g_label_loss) / 2

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
                save_image(fake_image.data[:25], "implementations/pixelshuffle/result/%d.png" % batches_done, nrow=5, normalize=True)

def main():
    batch_size = 32
    image_size = 128
    epochs = 150
    latent_dim = 200
    label_type, label_dim = 'i2v', 28
    # label_type, label_dim = 'year', 20

    dataset = LabeledAnimeFaceDataset(image_size)
    dataset = to_loader(dataset, batch_size)

    G = Generator(latent_dim=latent_dim, label_dim=label_dim)
    D = Discriminator(label_dim=label_dim)

    G.apply(weights_init_normal)
    D.apply(weights_init_normal)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    G.to(device)
    D.to(device)

    optimizer_G = optim.Adam(G.parameters())
    optimizer_D = optim.Adam(D.parameters())

    validity_criterion = nn.MSELoss()
    label_criterion = nn.CrossEntropyLoss()

    train(
        epochs=epochs,
        dataset=dataset,
        label_type=label_type,
        latent_dim=latent_dim,
        G=G,
        optimizer_G=optimizer_G,
        D=D,
        optimizer_D=optimizer_D,
        validity_criterion=validity_criterion,
        label_criterion=label_criterion,
        device=device,
        verbose_interval=100,
        save_interval=500
    )