
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import numpy as np

from .model import Generator, Discriminator

from ..general import AnimeFaceDataset, to_loader, save_args

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
            gt   = torch.from_numpy((1.2 - 0.7) * np.random.randn(image.size(0), 1) + 0.7)
            gt   = gt.type(torch.FloatTensor).to(device)
            fake = torch.from_numpy((0.3 - 0.0) * np.random.randn(image.size(0), 1) + 0.0)
            fake = fake.type(torch.FloatTensor).to(device)

            image = image.to(device)

            z = torch.from_numpy(np.random.normal(0, 10, (image.size(0), latent_dim)))
            z = z.type(torch.FloatTensor).to(device)

            # generate image
            fake_image = G(z)

            # detect real
            real_loss = criterion(D(image), gt)
            # detect fake
            fake_loss = criterion(D(fake_image.detach()), fake)
            # total loss
            d_loss = (real_loss + fake_loss) / 2

            # optimize
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            # train to fool D
            g_loss = criterion(D(fake_image), gt)

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
            
            if batches_done % save_interval == 0:
                save_image(fake_image.data[:25], "implementations/GAN/result/%d.png" % batches_done, nrow=5, normalize=True)

def add_arguments(parser):
    parser.add_argument('--epochs', default=10, type=int, help='epochs to train')
    parser.add_argument('--latent-dim', default=100, type=int, help='dimension of input latent')
    return parser

def main(parser):

    parser = add_arguments(parser)
    args = parser.parse_args()
    save_args(args)

    dataset = AnimeFaceDataset(args.image_size)
    dataset = to_loader(dataset, args.batch_size)

    G = Generator(latent_dim=args.latent_dim, image_shape=(3, args.image_size, args.image_size))
    D = Discriminator(image_shape=(3, args.image_size, args.image_size))

    if not args.disable_gpu:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    G.to(device)
    D.to(device)

    optimizer_G = optim.Adam(G.parameters())
    optimizer_D = optim.Adam(D.parameters())

    criterion = nn.BCELoss()

    train(
        epochs=args.epochs,
        dataset=dataset,
        latent_dim=args.latent_dim,
        G=G,
        optimizer_G=optimizer_G,
        D=D,
        optimizer_D=optimizer_D,
        criterion=criterion,
        device=device,
        verbose_interval=100,
        save_interval=100
    )