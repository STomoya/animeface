
import torch
import torch.optim as optim
from torchvision.utils import save_image

from .model import Generator, Discriminator

from dataset import AnimeFaceOneHot
from utils import Status, save_args, add_args
from nnutils import get_device, sample_nnoise, init
from nnutils.loss import LSGANLoss

def train(
    max_iters, dataset, latent_dim,
    G, optimizer_G,
    D, optimizer_D,
    device, save_interval
):

    status = Status(max_iters)
    loss   = LSGANLoss()

    while status.batches_done < max_iters:
        for index, (image, label) in enumerate(dataset, 1):
            real = image.to(device)
            z = sample_nnoise((real.size(0), latent_dim), device)
            label = label.float().to(device)

            # generate image
            fake = G(z, label)
            # D(real)
            real_prob = D(real, label)
            # D(G(z))
            fake_prob = D(fake.detach(), label)

            D_loss = loss.d_loss(real_prob, fake_prob)

            # optimize
            optimizer_D.zero_grad()
            D_loss.backward()
            optimizer_D.step()

            # train to fool D
            fake_prob = D(fake, label)
            G_loss = loss.g_loss(fake_prob)

            # optimize
            optimizer_G.zero_grad()
            G_loss.backward()
            optimizer_G.step()

            if status.batches_done % save_interval == 0:
                save_image(
                    fake, f'implementations/cGAN/result/{status.batches_done}.png',
                    normalize=True, value_range=(-1, 1))
                torch.save(G.state_dict(), f'implementations/cGAN/result/G_{status.batches_done}.pt')

            # updates
            loss_dict = dict(
                g=G_loss.item() if not torch.any(torch.isnan(G_loss)) else 0,
                d=D_loss.item() if not torch.any(torch.isnan(D_loss)) else 0,)
            status.update(**loss_dict)

            if status.batches_done == max_iters:
                break

    status.plot_loss()

def main(parser):
    parser = add_args(parser,
        dict(latent_dim=[200]))
    args = parser.parse_args()
    save_args(args)

    device = get_device(not args.disable_gpu)

    dataset = AnimeFaceOneHot.asloader(
        args.batch_size, (args.image_size, ),
        pin_memory=not args.disable_gpu)

    G = Generator(args.latent_dim, dataset.dataset.num_classes)
    D = Discriminator(dataset.dataset.num_classes)
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
