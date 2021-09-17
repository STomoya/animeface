
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from .model import Generator, Discriminator

from dataset import AnimeFaceLabel
from utils import add_args, save_args, Status
from nnutils import get_device, sample_nnoise, init
from nnutils.loss import LSGANLoss

def train(
    max_iters, dataset, latent_dim, num_class,
    G, optimizer_G,
    D, optimizer_D,
    label_criterion,
    device, save_interval
):

    status = Status(max_iters)
    loss = LSGANLoss()

    while status.batches_done < max_iters:
        for index, (image, label) in enumerate(dataset, 1):
            # real label
            real_label = label.to(device)
            # fake label
            fake_label = torch.randint(0, num_class, (image.size(0), )).to(device)

            real = image.to(device)
            z = sample_nnoise((image.size(0), latent_dim), device)

            # generate image
            fake= G(z, fake_label)

            '''
            Train Discriminator
            '''

            # D(real)
            real_prob, label_prob = D(real)
            # D(G(z))
            fake_prob, label_prob = D(fake.detach())

            # loss
            real_label_loss = label_criterion(label_prob, real_label)
            fake_label_loss = label_criterion(label_prob, fake_label)
            label_loss = (real_label_loss + fake_label_loss) / 2
            adv_loss = loss.d_loss(real_prob, fake_prob)
            # total loss
            D_loss = adv_loss + label_loss

            # optimize
            optimizer_D.zero_grad()
            D_loss.backward()
            optimizer_D.step()

            '''
            Train Generator
            '''

            # train to fool D
            fake_prob, label_prob = D(fake)
            fake_label_loss = label_criterion(label_prob, fake_label)
            adv_loss = loss.g_loss(fake_prob)
            G_loss = adv_loss + fake_label_loss

            # optimize
            optimizer_G.zero_grad()
            G_loss.backward()
            optimizer_G.step()


            if status.batches_done % save_interval == 0:
                save_image(
                    fake, f'implementations/ACGAN/result/{status.batches_done}.png',
                    normalize=True, value_range=(-1, 1))
                torch.save(G.state_dict(), f'implementations/ACGAN/result/G_{status.batches_done}.pt')

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

    dataset = AnimeFaceLabel.asloader(
        args.batch_size, (args.image_size, ),
        pin_memory=not args.disable_gpu)

    G = Generator(latent_dim=args.latent_dim, label_dim=dataset.dataset.num_classes)
    D = Discriminator(label_dim=dataset.dataset.num_classes)

    G.apply(init().N002)
    D.apply(init().N002)

    G.to(device)
    D.to(device)

    optimizer_G = optim.Adam(G.parameters())
    optimizer_D = optim.Adam(D.parameters())

    label_criterion = nn.CrossEntropyLoss()

    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs

    train(
        args.max_iters, dataset, args.latent_dim,
        dataset.dataset.num_classes,
        G, optimizer_G, D, optimizer_D,
        label_criterion,
        device, args.save
    )
