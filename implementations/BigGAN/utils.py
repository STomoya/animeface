
import torch
import torch.nn as nn
import torch.optim as optim

from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from .model import Generator, Discriminator, init_weight_ortho

from ..general import AnimeFaceDataset, DanbooruPortraitDataset, to_loader
from ..gan_utils import get_device, GANTrainingStatus, sample_nnoise, DiffAugment
from ..gan_utils.losses import HingeLoss

def grad_accumulate_train(
    max_iter, dataset, z_dim, batch_size, target_batch_size,
    G, G_ema, D, optimizer_G, optimizer_D,
    loss, device,
    verbose_interval=1000, save_interval=1000, save_folder='./implementations/BigGAN/result'
):
    status = GANTrainingStatus()
    const_z = sample_nnoise((25, z_dim))

    scaler = GradScaler()

    acc_size = target_batch_size // batch_size
    print(f'batch size will be {acc_size*batch_size}. Change target_batch_size or batch_size if this is not the desired batch size.')
    bar = tqdm(total=max_iter)

    d_iter = 0
    D_loss = 0
    G_loss = 0

    while True:
        for image in dataset:

            real = image.to(device)
            z = sample_nnoise((batch_size, z_dim), device=device)
            print('a')

            '''Discriminator'''
            with autocast():
                # D(x)
                real_prob = D(real)
                # D(G(z))
                fake = G(z)
                fake_prob = D(fake.detach())
                # loss
                D_loss += loss.d_loss(real_prob, fake_prob)

            d_iter += 1

            if d_iter == acc_size:
                optimizer_D.zero_grad()
                optimizer_G.zero_grad()

                # update discriminator after accumulating target_batch_size samples
                with autocast():
                    D_loss /= acc_size
                scaler.scale(D_loss).backward()
                scaler.step(optimizer_D)

                '''Generator'''
                # update generator using updated discriminator
                with autocast():
                    for _ in range(acc_size):
                        z = sample_nnoise((batch_size, z_dim), device=device)
                        # D(G(z))
                        fake = G(z)
                        fake_prob = D(fake)
                        # loss
                        G_loss += loss.g_loss(fake_prob)
                    G_loss /= acc_size

                scaler.scale(G_loss).backward()
                scaler.step(optimizer_G)

                scaler.update()

                # resets
                d_iter = 0
                D_loss = 0
                G_loss = 0

                # update stats
                status.append(G_loss.item(), D_loss.item())
                bar.update(1)
            
                # verbose and save
                if status.batches_done % verbose_interval == 0 or status.batches_done == 1:
                    print(status)
                if status.batches_done % save_interval == 0 or status.batches_done == 1:
                    status.save_image(save_folder, G, const_z)
                    torch.save(G.state_dict(), f'implementations/BigGAN/G_128_{target_batch_size}_{status.batches_done}.pt')


                if status.batches_done == max_iter:
                    break
            torch.cuda.empty_cache()
        if status.batches_done == max_iter:
            break
    status.plot_loss()


def train(
    epochs, dataset, z_dim, batch_size,
    G, G_ema, D, optimizer_G, optimizer_D,
    loss, device,
    verbose_interval=1000, save_interval=1000, save_folder='./implementations/BigGAN/result'
):
    status = GANTrainingStatus()
    const_z = sample_nnoise((25, z_dim))

    scaler = GradScaler()

    for epoch in range(epochs):
        for image in tqdm(dataset):
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()

            real = image.to(device)
            z = sample_nnoise((batch_size, z_dim), device=device)

            '''Discriminator'''
            with autocast():
                # D(x)
                real = DiffAugment(real, 'color,translation')
                real_prob = D(real)
                # D(G(z))
                fake = G(z)
                fake = DiffAugment(fake, 'color,translation')
                fake_prob = D(fake.detach())
                # loss
                D_loss = loss.d_loss(real_prob, fake_prob)

            scaler.scale(D_loss).backward()
            scaler.step(optimizer_D)

            '''Generator'''
            with autocast():
                # D(G(z))
                fake = G(z)
                fake = DiffAugment(fake, 'color,translation')
                fake_prob = D(fake)
                # loss
                G_loss = loss.g_loss(fake_prob)

            scaler.scale(G_loss).backward()
            scaler.step(optimizer_G)

            scaler.update()

            # G_ema.update(G)

            status.append(G_loss.item(), D_loss.item())
            
            if status.batches_done % verbose_interval == 0 or status.batches_done == 1:
                print(status)
            if status.batches_done % save_interval == 0 or status.batches_done == 1:
                status.save_image(save_folder, G, const_z)
        torch.save(G.state_dict(), f'implementations/BigGAN/G_128_{epoch}epoch_state_dict.pt')
    status.plot_loss()

def main(parser):
    # params
    # data
    image_size = 128
    batch_size = 12
    target_batch_size = 30
    # model
    channels = 64
    deep = False
    z_dim = 120
    decay = 0.9999
    # training
    epochs = 100
    d_lr = 2e-4
    g_lr = 5e-5
    betas = (0., 0.999)

    device = get_device()
    G = Generator(image_size, z_dim, deep, channels)
    D = Discriminator(image_size, deep, channels)
    G.apply(init_weight_ortho)
    D.apply(init_weight_ortho)
    G_ema = EMA(G, decay)

    G.to(device)
    D.to(device)

    dataset = AnimeFaceDataset(image_size)
    dataset = to_loader(dataset, batch_size)

    optimizer_G = optim.Adam(G.parameters(), lr=g_lr, betas=betas)
    optimizer_D = optim.Adam(D.parameters(), lr=d_lr, betas=betas)

    loss = HingeLoss()

    # grad_accumulate_train(
    #     epochs, dataset, z_dim, batch_size, target_batch_size,
    #     G, G_ema, D, optimizer_G, optimizer_D,
    #     loss, device,
    #     1000, 1000
    # )

    train(
        epochs, dataset, z_dim, batch_size,
        G, G_ema, D, optimizer_G, optimizer_D,
        loss, device,
        1000, 1000
    )
