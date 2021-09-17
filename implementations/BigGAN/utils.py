
from functools import partial

import torch
import torch.optim as optim

from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image

from .model import Generator, Discriminator

from dataset import AnimeFace, Danbooru
from utils import save_args, add_args, Status
from nnutils import get_device, sample_nnoise, update_ema, init
from nnutils.loss import HingeLoss
from thirdparty.diffaugment import DiffAugment

def train(
    max_iters, dataset, z_dim, batch_size,
    G, G_ema, D, optimizer_G, optimizer_D,
    loss, augment, device, save_interval=1000
):
    status = Status(max_iters)
    const_z = sample_nnoise((25, z_dim), device)

    scaler = GradScaler()

    while status.batches_done < max_iters:
        for image in dataset:
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()

            real = image.to(device)
            z = sample_nnoise((batch_size, z_dim), device=device)

            '''Discriminator'''
            with autocast():
                # D(x)
                real = augment(real)
                real_prob = D(real)
                # D(G(z))
                fake = G(z)
                fake = augment(fake)
                fake_prob = D(fake.detach())
                # loss
                D_loss = loss.d_loss(real_prob, fake_prob)

            scaler.scale(D_loss).backward()
            scaler.step(optimizer_D)

            '''Generator'''
            with autocast():
                # D(G(z))
                fake_prob = D(fake)
                # loss
                G_loss = loss.g_loss(fake_prob)

            scaler.scale(G_loss).backward()
            scaler.step(optimizer_G)

            if G_ema is not None:
                update_ema(G_ema, G)

            if status.batches_done % save_interval == 0:
                with torch.no_grad():
                    if G_ema is not None:
                        images = G_ema(const_z)
                        state_dict = G_ema.state_dict()
                    else:
                        G.eval()
                        images = G(const_z)
                        state_dict = G.state_dict()
                        G.train()
                save_image(
                    images, f'implementations/BigGAN/result/{status.batches_done}.png',
                    nrow=5, normalize=True, value_range=(-1, 1))
                torch.save(state_dict, f'implementations/BigGAN/result/G_{status.batches_done}.pt')

            # updates
            loss_dict = dict(
                g=G_loss.item() if not torch.any(torch.isnan(G_loss)) else 0,
                d=D_loss.item() if not torch.any(torch.isnan(D_loss)) else 0,)
            status.update(**loss_dict)
            if scaler is not None:
                scaler.update()

            if status.batches_done == max_iters:
                break

    status.plot_loss()

def main(parser):

    parser = add_args(parser,
        dict(
            channels = [64, 'channel width multiplier'],
            deep     = [False, 'use deep model'],
            z_dim    = [120, 'latent dimension'],
            g_lr     = [0.0002, 'learning rate for G'],
            d_lr     = [0.00005, 'learning rate for D'],
            betas    = [[0., 0.999], 'betas'],
            ema      = [False, 'use EMA'],
            policy   = ['color,translation', 'policy for DiffAugment'])
    )
    args = parser.parse_args()
    save_args(args)

    device = get_device(not args.disable_gpu)
    G = Generator(args.image_size, args.z_dim, args.deep, args.channels)
    if args.ema:
        G_ema = Generator(args.image_size, args.z_dim, args.deep, args.channels)
    else:
        G_ema = None
    D = Discriminator(args.image_size, args.deep, args.channels)
    G.apply(init().xavier)
    D.apply(init().xavier)

    G.to(device)
    D.to(device)

    if args.dataset == 'animeface':
        dataset = AnimeFace.asloader(
            args.batch_size, (args.image_size, args.min_year),
            pin_memory=not args.disable_gpu)
    elif args.dataset == 'danbboru':
        dataset = Danbooru.asloader(
            args.batch_size, (args.image_size, args.num_images),
            pin_memory=not args.disable_gpu)
    augment = partial(DiffAugment, policy=args.policy)

    optimizer_G = optim.Adam(G.parameters(), lr=args.g_lr, betas=args.betas)
    optimizer_D = optim.Adam(D.parameters(), lr=args.d_lr, betas=args.betas)

    loss = HingeLoss()

    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs

    train(
        args.max_iters, dataset, args.z_dim, args.batch_size,
        G, G_ema, D, optimizer_G, optimizer_D,
        loss, augment, device, args.save
    )
