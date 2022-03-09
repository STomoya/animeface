
import copy
import torch
import torch.optim as optim
from torchvision.utils import save_image

from dataset import AnimeFace, DanbooruPortrait
from utils import Status, add_args, save_args
from nnutils import loss, sample_nnoise, update_ema, freeze, MiniAccelerator, init

from .model import Generator, Discriminator

def train(args,
    max_iters, dataset, test_size, latent_dim,
    G, G_ema, D, optimizer_G, optimizer_D,
    gp_lambda,
    amp, save, logfile, loginterval
):
    status = Status(max_iters, logfile is None, logfile, loginterval)
    adv_fn = loss.NonSaturatingLoss()
    gp_fn  = loss.r1_regularizer()

    accelerator = MiniAccelerator(amp)
    G, G_ema, D, optimizer_G, optimizer_D, dataset = accelerator.prepare(
        G, G_ema, D, optimizer_G, optimizer_D, dataset)
    const_z = sample_nnoise(test_size, accelerator.device)

    status.log_training(args, G, D)

    while not status.is_end():
        for real in dataset:
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()
            z = sample_nnoise((real.size(0), latent_dim), accelerator.device)

            with accelerator.autocast():
                # G forward
                fake = G(z)

                # D forward (SG)
                real_prob = D(real)
                fake_prob = D(fake.detach())

                # loss
                adv_loss = adv_fn.d_loss(real_prob, fake_prob)
                gp_loss = 0
                if gp_lambda > 0:
                    gp_loss = gp_fn(real, D, accelerator.scaler) * gp_lambda
                D_loss = adv_loss + gp_loss

            accelerator.backward(D_loss)
            optimizer_D.step()

            with accelerator.autocast():
                # D forward
                fake_prob = D(fake)

                # loss
                G_loss = adv_fn.g_loss(fake_prob)

            accelerator.backward(G_loss)
            optimizer_G.step()
            update_ema(G, G_ema, 0.999, True)

            if status.batches_done % save == 0:
                with torch.no_grad():
                    images = G_ema(const_z)
                save_image(images, f'implementations/VAN/result/{status.batches_done}.jpg',
                    nrow=4, normalize=True, value_range=(-1, 1))
                torch.save(G.state_dict(), f'implementations/VAN/result/model{status.batches_done}.pth')
            save_image(fake, 'running.jpg', normalize=True, value_range=(-1, 1))

            accelerator.update()
            status.update(G=G_loss.item(), D=D_loss.item())
            if status.is_end():
                break
    status.plot_loss()

def main(parser):
    parser = add_args(parser,
        dict(
            num_test         = [16, 'number of samples for eval'],
            image_channels   = [3, 'image channels'],

            latent_dim       = [128, 'input latent dimension'],
            bottom           = [4, 'bottom width'],
            channels         = [64, 'channel width'],
            max_channels     = [int, 'maximum channel width'],
            blocks_per_scale = [2, 'number of blocks per scale'],
            norm_name        = ['ln', 'normalization layer name'],
            act_name         = ['gelu', 'activation function name'],
            layers           = [[3, 3, 9, 3], 'layers'],

            lr               = [0.0001, 'learning rate'],
            betas            = [[0.5, 0.99], 'betas'],
            ttur             = [False, 'use TTUR'],
            gp_lambda        = [10., 'lambda for gradient penalty']))
    args = parser.parse_args()
    save_args(args)

    amp = not args.disable_amp and not args.disable_gpu

    # dataset
    if args.dataset == 'animeface':
        dataset = AnimeFace.asloader(
            args.batch_size, (args.image_size, args.min_year),
            pin_memory=not args.disable_gpu)
    if args.dataset == 'danbooru':
        dataset = DanbooruPortrait.asloader(
            args.batch_size, (args.image_size, args.num_images),
            pin_memory=not args.disable_gpu)

    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs

    test_size = (args.num_test, args.latent_dim)

    # models
    G = Generator(
        args.latent_dim, args.image_size, args.bottom, args.channels,
        args.max_channels, args.blocks_per_scale, args.image_channels,
        args.norm_name, args.act_name)
    G.apply(init().N002)
    G_ema = copy.deepcopy(G)
    freeze(G_ema)
    D = Discriminator(
        args.layers, args.channels, args.max_channels,
        args.image_channels, args.norm_name, args.act_name)
    D.apply(init().N002)

    # optimizers
    if args.ttur: g_lr, d_lr = args.lr / 2, args.lr * 2
    else: g_lr, d_lr = args.lr, args.lr
    optimizer_G = optim.Adam(G.parameters(), lr=g_lr, betas=args.betas)
    optimizer_D = optim.Adam(D.parameters(), lr=d_lr, betas=args.betas)

    train(args,
        args.max_iters, dataset, test_size, args.latent_dim,
        G, G_ema, D, optimizer_G, optimizer_D,
        args.gp_lambda,
        amp, args.save, args.log_file, args.log_interval)
