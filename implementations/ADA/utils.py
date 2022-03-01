
import torch
import torch.optim as optim
from torchvision.utils import save_image

from dataset import AnimeFace, DanbooruPortrait
from nnutils.accelerate import MiniAccelerator
from utils import Status, save_args, add_args
from nnutils import update_ema, sample_nnoise, freeze, get_device
from nnutils.loss import NonSaturatingLoss, r1_regularizer

from .model import Generator, Discriminator, ADA

def train(
    max_iters, dataset, latent_dim, const_input,
    G, G_ema, D, optimizer_G, optimizer_D,
    gp_lambda, gp_every, augment,
    device, amp, save, log_file
):

    if log_file is not None:
        status = Status(max_iters, False, log_file)
    else: status = Status(max_iters)
    adv_fn = NonSaturatingLoss()
    gp_fn  = r1_regularizer()

    accelerator = MiniAccelerator(amp, device=device)
    G, G_ema, D, optimizer_G, optimizer_D, dataset, augment = accelerator.prepare(
        G, G_ema, D, optimizer_G, optimizer_D, dataset, augment)

    while not status.is_end():
        for real in dataset:
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()
            z = sample_nnoise((real.size(0), latent_dim), device)

            with accelerator.autocast():
                # G(z)
                fake = G(z)
                # augment
                real_aug = augment(real)
                fake_aug = augment(fake)

                # D(real)
                real_prob = D(real_aug)
                # D(G(z))
                fake_prob = D(fake_aug.detach())

                # loss
                adv_loss, gp_loss = 0, 0
                adv_loss = adv_fn.d_loss(real_prob, fake_prob)
                if gp_lambda > 0 and status.batches_done % gp_every == 0:
                    gp_loss = gp_fn(real, D, accelerator.scaler) * gp_lambda
                D_loss = adv_loss + gp_loss

            accelerator.backward(D_loss)
            optimizer_D.step()

            with accelerator.autocast():
                # D(G(z))
                fake_prob = D(fake_aug)

                # loss
                G_loss = adv_fn.g_loss(fake_prob)

            accelerator.backward(G_loss)
            optimizer_G.step()

            update_ema(G, G_ema, copy_buffers=True)
            augment.update_p(real_prob)

            if status.batches_done % save == 0:
                with torch.no_grad():
                    images = G_ema(const_input)
                save_image(
                    images, f'implementations/ADA/result/{status.batches_done}.jpg',
                    nrow=4, normalize=True, value_range=(-1, 1))
                torch.save(
                    G_ema.state_dict(),
                    f'implementations/ADA/result/G_{status.batches_done}.pt')
            save_image(fake, 'running.jpg', normalize=True, value_range=(-1, 1))

            accelerator.update()
            status.update(g=G_loss.item(), d=D_loss.item())

            if status.is_end():
                break

    status.plot_loss()

def main(parser):

    parser = add_args(parser,
        dict(
            num_test          = [16, 'number of images for eval'],
            image_channels    = [3, 'number of image channels'],

            # model
            latent_dim        = [512, 'latent dimension'],
            num_layers        = [14, 'number of layers in G'],
            map_num_layers    = [2, 'number of layers in mapping network'],
            channels          = [32, 'channel base'],
            max_channels      = [512, 'maximum channel width'],
            style_dim         = [512, 'style code dimension'],
            kernel_size       = [3, 'kernel size. 3'],
            no_pixel_norm     = [False, 'no pixel normalization'],
            output_scale      = [0.25, 'scale output tensor with'],
            margin_size       = [10, 'bigger size to work on'],
            first_cutoff      = [2., 'first cutoff'],
            first_stopband    = [2**2.1, 'first stopband'],
            last_stopband_rel = [2**0.3, 'last relative stopband'],
            d_channels        = [32, 'channel base for D'],
            d_max_channels    = [512, 'maximum channels in D'],
            mbsd_group_size   = [4, 'mini-batch stddev group size'],
            mbsd_channels     = [1, 'mini-batch stddev channels'],
            bottom            = [4, 'bottom width in D'],
            gaus_filter_size  = [4, 'filter size in D'],

            # training
            lr                = [0.0025, 'learning rate'],
            map_lr_scale      = [0.01, 'scale learning rate for mapping network with'],
            betas             = [[0., 0.99], 'betas'],
            gp_lambda         = [3., 'lambda for r1'],
            gp_every          = [16, 'calc penalty every'],

            # ADA
            ada_interval      = [4, 'update p every'],
            ada_target_kimg   = [500, 'target k images'],
            ada_threshold     = [0.6, 'ada threshold'],

            logfile           = [str, 'log file']))

    args = parser.parse_args()
    save_args(args)

    device = get_device(not args.disable_gpu)
    amp = not args.disable_amp and not args.disable_gpu

    # dataset
    if args.dataset == 'animeface':
        dataset = AnimeFace.asloader(
            args.batch_size, (args.image_size, args.min_year),
            pin_memory=not args.disable_gpu)
    elif args.dataset == 'danbooru':
        dataset = DanbooruPortrait.asloader(
            args.batch_size, (args.image_size, args.num_images),
            pin_memory=not args.disable_gpu)

    const_input = sample_nnoise((args.num_test, args.latent_dim), device)

    # model
    G = Generator(
        args.image_size, args.latent_dim, args.num_layers,
        args.map_num_layers, args.channels, args.max_channels,
        args.style_dim, not args.no_pixel_norm, args.image_channels,
        args.output_scale, args.margin_size, args.first_cutoff,
        args.first_stopband, args.last_stopband_rel, args.kernel_size)
    G_ema = Generator(
        args.image_size, args.latent_dim, args.num_layers,
        args.map_num_layers, args.channels, args.max_channels,
        args.style_dim, not args.no_pixel_norm, args.image_channels,
        args.output_scale, args.margin_size, args.first_cutoff,
        args.first_stopband, args.last_stopband_rel, args.kernel_size)
    freeze(G_ema)
    update_ema(G, G_ema, 0., copy_buffers=True)
    D = Discriminator(
        args.image_size, args.image_channels, args.d_channels,
        args.d_max_channels, 3, args.mbsd_group_size, args.mbsd_channels,
        args.bottom, args.gaus_filter_size, 'lrelu', 1)

    # optimizer
    optimizer_G = optim.Adam(
        [{'params': G.synthesis.parameters()},
         {'params': G.map.parameters(), 'lr': args.lr*args.map_lr_scale}],
        lr=args.lr, betas=args.betas)
    optimizer_D = optim.Adam(
        D.parameters(), lr=args.lr, betas=args.betas)

    # augment = partial(DiffAugment, policy=args.policy)
    augment = ADA(args.ada_interval, args.ada_target_kimg, args.ada_threshold, args.batch_size,
        xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1,
        brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)

    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs

    train(
        args.max_iters, dataset, args.latent_dim, const_input,
        G, G_ema, D, optimizer_G, optimizer_D,
        args.gp_lambda, args.gp_every, augment,
        device, amp, args.save, args.logfile)
