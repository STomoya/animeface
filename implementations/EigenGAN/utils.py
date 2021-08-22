
import functools

import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image

from dataset import AnimeFace, DanbooruPortrait
from utils import Status, save_args, add_args
from nnutils import get_device, sample_nnoise, init, update_ema
from nnutils.loss import HingeLoss, r1_regularizer
from thirdparty.diffaugment import DiffAugment

from .model import Generator, Discriminator

@torch.enable_grad()
def orthogonal_regularizer(model):
    loss = 0
    for name, param in model.named_parameters():
        if 'U' in name:
            flat = param.view(param.size(0), -1)
            sym = torch.mm(flat, flat.T)
            eye = torch.eye(sym.size(-1), device=sym.device)
            loss = loss + (sym - eye).abs().sum() * 0.5
    return loss

def train(
    max_iters, dataset, const_input,
    sampler, eps_dim, latent_dim,
    G, G_ema, D, optimizer_G, optimizer_D,
    gp_lambda, ortho_lambda,
    augment,
    device, amp, save=1000
):

    status = Status(max_iters)
    loss   = HingeLoss()
    gp     = r1_regularizer()
    scaler = GradScaler() if amp else None

    while status.batches_done < max_iters:
        for image in dataset:
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()
            real = image.to(device)

            with autocast(amp):
                # generate images
                B = real.size(0)
                eps = sampler((B, eps_dim))
                zs = [sampler((B, latent_dim)) for _ in range(G.num_layers)]
                fake = G(eps, zs)
                # augment
                real_aug = augment(real)
                fake_aug = augment(fake)

                # D(real)
                real_prob = D(real_aug)
                # D(fake)
                fake_prob = D(fake_aug.detach())

                # loss
                adv_loss = loss.d_loss(real_prob, fake_prob)
                if gp_lambda > 0:
                    gp_loss = gp(real, D, scaler) * gp_lambda
                else: gp_loss = 0

                D_loss = adv_loss + gp_loss

            if scaler is not None:
                scaler.scale(D_loss).backward()
                scaler.step(optimizer_D)
            else:
                D_loss.backward()
                optimizer_D.step()

            with autocast(amp):
                # D(fake)
                fake_prob = D(fake_aug)

                # loss
                adv_loss = loss.g_loss(fake_prob)
                if ortho_lambda > 0:
                    ortho_loss = orthogonal_regularizer(G) * ortho_lambda
                else: ortho_loss = 0
                G_loss = adv_loss + ortho_loss

            if scaler is not None:
                scaler.scale(G_loss).backward()
                scaler.step(optimizer_G)
            else:
                G_loss.backward()
                optimizer_G.step()

            if G_ema is not None:
                update_ema(G, G_ema)

            # save
            with torch.no_grad():
                if status.batches_done % save == 0:
                    if const_input[1] == None:
                        input = (const_input[0], [sampler((const_input[0].size(0), latent_dim)) for _ in range(G.num_layers)])
                    else:
                        input = const_input
                    if G_ema is not None:
                        images = G_ema(*input)
                        state_dict = G_ema.state_dict()
                    else:
                        G.eval()
                        images = G(*input)
                        G.train()
                        state_dict = G.state_dict()
                    save_image(images, f'implementations/EigenGAN/result/{status.batches_done}.jpg',
                        normalize=True, value_range=(-1, 1))
                    torch.save(state_dict, f'implementations/EigenGAN/result/G_{status.batches_done}.pt')
            save_image(fake, 'running.jpg', normalize=True, value_range=(-1, 1))

            # updates
            loss_dict = dict(
                G=G_loss.item() if not torch.any(torch.isnan(G_loss)) else 0,
                D=D_loss.item() if not torch.any(torch.isnan(D_loss)) else 0)
            status.update(**loss_dict)
            if scaler is not None:
                scaler.update()

    status.plot_loss()

def main(parser):
    parser = add_args(parser,
        dict(
            num_test       = [16, 'number of const for eval'],
            const_z        = [False, 'subspace input will be const'],
            image_channels = [3, 'image channels'],
            eps_dim        = [512, 'channels of eps input'],
            latent_dim     = [6, 'channels of subspace input'],
            bottom_width   = [4, 'bottom width'],
            g_channels     = [32, 'channel width multiplier'],
            g_max_channels = [512, 'maximum channel width'],
            g_disable_sn   = [False, 'disable spectral norm'],
            g_disable_bias = [False, 'disable bias'],
            g_norm_name    = ['in', 'normalization layer name'],
            g_act_name     = ['lrelu', 'activation function name'],
            d_channels     = [32, 'channel width multiplier'],
            d_max_channels = [512, 'maximum channel width'],
            d_disable_sn   = [False, 'disable spectral norm'],
            d_disable_bias = [False, 'disable bias'],
            d_norm_name    = ['in', 'normalization layer name'],
            d_act_name     = ['lrelu', 'activation function name'],
            lr             = [0.0002, 'learning rate'],
            betas          = [[0.5, 0.999], 'betas'],
            gp_lambda      = [0., 'lambda for r1'],
            ortho_lambda   = [1., 'lambda for orthogonal regularization'],
            policy         = ['color,translation', 'policy for diffaugment'],
            ema            = [False, 'use EMA']))
    args = parser.parse_args()
    save_args(args)

    device = get_device(not args.disable_gpu)
    amp = not args.disable_amp and not args.disable_gpu

    # data
    if args.dataset == 'animeface':
        dataset = AnimeFace.asloader(
            args.batch_size, (args.image_size, args.min_year),
            pin_memory=not args.disable_gpu)
    elif args.dataset == 'danbooru':
        dataset = DanbooruPortrait.asloader(
            args.batch_size, (args.image_size, args.num_images),
            pin_memory=not args.disable_gpu)

    sampler = functools.partial(sample_nnoise, device=device)
    const_eps = sampler((args.num_test, args.eps_dim))
    if args.const_z:
        const_zs  = sampler((args.num_test, args.latent_dim))
    else: const_zs = None
    const_input = (const_eps, const_zs)

    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs

    # model
    G = Generator(
        args.image_size, args.eps_dim, args.latent_dim,
        args.image_channels, args.bottom_width, args.g_channels,
        args.g_max_channels, not args.g_disable_sn, not args.g_disable_bias,
        args.g_norm_name, args.g_act_name
    )
    D = Discriminator(
        args.image_size, args.image_channels, args.bottom_width,
        args.d_channels, args.d_max_channels, not args.d_disable_sn,
        not args.d_disable_bias, args.d_norm_name, args.d_act_name
    )
    G.to(device)
    D.to(device)
    init_func = init().xavier
    G.apply(init_func)
    D.apply(init_func)
    if args.ema:
        G_ema = Generator(
            args.image_size, args.eps_dim, args.latent_dim,
            args.image_channels, args.bottom_width, args.g_channels,
            args.g_max_channels, not args.g_disable_sn, not args.g_disable_bias,
            args.g_norm_name, args.g_act_name
        )
        G_ema.to(device)
        G_ema.eval()
        for param in G_ema.parameters():
            param.requires_grad = False
        update_ema(G, G_ema, 0)
    else: G_ema = None

    # optimizers
    optimizer_G = optim.Adam(G.parameters(), lr=args.lr, betas=args.betas)
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr, betas=args.betas)

    train(
        args.max_iters, dataset, const_input,
        sampler, args.eps_dim, args.latent_dim,
        G, G_ema, D, optimizer_G, optimizer_D,
        args.gp_lambda, args.ortho_lambda,
        functools.partial(DiffAugment, policy=args.policy),
        device, amp, args.save
    )
