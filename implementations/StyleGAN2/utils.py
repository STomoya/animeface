
import functools

import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image
import numpy as np

from dataset import AnimeFace, DanbooruPortrait
from utils import Status, save_args, add_args
from nnutils import get_device, sample_nnoise, update_ema
from nnutils.loss import NonSaturatingLoss, r1_regularizer, calc_grad
from thirdparty.diffaugment import DiffAugment

from .model import Generator, Discriminator, init_weight_N01

def pl_penalty(styles, images, pl_mean, scaler=None):
    '''path length regularizer'''
    # mul noise
    num_pixels = images.size()[2:].numel()
    noise = torch.randn(images.size(), device=images.device) / np.sqrt(num_pixels)
    outputs = (images * noise).sum()

    gradients = calc_grad(outputs, styles, scaler)

    # calc penalty
    gradients = gradients.pow(2).sum(dim=1).sqrt()
    return (gradients - pl_mean).pow(2).mean()

def update_pl_mean(old, new, decay=0.99):
    '''exponential moving average'''
    return decay * old + (1 - decay) * new

def train(
    max_iter, dataset, sampler, const_z, latent_dim,
    G, G_ema, D, optimizer_G, optimizer_D,
    r1_lambda, pl_lambda, d_k, g_k, policy,
    device, amp,
    save=1000
):

    status  = Status(max_iter)
    pl_mean = 0.
    loss    = NonSaturatingLoss()
    r1_loss = r1_regularizer()
    scaler  = GradScaler() if amp else None
    augment = functools.partial(DiffAugment, policy=policy)

    if G_ema is not None:
        G_ema.eval()

    while status.batches_done < max_iter:
        for index, real in enumerate(dataset):
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            real = real.to(device)

            '''discriminator'''
            z = sampler((real.size(0), latent_dim))
            with autocast(amp):
                # D(x)
                real_aug = augment(real)
                real_prob = D(real_aug)
                # D(G(z))
                fake, _ = G(z)
                fake_aug = augment(fake)
                fake_prob = D(fake_aug.detach())
                # loss
                if status.batches_done % d_k == 0 \
                    and r1_lambda > 0 \
                    and status.batches_done != 0:
                    # lazy regularization
                    r1 = r1_loss(real, D, scaler)
                    D_loss = r1 * r1_lambda * d_k
                else:
                    # gan loss on other iter
                    D_loss = loss.d_loss(real_prob, fake_prob)

            if scaler is not None:
                scaler.scale(D_loss).backward()
                scaler.step(optimizer_D)
            else:
                D_loss.backward()
                optimizer_D.step()

            '''generator'''
            z = sampler((real.size(0), latent_dim))
            with autocast(amp):
                # D(G(z))
                fake, style = G(z)
                fake_aug = augment(fake)
                fake_prob = D(fake_aug)
                # loss
                if status.batches_done % g_k == 0 \
                    and pl_lambda > 0 \
                    and status.batches_done != 0:
                    # lazy regularization
                    pl = pl_penalty(style, fake, pl_mean, scaler)
                    G_loss = pl * pl_lambda * g_k
                    avg_pl = np.mean(pl.detach().cpu().numpy())
                    pl_mean = update_pl_mean(pl_mean, avg_pl)
                else:
                    # gan loss on other iter
                    G_loss = loss.g_loss(fake_prob)

            if scaler is not None:
                scaler.scale(G_loss).backward()
                scaler.step(optimizer_G)
            else:
                G_loss.backward()
                optimizer_G.step()

            if G_ema is not None:
                update_ema(G, G_ema)

            # save
            if status.batches_done % save == 0:
                with torch.no_grad():
                    images, _ = G_ema(const_z)
                save_image(images, f'implementations/StyleGAN2/result/{status.batches_done}.jpg', nrow=4, normalize=True, value_range=(-1, 1))
                torch.save(G_ema.state_dict(), f'implementations/StyleGAN2/result/G_{status.batches_done}.pt')
            save_image(fake, f'running.jpg', nrow=4, normalize=True, value_range=(-1, 1))

            # updates
            loss_dict = dict(
                G=G_loss.item() if not torch.isnan(G_loss).any() else 0,
                D=D_loss.item() if not torch.isnan(D_loss).any() else 0
            )
            status.update(**loss_dict)
            if scaler is not None:
                scaler.update()

            if status.batches_done == max_iter:
                break

    status.plot_loss()

def main(parser):

    parser = add_args(parser,
        dict(
            image_channels   = [3, 'number of channels for the generated image'],
            style_dim        = [512, 'style feature dimension'],
            channels         = [32, 'channel width multiplier'],
            max_channels     = [512, 'maximum channels'],
            block_num_conv   = [2, 'number of convolution layers in residual block'],
            map_num_layers   = [8, 'number of layers in mapping network'],
            map_lr           = [0.01, 'learning rate for mapping network'],
            disable_map_norm = [False, 'disable pixel normalization in mapping network'],
            mbsd_groups      = [4, 'number of groups in mini_batch standard deviation'],
            lr               = [0.001, 'learning rate'],
            beta1            = [0., 'beta1'],
            beta2            = [0.99, 'beta2'],
            g_k              = [8, 'for lazy regularization. calculate perceptual path length loss every g_k iters'],
            d_k              = [16, 'for lazy regularization. calculate gradient penalty each d_k iters'],
            r1_lambda        = [10, 'lambda for r1'],
            pl_lambda        = [0., 'lambda for perceptual path length loss'],
            policy           = ['color,translation', 'policy for DiffAugment'],))
    args = parser.parse_args()
    save_args(args)

    normalize = not args.disable_map_norm
    betas = (args.beta1, args.beta2)

    amp = not args.disable_amp
    device = get_device(not args.disable_gpu)

    # dataset
    if args.dataset == 'animeface':
        dataset = AnimeFace.asloader(
            args.batch_size, (args.image_size, args.min_year),
            pin_memory=not args.disable_gpu)
    elif args.dataset == 'danbooru':
        dataset = DanbooruPortrait.asloader(
            args.batch_size, (args.image_size, args.num_images),
            pin_memory=not args.disable_gpu)

    # random noise sampler
    sampler = functools.partial(sample_nnoise, device=device)
    # const input for eval
    const_z = sample_nnoise((16, args.style_dim), device=device)

    # models
    G = Generator(
            args.image_size, args.image_channels, args.style_dim, args.channels, args.max_channels,
            args.block_num_conv, args.map_num_layers, normalize, args.map_lr)
    G_ema = Generator(
            args.image_size, args.image_channels, args.style_dim, args.channels, args.max_channels,
            args.block_num_conv, args.map_num_layers, normalize, args.map_lr)
    D = Discriminator(
            args.image_size, args.image_channels, args.channels, args.max_channels,
            args.block_num_conv, args.mbsd_groups)
    ## init
    G.init_weight(
        map_init_func=functools.partial(init_weight_N01, lr=args.map_lr),
        syn_init_func=init_weight_N01)
    G_ema.eval()
    update_ema(G, G_ema, decay=0)
    D.apply(init_weight_N01)

    G.to(device)
    G_ema.to(device)
    D.to(device)

    # optimizer
    if args.pl_lambda > 0:
        g_ratio = args.g_k / (args.g_k + 1)
        g_lr = args.lr * g_ratio
        g_betas = (betas[0]**g_ratio, betas[1]**g_ratio)
    else: g_lr, g_betas = args.lr, betas

    if args.r1_lambda > 0:
        d_ratio = args.d_k / (args.d_k + 1)
        d_lr = args.lr * d_ratio
        d_betas = (betas[0]**d_ratio, betas[1]**d_ratio)
    else: d_lr, d_betas = args.lr, betas

    optimizer_G = optim.Adam(G.parameters(), lr=g_lr, betas=g_betas)
    optimizer_D = optim.Adam(D.parameters(), lr=d_lr, betas=d_betas)

    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs

    train(
        args.max_iters, dataset, sampler, const_z, args.style_dim,
        G, G_ema, D, optimizer_G, optimizer_D,
        args.r1_lambda, args.pl_lambda, args.d_k, args.g_k,
        args.policy, device, amp, args.save
    )
