
import functools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torchvision.utils import save_image

from dataset import AnimeFace, DanbooruPortrait
from utils import Status, save_args, add_args
from nnutils import sample_nnoise, update_ema, get_device
from nnutils.loss import NonSaturatingLoss, r1_regularizer
from thirdparty.diffaugment import DiffAugment

from .model import Generator, Discriminator, init_weight_normal

def train(
    max_iters, dataset, latent_dim, sampler, const_z,
    G, G_ema, D, optimizer_G, optimizer_D,
    policy, gp_lambda, ema_decay,
    device, amp, save=1000
):

    status = Status(max_iters)
    scaler = GradScaler() if amp else None

    loss = NonSaturatingLoss()
    gp = r1_regularizer()

    if policy is not '':
        augment = functools.partial(DiffAugment, policy=policy)
    else:
        augment = lambda x: x

    while status.batches_done < max_iters:
        for real in dataset:
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()

            real = real.to(device)
            z = sampler((real.size(0), latent_dim))

            '''Discriminator'''
            with autocast(amp):
                # D(real)
                real_aug = augment(real)
                real_prob = D(real_aug)
                # D(G(z))
                fake = G(z)
                fake_aug = augment(fake)
                fake_prob = D(fake_aug.detach())
                # loss
                D_loss = loss.d_loss(real_prob, fake_prob)
                D_loss = D_loss + gp(real, D, scaler) * gp_lambda

            if scaler is not None:
                scaler.scale(D_loss).backward()
                scaler.step(optimizer_D)
            else:
                D_loss.backward()
                optimizer_D.step()

            '''Generator'''
            with autocast(amp):
                # D(G(z))
                fake_prob = D(fake_aug)
                # loss
                G_loss = loss.g_loss(fake_prob)

            if scaler is not None:
                scaler.scale(G_loss).backward()
                scaler.step(optimizer_G)
            else:
                G_loss.backward()
                optimizer_G.step()

            if G_ema is not None:
                update_ema(G, G_ema, ema_decay)

            # save
            if status.batches_done % save == 0:
                with torch.no_grad():
                    if G_ema is not None:
                        images = G_ema(const_z)
                        state_dict = G_ema.state_dict()
                    else:
                        G.eval()
                        images = G(const_z)
                        G.train()
                        state_dict = G.state_dict()
                save_image(
                    images, f'implementations/TransGAN/result/{status.batches_done}.jpg',
                    nrow=4, normalize=True, value_range=(-1, 1))
                torch.save(state_dict, f'implementations/TransGAN/result/G_{status.batches_done}.pt')
            save_image(fake, f'./running.jpg', nrow=4, normalize=True, value_range=(-1, 1))

            # updates
            loss_dict = dict(
                G=G_loss.item() if not torch.isnan(G_loss).any() else 0.,
                D=D_loss.item() if not torch.isnan(D_loss).any() else 0.
            )
            status.update(**loss_dict)
            if scaler is not None:
                scaler.update()

            if status.batches_done == max_iters:
                break

    status.plot_loss()

def main(parser):

    parser = add_args(parser,
        dict(
            image_channels = [3, 'channels of the output image'],
            latent_dim     = [128, 'dimension of latent input'],
            g_depths       = [[5, 4, 2], 'number of transformer blocks in each reslution'],
            bottom_width   = [8, 'first resolution'],
            g_embed_dim    = [1024, 'dimension of embedding in G. times of 4'],
            g_num_heads    = [4, 'number of heads in multi-head attention in G'],
            g_mlp_ratio    = [4, 'ratio for dimension of features of hidden layers in mlp in G'],
            g_use_qkv_bias = [False, 'use bias for query, key and value in attention in G'],
            g_dropout      = [0.1, 'dropout probability in G'],
            g_attn_dropout = [0.1, 'dropout probability for heatmap in attetion in G'],
            g_act_name     = ['gelu', 'activation function in G'],
            g_norm_name    = ['ln', 'normalization layer name in G'],
            patch_size     = [8, 'size of each patch'],
            d_depth        = [7, 'number of encoders in D'],
            d_embed_dim    = [384, 'dimension of embedding in D. times of 4'],
            d_num_heads    = [4, 'number of heads in multi-head attention in D'],
            d_mlp_ratio    = [4, 'ratio for dimension of features of hidden layers in mlp in G'],
            d_use_qkv_bias = [False, 'use bias for query, key and value in attention in D'],
            d_dropout      = [0.1, 'dropout probability in D'],
            d_attn_dropout = [0.1, 'dropout probability for heatmap in attetion in D'],
            d_act_name     = ['gelu', 'activation function in D'],
            d_norm_name    = ['ln', 'normalization layer name in D'],
            lr             = [0.00001, 'learning rate'],
            ttur           = [False, 'use TTUR'],
            beta1          = [0.5, 'beta1'],
            beta2          = [0.999, 'beta2'],
            gp_lambda      = [10., 'lambda for gradient penalty'],
            policy         = ['color,translation', 'policy for DiffAugment'],
            ema            = [False, 'exponential moving average'],
            ema_decay      = [0.999, 'decay for EMA']))
    args = parser.parse_args()
    save_args(args)

    amp = not args.disable_amp and not args.disable_gpu
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

    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs
    sampler = functools.partial(sample_nnoise, device=device)
    const_z = sample_nnoise((16, args.latent_dim), device=device)

    # models
    expected_g_depths_len = Generator.depths_len_from_target_width(args.image_size, args.bottom_width)
    assert expected_g_depths_len == len(args.g_depths), \
          f'Expected number of args for "--g-depths" for your setting is {expected_g_depths_len}. ' \
        + f'You have {len(args.g_depths)}.'

    G = Generator(
        args.g_depths, args.latent_dim, args.image_channels,
        args.bottom_width, args.g_embed_dim, args.g_num_heads,
        args.g_mlp_ratio, args.g_use_qkv_bias, args.g_dropout, args.g_attn_dropout,
        args.g_act_name, args.g_norm_name
    )
    D = Discriminator(
        args.d_depth, args.image_size, args.patch_size,
        args.image_channels, args.d_embed_dim, args.d_num_heads,
        args.d_mlp_ratio, args.d_use_qkv_bias, args.d_dropout, args.d_attn_dropout,
        args.d_act_name, args.d_norm_name
    )
    G.apply(init_weight_normal)
    D.apply(init_weight_normal)
    G.to(device)
    D.to(device)

    if args.ema:
        G_ema = Generator(
            args.g_depths, args.latent_dim, args.image_channels,
            args.bottom_width, args.g_embed_dim, args.g_num_heads,
            args.g_mlp_ratio, args.g_use_qkv_bias, args.g_dropout, args.g_attn_dropout,
            args.g_act_name, args.g_norm_name
        )
        G_ema.to(device)
        update_ema(G, G_ema, 0.)
    else: G_ema = None

    # optimizers
    betas = (args.beta1, args.beta2)
    if args.ttur:
        g_lr = args.lr / 2
        d_lr = args.lr * 2
    else: g_lr, d_lr = args.lr, args.lr
    optimizer_G = optim.Adam(G.parameters(), lr=g_lr, betas=betas)
    optimizer_D = optim.Adam(D.parameters(), lr=d_lr, betas=betas)

    train(
        args.max_iters, dataset, args.latent_dim, sampler, const_z,
        G, G_ema, D, optimizer_G, optimizer_D,
        args.policy, args.gp_lambda, args.ema_decay,
        device, amp, args.save
    )
