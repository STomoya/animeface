
import functools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torchvision.utils import save_image

from ..general import YearAnimeFaceDataset, DanbooruPortraitDataset, to_loader
from ..general import save_args, get_device, Status
from ..gan_utils import sample_nnoise, DiffAugment, update_ema
from ..gan_utils.losses import HingeLoss, WGANLoss, GANLoss, GradPenalty

from .model import Generator, Discriminator, init_weight_normal

def train(
    max_iters, dataset, latent_dim, sampler, const_z,
    G, G_ema, D, optimizer_G, optimizer_D,
    loss_type, policy, gp_lambda, ema_decay,
    device, amp, save=1000
):
    
    status = Status(max_iters)
    scaler = GradScaler() if amp else None

    if loss_type == 'hinge':
        loss = HingeLoss()
    elif loss_type == 'wgan_gp':
        loss = WGANLoss()
        gp = GradPenalty().gradient_penalty
    elif loss_type == 'gan':
        loss = GANLoss()
        gp = GradPenalty().r1_regularizer
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
                if loss_type == 'wgan_gp':
                    D_loss = D_loss + gp(real, fake, D, scaler, center=1.) * gp_lambda
                if loss_type == 'gan':
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
                    nrow=4, normalize=True, range=(-1, 1))
                torch.save(state_dict, f'implementations/TransGAN/result/G_{status.batches_done}.pt')
            save_image(fake, f'./running.jpg', nrow=4, normalize=True, range=(-1, 1))

            # updates
            loss_dict = dict(
                G=G_loss.item() if not torch.isnan(G_loss).any() else 0.,
                D=D_loss.item() if not torch.isnan(D_loss).any() else 0.
            )
            status.update(loss_dict)
            if scaler is not None:
                scaler.update()

            if status.batches_done == max_iters:
                break


def add_arguments(parser):

    # model
    parser.add_argument('--image-channels', default=3, type=int, help='channels of the output image')
    # G
    parser.add_argument('--latent-dim', default=128, type=int, help='dimension of latent input')
    parser.add_argument('--g-depths', default=[5, 4, 2], type=int, nargs='+', help='number of transformer blocks in each reslution')
    parser.add_argument('--bottom-width', default=8, type=int, help='first resolution')
    parser.add_argument('--g-embed-dim', default=1024, type=int, help='dimension of embedding in G. times of 4')
    parser.add_argument('--g-num-heads', default=4, type=int, help='number of heads in multi-head attention in G')
    parser.add_argument('--g-mlp-ratio', default=4, type=int, help='ratio for dimension of features of hidden layers in mlp in G')
    parser.add_argument('--g-use-qkv-bias', default=False, action='store_true', help='use bias for query, key and value in attention in G')
    parser.add_argument('--g-dropout', default=0.1, type=float, help='dropout probability in G')
    parser.add_argument('--g-attn-dropout', default=0.1, type=float, help='dropout probability for heatmap in attetion in G')
    parser.add_argument('--g-act-name', default='gelu', choices=['gelu'], help='activation function in G')
    parser.add_argument('--g-norm-name', default='ln', choices=['ln'], help='normalization layer name in G')
    # D
    parser.add_argument('--patch-size', default=8, type=int, help='size of each patch')
    parser.add_argument('--d-depth', default=7, type=int, help='number of encoders in D')
    parser.add_argument('--d-embed-dim', default=384, type=int, help='dimension of embedding in D. times of 4')
    parser.add_argument('--d-num-heads', default=4, type=int, help='number of heads in multi-head attention in D')
    parser.add_argument('--d-mlp-ratio', default=4, type=int, help='ratio for dimension of features of hidden layers in mlp in G')
    parser.add_argument('--d-use-qkv-bias', default=False, action='store_true', help='use bias for query, key and value in attention in D')
    parser.add_argument('--d-dropout', default=0.1, type=float, help='dropout probability in D')
    parser.add_argument('--d-attn-dropout', default=0.1, type=float, help='dropout probability for heatmap in attetion in D')
    parser.add_argument('--d-act-name', default='gelu', choices=['gelu'], help='activation function in D')
    parser.add_argument('--d-norm-name', default='ln', choices=['ln'], help='normalization layer name in D')

    # training
    parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
    parser.add_argument('--ttur', default=False, action='store_true', help='use TTUR')
    parser.add_argument('--beta1', default=0.5, type=float, help='beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta2')
    parser.add_argument('--loss-type', default='gan', choices=['hinge', 'wgan_gp', 'gan'], help='loss type')
    parser.add_argument('--gp-lambda', default=10., type=float, help='lambda for gradient penalty')
    parser.add_argument('--policy', default='color,translation', type=str, help='policy for DiffAugment')
    parser.add_argument('--ema', default=False, action='store_true', help='exponential moving average')
    parser.add_argument('--ema-decay', default=0.999, type=float, help='decay for EMA')
    return parser

def main(parser):

    parser = add_arguments(parser)
    args = parser.parse_args()
    save_args(args)

    amp = not args.disable_amp and not args.disable_gpu
    device = get_device(not args.disable_gpu)

    # dataset
    if args.dataset == 'animeface':
        dataset = YearAnimeFaceDataset(args.image_size, args.min_year)
    elif args.dataset == 'danbooru':
        dataset = DanbooruPortraitDataset(args.image_size, num_images=args.num_images)
    dataset = to_loader(dataset, args.batch_size)

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
        args.loss_type, args.policy, args.gp_lambda, args.ema_decay,
        device, amp, args.save
    )
