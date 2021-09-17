
import functools
import itertools
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image
import numpy as np

from utils import Status, save_args, add_args
from nnutils import sample_nnoise, update_ema, get_device, init
from nnutils.loss import LSGANLoss

from .dataset import AnimeFace, DanbooruPortrait, Hair, Eye, Glass
from .model import Generator, Discriminator, init_weight_xavier

l1_loss = nn.L1Loss()

def feature_matching(real_feat, fake_feat):
    scale = len(real_feat)
    loss = 0
    for real, fake in zip(real_feat, fake_feat):
        loss = loss + l1_loss(real, fake)
    return loss / scale

_arange_choice = lambda x: int(np.random.choice(np.arange(x)))
def random_ij(tags, ij: tuple=None):
    if ij is None:
        i = _arange_choice(len(tags))
        j = _arange_choice(tags[i])
    else:
        i = ij[0]
        while True:
            j = _arange_choice(tags[i])
            if j != ij[1]:
                break
    return i, j

def train(
    max_iters, dataset, sampler, latent_dim,
    G, G_ema, D, optimizer_G, optimizer_D,
    num_tags, ema_decay,
    recons_lambda, style_lambda, feat_lambda,
    amp, device, save
):

    status = Status(max_iters)
    scaler = GradScaler() if amp else None
    loss = LSGANLoss()
    _refs = [None]*len(num_tags)

    def check_d_output(output):
        if isinstance(output, tuple):
            return output[0], output[1]
        else:
            return output, None

    while status.batches_done < max_iters:
        i, j = random_ij(num_tags)
        _, j_ = random_ij(num_tags, (i, j))
        real = dataset.sample(i, j)
        real = real.to(device)
        z    = sampler((real.size(0), latent_dim))
        # print(real.size(0), z.size(0))

        optimizer_G.zero_grad()
        optimizer_D.zero_grad()

        '''generate images'''
        with autocast(amp):
            # reconstruct
            recons = G(real)
            # reconstruct with style code of itself
            refs = copy.copy(_refs)
            refs[i] = (real, j)
            recons_self_trans = G(real, refs)
            # fake
            refs[i] = (z, j_)
            fake = G(real, refs)
            # reconstruct from fake
            refs[i] = (real, j)
            recons_fake_trans = G(fake, refs)

            '''Discriminator'''
            # D(real)
            output = D(real, i, j)
            real_prob, real_feat = check_d_output(output)
            # D(fake)
            output = D(fake.detach(), i, j_)
            fake_prob, fake_feat = check_d_output(output)
            # D(G(fake))
            output = D(recons_fake_trans.detach(), i, j)
            recons_prob, recons_feat = check_d_output(output)

            # loss
            D_loss = loss.d_loss(real_prob[:, 0], fake_prob[:, 0]) \
                + loss.d_loss(real_prob[:, 1], recons_prob[:, 1])
            if feat_lambda > 0 and real_feat is not None:
                D_loss = D_loss \
                    + feature_matching(real_feat, fake_feat) * feat_lambda

        if scaler is not None:
            scaler.scale(D_loss).backward()
            scaler.step(optimizer_D)
        else:
            D_loss.backward()
            optimizer_D.step()

        '''Generator'''
        with autocast(amp):
            # D(fake)
            output = D(fake, i, j_)
            fake_prob, fake_feat = check_d_output(output)
            # D(G(fake))
            output = D(recons_fake_trans, i, j)
            recons_prob, recons_feat = check_d_output(output)
            # style codes
            style_j_ = G.category_modules[i].map(z, j_)
            style_fake = G.category_modules[i].extract(fake, j_)

            # loss
            G_loss = loss.g_loss(fake_prob[:, 0]) + loss.g_loss(recons_prob[:, 1])
            G_loss = G_loss \
                + l1_loss(style_j_, style_fake) * style_lambda \
                + (l1_loss(recons, real) \
                    + l1_loss(recons_self_trans, real) \
                    + l1_loss(recons_fake_trans, real)) * recons_lambda

        if scaler is not None:
            scaler.scale(G_loss).backward()
            scaler.step(optimizer_G)
        else:
            G_loss.backward()
            optimizer_G.step()

        update_ema(G, G_ema, ema_decay)

        # save
        if status.batches_done % save == 0:
            with torch.no_grad():
                refs[i] = (z, j_)
                fake = G_ema(real, refs)
            images = _image_grid(real, fake)
            save_image(images, f'implementations/HiSD/result/{status.batches_done}_tag{i}_{j}to{j_}.jpg',
                nrow=4, normalize=True, value_range=(-1, 1))
            torch.save(G.state_dict(), f'implementations/HiSD/result/G_{status.batches_done}.pt',)
        save_image(fake, f'running.jpg', nrow=4, normalize=True, value_range=(-1, 1))

        # updates
        loss_dict = dict(
            G=G_loss.item() if not torch.isnan(G_loss).any() else 0,
            D=D_loss.item() if not torch.isnan(D_loss).any() else 0
        )
        status.update(**loss_dict)
        if scaler is not None:
            scaler.update()

    status.plot_loss()

def _image_grid(real, fake, num_images=8):
    reals = real.chunk(real.size(0), dim=0)
    fakes = fake.chunk(fake.size(0), dim=0)
    images = []
    for index, (real, fake) in enumerate(zip(reals, fakes)):
        images.extend([real, fake])
        if index == num_images-1:
            break
    return torch.cat(images, dim=0)

CATEGORY = {'hair': Hair, 'eye': Eye, 'glass': Glass}

def main(parser):

    # parser = add_arguments(parser)
    parser.set_defaults(max_iters=500000)
    add_args(parser,
        dict(
            style_dim             = [256, 'style code dimension'],
            latent_dim            = [128, 'input latent dimension'],
            enc_num_downs         = [2, 'number of downsampling res-blocks in encoder/decoder'],
            map_mid_dim           = [256, 'dimension of middle layers in mapper network'],
            map_num_shared_layers = [3, 'number of shared layers for all tags in a category'],
            map_num_tag_layers    = [3, 'number of layers for each tags in a category'],
            channels              = [32, 'channel width multiplier'],
            ex_bottom_width       = [8, 'minimum width before global avgpool in extractor network'],
            trans_num_blocks      = [7, 'number of res-blocks in translator network'],
            num_layers            = [3, 'number of layers in D'],
            norm_name             = ['in', 'normalization layer name'],
            act_name              = ['lrelu', 'activation function name'],
            no_bias               = [False, 'disable bias'],
            normalize_latent      = [False, 'use pixel norm to input latent'],
            single_path           = [False, 'use only one branch for all tags'],
            affine_each           = [False, 'affine input at each AdaIN layer'],
            ret_feat              = [False, 'return features from D'],
            category              = [['hair', 'eye', 'glass'], 'categories'],
            image_channels        = [3, 'image channels'],
            lr                    = [0.0001, 'learning rate'],
            map_lr                = [0.000001, 'learning rate for mapper network'],
            betas                 = [[0., 0.99], 'betas'],
            feat_lambda           = [10., 'lambda for feature matching loss'],
            recons_lambda         = [1., 'lambda for reconstruction loss'],
            style_lambda          = [1., 'lambda for style loss'],
            ema_decay             = [0.999, 'decay for EMA']))
    args = parser.parse_args()
    save_args(args)

    amp = not args.disable_amp and not args.disable_gpu
    device = get_device(not args.disable_gpu)

    # data
    dataset_classes = []
    for key in args.category:
        dataset_classes.append(CATEGORY[key])
    if args.dataset == 'animeface':
        dataset = AnimeFace(
            args.image_size, args.batch_size, image_dataset_class=dataset_classes
        )
    elif args.dataset == 'danbooru':
        dataset = DanbooruPortrait(
            args.image_size, args.batch_size, image_dataset_class=dataset_classes
        )

    data = dataset.sample(2, 1)
    save_image(data, 'sample.jpg', normalize=True, value_range=(-1, 1))

    sampler = functools.partial(sample_nnoise, device=device)
    const_z = [sample_nnoise((16, args.latent_dim), device=device) for _ in dataset_classes]

    # model
    G = Generator(
        dataset.num_tags, args.image_size, args.image_channels,
        args.image_channels, args.style_dim, args.latent_dim,
        args.enc_num_downs,
        args.map_mid_dim, args.map_num_shared_layers, args.map_num_tag_layers,
        args.channels, args.ex_bottom_width, args.trans_num_blocks,
        args.norm_name, args.act_name, not args.no_bias,
        args.normalize_latent, args.single_path, args.affine_each
    )
    G_ema = Generator(
        dataset.num_tags, args.image_size, args.image_channels,
        args.image_channels, args.style_dim, args.latent_dim,
        args.enc_num_downs,
        args.map_mid_dim, args.map_num_shared_layers, args.map_num_tag_layers,
        args.channels, args.ex_bottom_width, args.trans_num_blocks,
        args.norm_name, args.act_name, not args.no_bias,
        args.normalize_latent, args.single_path, args.affine_each
    )
    D = Discriminator(
        dataset.num_tags, args.image_channels, args.image_channels,
        args.num_layers, args.channels,
        args.norm_name, args.act_name, not args.no_bias,
        args.ret_feat, args.single_path
    )
    G.to(device)
    G_ema.to(device)
    G_ema.eval()
    D.to(device)
    G.apply(init().xavier)
    update_ema(G, G_ema, 0)
    D.apply(init().xavier)

    # optimizers
    g_params = [
        {'params': G.encode.parameters()},
        {'params': G.decode.parameters()},
        {'params': itertools.chain(*[cat.extract.parameters()   for cat in G.category_modules])},
        {'params': itertools.chain(*[cat.translate.parameters() for cat in G.category_modules])},
        {'params': itertools.chain(*[cat.map.parameters()       for cat in G.category_modules]),
            'lr': args.map_lr} # different lr for mapper network
    ]
    optimizer_G = optim.Adam(g_params, lr=args.lr, betas=args.betas)
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr, betas=args.betas)

    train(
        args.max_iters, dataset, sampler, args.latent_dim,
        G, G_ema, D, optimizer_G, optimizer_D,
        dataset.num_tags, args.ema_decay,
        args.recons_lambda, args.style_lambda, args.feat_lambda,
        amp, device, args.save
    )
