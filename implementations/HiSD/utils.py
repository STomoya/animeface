
import functools
import itertools
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image
import numpy as np

from ..general import get_device, Status, save_args
from ..gan_utils import sample_nnoise, update_ema
from ..gan_utils.losses import LSGANLoss

from .dataset import AnimeFace, DanbooruPortrait, Hair, Eye, Glass
from .model import Generator, Discriminator, init_weight_xavier, init_weight_kaiming

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
        status.update(loss_dict)
        if scaler is not None:
            scaler.update()
        
    status.plot()

def _image_grid(real, fake, num_images=8):
    reals = real.chunk(real.size(0), dim=0)
    fakes = fake.chunk(fake.size(0), dim=0)
    images = []
    for index, (real, fake) in enumerate(zip(reals, fakes)):
        images.extend([real, fake])
        if index == num_images-1:
            break
    return torch.cat(images, dim=0)

def add_arguments(parser):
    parser.set_defaults(max_iters=500000)

    # model
    parser.add_argument('--style-dim', default=256, type=int, help='dimension for style code')
    parser.add_argument('--latent-dim', default=128, type=int, help='input noise dimension')
    parser.add_argument('--enc-num-downs', default=2, type=int, help='number of downsampling res-blocks in encoder/decoder')
    parser.add_argument('--map-mid-dim', default=256, type=int, help='dimension of middle layers in mapper network')
    parser.add_argument('--map-num-shared-layers', default=3, type=int, help='number of shared layers for all tags in a category')
    parser.add_argument('--map-num-tag-layers', default=3, type=int, help='number of layers for each tags in a category')
    parser.add_argument('--channels', default=32, type=int, help='channel width multiplier')
    parser.add_argument('--ex-bottom-width', default=8, type=int, help='minimum width before global avgpool in extractor network')
    parser.add_argument('--trans-num-blocks', default=7, type=int, help='number of res-blocks in translator network')
    parser.add_argument('--num-layers', default=3, type=int, help='number of layers in D')
    parser.add_argument('--norm-name', default='in', choices=['in', 'bn'], help='name of normalization layer')
    parser.add_argument('--act-name', default='lrelu', choices=['relu', 'lrelu'], help='name of activation function')
    parser.add_argument('--no-bias', default=True, action='store_true', help='no bias')
    # experimental
    parser.add_argument('--normalize-latent', default=False, action='store_true', help='use pixel norm to input latent in mapper network')
    parser.add_argument('--single-path', default=False, action='store_true', help='use only one branch for all tags')
    parser.add_argument('--affine-each', default=False, action='store_true', help='affine input at each AdaIN layer')
    parser.add_argument('--ret-feat', default=False, action='store_true', help='return features from D. for feature matching loss')

    # data
    parser.add_argument('--category', default=['hair', 'eye', 'glass'], nargs='+', type=str, help='categories')
    parser.add_argument('--image-channels', default=3, type=int, help='channels of images')
    
    # training
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--map-lr', default=0.000001, type=float, help='learning rate for mapper network')
    parser.add_argument('--betas', default=[0., 0.99], type=float, nargs=2, help='betas')
    parser.add_argument('--feat-lambda', default=10., type=float, help='lambda for feature matching loss')
    parser.add_argument('--recons-lambda', default=1., type=float, help='lambda for reconstruction loss')
    parser.add_argument('--style-lambda', default=1., type=float, help='lambda for style loss')
    parser.add_argument('--ema-decay', default=0.999, type=float, help='decay for EMA')
    return parser

CATEGORY = {'hair': Hair, 'eye': Eye, 'glass': Glass}

def main(parser):
    
    parser = add_arguments(parser)
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
    G.apply(init_weight_xavier)
    update_ema(G, G_ema, 0)
    D.apply(init_weight_xavier)

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
