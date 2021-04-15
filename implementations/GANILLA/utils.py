
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import random_split
from torchvision.utils import save_image

from ..general import AnimeFaceCelebADataset, DanbooruPortraitCelebADataset, to_loader
from ..general import save_args, Status, get_device
from ..gan_utils.losses import LSGANLoss
from ..gan_utils import init

from .model import Generator, Discriminator

l1 = nn.L1Loss()

def train(
    max_iters, dataset, test,
    GA, GH, DA, DH, optimizer_G, optimizer_D,
    cycle_lambda,
    amp, device, save
):
    
    status = Status(max_iters)
    loss   = LSGANLoss()
    scaler = GradScaler() if amp else None

    while status.batches_done < max_iters:
        for anime, human in dataset:
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()

            anime = anime.to(device)
            human = human.to(device)

            with autocast(amp):
                '''generate images'''
                AH = GH(anime)
                HA = GA(human)
                AHA = GA(AH)
                HAH = GH(HA)

                '''discriminator'''
                real_anime, _ = DA(anime)
                real_human, _ = DH(human)
                fake_anime, _ = DA(HA.detach())
                fake_human, _ = DH(AH.detach())

                # loss
                adv_anime = loss.d_loss(real_anime, fake_anime)
                adv_human = loss.d_loss(real_human, fake_human)
                D_loss = adv_anime + adv_human
            
            if scaler is not None:
                scaler.scale(D_loss).backward()
                scaler.step(optimizer_D)
            else:
                D_loss.backward()
                optimizer_D.step()

            with autocast(amp):
                '''generator'''
                fake_anime, _ = DA(HA)
                fake_human, _ = DH(AH)

                # loss
                adv_anime = loss.g_loss(fake_anime)
                adv_human = loss.g_loss(fake_human)
                cycle_anime = l1(AHA, anime)
                cycle_human = l1(HAH, human)
                G_loss = adv_anime + adv_human \
                    + (cycle_anime + cycle_human) * cycle_lambda
            
            if scaler is not None:
                scaler.scale(G_loss).backward()
                scaler.step(optimizer_G)
            else:
                G_loss.backward()
                optimizer_G.step()

            # save
            if status.batches_done % save == 0:
                with torch.no_grad():
                    GA.eval()
                    GH.eval()
                    ah = GH(test[0])
                    ha = GH(test[1])
                    GA.train()
                    GH.train()
                image_grid = _image_grid(test[0], test[1], ah, ha)
                save_image(image_grid, f'implementations/GANILLA/result/{status.batches_done}.jpg',
                    nrow=4*3, normalize=True, range=(-1, 1))
                ckpt = dict(ga=GA.state_dict(), gh=GH.state_dict())
                torch.save(ckpt, f'implementations/GANILLA/result/G_{status.batches_done}.pt')
            save_image(AH, 'running_AH.jpg', normalize=True, range=(-1, 1))
            save_image(HA, 'running_HA.jpg', normalize=True, range=(-1, 1))

            # updates
            loss_dict = dict(
                G=G_loss.item() if not torch.isnan(G_loss).any() else 0,
                D=D_loss.item() if not torch.isnan(D_loss).any() else 0
            )
            status.update(loss_dict)
            if scaler is not None:
                scaler.update()
            
            if status.batches_done == max_iters:
                break
    status.plot()

def _image_grid(a, h, ah, ha):
    _split = lambda x: x.chunk(x.size(0), dim=0)
    a_s = _split(a)
    hs  = _split(h)
    ahs = _split(ah)
    has = _split(ha)
    images = []
    for a, h, ah, ha in zip(a_s, hs, ahs, has):
        images.extend([a, h, ha, ah])
    return torch.cat(images, dim=0)

def add_arguments(parser):
    parser.add_argument('--num-test', default=6, type=int, help='number of images to use for test')

    parser.add_argument('--image-channels', default=3, type=int, help='input/output image channels')
    parser.add_argument('--bottom-width', default=8, type=int, help='bottom width')
    parser.add_argument('--num-downs', default=None, type=int, help='number of down/up sampling')
    parser.add_argument('--num-feats', default=3, type=int, help='number of features to return from the encoder')
    parser.add_argument('--g-channels', default=32, type=int, help='channel width multiplier for G')
    parser.add_argument('--hid-channels', default=128, type=int, help='channels in decoder')
    parser.add_argument('--layer-num-blocks', default=2, type=int, help='number of blocks in one GANILLA layer')
    parser.add_argument('--g-disable-sn', default=False, action='store_true', help='disable spectral norm in G')
    parser.add_argument('--g-bias', default=False, action='store_true', help='enable bias in G')
    parser.add_argument('--g-norm-name', default='in', choices=['in', 'bn'], help='normalization name in G')
    parser.add_argument('--g-act-name', default='lrelu', choices=['lrelu', 'relu'], help='activation function in G')

    parser.add_argument('--num-layers', default=3, type=int, help='number of layers in D')
    parser.add_argument('--d-channels', default=32, type=int, help='channel width multiplier for D')
    parser.add_argument('--d-disable-sn', default=False, action='store_true', help='do not use spectral normalization in D')
    parser.add_argument('--d-disable-bias', default=False, action='store_true', help='do not use bias in D')
    parser.add_argument('--d-norm-name', default='in', choices=['bn', 'in'], help='normalization layer name in D')
    parser.add_argument('--d-act-name', default='relu', choices=['relu', 'lrelu'], help='activation function in D')

    parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
    parser.add_argument('--betas', default=[0.5, 0.999], type=float, nargs=2, help='betas')
    parser.add_argument('--cycle-lambda', default=10., type=float, help='lambda for cycle consistency loss')
    return parser

def main(parser):
    parser = add_arguments(parser)
    args = parser.parse_args()
    save_args(args)

    amp = not args.disable_amp and not args.disable_gpu
    device = get_device(not args.disable_gpu)

    # dataset
    if args.dataset == 'animeface':
        dataset = AnimeFaceCelebADataset(args.image_size, args.min_year)
    elif args.dataset == 'danbooru':
        dataset = DanbooruPortraitCelebADataset(args.image_size, num_images=args.num_images+args.num_test)
    dataset, test = random_split(dataset, [len(dataset)-args.num_test, args.num_test])
    # train
    dataset = to_loader(dataset, args.batch_size)
    # test
    test = to_loader(test, args.num_test, shuffle=False, use_gpu=False)
    test_batch = next(iter(test))
    test_batch = (test_batch[0].to(device), test_batch[1].to(device))

    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs

    # models
    GA = Generator(
        args.image_size, args.image_channels, args.bottom_width,
        args.num_downs, args.num_feats, args.g_channels, args.hid_channels,
        args.layer_num_blocks, not args.g_disable_sn, args.g_bias,
        args.g_norm_name, args.g_act_name
    )
    GH = Generator(
        args.image_size, args.image_channels, args.bottom_width,
        args.num_downs, args.num_feats, args.g_channels, args.hid_channels,
        args.layer_num_blocks, not args.g_disable_sn, args.g_bias,
        args.g_norm_name, args.g_act_name
    )
    DA = Discriminator(
        args.image_size, args.image_channels, args.num_layers,
        args.d_channels, not args.d_disable_sn, not args.d_disable_bias,
        args.d_norm_name, args.d_act_name
    )
    DH = Discriminator(
        args.image_size, args.image_channels, args.num_layers,
        args.d_channels, not args.d_disable_sn, not args.d_disable_bias,
        args.d_norm_name, args.d_act_name
    )
    GA.apply(init().N002)
    GH.apply(init().N002)
    DA.apply(init().N002)
    DH.apply(init().N002)
    GA.to(device)
    GH.to(device)
    DA.to(device)
    DH.to(device)

    # optimizers
    optimizer_G = optim.Adam(
        itertools.chain(GA.parameters(), GH.parameters()),
        lr=args.lr, betas=args.betas
    )
    optimizer_D = optim.Adam(
        itertools.chain(DA.parameters(), DH.parameters()),
        lr=args.lr, betas=args.betas
    )

    train(
        args.max_iters, dataset, test_batch,
        GA, GH, DA, DH,
        optimizer_G, optimizer_D,
        args.cycle_lambda,
        amp, device, args.save
    )