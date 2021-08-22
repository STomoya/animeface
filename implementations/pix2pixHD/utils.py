
import os
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image

from dataset import AnimeFaceXDoG, DanbooruPortraitXDoG, to_loader
from utils import Status, add_args, save_args
from nnutils import get_device
from nnutils.loss import LSGANLoss

from .model import Generator, Discriminator, init_weight_normal

l1 = nn.L1Loss()

def feature_matching(real_feats, fake_feats):
    feat_loss = 0
    layer_weight = 4 / len(real_feats)
    for real_feat, fake_feat in zip(real_feats, fake_feats):
        feat_loss += l1(real_feat, fake_feat) * layer_weight
    return feat_loss

def update_lr(optimizer, delta):
    for param_group in optimizer.param_groups:
        param_group['lr'] -= delta

def train_global(
    dataset, epochs,
    D, G, optimizer_D, optimizer_G,
    feat_lambda, d_num_scale, lr_delta,
    device, amp, save=1000
):
    assert lr_delta > 0
    max_iters = len(dataset) * epochs

    status = Status(max_iters)
    scaler = GradScaler()
    loss = LSGANLoss()
    D_input = lambda x, y: torch.cat([x, y], dim=1)

    while status.batches_done < max_iters:
        for rgb, line in dataset:
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()

            rgb = rgb.to(device)
            line = line.to(device)

            '''Discriminator'''
            with autocast(amp):
                # D(dst)
                real_outs = D(D_input(line, rgb))
                # D(G(src))
                _, fake = G.global_G(line)
                fake_outs = D(D_input(line, fake.detach()))
                # loss
                D_loss = 0
                for i in range(d_num_scale):
                    D_loss += loss.d_loss(real_outs[i][0], fake_outs[i][0])

            if scaler is not None:
                scaler.scale(D_loss).backward()
                scaler.step(optimizer_D)
            else:
                D_loss.backward()
                optimizer_D.step()

            '''Generator'''
            with autocast(amp):
                # D(dst)
                real_outs = D(D_input(line, rgb))
                # D(G(src))
                fake_outs = D(D_input(line, fake))
                # loss
                G_loss = 0
                for i in range(d_num_scale):
                    G_loss += loss.g_loss(fake_outs[i][0])
                    if feat_lambda > 0:
                        G_loss += feature_matching(real_outs[i][1], fake_outs[i][1]) \
                                  * feat_lambda / d_num_scale

            if scaler is not None:
                scaler.scale(G_loss).backward()
                scaler.step(optimizer_G)
            else:
                G_loss.backward()
                optimizer_G.step()

            # save
            if status.batches_done % save == 0:
                image_grid = _image_grid(line, fake, rgb, num_images=6)
                save_image(
                    image_grid, f'implementations/pix2pixHD/result/global_{status.batches_done}.jpg',
                    nrow=6, normalize=True, value_range=(-1, 1))

            # updates
            loss_dict = dict(
                G=G_loss.item() if not torch.isnan(G_loss).any() else 0,
                D=D_loss.item() if not torch.isnan(D_loss).any() else 0
            )
            status.update(**loss_dict)
            if scaler is not None:
                scaler.update()

            # exit training
            if status.batches_done == max_iters:
                break

        # linearly decrease lr to 0 each epoch
        if status.batches_done > max_iters // 2 + 1: # +1 to avoid lr=0
            update_lr(optimizer_G, lr_delta)
            update_lr(optimizer_D, lr_delta)

    save_dict = dict(
        G=G.state_dict(),
        D=D.state_dict()
    )
    torch.save(save_dict, 'implementations/pix2pixHD/result/global_final.pt')

def train_local(
    dataset, epochs,
    D, G, optimizer_D, optimizer_G,
    feat_lambda, d_num_scale, lr_delta,
    fine_from, lr, betas,
    device, amp, save=1000
):
    assert lr_delta > 0
    max_iters = len(dataset) * epochs
    fine_from_iter = len(dataset) * fine_from

    status = Status(max_iters)
    scaler = GradScaler()
    loss = LSGANLoss()
    D_input = lambda x, y: torch.cat([x, y], dim=1)

    while status.batches_done < max_iters:
        for rgb, line in dataset:
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()

            rgb = rgb.to(device)
            line = line.to(device)

            '''Discriminator'''
            with autocast(amp):
                # D(dst)
                real_outs = D(D_input(line, rgb))
                # D(G(src))
                fake, _ = G(line)
                fake_outs = D(D_input(line, fake.detach()))
                # loss
                D_loss = 0
                for i in range(d_num_scale):
                    D_loss += loss.d_loss(real_outs[i][0], fake_outs[i][0])

            if scaler is not None:
                scaler.scale(D_loss).backward()
                scaler.step(optimizer_D)
            else:
                D_loss.backward()
                optimizer_D.step()

            '''Generator'''
            with autocast(amp):
                # D(dst)
                real_outs = D(D_input(line, rgb))
                # D(G(src))
                fake_outs = D(D_input(line, fake))
                # loss
                G_loss = 0
                for i in range(d_num_scale):
                    G_loss += loss.g_loss(fake_outs[i][0])
                    if feat_lambda > 0:
                        G_loss += feature_matching(real_outs[i][1], fake_outs[i][1]) \
                                  * feat_lambda / d_num_scale

            if scaler is not None:
                scaler.scale(G_loss).backward()
                scaler.step(optimizer_G)
            else:
                G_loss.backward()
                optimizer_G.step()

            # save
            if status.batches_done % save == 0:
                image_grid = _image_grid(line, fake, rgb, num_images=6)
                save_image(
                    image_grid, f'implementations/pix2pixHD/result/{status.batches_done}.jpg',
                    nrow=6, normalize=True, value_range=(-1, 1))
                torch.save(
                    G.state_dict(), f'implementations/pix2pixHD/result/G_{status.batches_done}.pt')

            # updates
            loss_dict = dict(
                G=G_loss.item() if not torch.isnan(G_loss).any() else 0,
                D=D_loss.item() if not torch.isnan(D_loss).any() else 0
            )
            status.update(loss_dict)
            if scaler is not None:
                scaler.update()

            # start fine-tuning
            if status.batches_done == fine_from_iter:
                optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=betas)
                print('start fine-tuning.')
                if torch.cuda.is_available(): # reduce VRAM usage
                    torch.cuda.empty_cache()

            # exit training
            if status.batches_done == max_iters:
                break

        # linearly decrease lr to 0 each epoch
        if status.batches_done > max_iters // 2 + 1: # +1 to avoid lr=0
            update_lr(optimizer_G, lr_delta)
            update_lr(optimizer_D, lr_delta)

    status.plot_loss('./loss_local')

def _image_grid(src, gen, dst, num_images=6):
    B = src.size(0)
    srcs = src.expand(-1, 3, -1, -1).chunk(B, dim=0) # expand to match rgb image
    gens = gen.chunk(B, dim=0)
    dsts = dst.chunk(B, dim=0)

    images = []
    for index, (src, gen, dst) in enumerate(zip(srcs, gens, dsts)):
        images.extend([src, gen, dst])
        if index == num_images-1:
            break
    return torch.cat(images, dim=0)

def main(parser):

    parser = add_args(parser,
        dict(
            input_channels=[1, 'input image channels'],
            target_channels=[3, 'output image channels'],
            channels=[32, 'channel width multiplier'],
            local_num_blocks=[3, 'number of resblocks in local G'],
            global_num_blocks=[3, 'number of reblocks in global G'],
            global_num_downs=[4, 'number of down sampling blocks in global G'],
            g_norm_name=['in', 'normalization layer name'],
            g_act_name=['relu', 'activation function name'],
            d_num_scale=[3, 'number of scales'],
            d_norm_name=['in', 'normalization layer name'],
            d_act_name=['lrelu', 'activatoin function name'],
            g_epochs=[200, 'epochs to train global G'],
            g_d_scale=[2, 'number of scales in D when training global G'],
            l_epochs=[100, 'epochs to train local G'],
            l_d_scale=[3, 'number of scales in D when training local G'],
            fine_from=[10, 'when to start fine-tune when training local G'],
            lr=[0.0002, 'learning rate'],
            betas=[[0.5, 0.999], 'betas'],
            feat_lambda=[10., 'lambda for feature matching loss']))
    args = parser.parse_args()
    save_args(args)

    amp = not args.disable_amp
    device = get_device(not args.disable_gpu)

    def get_dataset(name, image_size_):
        if name == 'animeface': return AnimeFaceXDoG(image_size_, args.min_year)
        elif name == 'danbooru': return DanbooruPortraitXDoG(image_size_)
        else: raise Exception('no such dataset')

    G = Generator(
        args.input_channels, args.target_channels, args.channels,
        args.local_num_blocks, args.global_num_blocks, args.global_num_downs,
        args.g_norm_name, args.g_act_name
    )
    D = Discriminator(
        args.input_channels+args.target_channels, args.channels, args.d_num_scale,
        args.d_norm_name, args.d_act_name
    )
    G.apply(init_weight_normal)
    D.apply(init_weight_normal)
    G.to(device)
    D.to(device)

    if not os.path.exists('implementations/pix2pixHD/result/global_final.pt'):
        print('This will only train the global generator. Run the same command again to resume the training for the local generator.')
        # dataset
        dataset = get_dataset(args.dataset, args.image_size//2)
        dataset = to_loader(dataset, args.batch_size)
        # optimizer
        # only update global G
        optimizer_G = optim.Adam(G.global_G.parameters(), lr=args.lr, betas=args.betas)
        # only update finest 'g_d_scale' Ds
        iterables = [D.discriminates[i].parameters() for i in range(args.g_d_scale)]
        optimizer_D = optim.Adam(itertools.chain(*iterables), lr=args.lr, betas=args.betas)

        lr_delta = args.lr / args.g_epochs / 2

        train_global(
            dataset, args.g_epochs,
            D, G, optimizer_D, optimizer_G,
            args.feat_lambda, args.g_d_scale, lr_delta,
            device, amp, 1000
        )
    else:
        # load pretrained global G
        state_dict = torch.load('implementations/pix2pixHD/result/global_final.pt')
        G.load_state_dict(state_dict['G'])
        D.load_state_dict(state_dict['D'])
        # dataset
        dataset = get_dataset(args.dataset, args.image_size)
        dataset = to_loader(dataset, args.batch_size)
        # optimizer
        optimizer_G = optim.Adam(G.local_G.parameters(), lr=args.lr, betas=args.betas)
        optimizer_D = optim.Adam(D.parameters(), lr=args.lr, betas=args.betas)

        lr_delta = args.lr / args.l_epochs / 2

        train_local(
            dataset, args.l_epochs,
            D, G, optimizer_D, optimizer_G,
            args.feat_lambda, args.l_d_scale, lr_delta,
            args.fine_from, args.lr, args.betas,
            device, amp, 1000
        )
