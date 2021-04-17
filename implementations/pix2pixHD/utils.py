
import os
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image

from ..general import XDoGAnimeFaceDataset, XDoGDanbooruPortraitDataset, to_loader
from ..general import Status, get_device, save_args
from ..gan_utils import AdaBelief
from ..gan_utils.losses import LSGANLoss, GANLoss, HingeLoss

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

def train(
    dataset, max_iter,
    G, D, optimizer_G, optimizer_D,
    feat_lambda, d_num_scale,
    device, amp, save=1000
):

    status = Status(max_iter)
    loss = LSGANLoss()
    scaler = GradScaler() if amp else None
    D_input = lambda x, y: torch.cat([x, y], dim=1)

    while status.batches_done < max_iter:
        for rgb, line in dataset:
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

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
                for real_out, fake_out in zip(real_outs, fake_outs):
                    D_loss += loss.d_loss(real_out[0], fake_out[0]) / d_num_scale
            
            if scaler is not None:
                scaler.scale(D_loss).backward()
                scaler.step(optimizer_D)
            else:
                D_loss.backward()
                optimizer_D.step()

            with autocast(amp):
                # D(dst)
                real_outs = D(D_input(line, rgb))
                # D(G(src))
                fake_outs = D(D_input(line, fake))
                # loss
                G_loss = 0
                for real_out, fake_out in zip(real_outs, fake_outs):
                    G_loss += loss.g_loss(fake_out[0]) / d_num_scale
                    if feat_lambda > 0:
                        G_loss += feature_matching(real_out[1], fake_out[1]) / d_num_scale * feat_lambda
            
            if scaler is not None:
                scaler.scale(G_loss).backward()
                scaler.step(optimizer_G)
            else:
                G_loss.backward()
                optimizer_G.step()

            # save
            if status.batches_done % save == 0:
                image_grid = _image_grid(line, fake, rgb, num_images=6)
                save_image(image_grid, f'implementations/pix2pixHD/result/{status.batches_done}.jpg', nrow=6, normalize=True, value_range=(-1, 1))
                torch.save(G.state_dict(), f'implementations/pix2pixHD/result/G_{status.batches_done}.pt')
            
            # updates
            loss_dict = dict(
                G=G_loss.item() if not torch.isnan(G_loss).any() else 0,
                D=D_loss.item() if not torch.isnan(D_loss).any() else 0
            )
            status.update(loss_dict)
            if scaler is not None:
                scaler.update()

            if status.batches_done == max_iter:
                break

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
                save_image(image_grid, f'implementations/pix2pixHD/result/global_{status.batches_done}.jpg', nrow=6, normalize=True, value_range=(-1, 1))
            
            # updates
            loss_dict = dict(
                G=G_loss.item() if not torch.isnan(G_loss).any() else 0,
                D=D_loss.item() if not torch.isnan(D_loss).any() else 0
            )
            status.update(loss_dict)
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
                save_image(image_grid, f'implementations/pix2pixHD/result/{status.batches_done}.jpg', nrow=6, normalize=True, value_range=(-1, 1))
                torch.save(G.state_dict(), f'implementations/pix2pixHD/result/G_{status.batches_done}.pt')
            
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
                optimizer_G = AdaBelief(G.parameters(), lr=lr, betas=betas)
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

    status.plot('./loss_local')

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
    
def add_arguments(parser):
    parser.add_argument('--input-channels', default=1, type=int, help='number of channels of the src image')
    parser.add_argument('--target-channels', default=3, type=int, help='number of channels of the dst images')
    parser.add_argument('--channels', default=32, type=int, help='channelwidth multiplier')
    parser.add_argument('--local-num-blocks', default=3, type=int, help='number of residual blocks in the local G')
    parser.add_argument('--global-num-blocks', default=9, type=int, help='number of residual blocks in the global G')
    parser.add_argument('--global-num-downs', default=4, type=int, help='number of down-sampling blocks in global G')
    parser.add_argument('--g-norm-name', default='in', choices=['in', 'bn'], help='normalization layer name for G')
    parser.add_argument('--g-act-name', default='relu', choices=['relu', 'lrelu'], help='activation layer name for G')
    parser.add_argument('--d-num-scale', default=3, type=int, help='number of scales to discriminate')
    parser.add_argument('--d-norm-name', default='in', choices=['in', 'bn'], help='normalization layer name for D')
    parser.add_argument('--d-act-name', default='lrelu', choices=['relu', 'lrelu'], help='activation layer name for D')
    
    parser.add_argument('--train-once', default=False, action='store_true', help='train model at once')
    parser.add_argument('--g-epochs', default=100*2, type=int, help='epochs to train the global G')
    parser.add_argument('--g-d-scale', default=2, type=int, help='number of scales to discriminate when training global G')
    parser.add_argument('--l-epochs', default=50*2, type=int, help='epochs to train the local G')
    parser.add_argument('--l-d-scale', default=3, type=int, help='number of scales to discriminate when training local G')
    parser.add_argument('--fine-from', default=10, type=int, help='the epoch to start fine tuning on when training local G')

    parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.5, type=float, help='beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta2')
    parser.add_argument('--feat-lambda', default=10., type=float, help='lambda for feature matching loss')
    return parser


def main(parser):

    parser = add_arguments(parser)
    args = parser.parse_args()
    save_args(args)

    betas = (args.beta1, args.beta2)
    
    amp = not args.disable_amp
    device = get_device(not args.disable_gpu)

    def get_dataset(name, image_size_):
        if name == 'animeface': return XDoGAnimeFaceDataset(image_size_, args.min_year)
        elif name == 'danbooru': return XDoGDanbooruPortraitDataset(image_size_)
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

    if args.train_once:
        raise Exception('Not Tested')
        # dataset
        dataset = get_dataset(args.dataset, args.image_size)
        dataset = to_loader(dataset, args.batch_size)
        max_iter = len(dataset) * args.g_epochs
        # optimizer
        optimizer_G = AdaBelief(G.parameters(), lr=args.lr, betas=betas)
        optimizer_D = AdaBelief(D.parameters(), lr=args.lr, betas=betas)
        train(
            dataset, max_iter,
            G, D, optimizer_G, optimizer_D,
            args.feat_lambda, args.d_num_scale,
            device, amp, 1000
        )

    else:
        if not os.path.exists('implementations/pix2pixHD/result/global_final.pt'):
            print('This will only train the global generator. Run the same command again to resume the training for the local generator.')
            # dataset
            dataset = get_dataset(args.dataset, args.image_size//2)
            dataset = to_loader(dataset, args.batch_size)
            # optimizer
            # only update global G
            optimizer_G = AdaBelief(G.global_G.parameters(), lr=args.lr, betas=betas)
            # only update finest 'g_d_scale' Ds
            iterables = [D.discriminates[i].parameters() for i in range(args.g_d_scale)]
            optimizer_D = AdaBelief(itertools.chain(*iterables), lr=args.lr, betas=betas)

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
            optimizer_G = AdaBelief(G.local_G.parameters(), lr=args.lr, betas=betas)
            optimizer_D = AdaBelief(D.parameters(), lr=args.lr, betas=betas)

            lr_delta = args.lr / args.l_epochs / 2

            train_local(
                dataset, args.l_epochs,
                D, G, optimizer_D, optimizer_G,
                args.feat_lambda, args.l_d_scale, lr_delta,
                args.fine_from, args.lr, betas,
                device, amp, 1000
            )