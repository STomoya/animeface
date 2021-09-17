
import os
import functools

import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image

from ..StyleGAN2.model import Generator, Discriminator, init_weight_N01

from dataset._base import Image, make_default_transform, pilImage
from utils import Status, save_args, add_args
from nnutils import get_device, update_ema, sample_nnoise
from nnutils.loss import NonSaturatingLoss, r1_regularizer
from thirdparty.diffaugment import DiffAugment

'''dataset classes with Image + Blur'''
import glob
import random
class ImageBlur(Image):
    def __init__(self, image_size, resize_ratio=1.):
        transform = make_default_transform(image_size, resize_ratio)
        super().__init__(transform)
        self.blurs = self._load_blur()
    def _load_blur(self):
        raise NotImplementedError()
    def __getitem__(self, index):
        image = self.images[index]
        blur  = self.blurs[index]

        image = pilImage.open(image).convert('RGB')
        blur  = pilImage.open(blur).convert('RGB')

        image = self.transform(image)
        blur  = self.transform(blur)
        return image, blur
    def shuffle_blur(self):
        random.shuffle(self.blurs)

_year_from_path = lambda path: int(os.path.splitext(os.path.basename(path))[0].split('_')[-1])


class AnimeFaceBlur(ImageBlur):
    def __init__(self, image_size, min_year=2005, resize_ratio=1.):
        self.min_year = min_year
        super().__init__(image_size, resize_ratio)
    def _load(self):
        images = glob.glob('/usr/src/data/animefacedataset/images/*')
        if self.min_year != None:
            images = [path for path in images if _year_from_path(path) >= self.min_year]
        return images
    def _load_blur(self):
        blurs = [path.replace('images', 'blur') for path in self.images]
        random.shuffle(blurs)
        return blurs

class DanbooruPortraitBlur(ImageBlur):
    def __init__(self, image_size, num_images=None, resize_ratio=1.2):
        self.num_images = num_images
        super().__init__(image_size, resize_ratio)
    def _load(self):
        images = glob.glob('/usr/src/data/danbooru/2020/*/*.jpg', recursive=True)
        if self.num_images is not None:
            random.shuffle(images)
            images = images[:self.num_images]
        return images
    def _load_blur(self):
        blurs = [path.replace('portraits/portraits', 'portraits/blur') for path in self.images]
        random.shuffle(blurs)
        return blurs

def train(
    max_iter, dataset, sampler, const_z, style_dim,
    G, G_ema, D, optimizer_G, optimizer_D,
    r1_lambda, d_k, policy,
    edge_loss_from,
    device, amp,
    save=1000
):

    status  = Status(max_iter)
    loss    = NonSaturatingLoss()
    gp      = r1_regularizer()
    scaler  = GradScaler() if amp else None
    augment = functools.partial(DiffAugment, policy=policy)


    if G_ema is not None:
        G_ema.eval()

    while status.batches_done < max_iter:
        for index, (real, blur) in enumerate(dataset):
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            real = real.to(device)
            blur = blur.to(device)

            '''discriminator'''
            z = sampler((real.size(0), style_dim))
            with autocast(amp):
                # D(real)
                real_aug = augment(real)
                real_prob = D(real_aug)
                # D(blured)
                blur = augment(blur)
                blur_prob = D(blur)
                # D(G(z))
                fake, _ = G(z)
                fake_aug = augment(fake)
                fake_prob = D(fake_aug.detach())
                # loss
                if status.batches_done % d_k == 0 \
                    and r1_lambda > 0 \
                    and status.batches_done != 0:
                    # lazy regularization
                    r1 = gp(real, D, scaler)
                    D_loss = r1 * r1_lambda * d_k
                else:
                    # gan loss on other iter
                    D_loss = loss.d_loss(real_prob, fake_prob)
                    # add gan loss for blured edges
                    if edge_loss_from > status.batches_done:
                        D_loss = D_loss + loss.fake_loss(blur_prob)

            if scaler is not None:
                scaler.scale(D_loss).backward()
                scaler.step(optimizer_D)
            else:
                D_loss.backward()
                optimizer_D.step()

            '''generator'''
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
                update_ema(G, G_ema)

            # save
            if status.batches_done % save == 0:
                with torch.no_grad():
                    images, _ = G_ema(const_z)
                save_image(
                    images, f'implementations/edge/result/{status.batches_done}.jpg',
                    nrow=4, normalize=True, value_range=(-1, 1))
                torch.save(
                    G_ema.state_dict(),
                    f'implementations/edge/result/G_{status.batches_done}.pt')
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
        dataset.dataset.shuffle_blur()

    status.plot()

def main(parser):

    parser = add_args(parser,
        dict(
            image_channels=[3, 'number of channels for generated images'],
            style_dim=[512, 'style code dimension'],
            channels=[32, 'channel width multiplier'],
            max_channels=[512, 'maximum channels width'],
            block_num_conv=[2, 'number of conv in residual'],
            map_num_layers=[8, 'number of layers mapping network'],
            map_lr=[0.01, 'learning rate for mapping network'],
            disable_map_norm=[False, 'disable pixel norm'],
            mbsd_groups=[4, 'mini batch stddev groups'],
            lr=[0.001, 'learning rate'],
            betas=[[0., 0.99], 'betas'],
            d_k=[16, 'calc gp every'],
            r1_lambda=[10., 'lambda for gp'],
            policy=['color,translation', 'policy for DiffAugment'],
            wait_edge_epoch=[0, 'epochs to wait before adding edge adv loss']
        )
    )
    args = parser.parse_args()
    save_args(args)

    normalize = not args.disable_map_norm
    betas = args.betas

    amp = not args.disable_amp
    device = get_device(not args.disable_gpu)

    # dataset
    if args.dataset == 'animeface':
        dataset = AnimeFaceBlur.asloader(
            args.batch_size, (args.image_size, args.min_year),
            pin_memory=not args.disable_gpu)
    elif args.dataset == 'danbooru':
        dataset = DanbooruPortraitBlur.asloader(
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
    g_lr, g_betas = args.lr, betas

    if args.r1_lambda > 0:
        d_ratio = args.d_k / (args.d_k + 1)
        d_lr = args.lr * d_ratio
        d_betas = (betas[0]**d_ratio, betas[1]**d_ratio)
    else: d_lr, d_betas = args.lr, betas

    optimizer_G = optim.Adam(G.parameters(), lr=g_lr, betas=g_betas)
    optimizer_D = optim.Adam(D.parameters(), lr=d_lr, betas=d_betas)

    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs
    edge_loss_from = len(dataset) * args.wait_edge_epoch

    train(
        args.max_iters, dataset, sampler, const_z, args.style_dim,
        G, G_ema, D, optimizer_G, optimizer_D,
        args.r1_lambda, args.d_k,
        args.policy, edge_loss_from, device, amp, 1000
    )
