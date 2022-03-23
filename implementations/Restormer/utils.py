
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image

import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import cv2

from dataset import AAHQ, AnimeFace, DanbooruPortrait
from utils import Status, save_args, add_args, make_image_grid
from nnutils import get_device, loss

from .model import Generator, Discriminator

def optim_step(loss, optimizer, scaler):
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
    else:
        loss.backward()
        optimizer.step()


def train(args,
    max_iters, dataset,
    G, D, optim_G, optim_D,
    pixel_lambda,
    device, amp, save, logfile, loginterval
):

    status = Status(max_iters, False, logfile, loginterval, __name__)
    scaler = GradScaler() if amp else None
    adv_fn = loss.LSGANLoss()
    l1_fn  = nn.L1Loss()

    status.log_training(args, G, D)

    while not status.is_end():
        for line, real in dataset:
            optim_D.zero_grad()
            optim_G.zero_grad()
            line = line.to(device)
            real = real.to(device)

            with autocast(amp):
                # G forward
                fake = G(line)

                # D forward (SG)
                real_prob = D(torch.cat([real, line], dim=1))
                fake_prob = D(torch.cat([fake.detach(), line], dim=1))

                # loss
                D_loss = adv_fn.d_loss(real_prob, fake_prob)

            optim_step(D_loss, optim_D, scaler)

            with autocast(amp):
                # D forward
                fake_prob = D(torch.cat([fake, line], dim=1))

                # loss
                adv_loss = adv_fn.g_loss(fake_prob)
                pixel_loss = l1_fn(fake, real) * pixel_lambda
                G_loss = adv_loss + pixel_loss

            optim_step(G_loss, optim_G, scaler)

            if status.batches_done % save == 0:
                images = make_image_grid(real, line.expand(-1, 3, -1, -1), fake)
                save_image(images, f'implementations/Restormer/result/{status.batches_done}.jpg',
                    nrow=3*4, normalize=True, value_range=(-1, 1))
                torch.save(G.state_dict(),
                    f'implementations/Restormer/result/model_{status.batches_done}.pth')
            save_image(make_image_grid(line.expand(-1, 3, -1, -1), fake), 'running.jpg',
                nrow=2*3, normalize=True, value_range=(-1, 1))

            if scaler is not None: scaler.update()
            status.update(G=G_loss.item(), D=D_loss.item())

            if status.is_end():
                break
    status.plot_loss()

def xdog(
    image: Image.Image, sigma: float=1., k: float=1.6,
    p: float=200, eps: float=0, phi: float=2
) -> Image.Image:
    '''Edge detection via XDoG'''
    image = image.convert('L')
    image = np.array(image)

    if image.max() > 1:
        image = image / 255

    def sharp(image):
        g1 = cv2.GaussianBlur(image, (0, 0), sigma)
        g2 = cv2.GaussianBlur(image, (0, 0), sigma*k)
        return (1 + p) * g1 - p * g2

    def soft_threshold(s):
        T = 1 + np.tanh(phi * (s - eps))
        T[s > eps] = 1
        return T

    S = sharp(image)
    SI = np.multiply(image, S)
    T = soft_threshold(SI) * 255
    T = T.astype(np.uint8)

    image = Image.fromarray(T).convert('RGB')
    return image

def transform(image, image_size):
    line = xdog(image)
    image = TF.resize(image, (image_size, image_size))
    line  = TF.resize(line,  (image_size, image_size))
    image = TF.to_tensor(image)
    line  = TF.to_tensor(line)
    image = TF.normalize(image, 0.5, 0.5)
    line  = TF.normalize(line,  0.5, 0.5)
    return (line, image)

def transform_rand(image, image_size):
    sigma = np.random.uniform(1.0, 1.5)
    k     = np.random.uniform(1.1, 3.0)
    phi   = np.random.uniform(2, 10)
    line = xdog(image, sigma=sigma, k=k, phi=phi)
    image = TF.resize(image, int(image_size*1.1))
    image = TF.center_crop(image, image_size)
    line  = TF.resize(line,  int(image_size*1.1))
    line  = TF.center_crop(line,  image_size)
    image = TF.to_tensor(image)
    line  = TF.to_tensor(line)
    image = TF.normalize(image, 0.5, 0.5)
    line  = TF.normalize(line,  0.5, 0.5)
    return (line, image)

def main(parser):
    parser = add_args(parser, dict(
        image_channels = [3],

        num_blocks            = [[2, 6, 6, 8], 'number of transformer blocks for each scale'],
        num_heads             = [[1, 2, 4, 8], 'number of heads in attention module for each scale'],
        num_refinement_blocks = [4, 'number of transformer blocks in refinement.'],
        ff_expansion          = [2.66, 'feed forward expansion ratio'],
        g_channels            = [48, 'minimum channel width'],
        g_norm_name           = ['ln', 'normalization layer name'],
        g_act_name            = ['gelu', 'activation layer name'],
        skip_input            = [False, 'If true skip connect input to output.'],
        num_layers            = [3, 'number of layers'],
        d_channels            = [64, 'minimum channel width'],
        max_channels          = [512, 'maximum channel width'],
        d_norm_name           = ['bn', 'normalization layer name'],
        d_act_name            = ['lrelu', 'activation function name'],

        lr           = [0.0002, 'learning rate'],
        betas        = [[0.5, 0.999], 'betas'],
        pixel_lambda = [10., 'lambda for pixel-wise loss'],
        randomize_xdog = [False, 'use random xdog parameters']))
    args = parser.parse_args()
    save_args(args)

    device = get_device(not args.disable_gpu)
    amp    = not args.disable_amp and not args.disable_gpu

    if args.randomize_xdog: transform_func = transform_rand
    else:                   transform_func = transform
    # dataset
    if args.dataset == 'aahq':
        dataset = AAHQ.asloader(args.batch_size,
            (args.image_size, args.num_images, partial(transform_func, image_size=args.image_size)),
            pin_memory=not args.disable_gpu)
    elif args.dataset == 'animeface':
        dataset = AnimeFace.asloader(args.batch_size,
            (args.image_size, args.min_year, partial(transform_func, image_size=args.image_size)),
            pin_memory=not args.disable_gpu)
    elif args.dataset == 'danbooru':
        dataset = DanbooruPortrait.asloader(args.batch_size,
            (args.image_size, args.num_images, partial(transform_func, image_size=args.image_size)),
            pin_memory=not args.disable_gpu)

    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs

    # models
    G = Generator(
        args.num_blocks, args.num_heads, args.num_refinement_blocks, args.ff_expansion,
        args.g_channels, args.g_norm_name, args.g_act_name, False, args.image_channels,
        args.skip_input)
    D = Discriminator(
        args.num_layers, args.d_channels, args.max_channels,
        args.d_norm_name, args.d_act_name, args.image_channels*2)

    G.to(device)
    D.to(device)

    adam_eps = 1e-4 if amp else 1e-6
    # optimizers
    optim_G = optim.Adam(G.parameters(), lr=args.lr, betas=args.betas, eps=adam_eps)
    optim_D = optim.Adam(D.parameters(), lr=args.lr, betas=args.betas, eps=adam_eps)

    train(args,
        args.max_iters, dataset,
        G, D, optim_G, optim_D,
        args.pixel_lambda,
        device, amp, args.save, args.log_file, args.log_interval)
