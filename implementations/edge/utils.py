
import os
import functools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image

from ..StyleGAN2.model import Generator, Discriminator, init_weight_N01
from ..general.dataset_base import Image, make_default_transform
from ..general import to_loader
from ..general import get_device, Status, save_args
from ..gan_utils import sample_nnoise, AdaBelief, update_ema, DiffAugment
from ..gan_utils.losses import GANLoss, GradPenalty


'''dataset classes with Image + Blur'''
import glob
import random
from PIL import Image as pilImage
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
    
class AnimeFaceBlur(ImageBlur):
    def __init__(self, image_size, min_year=2005, resize_ratio=1.):
        super().__init__(image_size, resize_ratio)
        self.images = [path for path in self.images if self._year_from_path(path) >= min_year]
        self.blurs = self._load_blur()
        self.length = len(self.images)
    def _load(self):
        return glob.glob('/usr/src/data/animefacedataset/images/*')
    def _load_blur(self):
        blurs = [path.replace('images', 'blur') for path in self.images]
        random.shuffle(blurs)
        return blurs
    def _year_from_path(self, path):
        name, _ = os.path.splitext(os.path.basename(path))
        year = int(name.split('_')[-1])
        return year

class DanbooruPortraitBlur(ImageBlur):
    def __init__(self, image_size, num_images=None, resize_ratio=1.2):
        super().__init__(image_size, resize_ratio)
        if num_images:
            random.shuffle(self.images)
            self.images = self.images[:num_images]
            self.length = len(self.images)
            self.blurs  = self._load_blur()
    def _load(self):
        return glob.glob('/usr/src/data/danbooru/portraits/portraits/*')
    def _load_blur(self):
        blurs = [path.replace('portraits/portraits', 'portraits/blur') for path in self.images]
        random.shuffle(blurs)
        return blurs

softplus = nn.Softplus()
def edge_adv_loss(edge_prob):
    # train blured edge images as fake
    edge_loss = softplus(edge_prob).mean()
    return edge_loss

def train(
    max_iter, dataset, sampler, const_z,
    G, G_ema, D, optimizer_G, optimizer_D,
    r1_lambda, d_k, policy,
    edge_loss_from,
    device, amp,
    save=1000
):
    
    status  = Status(max_iter)
    pl_mean = 0.
    loss    = GANLoss()
    gp      = GradPenalty()
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
            z = sampler()
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
                    and status.batches_done is not 0:
                    # lazy regularization
                    r1 = gp.r1_regularizer(real, D, scaler)
                    D_loss = r1 * r1_lambda * d_k
                else:
                    # gan loss on other iter
                    D_loss = loss.d_loss(real_prob, fake_prob)
                    # add gan loss for blured edges
                    if edge_loss_from > status.batches_done:
                        D_loss = D_loss + edge_adv_loss(blur_prob)
            
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
                save_image(images, f'implementations/edge/result/{status.batches_done}.jpg', nrow=4, normalize=True, range=(-1, 1))
                torch.save(G_ema.state_dict(), f'implementations/edge/result/G_{status.batches_done}.pt')
            save_image(fake, f'running.jpg', nrow=4, normalize=True, range=(-1, 1))

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
        dataset.dataset.shuffle_blur()
    
    status.plot()

def add_argument(parser):
    # args for StyleGAN2
    # model
    parser.add_argument('--image-channels', default=3, type=int, help='number of channels for the generated image')
    parser.add_argument('--style-dim', default=512, type=int, help='style feature dimension')
    parser.add_argument('--channels', default=32, type=int, help='channel width multiplier')
    parser.add_argument('--max-channels', default=512, type=int, help='maximum channels')
    parser.add_argument('--block-num-conv', default=2, type=int, help='number of convolution layers in residual block')
    parser.add_argument('--map-num-layers', default=8, type=int, help='number of layers in mapping network')
    parser.add_argument('--map-lr', default=0.01, type=float, help='learning rate for mapping network')
    parser.add_argument('--disable-map-norm', default=False, action='store_true', help='disable pixel normalization in mapping network')
    parser.add_argument('--mbsd-groups', default=4, type=int, help='number of groups in mini-batch standard deviation')
    # training
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0., type=float, help='beta1')
    parser.add_argument('--beta2', default=0.99, type=float, help='beta2')
    # parser.add_argument('--g-k', default=8, type=int, help='for lazy regularization. calculate perceptual path length loss every g_k iters')
    parser.add_argument('--d-k', default=16, type=int, help='for lazy regularization. calculate gradient penalty each d_k iters')
    parser.add_argument('--r1-lambda', default=10, type=float, help='lambda for r1')
    # parser.add_argument('--pl-lambda', default=0., type=float, help='lambda for perceptual path length loss')
    parser.add_argument('--policy', default='color,translation', type=str, help='policy for DiffAugment')

    # args for edge
    parser.add_argument('--wait-edge-epoch', default=0, type=int, help='epochs to wait before adding edge adv loss')
    return parser

def main(parser):

    parser = add_argument(parser)
    args = parser.parse_args()
    save_args(args)

    normalize = not args.disable_map_norm
    betas = (args.beta1, args.beta2)

    amp = not args.disable_amp
    device = get_device(not args.disable_gpu)

    # dataset
    if args.dataset == 'animeface':
        dataset = AnimeFaceBlur(args.image_size, args.min_year)
    elif args.dataset == 'danbooru':
        dataset = DanbooruPortraitBlur(args.image_size, args.num_images)
    dataset = to_loader(
                dataset, args.batch_size, shuffle=True,
                num_workers=os.cpu_count(), use_gpu=torch.cuda.is_available())

    # random noise sampler
    sampler = functools.partial(sample_nnoise, (args.batch_size, args.style_dim), device=device)
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
        args.max_iters, dataset, sampler, const_z,
        G, G_ema, D, optimizer_G, optimizer_D,
        args.r1_lambda, args.d_k,
        args.policy, edge_loss_from, device, amp, 1000
    )
    