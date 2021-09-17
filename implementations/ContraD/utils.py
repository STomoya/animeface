
import functools

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image
import torchvision.transforms as T

from dataset import AnimeFace, DanbooruPortrait
from utils import Status, save_args, add_args
from nnutils import sample_nnoise, update_ema, get_device, freeze
from nnutils.loss import NonSaturatingLoss
from nnutils.loss.penalty import calc_grad, Variable
from thirdparty.diffaugment import DiffAugment

from ..StyleGAN2 import model as stylegan2
from .model import Discriminator

def supervised_contrastive_loss(out1, out2, others, temperature=0.1, normalized=False):
    '''
    Supervised contrastive loss
    from : https://github.com/jh-jeong/ContraD/blob/main/training/gan/contrad.py#L8-L32
    '''
    if not normalized:
        out1 = F.normalize(out1, p=2, dim=-1)
        out2 = F.normalize(out2, p=2, dim=-1)
        others = F.normalize(others, p=2, dim=-1)

    N = out1.size(0)

    outputs = torch.cat([out1, out2, others], dim=0)
    sim_matrix = outputs @ outputs.t()
    sim_matrix = sim_matrix / temperature
    sim_matrix.fill_diagonal_(-5e4)

    mask = torch.zeros_like(sim_matrix)
    mask[2*N:,2*N:] = 1
    mask.fill_diagonal_(0)

    sim_matrix = sim_matrix[2*N:]
    mask = mask[2*N:]
    mask = mask / mask.sum(1, keepdim=True)

    lsm = F.log_softmax(sim_matrix, dim=1)
    lsm = lsm * mask
    d_loss = -lsm.sum(1).mean()
    return d_loss

def nt_xent_loss(out1, out2, temperature=0.1, normalized=False):
    '''
    Normalized Temperature-scaled Cross Entropy Loss
    from : https://github.com/jh-jeong/ContraD/blob/main/training/criterion.py#L24-L45
    '''
    if not normalized:
        out1 = F.normalize(out1, p=2, dim=-1)
        out2 = F.normalize(out2, p=2, dim=-1)
    N = out1.size(0)

    outputs = torch.cat([out1, out2], dim=0)

    sim_matrix = outputs @ outputs.t()
    sim_matrix = sim_matrix / temperature

    sim_matrix.fill_diagonal_(-5e4)
    sim_matrix = F.log_softmax(sim_matrix, dim=1)
    loss = -torch.sum(sim_matrix[:N, N:].diag() + sim_matrix[N:, :N].diag()) / (2*N)

    return loss

class R1:
    def __init__(self) -> None:
        super().__init__()
        self._calc_grad = calc_grad
    '''grad penalty'''
    def penalty(self, real, D, scaler=None):
        '''R1 regularizer'''
        real_loc = Variable(real, requires_grad=True)
        d_real_loc, _, _ = D(real_loc, stop_grad=False)
        gradients = self._calc_grad(d_real_loc, real_loc, scaler)
        gradients = gradients.reshape(gradients.size(0), -1)
        penalty = gradients.norm(2, dim=1).pow(2).mean() / 2.
        return penalty

def train(
    max_iters, dataset, sampler, latent_dim, const_z,
    G, G_ema, D, optimizer_G, optimizer_D,
    r1_lambda, con_lambda, dis_lambda, temperature,
    augment_r, augment_f,
    device, amp, save
):

    status = Status(max_iters)
    loss = NonSaturatingLoss()
    gp   = R1()
    scaler = GradScaler() if amp else None

    while status.batches_done < max_iters:
        for real in dataset:
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()

            real = real.to(device)
            z = sampler((real.size(0), latent_dim))

            '''Discriminator'''
            with autocast(amp):
                # D(real)
                real_ = augment_r(real)
                real_prob, proj_con_1, proj_supcon_r_1 = D(real_, stop_grad=True)
                real_ = augment_r(real)
                _,         proj_con_2, proj_supcon_r_2 = D(real_, stop_grad=True)
                # D(G(z))
                fake, _ = G(z)
                fake_ = augment_f(fake)
                fake_prob, _,          proj_supcon_f   = D(fake.detach())

                # loss
                # no lazy regularization
                r1 = gp.penalty(real, D, scaler) * r1_lambda
                D_loss = loss.d_loss(real_prob, fake_prob) * dis_lambda + r1
                D_loss = D_loss + nt_xent_loss(
                    proj_con_1, proj_con_2, temperature
                )
                D_loss = D_loss + supervised_contrastive_loss(
                    proj_supcon_r_1, proj_supcon_r_2, proj_supcon_f, temperature
                ) * con_lambda

            if scaler is not None:
                scaler.scale(D_loss).backward()
                scaler.step(optimizer_D)
            else:
                D_loss.backward()
                optimizer_D.step()

            '''Generator'''
            with autocast(amp):
                # D(G(z))
                fake_prob, *_ = D(fake_, stop_grad=False)
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
                    if G_ema is not None:
                        images, _ = G_ema(const_z)
                        state_dict = G_ema.state_dict()
                    else:
                        G.eval()
                        images, _ = G(const_z)
                        G.train()
                        state_dict = G.state_dict()
                save_image(
                    images, f'implementations/ContraD/result/{status.batches_done}.jpg',
                    nrow=4, normalize=True, value_range=(-1, 1))
                torch.save(
                    state_dict, f'implementations/ContraD/result/G_{status.batches_done}.pt')

            save_image(fake, f'running.jpg', nrow=4, normalize=True, value_range=(-1, 1))

            # updates
            loss_dict = dict(
                G=G_loss.item() if not torch.isnan(G_loss).any() else 0,
                D=D_loss.item() if not torch.isnan(D_loss).any() else 0
            )
            status.update(**loss_dict)
            if scaler is not None:
                scaler.update()

            if status.batches_done == max_iters:
                break

    status.plot_loss()

class DeNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std
    def forward(self, x):
        return x.mul(self.std).add(self.mean)

def get_simclr_transform(image_size, resize_ratio, denorm=False):
    '''Augmentation used in SimCLR'''

    transform = []
    if denorm:
        transform.append(DeNormalize(0.5, 0.5))

    transform.extend([
        T.RandomApply(
            [T.ColorJitter(0.5, 0.5, 0.5, 0.2)],
            p=0.5
        ),
        T.RandomGrayscale(0.2),
        T.RandomHorizontalFlip(),
        T.RandomApply(
            [T.GaussianBlur((3, 3), (1., 2.))],
            p=0.5
        ),
        T.Resize(int(image_size*resize_ratio)),
        T.RandomCrop(image_size),
        T.Normalize(torch.tensor([0.5]), torch.tensor([0.5]))
    ])

    return nn.Sequential(*transform)

def main(parser):

    parser = add_args(parser,
        dict(
            image_channels      = [3, 'number of channels for the generated image'],
            style_dim           = [512, 'style feature dimension'],
            channels            = [32, 'channel width multiplier'],
            max_channels        = [512, 'maximum channels'],
            block_num_conv      = [2, 'number of convolution layers in residual block'],
            map_num_layers      = [4, 'number of layers in mapping network'],
            map_lr              = [0.01, 'learning rate for mapping network'],
            disable_map_norm    = [False, 'disable pixel norm'],
            mbsd_groups         = [4, 'mini batch stddev group size'],
            lr                  = [0.001, 'learning rate'],
            betas               = [[0., 0.99], 'betas'],
            r1_lambda           = [0.5, 'lambda for r1'],
            policy              = ['color,translation', 'policy for DiffAugment'],
            augmentation        = ['diff', 'augmentation to perform'],
            projection_features = [256, 'output feature dimensions for projection'],
            hidden_features     = [256, 'dimensions for hidden layers'],
            d_act_name          = ['lrelu', 'activation function for D'],
            con_lambda          = [1., 'lambda for contrastive loss'],
            dis_lambda          = [1., 'lambda for adversarial loss'],
            temperature         = [0.1, 'temperature used to calculate NTXent loss'])
    )
    args = parser.parse_args()
    save_args(args)

    amp = not args.disable_amp and not args.disable_amp
    device = get_device(not args.disable_gpu)

    # dataset
    if args.augmentation == 'simclr':
        transform = T.Compose([
            T.Resize((args.image_size, args.image_size)),
            T.ToTensor()
        ])
    else: transform = None
    if args.dataset == 'animeface':
        dataset = AnimeFace.asloader(
            args.batch_size, (args.image_size, args.min_year, transform),
            pin_memory=not args.disable_gpu)
    elif args.dataset == 'danbooru':
        dataset = DanbooruPortrait.asloader(
            args.batch_size, (args.image_size, args.num_images, transform),
            pin_memory=not args.disable_gpu)

    # random noise sampler
    sampler = functools.partial(sample_nnoise, device=device)
    # const input for eval
    const_z = sample_nnoise((16, args.style_dim), device=device)

    # models
    normalize = not args.disable_map_norm
    G = stylegan2.Generator(
            args.image_size, args.image_channels, args.style_dim, args.channels, args.max_channels,
            args.block_num_conv, args.map_num_layers, normalize, args.map_lr)
    G_ema = stylegan2.Generator(
            args.image_size, args.image_channels, args.style_dim, args.channels, args.max_channels,
            args.block_num_conv, args.map_num_layers, normalize, args.map_lr)

    # Discriminator
    # make StyleGAN2 D
    _D = stylegan2.Discriminator(
            args.image_size, args.image_channels, args.channels, args.max_channels,
            args.block_num_conv, args.mbsd_groups)
    # cut off last layers (act+linear)
    _D.blocks = nn.Sequential(*list(_D.blocks.children())[:-2])
    # set attr for ContraD wrapper requirements
    _D.out_channels = args.max_channels
    # make ContraD net
    D = Discriminator(
        _D, args.projection_features, args.hidden_features,
        weight_norm='elr', bias=True, act_name=args.d_act_name)

    ## init
    G.init_weight(
        map_init_func=functools.partial(stylegan2.init_weight_N01, lr=args.map_lr),
        syn_init_func=stylegan2.init_weight_N01)
    freeze(G_ema)
    update_ema(G, G_ema, decay=0)
    D.apply(stylegan2.init_weight_N01)

    G.to(device)
    G_ema.to(device)
    D.to(device)

    # optimizer
    g_lr, g_betas = args.lr, args.betas
    d_lr, d_betas = args.lr, args.betas

    optimizer_G = optim.Adam(G.parameters(), lr=g_lr, betas=g_betas)
    optimizer_D = optim.Adam(D.parameters(), lr=d_lr, betas=d_betas)

    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs

    # augmentation
    if args.augmentation == 'diff':
        augment_r = functools.partial(
            DiffAugment, policy=args.policy
        )
        augment_f = augment_r
    elif args.augmentation == 'simclr':
        augment_r = get_simclr_transform(
            args.image_size, resize_ratio=1. if args.dataset == 'animeface' else 1.1
        )
        # the output is [-1, 1] in G
        # so denormalization is performed before the transform
        augment_f = get_simclr_transform(
            args.image_size, 1., True
        )


    train(
        args.max_iters, dataset, sampler, args.style_dim, const_z,
        G, G_ema, D, optimizer_G, optimizer_D,
        args.r1_lambda, args.con_lambda, args.dis_lambda, args.temperature,
        augment_r, augment_f, device, amp, args.save
    )
