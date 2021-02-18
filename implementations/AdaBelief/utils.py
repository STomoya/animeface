
import functools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad, Variable
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from ..StyleGAN2.model import Generator, Discriminator, init_weight_N01
from ..general import YearAnimeFaceDataset, to_loader, save_args, get_device
from ..gan_utils import GANTrainingStatus, sample_nnoise, DiffAugment, AdaAugment
from ..gan_utils.losses import GANLoss
from .AdaBelief import AdaBelief

def r1_gp(D, real, scaler=None):
    inputs = Variable(real, requires_grad=True)
    outputs = D(inputs).sum()
    ones = torch.ones(outputs.size(), device=real.device)
    with autocast(False):
        if scaler is not None:
            outputs = scaler.scale(outputs)[0]
        gradients = grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        if scaler is not None:
            gradients = gradients / scaler.get_scale()
    gradients = gradients.view(gradients.size(0), -1)
    penalty = gradients.norm(2, dim=1).pow(2).mean()
    return penalty / 2

def train(
    max_iter, dataset, sample_noise, style_dim,
    G, D, optimizer_G, optimizer_D,
    gp_lambda, policy,
    device, amp, save_interval=1000
):
    
    status = GANTrainingStatus()
    loss = GANLoss()
    scaler = GradScaler() if amp else None
    const_z = sample_nnoise((16, style_dim))
    bar = tqdm(total=max_iter)

    if policy == 'ada':
        augment = AdaAugment(0.6, 500000, device)
    elif policy is not '':
        augment = functools.partial(DiffAugment, policy=policy)
    else:
        augment = lambda x: x
    
    while status.batches_done < max_iter:
        for index, real in enumerate(dataset):
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            real = real.to(device)
            z = sample_noise()

            '''Discriminator'''
            with autocast(amp):
                # D(x)
                real = augment(real)
                real_prob = D(real)
                # D(G(z))
                fake, _ = G(z)
                fake = augment(fake)
                fake_prob = D(fake.detach())
                # loss
                adv_loss = loss.d_loss(real_prob, fake_prob)
                if gp_lambda > 0:
                    r1_loss = r1_gp(D, real, scaler)
                else:
                    r1_loss = 0
                D_loss = adv_loss + r1_loss * gp_lambda

            if scaler is not None:
                scaler.scale(D_loss).backward()
                scaler.step(optimizer_D)
            else:
                D_loss.backward()
                optimizer_D.step()
            
            '''Generator'''
            z = sample_noise()
            with autocast(amp):
                # D(G(z))
                fake, _ = G(z)
                fake = augment(fake)
                fake_prob = D(fake)
                # loss
                G_loss = loss.g_loss(fake_prob)
            
            if scaler is not None:
                scaler.scale(G_loss).backward()
                scaler.step(optimizer_G)
            else:
                G_loss.backward()
                optimizer_G.step()

            if status.batches_done % save_interval == 0:
                G.eval()
                with torch.no_grad():
                    image, _ = G(const_z)
                G.train()
                status.save_image('implementations/AdaBelief/result', image, nrow=4)

            G_loss_pyfloat, D_loss_pyfloat = G_loss.item(), D_loss.item()
            status.append(G_loss_pyfloat, D_loss_pyfloat)
            if scaler is not None:
                scaler.update()
            if policy == 'ada':
                augment.update_p(real_prob)
            bar.set_postfix_str('G {:.5f} D {:.5f}'.format(G_loss_pyfloat, D_loss_pyfloat))
            bar.update(1)

            if status.batches_done == max_iter:
                break

def add_argument(parser):
    parser.add_argument('--image-channels', default=3, type=int, help='number of channels for the generated image')
    parser.add_argument('--style-dim', default=512, type=int, help='style feature dimension')
    parser.add_argument('--channels', default=32, type=int, help='channel width multiplier')
    parser.add_argument('--max-channels', default=512, type=int, help='maximum channels')
    parser.add_argument('--block-num-conv', default=2, type=int, help='number of convolution layers in residual block')
    parser.add_argument('--map-num-layers', default=8, type=int, help='number of layers in mapping network')
    parser.add_argument('--map-lr', default=0.01, type=float, help='learning rate for mapping network')
    parser.add_argument('--disable-map-norm', default=False, action='store_true', help='disable pixel normalization in mapping network')
    parser.add_argument('--mbsd-groups', default=4, type=int, help='number of groups in mini-batch standard deviation')

    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0., type=float, help='beta1')
    parser.add_argument('--beta2', default=0.99, type=float, help='beta2')
    parser.add_argument('--gp-lambda', default=10, type=float, help='lambda for r1')
    parser.add_argument('--policy', default='color,translation', type=str, help='policy for DiffAugment')
    return parser

def main(parser):

    parser = add_argument(parser)
    args = parser.parse_args()
    save_args(args)

    # # data
    # image_size = 128
    # min_year = 2010
    # batch_size = 32

    # # model
    # mapping_layers = 8
    # mapping_lr = 0.01
    # style_dim = 512

    # # training
    # max_iters = -1
    # lr = 0.0002
    betas = (args.beta1, args.beta2)
    normalize = not args.disable_map_norm
    # gp_lambda = 0
    # policy = 'color,translation'
    # policy = 'ada'

    amp = not args.disable_amp

    device = get_device(not args.disable_gpu)

    dataset = YearAnimeFaceDataset(args.image_size, args.min_year)
    dataset = to_loader(dataset, args.batch_size)

    sample_noise = functools.partial(sample_nnoise, size=(args.batch_size, args.style_dim), device=device)

    G = Generator(
            args.image_size, args.image_channels, args.style_dim, args.channels, args.max_channels,
            args.block_num_conv, args.map_num_layers, normalize, args.map_lr)
    D = Discriminator(
            args.image_size, args.image_channels, args.channels, args.max_channels, 
            args.block_num_conv, args.mbsd_groups)
    G.init_weight(
        map_init_func=functools.partial(init_weight_N01, lr=args.map_lr),
        syn_init_func=init_weight_N01)
    D.apply(init_weight_N01)
    G.to(device)
    D.to(device)

    optimizer_G = AdaBelief(G.parameters(), lr=args.lr, betas=betas)
    optimizer_D = AdaBelief(D.parameters(), lr=args.lr, betas=betas)

    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs

    train(
        args.max_iters, dataset, sample_noise, args.style_dim,
        G, D, optimizer_G, optimizer_D, args.gp_lambda, args.policy,
        device, amp, 1000
    )
