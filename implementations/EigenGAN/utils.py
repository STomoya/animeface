
import functools

import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image

from ..general import YearAnimeFaceDataset, DanbooruPortraitDataset, to_loader
from ..general import Status, get_device, save_args
from ..gan_utils import sample_nnoise, DiffAugment, init, update_ema
from ..gan_utils.losses import HingeLoss, GradPenalty

from .model import Generator, Discriminator

@torch.enable_grad()
def orthogonal_regularizer(model):
    loss = 0
    for name, param in model.named_parameters():
        if 'U' in name:
            flat = param.view(param.size(0), -1)
            sym = torch.mm(flat, flat.T)
            eye = torch.eye(sym.size(-1), device=sym.device)
            loss = loss + (sym - eye).pow(2).sum() * 0.5
    return loss

def train(
    max_iters, dataset, const_input,
    sampler, eps_dim, latent_dim,
    G, G_ema, D, optimizer_G, optimizer_D,
    gp_lambda, ortho_lambda,
    augment,
    device, amp, save=1000
):
    
    status = Status(max_iters)
    loss   = HingeLoss()
    gp     = GradPenalty()
    scaler = GradScaler() if amp else None

    while status.batches_done < max_iters:
        for image in dataset:
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()
            real = image.to(device)

            with autocast(amp):
                # generate images
                B = real.size(0)
                eps = sampler((B, eps_dim))
                zs = [sampler((B, latent_dim)) for _ in range(G.num_layers)]
                fake = G(eps, zs)
                # augment
                real_aug = augment(real)
                fake_aug = augment(fake)

                # D(real)
                real_prob = D(real_aug)
                # D(fake)
                fake_prob = D(fake_aug.detach())

                # loss
                adv_loss = loss.d_loss(real_prob, fake_prob)
                if gp_lambda > 0:
                    gp_loss = gp.r1_regularizer(real, D, scaler) * gp_lambda
                else: gp_loss = 0

                D_loss = adv_loss + gp_loss

            if scaler is not None:
                scaler.scale(D_loss).backward()
                scaler.step(optimizer_D)
            else:
                D_loss.backward()
                optimizer_D.step()

            with autocast(amp):
                # D(fake)
                fake_prob = D(fake_aug)

                # loss
                adv_loss = loss.g_loss(fake_prob)
                if ortho_lambda > 0:
                    ortho_loss = orthogonal_regularizer(G) * ortho_lambda
                else: ortho_loss = 0
                G_loss = adv_loss + ortho_loss
            
            if scaler is not None:
                scaler.scale(G_loss).backward()
                scaler.step(optimizer_G)
            else:
                G_loss.backward()
                optimizer_G.step()

            if G_ema is not None:
                update_ema(G, G_ema)

            # save
            with torch.no_grad():
                if status.batches_done % save == 0:
                    if const_input[1] == None:
                        input = (const_input[0], [sampler((const_input[0].size(0), latent_dim)) for _ in range(G.num_layers)])
                    else:
                        input = const_input
                    if G_ema is not None:
                        images = G_ema(*input)
                        state_dict = G_ema.state_dict()
                    else:
                        G.eval()
                        images = G(*input)
                        G.train()
                        state_dict = G.state_dict()
                    save_image(images, f'implementations/EigenGAN/result/{status.batches_done}.jpg',
                        normalize=True, value_range=(-1, 1))
                    torch.save(state_dict, f'implementations/EigenGAN/result/G_{status.batches_done}.pt')
            save_image(fake, 'running.jpg', normalize=True, value_range=(-1, 1))

            # updates
            loss_dict = dict(
                G=G_loss.item() if not torch.any(torch.isnan(G_loss)) else 0,
                D=D_loss.item() if not torch.any(torch.isnan(D_loss)) else 0)
            status.update(loss_dict)
            if scaler is not None:
                scaler.update()
    
    status.plot()
            
def add_arguments(parser):
    parser.add_argument('--num-test', default=16, type=int, help='number of const for eval')
    parser.add_argument('--const-z', default=False, action='store_true', help='subspace input will be const too.')
    parser.add_argument('--image-channels', default=3, type=int, help='channels of output/real images')
    # G
    parser.add_argument('--eps-dim', default=512, type=int, help='channels of eps input')
    parser.add_argument('--latent-dim', default=6, type=int, help='channels of subspace input')
    parser.add_argument('--bottom-width', default=4, type=int, help='bottom width of features in model')
    parser.add_argument('--g-channels', default=32, type=int, help='channel width multiplier for G')
    parser.add_argument('--g-max-channels', default=512, type=int, help='maximum channel width in G')
    parser.add_argument('--g-disable-sn', default=False, action='store_true', help='disable spectral norm in G')
    parser.add_argument('--g-disable-bias', default=False, action='store_true', help='disable bias in G')
    parser.add_argument('--g-norm-name', default='in', choices=['in', 'bn'], help='normalization layer name in G')
    parser.add_argument('--g-act-name', default='lrelu', choices=['relu', 'lrelu'], help='activation function name in G')
    # D
    parser.add_argument('--d-channels', default=32, type=int, help='channel width multiplier for D')
    parser.add_argument('--d-max-channels', default=512, type=int, help='maximum channel width in D')
    parser.add_argument('--d-disable-sn', default=False, action='store_true', help='disable spectral norm in D')
    parser.add_argument('--d-disable-bias', default=False, action='store_true', help='disable bias in D')
    parser.add_argument('--d-norm-name', default='in', choices=['in', 'bn'], help='normalization layer name in D')
    parser.add_argument('--d-act-name', default='lrelu', choices=['relu', 'lrelu'], help='activation function name in D')

    # training
    parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
    parser.add_argument('--betas', default=[0.5, 0.999], type=float, nargs=2, help='betas')
    parser.add_argument('--gp-lambda', default=0, type=float, help='lambda for gradient penalty')
    parser.add_argument('--ortho-lambda', default=1, type=float, help='lambda for orthogonal regularization to U')
    parser.add_argument('--policy', default='color,translation', help='policy for Diffaugment')
    # exprimental
    parser.add_argument('--ema', default=False, action='store_true', help='use EMA')
    return parser

def main(parser):
    
    parser = add_arguments(parser)
    args = parser.parse_args()
    save_args(args)

    device = get_device(not args.disable_gpu)
    amp = not args.disable_amp and not args.disable_gpu

    # data
    if args.dataset == 'animeface':
        dataset = YearAnimeFaceDataset(args.image_size, args.min_year)
    elif args.dataset == 'danbooru':
        dataset = DanbooruPortraitDataset(args.image_size, num_images=args.num_images)
    dataset = to_loader(dataset, args.batch_size)

    sampler = functools.partial(sample_nnoise, device=device)
    const_eps = sampler((args.num_test, args.eps_dim))
    if args.const_z:
        const_zs  = sampler((args.num_test, args.latent_dim))
    else: const_zs = None
    const_input = (const_eps, const_zs)

    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs

    # model
    G = Generator(
        args.image_size, args.eps_dim, args.latent_dim,
        args.image_channels, args.bottom_width, args.g_channels,
        args.g_max_channels, not args.g_disable_sn, not args.g_disable_bias,
        args.g_norm_name, args.g_act_name
    )
    D = Discriminator(
        args.image_size, args.image_channels, args.bottom_width,
        args.d_channels, args.d_max_channels, not args.d_disable_sn,
        not args.d_disable_bias, args.d_norm_name, args.d_act_name
    )
    G.to(device)
    D.to(device)
    init_func = init().N002
    G.apply(init_func)
    D.apply(init_func)
    if args.ema:
        G_ema = Generator(
            args.image_size, args.eps_dim, args.latent_dim,
            args.image_channels, args.bottom_width, args.g_channels,
            args.g_max_channels, not args.g_disable_sn, not args.g_disable_bias,
            args.g_norm_name, args.g_act_name
        )
        G_ema.to(device)
        G_ema.eval()
        for param in G_ema.parameters():
            param.requires_grad = False
        update_ema(G, G_ema, 0)
    else: G_ema = None
    
    # optimizers
    optimizer_G = optim.Adam(G.parameters(), lr=args.lr, betas=args.betas)
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr, betas=args.betas)

    train(
        args.max_iters, dataset, const_input,
        sampler, args.eps_dim, args.latent_dim,
        G, G_ema, D, optimizer_G, optimizer_D,
        args.gp_lambda, args.ortho_lambda,
        functools.partial(DiffAugment, policy=args.policy),
        device, amp, args.save
    )
