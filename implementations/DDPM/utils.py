
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image

from dataset import AnimeFace, DanbooruPortrait
from utils import Status, save_args, add_args
from nnutils import get_device, update_ema, freeze

from .model import UNet, GaussianDiffusion

def train(
    max_iters, dataset, timesteps, test_shape,
    denoise_model, ema_model, gaussian_diffusion,
    optimizer,
    device, amp, save, sample=1000
):

    status  = Status(max_iters)
    scaler  = GradScaler() if amp else None
    loss_fn = nn.MSELoss()
    const_z = torch.randn(test_shape, device=device)

    while status.batches_done < max_iters:
        for real in dataset:
            optimizer.zero_grad()
            real = real.to(device)
            t = torch.randint(0, timesteps, (real.size(0), ), device=device)

            with autocast(amp):
                x_noisy, noise = gaussian_diffusion.q_sample(real, t)
                recon = denoise_model(x_noisy, t)
                loss  = loss_fn(recon, noise)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
            else:
                loss.backward()
                optimizer.step()

            update_ema(denoise_model, ema_model)

            if status.batches_done % sample == 0 and status.batches_done != 0:
                images = gaussian_diffusion.p_sample_loop(
                    ema_model, test_shape, const_z)
                save_image(
                    images, f'implementations/DDPM/result/{status.batches_done}.jpg',
                    normalize=True, value_range=(-1, 1), nrow=2*4)

            if status.batches_done % save == 0:
                torch.save(
                    dict(denoise_model=denoise_model.state_dict(), gaussian_diffusion=gaussian_diffusion.state_dict()),
                    f'implementations/DDPM/result/DDPM_{status.batches_done}.pt')

            status.update(loss=loss.item() if not torch.any(loss.isnan()) else 0.)
            if scaler is not None:
                scaler.update()
            if status.batches_done == max_iters:
                break

    status.plot_loss()

def main(parser):

    parser = add_args(parser,
        dict(
            num_test       = [16, 'number of test smaples'],
            image_channels = [3, 'image channels'],
            # model
            bottom         = [16, 'bottom width'],
            channels       = [32, 'channel width mutiplier'],
            attn_resls     = [[16], 'resolution to apply attention'],
            attn_head      = [8, 'heads for MHA'],
            time_affine    = [False, 'adaptive normalization'],
            dropout        = [0., 'dropout'],
            num_res        = [1, 'number of residual blocks in one resolution'],
            norm_name      = ['gn', 'normalization layer name'],
            act_name       = ['swish', 'activation layer name'],
            # diffusion
            timesteps      = [1000, 'number of time steps in forward/backward diffusion process'],
            # optimization
            lr             = [2e-5, 'learning rate'],
            betas          = [[0.9, 0.999], 'betas'],
            sample         = [10000, 'sample very']
        ))
    args = parser.parse_args()
    save_args(args)

    device = get_device(not args.disable_gpu)
    amp    = not args.disable_gpu and not args.disable_amp

    # dataset
    if args.dataset == 'animeface':
        dataset = AnimeFace.asloader(
            args.batch_size, (args.image_size, args.min_year),
            pin_memory=not args.disable_gpu)
    elif args.dataset == 'danbooru':
        dataset = DanbooruPortrait.asloader(
            args.batch_size, (args.image_size, args.min_year),
            pin_memory=not args.disable_gpu)
    test_shape = (args.num_test, args.image_channels, args.image_size, args.image_size)

    # model
    denoise_model = UNet(
        args.image_size, args.bottom, args.image_channels, args.channels,
        args.attn_resls, args.attn_head, args.time_affine,
        args.dropout, args.num_res, args.norm_name, args.act_name)
    ema_model = UNet(
        args.image_size, args.bottom, args.image_channels, args.channels,
        args.attn_resls, args.attn_head, args.time_affine,
        args.dropout, args.num_res, args.norm_name, args.act_name)
    freeze(ema_model)
    update_ema(denoise_model, ema_model, 0.)
    # diffusion
    gaussian_diffusion = GaussianDiffusion(
        args.timesteps)

    denoise_model.to(device)
    ema_model.to(device)
    gaussian_diffusion.to(device)

    # optimizer
    optimizer = optim.Adam(denoise_model.parameters(), lr=args.lr, betas=args.betas)

    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs

    train(
        args.max_iters, dataset, args.timesteps, test_shape,
        denoise_model, ema_model, gaussian_diffusion,
        optimizer, device, amp, args.save, args.sample)
