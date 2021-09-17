
import functools
from utils.argument import add_args

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image

from dataset import AnimeFace, DanbooruPortrait
from utils import Status, save_args, add_args
from nnutils import sample_nnoise, get_device

from .model import VAE, init_weight

recons = nn.MSELoss(reduction='sum')
def KL_divergence(mu, logvar):
    return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def train(
    dataset, max_iter, test_sampler,
    model, optimizer,
    device, amp, save=1000
):

    status = Status(max_iter)
    scaler = GradScaler() if amp else None

    while status.batches_done < max_iter:
        for src in dataset:
            optimizer.zero_grad()

            src = src.to(device)

            with autocast(amp):
                # VAE(rsc)
                dst, _, mu, logvar = model(src)
                # loss
                recons_loss = recons(dst, src)
                kld = KL_divergence(mu, logvar)
                loss = recons_loss + kld

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
            else:
                loss.backward()
                optimizer.step()

            # save
            if status.batches_done % save == 0:
                model.eval()
                with torch.no_grad():
                    images = model.decoder(test_sampler())
                model.train()
                save_image(
                    images, f'implementations/VAE/result/{status.batches_done}.jpg',
                    nrow=4, normalize=True, value_range=(-1, 1))
                recons_images = _image_grid(src, dst)
                save_image(
                    recons_images, f'implementations/VAE/result/recons_{status.batches_done}.jpg',
                    nrow=6, normalize=True, value_range=(-1, 1))
                torch.save(model.state_dict(), f'implementations/VAE/result/model_{status.batches_done}.pt')

            # updates
            loss_dict = dict(
                loss=loss.item() if not torch.isnan(loss).any() else 0
            )
            status.update(**loss_dict)
            if scaler is not None:
                scaler.update()

            if status.batches_done == max_iter:
                break

    status.plot_loss()

def _image_grid(src, dst, num_images=6):
    srcs = src.chunk(src.size(0), dim=0)
    dsts = dst.chunk(dst.size(0), dim=0)

    images = []
    for index, (src, dst) in enumerate(zip(srcs, dsts)):
        images.extend([src, dst])
        if index == num_images - 1:
            break

    return torch.cat(images, dim=0)

def main(parser):

    parser = add_args(parser,
        dict(
            image_channels  = [3, 'number of channels in input images'],
            z_dim           = [256, 'dimension of extracted feature vector z'],
            channels        = [32, 'channel width multiplier'],
            max_channels    = [1024, 'maximum channels'],
            enc_target_resl = [4, 'resolution to dwonsample to before faltten'],
            disable_bias    = [False, 'do not use bias'],
            norm_name       = ['bn', 'normalization layer name'],
            act_name        = ['relu', 'activation function name'],
            lr              = [0.0002, 'learning rate'],
            beta1           = [0.9, 'beta1'],
            beta2           = [0.999, 'beta2'],
            test_images     = [16, 'number of images for evaluation']))
    args = parser.parse_args()
    save_args(args)

    use_bias = not args.disable_bias
    betas = (args.beta1, args.beta2)

    amp = not args.disable_amp
    device = get_device(not args.disable_gpu)

    # dataset
    dataset = AnimeFace.asloader(
        args.batch_size, (args.image_size, args.min_year),
        pin_memory=not args.disable_gpu)
    test_sampler = functools.partial(
        sample_nnoise, size=(args.test_images, args.z_dim), device=device
    )
    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs

    model = VAE(
        args.image_size, args.z_dim, args.image_channels,
        args.channels, args.max_channels, args.enc_target_resl,
        use_bias, args.norm_name, args.act_name
    )
    # model.apply(init_weight)
    model.to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=betas)

    train(
        dataset, args.max_iters, test_sampler,
        model, optimizer,
        device, amp, 1000
    )
