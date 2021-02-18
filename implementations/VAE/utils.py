
import functools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image

from ..general import YearAnimeFaceDataset, DanbooruPortraitDataset, to_loader
from ..general import get_device, Status, save_args
from ..gan_utils import sample_nnoise, AdaBelief

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
                save_image(images, f'implementations/VAE/result/{status.batches_done}.jpg', nrow=4, normalize=True, range=(-1, 1))
                recons_images = _image_grid(src, dst)
                save_image(recons_images, f'implementations/VAE/result/recons_{status.batches_done}.jpg', nrow=6, normalize=True, range=(-1, 1))
                torch.save(model.state_dict(), f'implementations/VAE/result/model_{status.batches_done}.pt')

            # updates
            loss_dict = dict(
                loss=loss.item() if not torch.isnan(loss).any() else 0
            )
            status.update(loss_dict)
            if scaler is not None:
                scaler.update()

            if status.batches_done == max_iter:
                break

def _image_grid(src, dst, num_images=6):
    srcs = src.chunk(src.size(0), dim=0)
    dsts = dst.chunk(dst.size(0), dim=0)

    images = []
    for index, (src, dst) in enumerate(zip(srcs, dsts)):
        images.extend([src, dst])
        if index == num_images - 1:
            break
    
    return torch.cat(images, dim=0)

def add_arguments(parser):
    parser.add_argument('--image-channels', default=3, type=int, help='number of channels in input images')
    parser.add_argument('--z-dim', default=256, type=int, help='dimension of extracted feature vector z')
    parser.add_argument('--channels', default=32, type=int, help='channel width multiplier')
    parser.add_argument('--max-channels', default=1024, type=int, help='maximum channels')
    parser.add_argument('--enc-target-resl', default=4, type=int, help='resolution to dwon-sample to before faltten')
    parser.add_argument('--disable-bias', default=False, action='store_true', help='do not use bias')
    parser.add_argument('--norm-name', default='bn', choices=['bn', 'in'], help='normalization layer name')
    parser.add_argument('--act-name', default='relu', choices=['relu', 'lrelu'], help='activation function name')
    parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta2')
    parser.add_argument('--test-images', default=16, type=int, help='number of images for evaluation')
    return parser

def main(parser):

    parser = add_arguments(parser)
    args = parser.parse_args()
    save_args(args)

    use_bias = not args.disable_bias
    betas = (args.beta1, args.beta2)

    amp = not args.disable_amp
    device = get_device(not args.disable_gpu)

    # dataset
    dataset = YearAnimeFaceDataset(args.image_size, args.min_year)
    dataset = to_loader(dataset, args.batch_size)
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