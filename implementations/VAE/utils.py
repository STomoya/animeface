
import functools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image

from ..general import YearAnimeFaceDataset, DanbooruPortraitDataset, to_loader
from ..general import get_device, Status
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


def main(parser):
    
    # param
    # data
    image_size = 256
    min_year = 2010
    image_channels = 3
    batch_size = 64

    # model
    z_dim = 256
    channels = 32
    max_channels = 2 ** 10
    enc_target_resl = 4
    use_bias = True
    norm_name = 'bn'
    act_name = 'relu'

    # training
    max_iter = -1
    lr = 0.0002
    betas = (0.9, 0.999)
    test_num_images = 16

    amp = True
    device = get_device()

    # dataset
    dataset = YearAnimeFaceDataset(image_size, min_year)
    dataset = to_loader(dataset, batch_size)
    test_sampler = functools.partial(
        sample_nnoise, size=(test_num_images, z_dim), device=device
    )
    if max_iter < 0:
        max_iter = len(dataset) * 200

    model = VAE(
        image_size, z_dim, image_channels,
        channels, max_channels, enc_target_resl,
        use_bias, norm_name, act_name
    )
    # model.apply(init_weight)
    model.to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)
    
    train(
        dataset, max_iter, test_sampler,
        model, optimizer,
        device, amp, 1000
    )