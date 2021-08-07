
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as T
from torchvision.utils import save_image
from tqdm import tqdm

from .model import AE

# from ..general import YearAnimeFaceDataset, to_loader, save_args, get_device
from utils import save_args, add_args, Status
from nnutils import get_device
from dataset import AnimeFace

def train(
    max_iters, dataset,
    model, optimizer, criterion,
    device, amp, save_iterval=1000, image_range=(0, 1)
):

    status = Status(max_iters)
    scaler = GradScaler()
    best_loss = 1e10

    while status.batches_done < max_iters:
        epoch_loss = 0
        for index, data in enumerate(dataset):
            optimizer.zero_grad()

            data = data.to(device)

            with autocast(amp):
                output = model(data)
                batch_loss = criterion(output, data)

            if amp:
                scaler.scale(batch_loss).backward()
                scaler.step(optimizer)
            else:
                batch_loss.backward()
                optimizer.step()

            if status.batches_done % save_iterval == 0:
                save_image(
                    output[:9], f'implementations/AE/result/recon_{status.batches_done}.png',
                    nrow=3, normalize=True, value_range=image_range)

            batch_loss_pyfloat = batch_loss.item()
            epoch_loss += batch_loss_pyfloat

            status.update(loss=batch_loss_pyfloat)
            if amp:
                scaler.update()

            if status.batches_done == max_iters:
                break

        epoch_loss /= len(dataset)
        if best_loss > epoch_loss:
            torch.save(model.state_dict(), 'implementations/AE/result/best_model.pt')
            best_loss = epoch_loss

    status.plot_loss()

def main(parser):

    parser = add_args(parser,
        dict(
            enc_dim      = [128,        'dimension to encode to'],
            min_size     = [8,          'minimum size before flatten'],
            num_layers   = [int,        'number of layers in encoder. if not given, will be calculated from --min-size'],
            img_channels = [3,          'number of channels of the images'],
            channels     = [64,         'channel width multiplier'],
            norm_name    = ['bn',       'normalization layer name'],
            act_name     = ['relu',     'activation function name'],
            up_mode      = ['bilinear', 'upsample mode'],
            output_act   = ['tanh',     'output activation.'])
    )
    args = parser.parse_args()
    save_args(args)

    amp = not args.disable_amp
    image_range = (0, 1) if args.output_act == 'sigmoid' else (-1, 1)

    # define dataset
    dataset = AnimeFace.asloader(
        args.batch_size, (args.image_size, args.min_year),
        pin_memory=not args.disable_gpu)
    # transform w/o normalization
    if args.output_act == 'sigmoid':
        dataset.dataset.transform = T.Compose([
            T.Resize((args.image_size, args.image_size)),
            T.ToTensor()
        ])

    device = get_device(not args.disable_gpu)

    # define model
    model = AE(
        args.enc_dim, args.image_size, args.min_size, args.num_layers,
        args.img_channels, args.channels, args.norm_name, args.act_name,
        args.up_mode, args.output_act
    )
    model.to(device)

    # define optimizer
    optimizer = optim.Adam(model.parameters())

    # define loss function
    criterion = nn.BCEWithLogitsLoss() if args.output_act == 'sigmoid' else nn.MSELoss()

    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs

    train(
        args.max_iters, dataset, model,
        optimizer, criterion, device,
        amp, 1000, image_range
    )
