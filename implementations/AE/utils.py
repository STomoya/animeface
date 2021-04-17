
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as T
from torchvision.utils import save_image
from tqdm import tqdm

from .model import AE

from ..general import YearAnimeFaceDataset, to_loader, save_args, get_device

def train(
    dataset, epochs,
    model, optimizer, criterion,
    device, amp, save_iterval=1000, image_range=(0, 1)
):
    
    best_loss = 1e10
    best_epoch = 0
    batches_done = 0
    bar = tqdm(total=epochs*len(dataset))
    scaler = GradScaler()

    for epoch in range(epochs):
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

            if batches_done % save_iterval == 0:
                save_image(output[:9], f'implementations/AE/result/recon_{batches_done}.png',
                           nrow=3, normalize=True, value_range=image_range)

            batch_loss_pyfloat = batch_loss.item()
            epoch_loss += batch_loss_pyfloat

            if amp:
                scaler.update()
            bar.set_postfix_str('loss : {:.5f}'.format(batch_loss_pyfloat))
            bar.update(1)
            batches_done += 1
        
        epoch_loss /= len(dataset)
        if best_loss > epoch_loss:
            torch.save(model.state_dict(), 'implementations/AE/result/best_model.pt')
            best_loss = epoch_loss
            best_epoch = epoch

def add_arguments(parser):
    parser.add_argument('--enc-dim', default=128, type=int, help='dimension to encode to')
    parser.add_argument('--min-size', default=8, type=int, help='minimum size before flatten')
    parser.add_argument('--num-layers', default=None, type=int, help='number of layers in encoder. if not given, will be calculated from --min-size')
    parser.add_argument('--img-channels', default=3, type=int, help='number of channels for the input image')
    parser.add_argument('--channels', default=64, type=int, help='channel width multiplier')
    parser.add_argument('--norm-name', default='bn', choices=['bn', 'in'], help='normalization layer name')
    parser.add_argument('--act-name', default='relu', choices=['relu', 'lrelu'], help='activation function')
    parser.add_argument('--up-mode', default='bilinear', type=str, help='upsample mode')
    parser.add_argument('--output-act', default='tanh', choices=['tanh', 'sigmoid'], help='activation function on output')
    return parser

def main(parser):

    parser = add_arguments(parser)
    args = parser.parse_args()
    save_args(args)
    
    amp = not args.disable_amp
    image_range = (0, 1) if args.output_act == 'sigmoid' else (-1, 1)

    # define dataset
    dataset = YearAnimeFaceDataset(args.image_size, args.min_year)
    # transform w/o normalization
    if args.output_act == 'sigmoid':
        dataset.transform = T.Compose([
            T.Resize((args.image_size, args.image_size)),
            T.ToTensor()
        ])
    dataset = to_loader(dataset, args.batch_size, use_gpu=torch.cuda.is_available())

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

    train(
        dataset, args.default_epochs, model,
        optimizer, criterion, device,
        amp, 1000, image_range
    )