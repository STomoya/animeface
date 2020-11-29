
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as T
from torchvision.utils import save_image
from tqdm import tqdm

from .model import AE

from ..general import YearAnimeFaceDataset, to_loader
from ..gan_utils import get_device

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
                           nrow=3, normalize=True, range=image_range)

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

def main():

    # data
    min_year = 2010
    image_size = 256
    batch_size = 64

    # model
    enc_dim = 128
    min_size = 8
    num_layers = None
    img_channels = 3
    channels = 64
    norm_name = 'bn'
    act_name = 'relu'
    up_mode = 'bilinear'
    output_act = 'tanh' # use 'tanh' instead when normlizing the data to [-1, 1]

    # training
    epochs = 100
    amp = True
    image_range = (0, 1) if output_act == 'sigmoid' else (-1, 1)

    # define dataset
    dataset = YearAnimeFaceDataset(image_size, min_year)
    # transform w/o normalization
    if output_act == 'sigmoid':
        dataset.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor()
        ])
    dataset = to_loader(dataset, batch_size, use_gpu=torch.cuda.is_available())

    device = get_device()

    # define model
    model = AE(
        enc_dim, image_size, min_size, num_layers,
        img_channels, channels, norm_name, act_name,
        up_mode, output_act
    )
    model.to(device)

    # define optimizer
    optimizer = optim.Adam(model.parameters())

    # define loss function
    criterion = nn.BCEWithLogitsLoss() if output_act == 'sigmoid' else nn.MSELoss()

    train(
        dataset, epochs, model,
        optimizer, criterion, device,
        amp, 1000, image_range
    )