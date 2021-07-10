
import glob
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import save_image

from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np

from ..general import save_args, Status, get_device
from ..gan_utils import gif_from_files

from .model import MLP

def train(
    max_iters, train_data, test_data,
    model, optimizer, save
):

    status = Status(max_iters)
    criterion = nn.MSELoss()
    best_psnr = -999.

    while status.batches_done < max_iters:
        optimizer.zero_grad()

        output = model(train_data[0])
        loss = criterion(output, train_data[1])
        psnr = - 10 * torch.log10(loss)

        if best_psnr < psnr:
            best_psnr = psnr
            best_model = model.state_dict()

        loss.backward()
        optimizer.step()

        if status.batches_done % save == 0:
            file_id = str(status.batches_done).zfill(len(str(max_iters)))
            save_image(torch.cat([output, train_data[1]], dim=0),
                f'implementations/FourierFeatures/result/{file_id}.jpg',
                normalize=True, value_range=(0, 1))

        # updates
        loss_dict = dict(PSNR=psnr.item())
        status.update(loss_dict)
    gif_from_files(
        sorted(glob.glob(f'implementations/FourierFeatures/result/*.jpg')),
        'implementations/FourierFeatures/result/train-seq.gif')

    torch.save(best_model,
        f'implementations/FourierFeatures/result/psnr_best.pt')
    model.load_state_dict(best_model)
    model.eval()
    with torch.no_grad():
        image = model(test_data[0])
        save_image(torch.cat([image, test_data[1]], dim=0),
        f'implementations/FourierFeatures/result/best.jpg',
            normalize=True, value_range=(0, 1))

    status.plot()

def prepair_data(path, target_size):
    # image
    image = Image.open(path).convert('RGB')
    image = TF.resize(image, target_size)
    image = TF.center_crop(image, target_size)
    image = TF.to_tensor(image).unsqueeze(0)

    # input
    coords = np.linspace(0, 1, target_size, endpoint=False).astype(np.float32)
    xy_grid = np.stack(np.meshgrid(coords, coords), -1)
    xy_grid = torch.tensor(xy_grid).permute(2, 0, 1).unsqueeze(0)
    input = xy_grid.contiguous()

    # train
    # work with smaller image size when training
    train_data = (input[:, :, ::2, ::2], image[:, :, ::2, ::2])

    return train_data, (input, image)

def add_arguments(parser):
    parser.add_argument('--path', default='/usr/src/data/danbooru/2020/0638/1115638.jpg', help='path to an image')

    parser.add_argument('--no-map', default=False, action='store_true')
    parser.add_argument('--map-size', default=256, type=int)
    parser.add_argument('--map-scale', default=10., type=float)
    parser.add_argument('--num-layers', default=4, type=int)
    parser.add_argument('--hid-channels', default=256, type=int)
    parser.add_argument('--act-name', default='relu', choices=['relu', 'swish'])
    parser.add_argument('--norm-name', default=None, choices=['bn'])

    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--betas', default=[0.9, 0.999], type=float, nargs=2)

    return parser

def main(parser):
    parser = add_arguments(parser)
    args = parser.parse_args()
    save_args(args)

    device = get_device(not args.disable_gpu)
    # will not use amp for this

    # data
    train_data, test_data = prepair_data(args.path, args.image_size)
    map_device = lambda data: tuple(map(lambda x: x.to(device), data))
    train_data = map_device(train_data)
    test_data  = map_device(test_data)

    # model
    Net = MLP(
        not args.no_map, args.map_size, args.map_scale, args.num_layers,
        args.hid_channels, args.act_name, args.norm_name)
    Net.to(device)

    # optimizer
    optimizer = optim.Adam(Net.parameters(), lr=args.lr, betas=args.betas)

    if args.max_iters < 0:
        args.max_iters = args.default_epochs

    train(
        args.max_iters, train_data, test_data,
        Net, optimizer, args.save
    )
