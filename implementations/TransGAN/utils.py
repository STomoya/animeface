
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torchvision.utils import save_image

from ..general import YearAnimeFaceDataset, DanbooruPortraitDataset, to_loader
from ..general import save_args, get_device, Status
from ..gan_utils import sample_nnoise
from ..gan_utils.losses import HingeLoss, WGANLoss, GradPenalty

from .model import Generator, Discriminator

def train():
    pass

def add_arguments(parser):
    return parser

def main(parser):

    parser = add_arguments(parser)
    args = parser.parse_args()
    save_args(args)

    train()
