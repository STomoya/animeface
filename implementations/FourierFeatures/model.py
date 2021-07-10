
import math
import torch
import torch.nn as nn
from opt_einsum import contract

def get_activation(name='relu'):
    if name == 'relu': return nn.ReLU(True)
    if name == 'swish': return nn.SiLU(True)
    raise Exception('activation')

def get_normalization(name, channels):
    if name == 'bn': return nn.BatchNorm2d(channels)
    if name is None: return nn.Identity()
    raise Exception('normalization')

class GaussianFourierFeatureMapping(nn.Module):
    '''
    [code] https://github.com/ndahlquist/pytorch-fourier-feature-networks/blob/master/fourier_feature_transform.py
    modified to use einsum when applying B to input
    '''
    def __init__(self,
        in_channels, map_size, scale=10.
    ) -> None:
        super().__init__()
        self.register_buffer('B', torch.randn(in_channels, map_size//2) * scale)

    def forward(self, x):
        x = contract('bchw,cm->bmhw', x, self.B)
        x = 2 * math.pi * x
        return torch.cat([torch.cos(x), torch.sin(x)], dim=1)

class MLP(nn.Module):
    def __init__(self,
        map: bool, map_size=256, map_scale=10.,
        num_layers=4, hid_channels=256,
        act_name='relu', norm_name=None
    ) -> None:
        super().__init__()

        layers = []
        if map:
            layers.extend([
                GaussianFourierFeatureMapping(2, map_size, map_scale),
                nn.Conv2d(map_size, hid_channels, 1)])
        else:
            layers.append(
                nn.Conv2d(2, hid_channels, 1))

        for _ in range(num_layers-2):
            layers.extend([
                get_normalization(norm_name, hid_channels),
                get_activation(act_name),
                nn.Conv2d(hid_channels, hid_channels, 1)])
        layers.extend([
            get_normalization(norm_name, hid_channels),
            get_activation(act_name),
            nn.Conv2d(hid_channels, 3, 1)])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        assert x.ndim == 4, 'input requires 4 dim'
        x = self.net(x)
        return torch.sigmoid(x)
