
import torch
import torch.nn as nn
import numpy as np

def get_activation(name, inplace=True):
    if name == 'relu': return nn.ReLU(inplace)
    if name == 'lrelu': return nn.LeakyReLU(0.2, inplace=inplace)
    if name == 'prelu': return nn.PReLU()
    if name == 'sigmoid': return nn.Sigmoid()
    if name == 'tanh': return nn.Tanh()

def get_normalization(name, channels):
    if name == 'bn': return nn.BatchNorm2d(channels)
    if name == 'in': return nn.InstanceNorm2d(channels)

def _support_sn(sn, layer):
    if sn: return nn.utils.spectral_norm(layer)
    return layer

def Conv2d(sn, *args, **kwargs):
    layer = nn.Conv2d(*args, **kwargs)
    return _support_sn(sn, layer)
def Linear(sn, *args, **kwargs):
    layer = nn.Linear(*args, **kwargs)
    return _support_sn(sn, layer)
def ConvTranspose2d(sn, *args, **kwargs):
    layer = nn.ConvTranspose2d(*args, **kwargs)
    return _support_sn(sn, layer)

class Res(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    def forward(self, x):
        h = self.module(x)
        return (h + x) / np.sqrt(2)

class Block(nn.Module):
    def __init__(self,
        channels,
        sn=True, bias=True, norm_name='in', act_name='prelu'
    ):
        super().__init__()
        self.block = nn.Sequential(
            get_normalization(norm_name, channels),
            get_activation(act_name),
            Conv2d(sn, channels, channels, 3, 1, 1, bias=bias),
            get_normalization(norm_name, channels),
            get_activation(act_name),
            Conv2d(sn, channels, channels, 3, 1, 1, bias=bias)
        )
    def forward(self, x):
        return self.block(x)

class Up(nn.Module):
    def __init__(self,
        in_channels, out_channels,
        sn=True, bias=True, act_name='prelu'
    ):
        super().__init__()
        self.up = nn.Sequential(
            get_activation(act_name),
            Conv2d(sn, in_channels, out_channels*4, 3, 1, 1, bias=bias),
            nn.PixelShuffle(2)
        )
    def forward(self, x):
        return self.up(x)

class Generator(nn.Module):
    def __init__(self,
        scale, image_channels=3, channels=64, num_blocks=5,
        sn=True, bias=True, norm_name='in', act_name='prelu'
    ):
        super().__init__()
        num_ups = int(np.log2(scale))

        self.input = Conv2d(
            sn, image_channels, channels, 7, 1, 3
        )
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                Res(Block(
                    channels, sn, bias,
                    norm_name, act_name
                ))
            )
        blocks.extend([
            get_normalization(norm_name, channels),
            get_activation(act_name),
            Conv2d(sn, channels, channels, 3, 1, 1, bias=bias)
        ])
        self.res_blocks = Res(
            nn.Sequential(*blocks)
        )
        ups = []
        for _ in range(num_ups):
            ups.append(
                Up(
                    channels, channels,
                    sn, bias, act_name
                )
            )
        self.ups = nn.Sequential(*ups)
        self.output = nn.Sequential(
            get_activation(act_name),
            Conv2d(sn, channels, image_channels, 7, 1, 3, bias=bias),
            get_activation('tanh')
        )
    def forward(self, x):
        x = self.input(x)
        x = self.res_blocks(x)
        x = self.ups(x)
        x = self.output(x)
        return x

class SingleScaleDiscriminator(nn.Module):
    def __init__(self,
        in_channels, num_layers=3, channels=32,
        sn=True, bias=True, norm_name='in', act_name='lrelu', 
    ):
        super().__init__()

        ochannels = channels
        layers = [
            nn.Sequential(
                Conv2d(sn, in_channels, ochannels, 4, 2, 1, bias=bias),
                get_activation(act_name)
            )
        ]
        for _ in range(num_layers):
            layers.append(
                nn.Sequential(
                    Conv2d(sn, channels, channels*2, 4, 2, 1, bias=bias),
                    get_normalization(norm_name, channels*2),
                    get_activation(act_name)
                )
            )
            channels *= 2
        layers.append(
            Conv2d(sn, channels, 1, 4, 1, 1, bias=bias)
        )
        self.disc = nn.ModuleList(layers)
    def forward(self, x):
        out = []
        for module in self.disc:
            x = module(x)
            out.append(x)
        return out[-1], out[:-1]

class Discriminator(nn.ModuleList):
    def __init__(self,
        in_channels=3, num_scale=2, num_layers=3, channels=32,
        sn=True, bias=True, norm_name='in', act_name='lrelu'
    ):
        super().__init__()

        self.discs = nn.ModuleList()
        for _ in range(num_scale):
            self.discs.append(
                SingleScaleDiscriminator(
                    in_channels, num_layers, channels,
                    sn, bias, norm_name, act_name
                )
            )
        self.down = nn.AvgPool2d(2)
    def forward(self, x):
        out = []
        for module in self.discs:
            out.append(module(x))
            x = self.down(x)
        return out

def init_weight_N002(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.02)
        if not m.bias == None:
            m.bias.data.fill_(0.)
    elif isinstance(m, nn.BatchNorm2d):
        if not m.weight == None:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(0.)

def init_weight_xavier(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        if not m.bias == None:
            m.bias.data.fill_(0.)
    elif isinstance(m, nn.BatchNorm2d):
        if not m.weight == None:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(0.)

def init_weight_kaiming(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if not m.bias == None:
            m.bias.data.fill_(0.)
    elif isinstance(m, nn.BatchNorm2d):
        if not m.weight == None:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(0.)