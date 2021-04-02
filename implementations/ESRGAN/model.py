
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
    if name == 'in': return nn.InstanceNorm2d(channels)
    if name == 'bn': return nn.BatchNorm2d(channels)

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

class DenseBlock(nn.Module):
    def __init__(self,
        channels, hid_channels, num_conv=5,
        sn=True, bias=True, norm_name='', act_name='lrelu'
    ):
        super().__init__()

        def _make_layer(ichannels, ochannels, act=True):
            layers = [Conv2d(sn, ichannels, ochannels, 3, 1, 1, bias=bias)]
            if norm_name != '':
                layers.append(
                    get_normalization(norm_name, ochannels))
            layers.append(get_activation(act_name))
            return nn.Sequential(*layers)

        self.convs = nn.ModuleList()
        for i in range(num_conv-1):
            self.convs.append(
                _make_layer(
                    channels+i*hid_channels, hid_channels)
            )
        self.convs.append(
            Conv2d(sn, channels+(num_conv-1)*hid_channels, channels, 3, 1, 1, bias=bias)
        )
    def forward(self, x):
        x_ = self.convs[0](x)
        feats = [x]
        for module in self.convs[1:]:
            feats.append(x_)
            x_ = module(torch.cat(feats, dim=1))
        return x_

class ResinResDenseBlock(nn.Module):
    def __init__(self,
        channels, hid_channels, num_dense=3, num_conv=5,
        sn=True, bias=True, norm_name='', act_name='lrelu'
    ):
        super().__init__()
        
        layers = []
        for _ in range(num_dense):
            layers.append(
                Res(DenseBlock(
                    channels, hid_channels, num_conv,
                    sn, bias, norm_name, act_name))
            )
        res_dense = nn.Sequential(*layers)
        self.rrd_block = Res(res_dense)
    
    def forward(self, x):
        return self.rrd_block(x)

class Generator(nn.Module):
    def __init__(self,
        scale, image_channels=3, channels=64, hid_channels=32, num_rrdb=15,
        num_rd=3, num_conv=5,
        sn=True, bias=True, norm_name='', act_name='lrelu'
    ):
        super().__init__()
        num_ups = int(np.log2(scale))

        self.input = Conv2d(sn, image_channels, channels, 7, 1, 3)

        rrdbs = []
        for _ in range(num_rrdb):
            rrdbs.append(
                ResinResDenseBlock(
                    channels, hid_channels, num_rd,
                    num_conv, sn, bias,
                    norm_name, act_name
                )
            )
        rrdbs.append(
            Conv2d(sn, channels, channels, 3, 1, 1, bias=bias)
        )
        self.res_blocks = Res(nn.Sequential(*rrdbs))
        
        ups = []
        for _ in range(num_ups):
            ups.extend([
                nn.Upsample(scale_factor=2),
                Conv2d(sn, channels, channels, 3, 1, 1, bias=bias)
            ])
        self.ups = nn.Sequential(*ups)

        self.output = nn.Sequential(
            Conv2d(sn, channels, image_channels, 3, 1, 1, bias=bias),
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
