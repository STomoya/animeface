
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class View(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
    def forward(self, x):
        return x.reshape(x.size(0), *self.size)

def get_normalization(name, channels, **kwargs):
    if name == 'bn': return nn.BatchNorm2d(channels, **kwargs)
    elif name == 'in': return nn.InstanceNorm2d(channels, **kwargs)

def get_activation(name, inplace=True):
    if name == 'relu': return nn.ReLU(inplace=inplace)
    elif name == 'lrelu': return nn.LeakyReLU(0.2, inplace=inplace)

SN = nn.utils.spectral_norm

def Conv2d(sn, *args, **kwargs):
    layer = nn.Conv2d(*args, **kwargs)
    if sn: return SN(layer)
    return layer

def ConvTranspose2d(sn, *args, **kwargs):
    layer = nn.ConvTranspose2d(*args, **kwargs)
    if sn: return SN(layer)
    return layer

def Linear(sn, *args, **kwargs):
    layer = nn.Linear(*args, **kwargs)
    if sn: return SN(layer)
    return layer

class SPADE(nn.Module):
    def __init__(
        self, channels, in_channels, hidden_channels=128,
        norm_name='in', act_name='relu', use_bias=True
    ):
        super().__init__()
        use_sn = False

        self.norm = get_normalization(norm_name, channels, affine=False)
        self.shared = nn.Sequential(
            Conv2d(use_sn, in_channels, hidden_channels, 3, padding=1, bias=use_bias),
            get_activation(act_name)
        )
        self.gamma = Conv2d(use_sn, hidden_channels, channels, 3, padding=1, bias=use_bias)
        self.beta  = Conv2d(use_sn, hidden_channels, channels, 3, padding=1, bias=use_bias)
    
    def forward(self, x, input):
        norm = self.norm(x)
        input = self.resize(input, norm.size()[2:])
        input = self.shared(input)
        gamma, beta = self.gamma(input), self.beta(input)
        return gamma * norm + beta

    def resize(self, input, size):
        return F.interpolate(input, size, mode='nearest')

class SPADEResBlock(nn.Module):
    def __init__(self,
        in_channels, out_channels, spade_in_channels,
        spade_hidden_channels=128, num_conv=2,
        norm_name='in', act_name='lrelu', use_sn=True, use_bias=True
    ):
        super().__init__()

        layers = [
            SPADE(
                in_channels, spade_in_channels, spade_hidden_channels,
                norm_name, act_name, use_bias),
            get_activation(act_name, False),
            Conv2d(use_sn, in_channels, out_channels, 3, padding=1, bias=use_bias)
        ]
        for _ in range(num_conv-1):
            layers.extend([
                SPADE(
                    out_channels, spade_in_channels, spade_hidden_channels,
                    norm_name, act_name, use_bias),
                get_activation(act_name, False),
                Conv2d(use_sn, out_channels, out_channels, 3, padding=1, bias=use_bias)
            ])
        self.block = nn.ModuleList(layers)
        
        if in_channels is not out_channels:
            self.skip = nn.ModuleList([
                SPADE(
                    in_channels, spade_in_channels, spade_hidden_channels,
                    norm_name, act_name, use_bias),
                get_activation(act_name, False),
                Conv2d(use_sn, in_channels, out_channels, 3, padding=1, bias=use_bias)
            ])
        else: self.skip = []

    def _loop_forward(self, x, input, module_list):
        for module in module_list:
            if isinstance(module, SPADE):
                x = module(x, input)
            else:
                x = module(x)
        return x

    def forward(self, x, input):
        h = x
        h = self._loop_forward(h, input, self.block)
        x = self._loop_forward(x, input, self.skip)
        return x + h

class Generator(nn.Module):
    def __init__(self,
        image_size, z_dim, in_channels=1, out_channels=3, channels=32, max_channels=2**10,
        block_num_conv=2, spade_hidden_channels=128,
        norm_name='in', act_name='lrelu', use_sn=False, use_bias=True
    ):
        super().__init__()
        
        num_ups = int(np.log2(image_size) - 2)
        channels = channels * 2 ** num_ups
        ochannels = min(max_channels, channels)

        self.input = nn.Sequential(
            Linear(use_sn, z_dim, ochannels*4**2, bias=use_bias),
            View((ochannels, 4, 4))
        )
        self.up = nn.Upsample(scale_factor=2)
        self.blocks = nn.ModuleList()
        for _ in range(num_ups):
            channels = channels // 2
            ichannels, ochannels = ochannels, min(max_channels, channels)
            self.blocks.append(
                SPADEResBlock(
                    ichannels, ochannels, in_channels, spade_hidden_channels,
                    block_num_conv, norm_name, act_name, use_sn, use_bias
                )
            )
        self.output = nn.Sequential(
            Conv2d(use_sn, ochannels, out_channels, 3, padding=1, bias=use_bias),
            nn.Tanh()
        )
    def forward(self, x, input):
        x = self.input(x)
        for module in self.blocks:
            x = module(x, input)
            x = self.up(x)
        x = self.output(x)
        return x

class SingleScaleDiscriminator(nn.Module):
    def __init__(self,
        image_size, in_channels, num_layers=3, channels=32,
        norm_name='in', act_name='lrelu', use_sn=True, use_bias=True
    ):
        super().__init__()

        ochannels = channels
        layers = [
            nn.Sequential(
                Conv2d(use_sn, in_channels, ochannels, 4, 2, 1, bias=use_bias),
                get_activation(act_name)
            )
        ]
        for _ in range(num_layers):
            layers.append(
                nn.Sequential(
                    Conv2d(use_sn, channels, channels*2, 4, 2, 1, bias=use_bias),
                    get_normalization(norm_name, channels*2),
                    get_activation(act_name)
                )
            )
            channels *= 2
        layers.append(
            Conv2d(use_sn, channels, 1, 4, 1, 1, bias=use_bias)
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
        image_size, in_channels=3, num_scale=3, num_layers=3, channels=32,
        norm_name='in', act_name='lrelu', use_sn=True, use_bias=True
    ):
        super().__init__()

        self.discs = nn.ModuleList()
        for _ in range(num_scale):
            self.discs.append(
                SingleScaleDiscriminator(
                    image_size, in_channels, num_layers, channels,
                    norm_name, act_name, use_sn, use_bias
                )
            )
        self.down = nn.AvgPool2d(2)
    def forward(self, x):
        out = []
        for module in self.discs:
            out.append(module(x))
            x = self.down(x)
        return out

def ConvBlock(
    in_channels, out_channels, use_sn=True, use_bias=True,
    norm_name='bn', act_name='relu'
):
    return nn.Sequential(
        Conv2d(use_sn, in_channels, out_channels, 3, 2, 1, bias=use_bias),
        get_normalization(norm_name, out_channels),
        get_activation(act_name)
    )

class Encoder(nn.Module):
    def __init__(self,
        image_size, z_dim, in_channels=3, target_resl=4, channels=32, max_channels=512,
        use_sn=True, use_bias=True, norm_name='in', act_name='lrelu'
    ):
        super().__init__()
        convb_func = functools.partial(
            ConvBlock, use_sn=use_sn, use_bias=use_bias, norm_name=norm_name, act_name=act_name
        )
        
        ochannels = channels
        layers = [convb_func(in_channels, ochannels)]
        image_size = image_size // 2
        while image_size > target_resl:
            image_size = image_size // 2
            channels *= 2
            ichannels, ochannels = ochannels, min(max_channels, channels)
            layers.append(convb_func(ichannels, ochannels))
        layers.extend([
            nn.Flatten()
        ])
        self.extract = nn.Sequential(*layers)
        num_features = ochannels * (target_resl ** 2)
        self.mu  = nn.Linear(num_features, z_dim, bias=use_bias)
        self.var = nn.Linear(num_features, z_dim, bias=use_bias)

    def forward(self, x):
        feat = self.extract(x)
        mu, var = self.mu(feat), self.var(feat)
        z = self.reparameterize(mu, var)
        return z, mu, var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z

# glorot init.
def init_weight_xavier(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight.data, 0.02)
        if not m.bias == None:
            m.bias.data.fill_(0.)
    elif isinstance(m, nn.BatchNorm2d):
        if not m.weight == None:
            m.weight.data.normal_(1., 0.02)
        if not m.bias == None:
            m.bias.data.fill_(0.)

if __name__ == "__main__":
    g = Generator(256, 256)
    d = Discriminator(256)
    e = Encoder(256, 256)
    real = torch.randn(32, 3, 256, 256)
    input = torch.randn(32, 1, 256, 256)
    z, mu, var = e(real)
    img = g(z, input)
    probs = d(img)
    print(img.size(), mu.size(), var.size(), z.size())
    for prob in probs:
        print(prob[0].size())