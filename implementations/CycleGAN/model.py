
import torch
import torch.nn as nn
import numpy as np

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

class Down(nn.Module):
    def __init__(self,
        in_channels, out_channels,
        use_sn=False, use_bias=True, norm_name='in', act_name='relu'
    ):
        super().__init__()

        self.down = nn.Sequential(
            Conv2d(use_sn, in_channels, out_channels, 3, 2, 1, bias=use_bias),
            get_normalization(norm_name, out_channels),
            get_activation(act_name)
        )
    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    def __init__(self,
        in_channels, out_channels,
        use_sn=False, use_bias=True, norm_name='in', act_name='relu'
    ):
        super().__init__()

        self.up = nn.Sequential(
            ConvTranspose2d(use_sn, in_channels, out_channels, 3, 2, 1, output_padding=1, bias=use_bias),
            get_normalization(norm_name, out_channels),
            get_activation(act_name)
        )
    def forward(self, x):
        return self.up(x)

class ResBlock(nn.Module):
    def __init__(self,
        channels, num_conv=2,
        use_sn=False, use_bias=True, norm_name='in', act_name='relu'
    ):
        super().__init__()

        layers = []
        for _ in range(num_conv):
            layers.extend([
                Conv2d(use_sn, channels, channels, 3, 1, 1, padding_mode='reflect', bias=use_bias),
                get_normalization(norm_name, channels),
                get_activation(act_name)
            ])
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        h = x
        h = self.block(h)
        return x + h

class Generator(nn.Module):
    def __init__(self,
        image_size, in_channels, out_channels, target_resl=32,
        channels=32, max_channels=256, num_blocks=6, block_num_conv=2,
        use_sn=False, use_bias=True, norm_name='in', act_name='relu'
    ):
        super().__init__()

        num_downs = int(np.log2(image_size) - np.log2(target_resl))

        ochannels = channels
        self.input = nn.Sequential(
            Conv2d(use_sn, in_channels, ochannels, 7, 1, 3, padding_mode='reflect', bias=use_bias),
            get_normalization(norm_name, ochannels),
            get_activation(act_name)
        )

        downs = []
        for _ in range(num_downs):
            channels *= 2
            ichannels, ochannels = ochannels, min(max_channels, channels)
            downs.append(
                Down(
                    ichannels, ochannels, use_sn, use_bias,
                    norm_name, act_name
                )
            )
        self.downs = nn.Sequential(*downs)

        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                ResBlock(
                    ochannels, block_num_conv, use_sn, use_bias,
                    norm_name, act_name
                )
            )
        self.blocks = nn.Sequential(*blocks)

        ups = []
        for _ in range(num_downs):
            channels = channels // 2
            ichannels, ochannels = ochannels, min(max_channels, channels)
            ups.append(
                Up(
                    ichannels, ochannels, use_sn, use_bias,
                    norm_name, act_name
                )
            )
        self.ups = nn.Sequential(*ups)

        self.output = nn.Sequential(
            Conv2d(use_sn, ochannels, out_channels, 7, 1, 3, padding_mode='reflect', bias=use_bias),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.input(x)
        x = self.downs(x)
        x = self.blocks(x)
        x = self.ups(x)
        x = self.output(x)
        return x

class Discriminator(nn.Module):
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
        self.disc = nn.Sequential(*layers)
    def forward(self, x):
        return self.disc(x)

def init_weight_normal(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        m.weight.data.normal_(0, 0.02)
        if not m.bias == None:
            m.bias.data.fill_(0.)
    if isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1., 0.02)
        m.bias.data.fill_(0.)

if __name__ == "__main__":
    x = torch.randn(32, 1, 128, 128)
    g = Generator(128, 1, 3)
    d = Discriminator(128, 3)
    img = g(x)
    prob = d(img)
    print(img.size(), prob.size())
