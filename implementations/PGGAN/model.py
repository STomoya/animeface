
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


'''
Modules
'''

class PixelNorm(nn.Module):
    def forward(self, x):
        x_square_mean = x.pow(2).mean(dim=1, keepdim=True)
        denom = torch.rsqrt(x_square_mean + 1e-8)
        return x * denom

class EqualizedLR(nn.Module):
    def __init__(self, layer, gain=2):
        super(EqualizedLR, self).__init__()

        self.wscale = (gain / layer.weight[0].numel()) ** 0.5
        self.layer = layer

    def forward(self, x, gain=2):
        x = self.layer(x * self.wscale)
        return x

class MiniBatchStd(nn.Module):
    def forward(self, x):
        std = torch.std(x).expand(x.shape[0], 1, *x.shape[2:])
        return torch.cat([x, std], dim=1)

class GaussianNoise(nn.Module):
    def __init__(self, resl):
        super(GaussianNoise, self).__init__()
        self.magnitude = 0
        self.noise = Variable(torch.zeros(resl, resl)).cuda()
    def forward(self, x):
        if self.training:
            self.noise.data.normal_(0, std=0.1)
            x = x + (self.magnitude * self.noise)
        return x

class Conv2d(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        equalized=True
    ):
        super(Conv2d, self).__init__()

        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        conv.bias.data.fill_(0)
        conv.weight.data.normal_(0, 1)

        if equalized:
            self.conv = EqualizedLR(conv)
        else:
            self.conv = conv

    def forward(self, x):
        x = self.conv(x)
        return x

class ConvTranspose2d(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        equalized=True
    ):
        super(ConvTranspose2d, self).__init__()

        conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        conv.bias.data.fill_(0)
        conv.weight.data.normal_(0, 1)

        if equalized:
            self.conv = EqualizedLR(conv)
        else:
            self.conv = conv

    def forward(self, x):
        x = self.conv(x)
        return x

class ToRGB(nn.Module):
    '''
    To RGB layer
    '''
    def __init__(self,
        in_channels,
        out_channels=3
    ):
        super(ToRGB, self).__init__()

        self.to_rgb = nn.Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1
            ),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.to_rgb(x)
        return x

class FromRGB(nn.Module):
    '''
    From RGB layer
    '''
    def __init__(self,
        out_channels,
        in_channels=3
    ):
        super(FromRGB, self).__init__()

        self.from_rgb = nn.Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1
            )
        )

    def forward(self, x):
        x = self.from_rgb(x)
        return x

class ResolutionBlock(nn.Module):
    '''
    Resl Block
    '''
    def __init__(self,
        in_channels,
        out_channels,
        is_first=False
    ):
        super(ResolutionBlock, self).__init__()

        if is_first:
            self.block = nn.Sequential(
                ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                ),
                PixelNorm(),
                nn.LeakyReLU(0.2),
                Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1
                ),
                PixelNorm(),
                nn.LeakyReLU(0.2)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2),
                Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1
                ),
                PixelNorm(),
                nn.LeakyReLU(0.2),
                Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1
                ),
                PixelNorm(),
                nn.LeakyReLU(0.2)
            )
    def forward(self, x):
        x = self.block(x)
        return x

class DownResolutionBlock(nn.Module):
    def __init__(self,
        resl,
        in_channels,
        out_channels,
        is_last=False
    ):
        super(DownResolutionBlock, self).__init__()

        if is_last:
            self.block = nn.Sequential(
                MiniBatchStd(),
                GaussianNoise(resl),
                Conv2d(
                    in_channels=in_channels+1,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1
                ),
                nn.LeakyReLU(0.2),
                GaussianNoise(resl),
                Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=4
                ),
                nn.LeakyReLU(0.2),
                Conv2d(
                    in_channels=out_channels,
                    out_channels=1,
                    kernel_size=1
                )
            )
        else:
            self.block = nn.Sequential(
                GaussianNoise(resl),
                Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1
                ),
                nn.LeakyReLU(0.2),
                GaussianNoise(resl),
                Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1
                ),
                nn.AvgPool2d(2)
            )

    def forward(self, x):
        x = self.block(x)
        return x

'''
Generator
'''

class Generator(nn.Module):
    def __init__(self,
        latent_dim
    ):
        super(Generator, self).__init__()
        self.train_depth = 0
        self.resl2param = {
            4   : (latent_dim, 512, True),
            8   : (       512, 512, False),
            16  : (       512, 256, False),
            32  : (       256, 128, False),
            64  : (       128,  64, False),
            128 : (        64,  32, False)
        }

        resolution_blocks = []
        rgb_layers = []
        for resl in self.resl2param:
            param = self.resl2param[resl]
            resolution_blocks.append(
                ResolutionBlock(
                    in_channels=param[0],
                    out_channels=param[1],
                    is_first=param[2]
                )
            )
            rgb_layers.append(
                ToRGB(in_channels=param[1])
            )

        self.resolution_blocks = nn.ModuleList(resolution_blocks)
        self.rgb_layers        = nn.ModuleList(rgb_layers)

        self.alpha = 0

    def grow(self):
        self.train_depth += 1
        self.alpha = 0

    def forward(self, x, phase):
        x = x.view(x.size(0), x.size(1), 1, 1)
        if phase == 't':
            return self.transition_forward(x)
        else:
            return self.stablization_forward(x)

    def transition_forward(self, x):
        for index, block in enumerate(self.resolution_blocks):
            x = block(x)
            if index == self.train_depth-1:
                x_pre = F.interpolate(x, scale_factor=2)
            if index == self.train_depth:
                break

        rgb_pre = self.rgb_layers[index-1](x_pre)
        rgb_cur = self.rgb_layers[index](x)
        return (1 - self.alpha) * rgb_pre + self.alpha * rgb_cur

    def stablization_forward(self, x):
        for index, block in enumerate(self.resolution_blocks):
            x = block(x)
            if index == self.train_depth:
                break

        rgb = self.rgb_layers[index](x)
        return rgb

    def update_alpha(self, delta, phase):
        if phase == 't':
            self.alpha = min(1, self.alpha+delta)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.train_depth = 0

        self.resl2param = {
            4   : (512, 512, True),
            8   : (512, 512, False),
            16  : (256, 512, False),
            32  : (128, 256, False),
            64  : ( 64, 128, False),
            128 : ( 32,  64, False)
        }

        resolution_blocks = []
        rgb_layers = []
        for resl in self.resl2param:
            param = self.resl2param[resl]
            resolution_blocks.append(
                DownResolutionBlock(
                    resl=resl,
                    in_channels=param[0],
                    out_channels=param[1],
                    is_last=param[2]
                )
            )
            rgb_layers.append(FromRGB(out_channels=param[0]))

        self.resolution_blocks = nn.ModuleList(resolution_blocks)
        self.rgb_layers        = nn.ModuleList(rgb_layers)

        self.alpha = 0

    def grow(self):
        self.train_depth += 1
        self.alpha = 0

    def forward(self, x, phase):
        if phase == 't':
            return self.transition_forward(x)
        else:
            return self.stablization_forward(x)

    def transition_forward(self, x):
        size = x.size(2)

        x_down = F.avg_pool2d(x, 2)
        x_pre = self.rgb_layers[self.train_depth-1](x_down)

        x = self.rgb_layers[self.train_depth](x)
        x_cur = self.resolution_blocks[self.train_depth](x)

        x = (1 - self.alpha) * x_pre + self.alpha * x_cur
        for block in self.resolution_blocks[self.train_depth-1::-1]:
            x = block(x)

        return x.view(x.size(0), -1)

    def stablization_forward(self, x):

        x = self.rgb_layers[self.train_depth](x)
        for block in self.resolution_blocks[self.train_depth::-1]:
            x = block(x)

        return x.view(x.size(0), -1)

    def update_alpha(self, delta, phase):
        if phase == 't':
            self.alpha = min(1, self.alpha+delta)

    def update_noise(self, magnitude):
        for block in self.resolution_blocks:
            for m in block.modules():
                if isinstance(m, GaussianNoise):
                    m.magnitude = magnitude

if __name__ == "__main__":
    G = Generator(100).cuda()
    D = Discriminator().cuda()
    for _ in range(5):
        G.grow()
        D.grow()

        z = torch.randn(3, 100).cuda()

        D.update_noise(0.1)

        output = G(z, 't')
        print(output.mean())

        output = D(output, 't')
        print(output.mean())
