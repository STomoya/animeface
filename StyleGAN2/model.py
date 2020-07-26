
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


'''
layers
'''

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

class EqualizedLinear(nn.Module):
    def __init__(self,
        in_features, out_features, gain=2, lr=1.
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.weight.data.normal_(0, 1/lr)
        
        self.scale = ((gain / self.linear.weight[0].numel()) ** 0.5) * lr
    def forward(self, x):
        x = x * self.scale
        x = self.linear(x)
        return x

class EqualizedConv2d(nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel_size, gain=2, **kwargs
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, **kwargs
        )
        self.conv.weight.data.normal_(0, 1)

        self.scale = (gain / self.conv.weight[0].numel()) ** 0.5
    def forward(self, x):
        x = self.scale * x
        x = self.conv(x)
        return x

class EqualizedModulatedConv2d(nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel_size, style_dim, stride=1, padding=0,
        demod=True, gain=2,
    ):
        super().__init__()
        self.demod = demod
        self.stride, self.padding = stride, padding

        self.elr_scale = (gain / (in_channels*kernel_size**2)) ** 0.5

        weight_size = [out_channels, in_channels, kernel_size, kernel_size]
        self.weight = nn.Parameter(torch.rand(*weight_size))
        self.weight.data.normal_(0, 1)

        self.style_fc = nn.Sequential(
            EqualizedLinear(style_dim, in_channels),
            Bias(in_channels, bias_init=1.)
        )

    def forward(self, x, style):
        B, C, H, W = x.size()
        oC, iC, kH, kW = self.weight.size()

        affined_style = self.style_fc(style)
        weight = self.elr_scale * self.weight.view(1, oC, iC, kH, kW) * affined_style.view(B, 1, iC, 1, 1)
    
        if self.demod:
            norm = 1 / ((weight**2).sum([2, 3, 4]) + 1.e-8)**0.5
            weight = weight * norm.view(B, oC, 1, 1, 1)

        out = F.conv2d(
            x.contiguous().view(1, B*iC, H, W), weight.view(B*oC, iC, kH, kW),
            stride=self.stride, padding=self.padding, groups=B
        )

        _, _, H, W = out.size()
        out = out.view(B, -1, H, W)
        return out

class Bias(nn.Module):
    def __init__(self, out_channels, bias_init=0, lr=1.):
        super().__init__()
        self.bias = nn.Parameter(torch.rand(out_channels))
        self.bias.data.fill_(bias_init)
        self.lr = lr
    def forward(self, x):
        rest_dim = [1] * (x.ndim - self.bias.ndim - 1)
        x = x + self.bias.view(1, self.bias.size(0), *rest_dim) * self.lr
        return x

class FusedLeakyReLU(nn.Module):
    def __init__(self, out_channels, bias_init=0, bias_lr=1.0, scale=True):
        super().__init__()
        self.bias = Bias(out_channels, bias_init, bias_lr)
        self.scale = 2**0.5 if scale else 1.
    def forward(self, x):
        x = F.leaky_relu(self.bias(x), negative_slope=0.2) * self.scale
        return x

class ScaleNoise(nn.Module):
    def __init__(self):
        super().__init__()
        self.scaler = nn.Conv2d(1, 1, 1, bias=False)
        self.scaler.weight.data.fill_(0)
    def forward(self, x):
        x = self.scaler(x)
        return x


class MiniBatchStd(nn.Module):
    '''
    minibatch standard deviation
    '''
    def forward(self, x):
        std = torch.std(x).expand(x.shape[0], 1, *x.shape[2:])
        return torch.cat([x, std], dim=1)

class Blur2d(nn.Module):
    '''
    blur2d
    '''
    def __init__(self, stride=1):
        super().__init__()
        self.stride = stride
        kernel = torch.tensor([[[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]]], dtype=torch.float32)
        kernel /= kernel.sum()
        self.register_buffer('kernel', kernel)
    def forward(self, x):
        C = x.size(1)
        x = F.conv2d(x, self.kernel.expand(C, -1, -1, -1), stride=self.stride, padding=1, groups=C)
        return x

class UpsampleBlur(nn.Module):
    '''
    upsample -> blur
    '''
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.blur = Blur2d()
    def forward(self, x):
        x = self.upsample(x)
        x = self.blur(x)
        return x

class BlurDownsample(nn.Module):
    '''
    blur -> downsample
    '''
    def __init__(self):
        super().__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2) # equivalent to bilinear downsample
        self.blur = Blur2d()
    def forward(self, x):
        x = self.blur(x)
        x = self.downsample(x)
        return x

class ToRGB(nn.Module):
    def __init__(self,
        in_channels, style_dim
    ):
        super().__init__()
        self.to_rgb = EqualizedModulatedConv2d(
            in_channels, 3, 1, style_dim, demod=False
        )
        self.upsample = UpsampleBlur()
    def forward(self, x, style, summed=None):
        x = self.to_rgb(x, style)
        if not summed == None:
            summed = self.upsample(summed)
            x + summed
        return x

'''
Block
'''

class StyleBlock(nn.Module):
    '''
    (upsample) -> modconv -> add noise -> bias -> lrelu
    '''
    def __init__(self,
        in_channels, out_channels, kernel_size, style_dim, stride=1, padding=0, upsample=True
    ):
        super().__init__()

        self.upsample = UpsampleBlur() if upsample else upsample
        self.conv = EqualizedModulatedConv2d(
            in_channels, out_channels, kernel_size, style_dim,
            stride=stride, padding=padding, demod=True
        )
        self.scale_noise = ScaleNoise()
        self.activation = FusedLeakyReLU(out_channels)

    def forward(self, x, style, noise=None):
        if self.upsample:
            x = self.upsample(x)
        
        x = self.conv(x, style)

        if not noise:
            B, _, H, W = x.size()
            noise = torch.randn(B, 1, H, W, device=x.device)
        
        noise = self.scale_noise(noise)
        x = x + noise

        x = self.activation(x)

        return x

class Mapping(nn.Module):
    def __init__(self,
        n_layers, style_dim, lr=0.01
    ):
        super().__init__()

        layers = []
        for _ in range(n_layers):
            layers += [
                EqualizedLinear(style_dim, style_dim, lr=lr),
                FusedLeakyReLU(style_dim, bias_lr=lr)
            ]
        self.map = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.map(x)
        return x

class ResBlock(nn.Module):
    def __init__(self,
        in_channels, out_channels, 
    ):
        super().__init__()

        self.conv = nn.Sequential(
            EqualizedConv2d(
                in_channels, out_channels, 3, padding=1, bias=False
            ),
            FusedLeakyReLU(out_channels),
            EqualizedConv2d(
                out_channels, out_channels, 3, padding=1, bias=False
            ),
            FusedLeakyReLU(out_channels),
            BlurDownsample()
        )
        self.down = nn.Sequential(
            EqualizedConv2d(
                in_channels, out_channels, 1, bias=False
            ),
            BlurDownsample()
        )
    
    def forward(self, x):
        out = self.conv(x)
        skip = self.down(x)

        out = (out + skip) / 2**0.5
        return out

'''
Generator
'''

class Generator(nn.Module):
    def __init__(self,
        style_n_layers=8, style_dim=512, style_lr=0.01
    ):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, style_dim, 4, 4))
        self.style_map = Mapping(n_layers=style_n_layers, style_dim=style_dim, lr=style_lr)
        
        resl2ch = {
            4   : 512,
            8   : 512,
            16  : 256,
            32  : 128,
            64  :  64,
            128 :  32,
        }
        resl = 4

        self.first_conv = StyleBlock(
            style_dim, resl2ch[resl], 3, style_dim, padding=1, upsample=False
        )
        self.first_rgb = ToRGB(resl2ch[resl], style_dim)

        self.n_layers = 2

        self.conv0s = nn.ModuleList()
        self.conv1s = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        for resl in resl2ch:
            self.conv0s.append(
                StyleBlock(resl2ch[resl], resl2ch[resl*2], 3, style_dim, padding=1)
            )
            self.conv1s.append(
                StyleBlock(resl2ch[resl*2], resl2ch[resl*2], 3, style_dim, padding=1, upsample=False)
            )
            self.to_rgbs.append(
                ToRGB(resl2ch[resl*2], style_dim)
            )
            self.n_layers += 3
            if resl == list(resl2ch.keys())[-2]:
                break
        
    def forward(self, x, injection=None):

        # style mixing
        if isinstance(x, list) or isinstance(x, tuple):
            x0, x1 = x
            style0, style1 = self.style_map(x0), self.style_map(x1)
            if not injection:
                injection = random.randint(1, self.n_layers-1)
            styles = [style0 if i < injection else style1 for i in range(self.n_layers)]
        else:
            style = self.style_map(x)
            styles = [style for _ in range(self.n_layers)]

        # first layers
        x = self.first_conv(self.input.expand(styles[0].size(0), -1, -1, -1), styles[0])
        img = self.first_rgb(x, styles[1])

        # other layers
        i = 2
        for conv0, conv1, to_rgb in zip(self.conv0s, self.conv1s, self.to_rgbs):
            x = conv0(x, styles[i])
            x = conv1(x, styles[i+1])
            img = to_rgb(x, styles[i+2], img)

            i+=3

        return img


'''
Discriminator
'''

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        resl2ch = {
            4   : 512,
            8   : 512,
            16  : 256,
            32  : 128,
            64  :  64,
            128 :  32,
        }

        self.from_rgb = nn.Sequential(
            EqualizedConv2d(3, resl2ch[128], 1, bias=False),
            FusedLeakyReLU(resl2ch[128])
        )

        layers = []
        for resl in list(resl2ch.keys())[::-1]:
            layers += [
                ResBlock(resl2ch[resl], resl2ch[resl//2])
            ]
            if resl == list(resl2ch.keys())[1]:
                break
        
        layers.append(MiniBatchStd())
        layers += [
            EqualizedConv2d(resl2ch[4]+1, resl2ch[4], 3, padding=1, bias=False),
            FusedLeakyReLU(resl2ch[4]),
            Flatten(),
            EqualizedLinear(resl2ch[4]*4**2, resl2ch[4]),
            FusedLeakyReLU(resl2ch[4]),
            EqualizedLinear(resl2ch[4], 1)
        ]

        self.dis = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.from_rgb(x)
        x = self.dis(x)
        return x


if __name__ == "__main__":
    g = Generator()
    d = Discriminator()

    z = [torch.randn(2, 512)] * 2

    print(d(g(z)).size())