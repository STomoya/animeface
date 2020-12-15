
'''
Self re-implementation of StyleGAN2
with missing elements and a more flexable structure

Major change
- proper implementation of mini-batch standard deviation
- pixel normalization on input of mapping network
- coef of ELR with He init.

Minor change
- optional image channel size
- generate models depending on the image size
- adjustable number of convolution layer per reslution 
    - for reducing memory size if needed
- simpler implementation
    - eliminate redundant layers (e.g. Bias)
- init weights using ".apply()" function
'''

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''equalized learning rate'''
class ELR(nn.Module):
    def __init__(self, layer, gain=1.):
        super().__init__()
        self.coef = gain / (layer.weight[0].numel() ** 0.5)
        self.layer = layer
    def forward(self, x):
        # x*w* coef + bias
        x = x * self.coef
        return self.layer(x)

'''
functions
'''

'''linear with elr option'''
def Linear(name, *args, **kwargs):
    linear = nn.Linear(*args, **kwargs)
    if name == 'elr': return ELR(linear)
    return linear

'''conv with elr option'''
def Conv2d(name, *args, **kwargs):
    conv = nn.Conv2d(*args, **kwargs)
    if name == 'elr': return ELR(conv)
    return conv

'''2x scale upsample'''
def Upsample2x(name):
    if name == 'pixelshuffle': return nn.PixelShuffle(2)
    return nn.Upsample(scale_factor=2, mode=name, align_corners=False)

'''0.5x scale downsample'''
def Downsample2x(name):
    if name == 'max': return nn.MaxPool2d(2)
    elif name == 'avg': return nn.AvgPool2d(2)

'''Flatten'''
class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)

'''linear with learning rate'''
class MapLinear(nn.Module):
    def __init__(self, *args, lr=0.01, **kwargs):
        super().__init__()
        self.linear = Linear('elr', *args, **kwargs)
        self.lr = lr
    def forward(self, x):
        # (x*w* coef + bias) * lr
        return self.linear(x) * self.lr

'''noise injection'''
class InjectNoise(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        B, _, H, W = x.size()
        noise = torch.randn(B, 1, H, W, device=x.device)
        return x + noise

'''modulated convolution with affine layer'''
class ModulatedConv2d(nn.Module):
    def __init__(self,
        in_channels, out_channels, style_dim, kernel_size,
        stride=1, demod=True
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.demod = demod
        
        self.affine = Linear('elr', style_dim, in_channels)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(1, out_channels, 1, 1))
    def forward(self, x, y):
        B, _, H, W = x.size()

        # affine transform
        y = self.affine(y) + 1 # init bias with 1
                               # a little bit forcible method but still works
                               # for init weights with .apply()

        # modulate
        weight = self.weight[None, :, :, :, :] * y[:, None, :, None, None]

        # demodulate
        if self.demod:
            d = torch.rsqrt(weight.pow(2).sum([2, 3, 4], keepdim=True) + 1e-4)
            weight = weight * d

        # reshaping for conv input
        x = x.reshape(1, -1, H, W)
        _, _, *ws = weight.size()
        weight = weight.reshape(B*self.out_channels, *ws)
        pad = self._get_same_padding(H)

        # conv
        x = F.conv2d(x, weight, padding=pad, groups=B)

        # return with bias
        return x.reshape(B, self.out_channels, H, W) + self.bias
    
    def _get_same_padding(self, size):
        return ((size - 1) * (self.stride - 1) + (self.kernel_size - 1)) // 2

'''Gaussian Blur'''
class Blur2d(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = torch.tensor([[[1., 2., 1.],
                                [2., 4., 2.],
                                [1., 2., 1.]]])
        kernel /= kernel.sum()
        self.register_buffer('kernel', kernel)
    def forward(self, x):
        C = x.size(1)
        x = F.conv2d(x, self.kernel.expand(C, -1, -1, -1), padding=1, groups=C)
        return x

'''Generator Block ver. Skip
upsample -> conv*num_conv
'''
class StyleBlock(nn.Module):
    def __init__(self,
        in_channels, out_channels, style_dim, num_conv=2, up_name='bilinear'
    ):
        super().__init__()
        self.block = nn.ModuleList([
            Upsample2x(up_name),
            Blur2d(),
            ModulatedConv2d(in_channels, out_channels, style_dim, 3),
            InjectNoise(),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        for _ in range(num_conv-1):
            self.block.extend([
                ModulatedConv2d(out_channels, out_channels, style_dim, 3),
                InjectNoise(),
                nn.LeakyReLU(0.2, inplace=True)
            ])
    def forward(self, x, y):

        for module in self.block:
            if isinstance(module, ModulatedConv2d):
                x = module(x, y)
            else:
                x = module(x)
        
        return x

'''Discriminator Block ver. Residual
-- conv*num_conv -> downsample -> add ->
 |------- conv -> downsample ------|
'''
class DBlock(nn.Module):
    def __init__(self,
        in_channels, out_channels, num_conv=2, down_name='avg'
    ):
        super().__init__()
        layers = [
            Conv2d('elr', in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        for _ in range(num_conv-1):
            layers.extend([
                Conv2d('elr', out_channels, out_channels, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        self.block = nn.Sequential(*layers)
        self.down = Downsample2x(down_name)
        self.skip = Conv2d('elr', in_channels, out_channels, 1)

    def forward(self, x):
        t = x
        
        x = self.block(x)
        t = self.skip(t)

        x = self.down(x)
        t = self.down(t)
        return (x + t) / np.sqrt(2)

'''Mini Batch Standard Deviation'''
class MiniBatchStdDev(nn.Module):
    def __init__(self, group_size, eps=1e-4):
        super().__init__()
        self.group_size = group_size
        self.eps = eps

    def forward(self, x):
        B, C, H, W = x.size()
        y = x
        groups = self.__check_group_size(B)
        y   = y.view(groups, -1, C, H, W).contiguous() # [GMCHW] split to groups
        y   = y - y.mean(0, keepdim=True)              # [GMCHW] sub mean
        y   = y.square().mean(0)                       # [MCHW]  varience
        y   = y.add_(self.eps).sqrt()                  # [MCHW]  stddev
        y   = y.mean([1, 2, 3], keepdim=True)          # [M111]  mean over fmaps and pixels
        y   = y.repeat(groups, 1, H, W)                # [NCHW]  replicate over group and pixels
        out = torch.cat([x, y], dim=1)
        return out

    def __check_group_size(self, batch_size):
        if self.group_size % batch_size == 0: return self.group_size
        else:                                 return batch_size

'''to RGB'''
class ToImage(nn.Module):
    def __init__(self, in_channels, image_channels, style_dim, upsample=True, up_name='bilinear'):
        super().__init__()
        self.conv = ModulatedConv2d(in_channels, image_channels, style_dim, 1, demod=False)
        self.upsample = Upsample2x(up_name) if upsample else None
    def forward(self, x, y, pre=None):
        x = self.conv(x, y)
        if not pre == None:
            x = x + pre
        if self.upsample is not None:
            x = self.upsample(x)
        return x

'''Pixel Normalization'''
class PixelNorm(nn.Module):
    def forward(self, x):
        x = x / x.pow(2).mean(dim=1, keepdim=True).sqrt().add_(1e-4)
        return x

'''
Networks
'''

'''Mapping Network'''
class Mapping(nn.Module):
    def __init__(self, style_dim, num_layers=8, lr=0.01):
        super().__init__()

        self.normalize = PixelNorm()

        layers = []
        for _ in range(num_layers):
            layers.append(
                MapLinear(style_dim, style_dim, lr=lr)
            )
        self.map = nn.Sequential(*layers)

    def forward(self, x):
        x = self.normalize(x)
        return self.map(x)

'''Synthesis Network ver. Skip'''
class Synthesis(nn.Module):
    def __init__(self, image_size, image_channels, style_dim, channels=32, max_channels=512, num_conv=2):
        super().__init__()
        check_c = functools.partial(min, max_channels)
        resl = 4

        channels = channels * (2 ** int(np.log2(image_size) - 2))
        ochannels = check_c(channels)
        self.input = ModulatedConv2d(style_dim, ochannels, style_dim, 3)
        self.input_to_image = ToImage(ochannels, image_channels, style_dim)
        self.num_layers = 1

        self.blocks = nn.ModuleList()
        self.to_images = nn.ModuleList()
        while resl < image_size:
            resl *= 2
            channels = channels // 2
            ichannels, ochannels = ochannels, check_c(channels)
            self.blocks.append(
                StyleBlock(ichannels, ochannels, style_dim, num_conv)
            )
            self.to_images.append(
                ToImage(ochannels, image_channels, style_dim, upsample=True if resl < image_size else False)
            )
            self.num_layers += 1
        self.tanh = nn.Tanh()
    
    def forward(self, x, y, injection=None):

        # for style mixing
        if isinstance(y, (list, tuple)):
            assert len(y) == 2
            if injection is None or injection > self.num_layers:
                injection = np.random.randint(0, self.num_layers)
            y = [y[0] for _ in range(injection)] \
                + [y[1] for _ in range(self.num_layers - injection)]
        else:
            y = [y for _ in range(self.num_layers)]

        x = self.input(x, y[0])
        pre = self.input_to_image(x, y[0])

        for block, to_image, y in zip(self.blocks, self.to_images, y[1:]):
            x = block(x, y)
            image = to_image(x, y, pre)
            pre = image
        
        return self.tanh(image)

'''Generator ver. Skip'''
class Generator(nn.Module):
    def __init__(self,
        image_size=128, image_channels=3, style_dim=512, channels=32, max_channels=512, block_num_conv=2, map_num_layers=8, map_lr=0.01
    ):
        super().__init__()

        self.map = Mapping(style_dim, map_num_layers, map_lr)
        self.synthesis = Synthesis(
            image_size, image_channels, style_dim,
            channels, max_channels, block_num_conv
        )
        self.const = nn.Parameter(torch.empty(1, style_dim, 4, 4))
        self.const.data.normal_(0, 1)
    
    def forward(self, z, injection=None):

        # for style mixing
        if isinstance(z, (list, tuple)):
            style = [self.map(z[0]), self.map(z[1])]
            B = z[0].size(0)
        else:
            style = self.map(z)
            B = z.size(0)
        
        input = self.const.expand(B, -1, -1, -1)
        image = self.synthesis(input, style, injection)
        return image

    def init_weight(self, map_init_func, syn_init_func):
        self.map.apply(map_init_func)
        self.synthesis.apply(syn_init_func)

'''Discriminator ver. Residual'''
class Discriminator(nn.Module):
    def __init__(self, image_size=128, image_channels=3, channels=32, max_channels=512, block_num_conv=2, mbsd_groups=4):
        super().__init__()
        check_c = functools.partial(min, max_channels)
        ochannels = channels
        self.from_rgb = nn.Sequential(
            Conv2d('elr', image_channels, ochannels, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        resl = image_size
        blocks = []
        while resl > 4:
            resl = resl // 2
            channels *= 2
            ichannels, ochannels = ochannels, check_c(channels)
            blocks.append(
                DBlock(ichannels, ochannels, block_num_conv)
            )
        blocks.append(MiniBatchStdDev(mbsd_groups))
        blocks.extend([
            Conv2d('elr', ochannels+1, ochannels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            Flatten(),
            Linear('elr', ochannels*(resl**2), ochannels),
            nn.LeakyReLU(0.2, inplace=True),
            Linear('elr', ochannels, 1)
        ])
        self.blocks = nn.Sequential(*blocks)
    def forward(self, x):
        x = self.from_rgb(x)
        x = self.blocks(x)
        return x

'''init weights function'''
def init_weight_N01(m, lr=1):
    '''init weight with N(0, 1/lr)'''
    if isinstance(m, (nn.Linear, nn.Conv2d, ModulatedConv2d)):
        m.weight.data.normal_(0., 1/lr)
        m.bias.data.fill_(0.)

if __name__ == "__main__":
    z = torch.randn(32, 512)

    g = Generator(64, 1, 512)
    d = Discriminator(64, 1)
    g.init_weight(
        functools.partial(init_weight_N01, lr=0.01),
        init_weight_N01
    )
    g.apply(init_weight_N01)
    image = g((z, z))
    prob = d(image)
    print(image.size(), prob.size())