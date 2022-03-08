
from collections import OrderedDict
import math
import torch
import torch.nn as nn

def _default(value, default_value):
    return value if value is not None else default_value

def get_normalization(name, channels):
    if   name == 'bn': return nn.BatchNorm2d(channels)
    elif name == 'in': return nn.InstanceNorm2d(channels)
    elif name == 'ln': return nn.GroupNorm(1, channels)
    elif name == 'gn': return nn.GroupNorm(32, channels)

def get_activation(name):
    if   name == 'relu': return nn.ReLU(True)
    elif name == 'lrelu': return nn.LeakyReLU(0.2, True)
    elif name == 'gelu': return nn.GELU()
    elif name == 'swish': return nn.SiLU()

class ConvNeXtBlock(nn.Module):
    '''ConvNeXt block'''
    def __init__(self,
        channels, expansion=4, norm_name='ln', act_name='gelu'
    ) -> None:
        super().__init__()

        self.dwconv  = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm    = get_normalization(norm_name, channels)
        self.pwconv1 = nn.Conv2d(channels, channels*expansion, 1)
        self.act     = get_activation(act_name)
        self.pwconv2 = nn.Conv2d(channels*expansion, channels, 1)
        self.gamma   = nn.Parameter(torch.zeros([]))

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return residual + x * self.gamma

class ConvNextBlockFlex(nn.Module):
    '''
    Flexible implementation of a resnext block with ability to
    reproduce structures studied in the ConvNeXt paper.
    NOTE: Only changes within the block.
    '''
    def __init__(self,
        channels, ratio=4, norm_name='bn', act_name='relu',
        invert=False, input_dconv=False, large_kernel=False, fewer_act=False, fewer_norm=False
    ) -> None:
        super().__init__()
        if invert: mid_channels = channels * ratio
        else: mid_channels = channels // ratio

        if large_kernel: kernel_size = 7
        else: kernel_size = 3
        padding = kernel_size // 2

        self.fewer_act = fewer_act
        self.act = get_activation(act_name)

        if input_dconv:
            self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels)
            self.conv2 = nn.Conv2d(channels, mid_channels, 1)
        else:
            self.conv1 = nn.Conv2d(channels, mid_channels, 1)
            self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=padding, groups=mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, channels, 1)

        self.norm1 = get_normalization(norm_name, channels if input_dconv else mid_channels)
        if not fewer_norm:
            self.norm2 = get_normalization(norm_name, mid_channels)
            self.norm3 = get_normalization(norm_name, channels)

        self.gamma = nn.Parameter(torch.zeros([]))

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        if not self.fewer_act:
            x = self.act(x)
        x = self.conv2(x)
        if hasattr(self, 'norm2'):
            x = self.norm2(x)
        x = self.act(x)
        x = self.conv3(x)
        if hasattr(self, 'norm3'):
            x = self.norm3(x)
        x = residual + x * self.gamma
        if not self.fewer_act:
            x = self.act(x)
        return x

BLOCKS = {
    # resnext                   (Sec. 2.3)
    'resnext':  dict(norm_name='bn', act_name='relu', invert=False, input_dconv=False, large_kernel=False, fewer_act=False, fewer_norm=False),
    # inverted bottleneck block (Sec. 2.4)
    'invert':   dict(norm_name='bn', act_name='relu', invert=True, input_dconv=False, large_kernel=False, fewer_act=False, fewer_norm=False),
    # large kernel sizes        (Sec. 2.5)
    'kernel':   dict(norm_name='bn', act_name='gelu', invert=True, input_dconv=True, large_kernel=True, fewer_act=False, fewer_norm=False),
    # ReLU to GELU              (Sec. 2.6 - Replacing ReLU with GELU)
    'gelu':     dict(norm_name='bn', act_name='gelu', invert=True, input_dconv=True, large_kernel=True, fewer_act=False, fewer_norm=False),
    # fewer activation          (Sec. 2.6 - Fewer activation functions)
    'fewact':   dict(norm_name='bn', act_name='gelu', invert=True, input_dconv=True, large_kernel=True, fewer_act=True, fewer_norm=False),
    # fewer normlization        (Sec. 2.6 - Fewer normalization layers)
    'fewnorm':  dict(norm_name='bn', act_name='gelu', invert=True, input_dconv=True, large_kernel=True, fewer_act=True, fewer_norm=True),
    # convnext block            (Sec. 2.6 - Substituting BN with LN)
    'convnext': dict(norm_name='ln', act_name='gelu', invert=True, input_dconv=True, large_kernel=True, fewer_act=True, fewer_norm=True),
    # (Experimental) convnext small kernel size
    'smallkernel': dict(norm_name='ln', act_name='gelu', invert=True, input_dconv=True, large_kernel=False, fewer_act=True, fewer_norm=True)
}


class Stage(nn.Sequential):
    def __init__(self,
        in_channels, out_channels, num_blocks=2, block_kwargs=BLOCKS['convnext']
    ) -> None:

        layers = [
            ('norm', get_normalization(block_kwargs['norm_name'], in_channels)),
            ('up', nn.Upsample(scale_factor=2)),
            ('conv', nn.Conv2d(in_channels, out_channels, 3, 1, 1))]
        for i in range(num_blocks):
            layers.append((f'block{i}', ConvNextBlockFlex(out_channels, 4, **block_kwargs)))
        super().__init__(OrderedDict(layers))

class Generator(nn.Module):
    def __init__(self,
        latent_dim, image_size=128, bottom=4, channels=96, max_channels=None,
        block_type='convnext', blocks_per_scale=2, out_channels=3
    ) -> None:
        super().__init__()
        num_ups = int(math.log2(image_size)-math.log2(bottom))
        max_channels = _default(max_channels, channels * 16)
        channels = channels * 2 ** num_ups
        ochannels = min(max_channels, channels)

        block_kwargs = BLOCKS[block_type]

        self.input = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(latent_dim, ochannels*bottom**2)),
            ('act', get_activation(block_kwargs['act_name']))]))
        self.input_shape = (ochannels, bottom, bottom)

        stages = []
        for i in range(num_ups):
            channels //= 2
            ichannels, ochannels = ochannels, min(max_channels, channels)
            stages.append((f'stage{i}', Stage(
                ichannels, ochannels, blocks_per_scale, block_kwargs)))
        self.stages = nn.Sequential(OrderedDict(stages))

        self.output = nn.Sequential(OrderedDict([
            ('norm', get_normalization(block_kwargs['norm_name'], ochannels)
                if block_type in ['convnext', 'fewnorm'] else nn.Identity()),
            ('conv', nn.Conv2d(ochannels, out_channels, 3, 1, 1)),
            ('tanh', nn.Tanh())]))

    def forward(self, x):
        x = self.input(x)
        x = x.reshape(-1, *self.input_shape)
        x = self.stages(x)
        x = self.output(x)
        return x

class Discriminator(nn.Module):
    '''ConvNeXt'''
    def __init__(self,
            in_chans=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]
        ):
        super().__init__()

        layers = []
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            nn.GroupNorm(1, dims[0], eps=1e-6)
        )
        cur = 0
        for i in range(4):
            if i == 0: downsample_layer = stem
            else:
                downsample_layer = nn.Sequential(
                    nn.GroupNorm(1, dims[i-1], eps=1e-6),
                    nn.Conv2d(dims[i-1], dims[i], kernel_size=2, stride=2),
                )
            stage = nn.Sequential(
                *[ConvNeXtBlock(dims[i]) for _ in range(depths[i])]
            )
            cur += depths[i]
            layers.append((f'layer{i+1}', nn.Sequential(OrderedDict([
                ('down', downsample_layer), ('stage', stage)
            ]))))
        layers.extend([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', nn.Flatten(1)),
            ('norm',    nn.LayerNorm(dims[-1], eps=1e-6))])
        self.feature_blocks = nn.Sequential(OrderedDict(layers))
        self.head = nn.Linear(dims[-1], 1)

    def forward(self, x):
        x = self.feature_blocks(x)
        x = self.head(x)
        return x
