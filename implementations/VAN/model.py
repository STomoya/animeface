
from collections import OrderedDict
import math
import torch
import torch.nn as nn

def get_activation(name):
    if name == 'relu': return nn.ReLU(True)
    if name == 'lrelu': return nn.LeakyReLU(0.2, True)
    if name == 'swish': return nn.SiLU()
    if name == 'gelu': return nn.GELU()
    raise Exception(f'Activation: {name}')

def get_normalization(name, channels):
    if name == 'bn': return nn.BatchNorm2d(channels)
    if name == 'in': return nn.InstanceNorm2d(channels)
    if name == 'ln': return nn.GroupNorm(1, channels)
    if name == 'gn': return nn.GroupNorm(16, channels)
    raise Exception(f'Normalization: {name}')

_default = lambda value, default_value: value if value is not None else default_value

class AttentionModule(nn.Module):
    def __init__(self,
        channels
    ) -> None:
        super().__init__()
        self.dwconv  = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.dwdconv = nn.Conv2d(channels, channels, 7, padding=9, dilation=3, groups=channels)
        self.pwconv  = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        attn = self.dwconv(x)
        attn = self.dwdconv(attn)
        attn = self.pwconv(attn)
        return x * attn

class MLP(nn.Sequential):
    def __init__(self,
        channels, expansion=4, act_name='gelu'
    ) -> None:
        super().__init__(OrderedDict([
            ('fc1', nn.Conv2d(channels, channels*expansion, 1)),
            ('act', get_activation(act_name)),
            ('fc2', nn.Conv2d(channels*expansion, channels, 1))]))

class Block(nn.Module):
    def __init__(self,
        channels, expansion=4, norm_name='ln', act_name='gelu', layer_scale_init_value=1e-6
    ) -> None:
        super().__init__()

        self.norm1 = get_normalization(norm_name, channels)
        self.attention = AttentionModule(channels)
        self.norm2 = get_normalization(norm_name, channels)
        self.mlp   = MLP(channels, expansion, act_name)

        self.layer_scale1 = nn.Parameter(torch.ones([]) * layer_scale_init_value)
        self.layer_scale2 = nn.Parameter(torch.ones([]) * layer_scale_init_value)

    def forward(self, x):
        x = x + self.layer_scale1 * self.attention(self.norm1(x))
        x = x + self.layer_scale2 * self.mlp(self.norm2(x))
        return x

class Stage(nn.Sequential):
    def __init__(self,
        in_channels, out_channels, num_blocks=2, norm_name='ln', act_name='gelu', input=False
    ) -> None:

        if input: layers = []
        else: layers = [('norm', get_normalization(norm_name, in_channels))]
        layers.extend([
            ('up',   nn.Upsample(scale_factor=2)),
            ('conv', nn.Conv2d(in_channels, out_channels, 3, 1, 1))])

        for i in range(num_blocks):
            layers.append((f'block{i+1}', Block(out_channels, 4, norm_name, act_name)))

        super().__init__(OrderedDict(layers))

class Generator(nn.Module):
    def __init__(self,
        latent_dim, image_size, bottom=4, channels=64, max_channels=None,
        blocks_per_scale=2, out_channels=3, norm_name='ln', act_name='gelu'
    ) -> None:
        super().__init__()
        num_ups = int(math.log2(image_size)-math.log2(bottom))
        max_channels = _default(max_channels, channels*16)

        channels = channels * 2 ** num_ups
        ochannels = min(max_channels, channels)
        self.input = nn.Sequential(
            nn.Linear(latent_dim, ochannels*bottom**2),
            get_activation(act_name))
        self.input_shape = (ochannels, bottom, bottom)

        stages = []
        for i in range(num_ups):
            is_first = len(stages) == 0
            channels //= 2
            ichannels, ochannels = ochannels, min(max_channels, channels)
            stages.append((f'stage{i+1}', Stage(
                ichannels, ochannels, blocks_per_scale, norm_name,
                act_name, is_first)))
        self.stages = nn.Sequential(OrderedDict(stages))
        self.output = nn.Sequential(OrderedDict([
            ('norm', get_normalization(norm_name, ochannels)),
            ('conv', nn.Conv2d(ochannels, out_channels, 3, 1, 1)),
            ('tanh', nn.Tanh())]))

    def forward(self, x):
        x = self.input(x)
        x = x.reshape(-1, *self.input_shape)
        x = self.stages(x)
        x = self.output(x)
        return x

class PatchEmbed(nn.Sequential):
    def __init__(self,
        in_channels, out_channels, patch_size=3, stride=2, norm_name='ln'
    ) -> None:
        padding = patch_size // 2

        super().__init__(OrderedDict([
            ('proj', nn.Conv2d(in_channels, out_channels, patch_size, stride, padding=padding)),
            ('norm', get_normalization(norm_name, out_channels))
        ]))

class DStage(nn.Sequential):
    def __init__(self,
        in_channels, out_channels, num_blocks, norm_name='ln', act_name='gelu', input=False
    ) -> None:
        if input: layers = [('embed', PatchEmbed(in_channels, out_channels, 7, 4, norm_name))]
        else: layers = [('embed', PatchEmbed(in_channels, out_channels, norm_name=norm_name))]
        for i in range(num_blocks):
            layers.append((f'block{i}', Block(out_channels, 4, norm_name, act_name)))
        super().__init__(OrderedDict(layers))

class Discriminator(nn.Module):
    def __init__(self,
        layers=[3, 3, 9, 3], channels=64, max_channels=None,
        in_channels=3, norm_name='ln', act_name='gelu'
    ) -> None:
        super().__init__()
        stages = []
        ochannels = channels
        max_channels = _default(max_channels, channels*16)
        for i in range(len(layers)):
            channels *= 2
            ichannels, ochannels = ochannels, min(max_channels, channels)
            stages.append((f'stage{i}', DStage(
                ichannels if i!=0 else in_channels, ochannels, layers[i],
                norm_name, act_name, i==0)))
        self.stages = nn.Sequential(OrderedDict(stages))
        self.output = nn.Sequential(OrderedDict([
            ('norm', get_normalization(norm_name, ochannels)),
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', nn.Flatten()),
            ('fc', nn.Linear(ochannels, 1))]))

    def forward(self, x):
        x = self.stages(x)
        x = self.output(x)
        return x
