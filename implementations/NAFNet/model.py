
import math
import warnings
import torch
import torch.nn as nn

def get_normalization(name, channels, **kwargs):
    if name == 'ln': return nn.GroupNorm(1, channels, **kwargs)
    if name == 'bn': return nn.BatchNorm2d(channels, **kwargs)
    if name == 'in': return nn.InstanceNorm2d(channels, **kwargs)
    raise Exception(f'Normalization: {name}')

def get_activation(name):
    if name == 'sigmoid': return nn.Sigmoid()
    if name == 'relu':    return nn.ReLU()
    if name == 'lrelu':   return nn.LeakyReLU(0.2, True)
    if name == 'tanh':    return nn.Tanh()
    raise Exception(f'Activation: {name}')

class SimpleGate(nn.Module):
    def __init__(self, act_name=None) -> None:
        super().__init__()
        self.act = nn.Identity() if act_name is None else get_activation(act_name)
    def forward(self, x):
        x, gate = x.chunk(2, dim=1)
        return x * self.act(gate)


class MLP(nn.Module):
    def __init__(self,
        channels, mlp_ratio, act_name=None
    ) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels*mlp_ratio, 1)
        self.act = SimpleGate(act_name)
        self.fc2 = nn.Conv2d(channels*mlp_ratio//2, channels, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class SCA(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.global_content = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1))

    def forward(self, x):
        gc = self.global_content(x)
        return x * gc


class ConvBlock(nn.Module):
    def __init__(self,
        channels, act_name=None
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels*2, 1)
        self.conv2 = nn.Conv2d(channels*2, channels*2, 3, 1, 1, groups=channels*2)
        self.act = SimpleGate(act_name)
        self.sca   = SCA(channels)
        self.conv3 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.sca(x)
        x = self.conv3(x)
        return x


class NAFBlock(nn.Module):
    def __init__(self,
        channels, mlp_ratio=1, norm_name='ln', act_name=None
    ) -> None:
        super().__init__()

        self.norm1 = get_normalization(norm_name, channels)
        self.conv  = ConvBlock(channels, act_name)
        self.norm2 = get_normalization(norm_name, channels)
        self.mlp   = MLP(channels, mlp_ratio, act_name)

        self.layer_scale1 = nn.Parameter(torch.ones([])*1e-3)
        self.layer_scale2 = nn.Parameter(torch.ones([])*1e-3)

    def forward(self, x):
        x = x + self.layer_scale1 * self.conv(self.norm1(x))
        x = x + self.layer_scale2 * self.conv(self.norm2(x))
        return x


class Upsample(nn.Sequential):
    def __init__(self,
        in_channels, out_channels
    ) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels*4, 3, 1, 1),
            nn.PixelShuffle(2))
class Downsample(nn.Sequential):
    def __init__(self,
        in_channels, out_channels
    ) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels//4, 3, 1, 1),
            nn.PixelUnshuffle(2))


class NAFNet(nn.Module):
    def __init__(self,
        image_size, bottom, channels=32, max_channels=512, blocks_per_scale=2, mid_blocks=2,
        mlp_ratio=1, norm_name='ln', act_name=None, io_channels=3
    ) -> None:
        super().__init__()
        if act_name is not None:
            warnings.warn(f'Adding non-linearity: {act_name}')
        if not isinstance(io_channels, tuple):
            io_channels = (io_channels, io_channels)

        num_sampling = int(math.log2(image_size) - math.log2(bottom))
        ochannels = channels

        self.input = nn.Conv2d(io_channels[0], ochannels, 3, 1, 1)
        self.downs = nn.ModuleList()
        for _ in range(num_sampling):
            channels *= 2
            ichannels, ochannels = ochannels, min(channels, max_channels)
            self.downs.append(nn.ModuleList([
                nn.Sequential(*[
                    NAFBlock(ichannels, mlp_ratio, norm_name, act_name)
                    for _ in range(blocks_per_scale)]),
                Downsample(ichannels, ochannels)]))

        self.middle = nn.Sequential(*[
            NAFBlock(ochannels, mlp_ratio, norm_name, act_name)
            for _ in range(mid_blocks)])

        self.ups = nn.ModuleList()
        for _ in range(num_sampling):
            channels //= 2
            ichannels, ochannels = ochannels, min(max_channels, channels)
            self.ups.append(nn.ModuleList([
                Upsample(ichannels, ochannels),
                nn.Sequential(*[
                    NAFBlock(ochannels, mlp_ratio, norm_name, act_name)
                    for _ in range(blocks_per_scale)])
            ]))

        self.output = nn.Conv2d(ochannels, io_channels[1], 3, 1, 1)

    def forward(self, x):
        x = self.input(x)
        feats = []
        for encode, down in self.downs:
            x = encode(x)
            feats.append(x)
            x = down(x)
        x = self.middle(x)
        for (up, decode), dfeat in zip(self.ups, feats[::-1]):
            x = up(x)
            x = x + dfeat
            x = decode(x)
        x = self.output(x)
        return x


class Discriminator(nn.Sequential):
    def __init__(self,
        num_layers=3, channels=64, max_channels=512, norm_name='bn', act_name='lrelu',
        in_channels=3
    ) -> None:

        ochannels = channels
        layers = [
            nn.Conv2d(in_channels, ochannels, 4, 2, 1),
            get_activation(act_name)]
        for _ in range(num_layers-1):
            channels *= 2
            ichannels, ochannels = ochannels, min(max_channels, channels)
            layers.extend([
                nn.Conv2d(ichannels, ochannels, 4, 2, 1),
                get_normalization(norm_name, ochannels),
                get_activation(act_name)])
        layers.extend([
            nn.Conv2d(ochannels, ochannels*2, 4, 1, 1),
            get_normalization(norm_name, ochannels*2),
            get_activation(act_name),
            nn.Conv2d(ochannels*2, 1, 4, 1, 1)])
        super().__init__(*layers)
