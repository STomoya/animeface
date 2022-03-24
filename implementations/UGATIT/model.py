
import math
import torch
import torch.nn as nn


def get_activation(name):
    if name == 'relu': return nn.ReLU(True)
    elif name == 'lrelu': return nn.LeakyReLU(0.2, True)
    elif name == 'gelu':  return nn.GELU()
    elif name in ['swish', 'silu']: return nn.SiLU()
    raise Exception(f'Activation: {name}')


def get_normalization(name, channels, **kwargs):
    if name == 'bn': return nn.BatchNorm2d(channels, **kwargs)
    elif name == 'in': return nn.InstanceNorm2d(channels, **kwargs)
    elif name == 'ln': return nn.GroupNorm(1, channels, **kwargs)
    elif name == 'gn': return nn.GroupNorm(16, channels, **kwargs)
    raise Exception(f'Normalization: {name}')


# def SNConv2d(*args, **kwargs):
#     return nn.utils.spectral_norm(nn.Conv2d(*args, **kwargs))
def SNConv2d(*args, **kwargs):
    return nn.Conv2d(*args, **kwargs)


class CAM(nn.Module):
    def __init__(self, channels, act_name='relu', sn=False) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool_fc = nn.Linear(channels, 1, bias=False)
        self.maxpool_fc = nn.Linear(channels, 1, bias=False)
        conv_cls = SNConv2d if sn else nn.Conv2d
        self.conv = conv_cls(channels*2, channels, 1, bias=False)
        self.act  = get_activation(act_name)

    def forward(self, x):
        gap = self.avgpool(x)
        gap_logit = self.avgpool_fc(gap.flatten(1))
        gap_weight = self.avgpool_fc.weight.clone()
        gap = x * gap_weight[:, :, None, None]

        gmp = self.maxpool(x)
        gmp_logit = self.maxpool_fc(gmp.flatten(1))
        gmp_weight = self.maxpool_fc.weight.detach().clone()
        gmp = x * gmp_weight[:, :, None, None]

        cam_logit = torch.cat([gap_logit, gmp_logit], dim=1)
        x = torch.cat([gap, gmp], dim=1)
        x = self.conv(x)
        x = self.act(x)

        heatmap = torch.sum(x, dim=1, keepdim=True)

        return x, cam_logit, heatmap


class GammaBeta(nn.Module):
    def __init__(self,
        in_features, out_features, act_name='relu'
    ) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features, bias=False),
            get_activation(act_name),
            nn.Linear(out_features, out_features*2, bias=False))

    def forward(self, x):
        x = self.mlp(x)
        gamma, beta = x.chunk(2, dim=1)
        return gamma, beta


class LIN(nn.Module):
    def __init__(self, channels, affine=True) -> None:
        super().__init__()
        self.layer_norm = get_normalization('ln', channels, affine=False)
        self.instance_norm = get_normalization('in', channels, affine=False)
        self.rho   = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.5)
        if affine:
            self.affine = True
            self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
            self.beta  = nn.Parameter(torch.zeros(1, channels, 1, 1))
        else: self.affine = False

    def forward(self, x):
        layer_norm = self.layer_norm(x)
        instance_norm = self.instance_norm(x)
        x = self.rho * instance_norm + (1 - self.rho) * layer_norm
        if self.affine:
            x = self.gamma * x + self.beta
        return x


class AdaLIN(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.lin = LIN(channels, affine=False)
    def forward(self, x, gamma, beta):
        x = self.lin(x)
        x = gamma[:, :, None, None] * x + beta[:, :, None, None]
        return x


class ResBlock(nn.Module):
    def __init__(self,
        channels, norm_name='in', act_name='relu'
    ) -> None:
        super().__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(channels, channels, 3, bias=False)
        self.norm1 = get_normalization(norm_name, channels)
        self.act   = get_activation(act_name)
        self.conv2 = nn.Conv2d(channels, channels, 3, bias=False)
        self.norm2 = get_normalization(norm_name, channels)

    def forward(self, x):
        skip = x
        x = self.conv1(self.pad(x))
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(self.pad(x))
        x = self.norm2(x)
        return skip + x


class AdaLINResBlock(nn.Module):
    def __init__(self,
        channels, act_name='relu'
    ) -> None:
        super().__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(channels, channels, 3, bias=False)
        self.norm1 = AdaLIN(channels)
        self.act   = get_activation(act_name)
        self.conv2 = nn.Conv2d(channels, channels, 3, bias=False)
        self.norm2 = AdaLIN(channels)

    def forward(self, x, gamma, beta):
        skip = x
        x = self.conv1(self.pad(x))
        x = self.norm1(x, gamma, beta)
        x = self.act(x)
        x = self.conv2(self.pad(x))
        x = self.norm2(x, gamma, beta)
        return skip + x


class Generator(nn.Module):
    def __init__(self,
        image_size, bottom=None, channels=64, max_channels=512,
        resblocks=6, adalinresblocks=6, act_name='relu', norm_name='in',
        light=False, io_channels=3
    ) -> None:
        super().__init__()
        bottom = bottom if bottom is not None else image_size // (2**2)
        num_sampling = int(math.log2(image_size)-math.log2(bottom))

        ochannels = channels
        self.input = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(io_channels, ochannels, 3, bias=False),
            get_activation(act_name))

        down = []
        resl = image_size
        for _ in range(num_sampling):
            channels *= 2
            resl //= 2
            ichannels, ochannels = ochannels, min(max_channels, channels)
            down.extend([
                nn.ReflectionPad2d(1),
                nn.Conv2d(ichannels, ochannels, 3, 2, bias=False),
                get_normalization(norm_name, ochannels),
                get_activation(act_name)])
        self.down = nn.Sequential(*down)

        blocks = []
        for _ in range(resblocks):
            blocks.append(ResBlock(ochannels, norm_name, act_name))
        self.resblocks = nn.Sequential(*blocks)

        self.cam  = CAM(ochannels, act_name)

        if not light: infeat = ochannels * resl ** 2
        else: infeat = ochannels
        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d() if light else nn.Identity(),
            nn.Flatten())
        self.gammabeta = GammaBeta(infeat, ochannels, act_name)

        blocks = []
        for _ in range(adalinresblocks):
            blocks.append(AdaLINResBlock(ochannels, act_name))
        self.adalin_resblocks = nn.ModuleList(blocks)

        up = []
        for _ in range(num_sampling):
            channels //= 2
            ichannels, ochannels = ochannels, min(max_channels, channels)
            up.extend([
                nn.Upsample(scale_factor=2),
                nn.ReflectionPad2d(1),
                nn.Conv2d(ichannels, ochannels, 3, bias=False),
                LIN(ochannels),
                get_activation(act_name)])
        self.up = nn.Sequential(*up)

        self.output = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ochannels, io_channels, 3, bias=False),
            nn.Tanh())

    def forward(self, x, return_heatmap=False):
        x = self.input(x)
        x = self.down(x)
        x = self.resblocks(x)
        x, cam_logit, heatmap = self.cam(x)
        gamma, beta = self.gammabeta(self.flatten(x))

        for block in self.adalin_resblocks:
            x = block(x, gamma, beta)
        x = self.up(x)
        x = self.output(x)

        if return_heatmap: # we don't need heatmap when training
            return x, cam_logit, heatmap
        return x, cam_logit


class Discriminator(nn.Module):
    def __init__(self,
        num_layers=3, channels=64, max_channels=512, act_name='lrelu', in_channels=3
    ) -> None:
        super().__init__()

        ochannels = channels
        layers = [
            nn.ReflectionPad2d(1),
            SNConv2d(in_channels, ochannels, 4, 2),
            get_activation(act_name)]

        for _ in range(num_layers-1):
            channels *= 2
            ichannels, ochannels = ochannels, min(max_channels, channels)
            layers.extend([
                nn.ReflectionPad2d(1),
                SNConv2d(ichannels, ochannels, 4, 2),
                get_activation(act_name)])

        channels *= 2
        ichannels, ochannels = ochannels, min(max_channels, channels)

        layers.extend([
            nn.ReflectionPad2d(1),
            SNConv2d(ichannels, ochannels, 4),
            get_activation(act_name)])
        self.extract = nn.Sequential(*layers)

        self.cam = CAM(ochannels, act_name, True)

        self.output = nn.Sequential(
            nn.ReflectionPad2d(1), SNConv2d(ochannels, 1, 4))

    def forward(self, x):
        x = self.extract(x)
        x, cam_logit, _ = self.cam(x) # we never need heatmap from D
        x = self.output(x)
        return x, cam_logit


class MultiScaleD(nn.Module):
    def __init__(self,
        num_scale=2,
        num_layers=3, channels=64, max_channels=512, act_name='lrelu', in_channels=3
    ) -> None:
        super().__init__()

        self.discs = nn.ModuleList([
            Discriminator(num_layers, channels, max_channels, act_name, in_channels)
            for _ in range(num_scale)])
        self.downsample = nn.AvgPool2d(2)

    def forward(self, x):
        output = []
        for disc in self.discs:
            output.append(disc(x))
            x = self.downsample(x)
        probs = torch.cat([o[0].flatten() for o in output])
        cam_logits = torch.cat([o[1].flatten() for o in output])
        return probs, cam_logits
