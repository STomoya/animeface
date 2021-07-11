
from functools import partial
import math
import random
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract
import numpy as np

class _FusedAct(nn.Module):
    def __init__(self,
        act, scale=math.sqrt(2)
    ) -> None:
        super().__init__()
        self.act = act
        self.scale = scale
    def forward(self, x):
        return self.act(x) * self.scale

class ReLU(_FusedAct):
    def __init__(self, inplace=False) -> None:
        act = nn.ReLU(inplace)
        super().__init__(act)
class LeakyReLU(_FusedAct):
    def __init__(self, negative_slope=0.2, inplace=False) -> None:
        act = nn.LeakyReLU(negative_slope, inplace)
        super().__init__(act)

def get_activation(name, inplace=True):
    if name == 'relu': return ReLU(inplace)
    if name == 'lrelu': return LeakyReLU(inplace=inplace)
    raise Exception('activation')

def _binomial_filter(filter_size):
    def c(n,k):
        if(k<=0 or n<=k): return 1
        else: return c(n-1, k-1) + c(n-1, k)
    return [c(filter_size-1, j) for j in range(filter_size)]

class Blur(nn.Module):
    def __init__(self, filter_size=4, no_pad=False) -> None:
        super().__init__()
        filter = torch.tensor(_binomial_filter(filter_size), dtype=torch.float32)
        kernel = contract('i,j->ij', filter, filter)
        kernel /= kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.register_buffer('kernel', kernel)

        if not no_pad:
            if filter_size % 2 == 1:
                pad1, pad2 = filter_size//2, filter_size//2
            else:
                pad1, pad2 = filter_size//2, (filter_size-1)//2
            self.padding = (pad1, pad2, pad1, pad2)

    def forward(self, x):
        C = x.size(1)
        if hasattr(self, 'padding'):
            x = F.pad(x, self.padding)
        weight = self.kernel.expand(C, -1, -1, -1)
        x = F.conv2d(x, weight, groups=C)
        return x

class Upsample(nn.Sequential):
    def __init__(self,
        filter_size=4
    ) -> None:
        super().__init__(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            Blur(filter_size))
class Downsample(nn.Sequential):
    def __init__(self,
        filter_size=4
    ) -> None:
        super().__init__(
            Blur(filter_size),
            nn.AvgPool2d(2))

class ELR(nn.Module):
    def __init__(self,
        module, gain=1.
    ) -> None:
        super().__init__()
        self.module = module
        self.scale = gain / (module.weight[0].numel() ** 0.5)
    def forward(self, x):
        x = x * self.scale
        return self.module(x)

def Linear(*args, **kwargs):
    return ELR(nn.Linear(*args, **kwargs))
def Conv2d(*args, **kwargs):
    return ELR(nn.Conv2d(*args, **kwargs))

class ModulatedConv2d(nn.Module):
    def __init__(self,
        in_channels, style_dim, out_channels, kernel_size,
        padding, no_pad=False,
        bias=True, demod=True, gain=1.
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding= padding if not no_pad else 0
        self.demod = demod
        self.no_pad = no_pad

        self.affine = Linear(style_dim, in_channels)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        else:
            self.bias = None
        self.scale = gain / (self.weight[0].numel() ** 0.5)

    def forward(self, x, style):
        B, C, H, W = x.size()

        style = self.affine(style) + 1.
        weight = self.weight * style[:, None, :, None, None] * self.scale

        if self.demod:
            d = torch.rsqrt(weight.pow(2).sum([2, 3, 4], keepdim=True) + 1e-4)
            weight = weight * d

        x = x.reshape(1, -1, H, W)
        _, _, *ws = weight.size()
        weight = weight.reshape(B*self.out_channels, *ws)

        x = F.conv2d(x, weight, padding=self.padding, groups=B)

        if self.no_pad:
            decrease = self.kernel_size - 1
            out_size = (B, self.out_channels, H-decrease, W-decrease)
        else:
            out_size = (B, -1, H, W)
        x = x.reshape(*out_size)

        if self.bias is not None:
            x = x + self.bias

        return x

class InjectNoise(nn.Module):
    def __init__(self, resolution) -> None:
        super().__init__()
        self.resolution = resolution
        self.register_buffer('const_noise',
            torch.randn(1, 1, resolution, resolution))
        self.scale = nn.Parameter(torch.zeros(1))
    def forward(self, x, noise=None):
        if isinstance(noise, torch.Tensor):
            x = x + noise * self.scale
        elif noise is None:
            B, _, H, W = x.size()
            x = x + torch.randn(B, 1, H, W, device=x.device) * self.scale
        else:
            x = x + self.const_noise.expand(x.size(0), -1, -1, -1) * self.scale
        return x

    def make_noise(self, batch_size, scale=1.):
        size = int(self.resolution * scale)
        return torch.randn(batch_size, 1, size, size)

class StyleBlock(nn.Module):
    def __init__(self,
        in_channels, style_dim, out_channels, resolution,
        no_pad=False, filter_size=4, act_name='lrelu', gain=1.
    ) -> None:
        super().__init__()
        self.no_pad = no_pad
        self.filter_size = filter_size
        if no_pad: resls = [resolution-2, resolution*2]
        else: resls = [resolution, resolution*2]

        self.conv1 = ModulatedConv2d(
            in_channels, style_dim, out_channels, 3, 1, no_pad,
            True, True, gain)
        self.add_noise1 = InjectNoise(resls[0])
        self.blur = Blur(filter_size, no_pad)
        self.conv2 = ModulatedConv2d(
            out_channels, style_dim, out_channels, 3, 1, no_pad,
            True, True, gain)
        self.add_noise2 = InjectNoise(resls[1])
        self.act = get_activation(act_name)

    def forward(self,
        x: torch.Tensor, style: List[torch.Tensor],
        noise: List[torch.Tensor]=[None, None]):
        assert len(style) == 2
        assert len(noise) == 2
        _, _, H, W = x.size()

        x = self.conv1(x, style[0])
        x = self.add_noise1(x, noise[0])
        x = self.act(x)

        if self.no_pad:
            blur_pad = self.filter_size - 1
            upsize = [H*2+2+blur_pad, W*2+2+blur_pad]
        else:
            upsize = [H*2, W*2]
        x = F.interpolate(
            x, size=upsize, mode='bilinear', align_corners=True)
        x = self.blur(x)

        x = self.conv2(x, style[1])
        x = self.add_noise2(x, noise[1])
        x = self.act(x)

        return x

    def make_noise(self, batch_size, scale=1.):
        return [
            self.add_noise1.make_noise(batch_size, scale),
            self.add_noise2.make_noise(batch_size, scale)]

class ToRGB(nn.Module):
    def __init__(self,
        in_channels, style_dim, out_channels=3
    ) -> None:
        super().__init__()
        self.conv = ModulatedConv2d(
            in_channels, style_dim, out_channels, 1, 0, demod=False)
    def forward(self, x, style):
        return self.conv(x, style)

class ConstInput(nn.Module):
    def __init__(self, size) -> None:
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, *size))
    def forward(self, x):
        return self.input.expand(x.size(0), -1, -1, -1)


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal Positional Embedding 1D or 2D (SPE/SPE2d).
    This module is a modified from:
    https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py # noqa
    Based on the original SPE in single dimension, we implement a 2D sinusoidal
    positional encodding (SPE2d), as introduced in Positional Encoding as
    Spatial Inductive Bias in GANs, CVPR'2021.

    modified by STomoya
    """

    def __init__(self,
                 embedding_dim,
                 padding_idx,
                 init_size=1024,
                 div_half_dim=False,
                 center_shift=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.div_half_dim = div_half_dim
        self.center_shift = center_shift

        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size, embedding_dim, padding_idx, self.div_half_dim)

        self.register_buffer('_float_tensor', torch.FloatTensor(1))

        self.max_positions = int(1e5)

    @staticmethod
    def get_embedding(num_embeddings,
                      embedding_dim,
                      padding_idx=None,
                      div_half_dim=False):
        # there is a little difference from the original paper.
        half_dim = embedding_dim // 2
        if not div_half_dim:
            emb = np.log(10000) / (half_dim - 1)
        else:
            emb = np.log(1e4) / half_dim
        # compute exp(-log10000 / d * i)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(
            num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)],
                        dim=1).view(num_embeddings, -1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb

    def forward(self, input, **kwargs):
        b, seq_len = input.shape
        max_pos = self.padding_idx + 1 + seq_len

        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embedding if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx)
        self.weights = self.weights.to(self._float_tensor)

        positions = self.make_positions(input, self.padding_idx).to(
            self._float_tensor.device)

        return self.weights.index_select(0, positions.view(-1)).view(
            b, seq_len, self.embedding_dim).detach()

    def make_positions(self, input, padding_idx):
        mask = input.ne(padding_idx).int()
        return (torch.cumsum(mask, dim=1).type_as(mask) *
                mask).long() + padding_idx

    def make_grid2d(self, height, width, num_batches=1):
        h, w = height, width
        center_shift = self.center_shift

        h_shift = 0
        w_shift = 0
        if center_shift is not None:
            # if h/w is even, the left center should be aligned with
            # center shift
            if h % 2 == 0:
                h_left_center = h // 2
                h_shift = center_shift - h_left_center
            else:
                h_center = h // 2 + 1
                h_shift = center_shift - h_center

            if w % 2 == 0:
                w_left_center = w // 2
                w_shift = center_shift - w_left_center
            else:
                w_center = w // 2 + 1
                w_shift = center_shift - w_center

        # Note that the index is started from 1 since zero will be padding idx.
        # axis -- (b, h or w)
        x_axis = torch.arange(1, w + 1).unsqueeze(0).repeat(num_batches, 1) + w_shift
        y_axis = torch.arange(1, h + 1).unsqueeze(0).repeat(num_batches, 1) + h_shift

        # emb -- (b, emb_dim, h or w)
        x_emb = self(x_axis).transpose(1, 2)
        y_emb = self(y_axis).transpose(1, 2)

        # make grid for x/y axis
        x_grid = x_emb.unsqueeze(2).repeat(1, 1, h, 1)
        y_grid = y_emb.unsqueeze(3).repeat(1, 1, 1, w)

        # cat grid -- (b, 2 x emb_dim, h, w)
        grid = torch.cat([x_grid, y_grid], dim=1)
        return grid.detach()

class Synthesis(nn.Module):
    def __init__(self,
        image_size=128, bottom=4, spe=True,
        in_channels=512, out_channels=3, style_dim=512,
        channels=64, max_channels=512,
        no_pad=False, filter_size=4, act_name='lrelu'
    ) -> None:
        super().__init__()
        self.no_pad = no_pad

        num_ups = int(math.log2(image_size) - math.log2(bottom))
        resl = bottom
        if no_pad:
            bottom += 2

        if spe:
            self.spe = SinusoidalPositionalEmbedding(in_channels//2, 0, center_shift=100)
            self.bottom = bottom
            self.base_bottom = resl
        else:
            self.const_input = ConstInput((in_channels, bottom, bottom))

        channels = channels * 2 ** num_ups
        ochannels = min(max_channels, channels)
        self.conv1 = ModulatedConv2d(
            in_channels, style_dim, ochannels, 3, 1, no_pad)
        self.add_noise1 = InjectNoise(bottom)
        self.act1 = get_activation(act_name)
        self.to_rgb = ToRGB(ochannels, style_dim, out_channels)
        self.num_layers = 1

        self.blocks = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        for _ in range(num_ups):
            channels = channels // 2
            ichannels, ochannels = ochannels, min(max_channels, channels)
            self.blocks.append(
                StyleBlock(
                    ichannels, style_dim, ochannels,
                    resl, no_pad, filter_size, act_name))
            self.to_rgbs.append(
                ToRGB(ochannels, style_dim, out_channels))
            resl *= 2
            self.num_layers += 2
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self,
        style: List[torch.Tensor],
        noise: List[torch.Tensor]=None):
        if hasattr(self, 'spe'):
            x = self.spe.make_grid2d(
                self.bottom, self.bottom, style[0].size(0))
        else:
            x = self.const_input(style[0])

        if noise is None:
            noise = [None for _ in range(self.num_layers)]

        x = self.conv1(x, style[0])
        x = self.add_noise1(x, noise[0])
        x = self.act1(x)
        image = self.to_rgb(x, style[0])

        for i, (block, to_rgb) in enumerate(zip(self.blocks, self.to_rgbs)):
            style_ = style[i*2+1:i*2+3]
            noise_ = noise[i*2+1:i*2+3]
            x = block(x, style_, noise_)
            image = self.up(image) + to_rgb(x, style_[-1])

        return image

    def make_noise(self, batch_size, scale=1.):
        noise = [self.add_noise1.make_noise(batch_size, scale)]
        for block in self.blocks:
            noise.extend(block.make_noise(batch_size, scale))
        return noise

class PixelNorm(nn.Module):
    def forward(self, x):
        x = x / x.pow(2).mean(dim=1, keepdim=True).sqrt().add(1e-4)
        return x

class MapLinear(nn.Module):
    def __init__(self,
        in_features, out_features, lr=.01
    ) -> None:
        super().__init__()
        self.lr = lr
        self.linear = Linear(in_features, out_features)
    def forward(self, x):
        return self.linear(x) * self.lr

class Mapping(nn.Module):
    def __init__(self,
        in_channels=512, style_dim=512, num_layers=8,
        act_name='lrelu', pixelnorm=True, lr=.01
    ) -> None:
        super().__init__()
        self.style_dim = style_dim
        if pixelnorm: layers = [PixelNorm()]
        else: layers = []

        layers.extend([MapLinear(in_channels, style_dim, lr), get_activation(act_name)])
        for _ in range(num_layers-1):
            layers.extend([
                MapLinear(style_dim, style_dim, lr),
                get_activation(act_name)])
        self.map = nn.Sequential(*layers)
    def forward(self, z):
        w = self.map(z)
        return w

    def avg_w(self, num_samples, device):
        noise = torch.randn(num_samples, self.style_dim, device=device)
        avg_w = self.forward(noise).mean(dim=0, keepdim=True)
        return avg_w

    def truncate_w(self,
        w, truncation_psi=0.7, num_samples=2000):
        return self.avg_w(num_samples, w.device).lerp(w, truncation_psi)

class Generator(nn.Module):
    def __init__(self,
        image_size: int=128,   # image size
        bottom: int=4,         # bottom size
        spe: bool=False,       # use positional encoding
        latent_dim: int=512,   # latent input size
        in_channels: int=512,  # synthesis input size
        style_dim: int=512,    # style code dim
        out_channels: int=3,   # image channels
        channels: int=32,      # channel width multiplier
        max_channels: int=512, # maximum channel width
        no_pad: bool=False,    # no padding mode
        map_num_layers: int=8, # number of layers in mapping
        map_lr: float=0.01,    # lr for mapping network
        pixelnorm: bool=True,  # apply pixel norm to latent input
        filter_size: int=4,    # binomial filter size
        act_name: str='lrelu'  # activation name
    ) -> None:
        super().__init__()

        self.map = Mapping(
            latent_dim, style_dim, map_num_layers,
            act_name, pixelnorm, map_lr)
        self.synthesis = Synthesis(
            image_size, bottom, spe,
            in_channels, out_channels, style_dim,
            channels, max_channels, no_pad,
            filter_size, act_name)

        self.make_noise = self.synthesis.make_noise
        self.num_layers = self.synthesis.num_layers

    def forward(self,
        z: torch.Tensor,
        noise: List[torch.Tensor]=None,
        truncation_psi=1.,
        mix_prob: float=0.5,
        mix_index: Union[list, int]=None
    ):
        w = self.map(z)

        if truncation_psi != 1.:
            w = self.map.truncate_w(w, truncation_psi)

        if w.ndim == 3:
            w = w.unbind(dim=1)
        style = self.to_syn_input(w, mix_prob, mix_index)

        if noise is None:
            noise = [None for _ in range(self.num_layers)]
        image = self.synthesis(style, noise)

        return image

    def to_syn_input(self,
        w,
        prob: float,
        indices: Union[list, int]=None
    ) -> List[torch.Tensor]:
        if isinstance(w, tuple):
            if indices is None:
                _perm = torch.randperm(self.num_layers-2) + 1
                indices = _perm[:len(w)-1].tolist()
            if self.training and random.random() > prob:
                indices = []
            indices = [0] + sorted(indices) + [self.num_layers]
            ws = []
            for i, index in enumerate(indices[:-1]):
                ws.extend([w[i].clone() for _ in range(index, indices[i+1])])
            return ws
        else:
            return [w.clone() for _ in range(self.num_layers)]

    def scale_image(self, scale=1.):
        '''input bigger embedding for bigger image'''
        self.synthesis.bottom = int(self.synthesis.base_bottom * scale)
        if self.synthesis.no_pad:
            self.synthesis.bottom += 2
    def reset_scale(self):
        '''reset to default image size'''
        self.synthesis.bottom = self.synthesis.base_bottom
        if self.synthesis.no_pad:
            self.synthesis.bottom += 2

    def init_weight(self, func, lr=0.01):
        self.map.apply(partial(func, lr=lr))
        self.synthesis.apply(func)

class MiniBatchStdDev(nn.Module):
    '''Mini-Batch Standard Deviation'''
    def __init__(self, group_size=4, eps=1e-4):
        super().__init__()
        self.group_size = group_size
        self.eps = eps

    def forward(self, x):
        B, C, H, W = x.size()
        y = x
        groups = self._check_group_size(B)
        # calc stddev and concatenate
        y = y.view(groups, -1, C, H, W)
        y = y - y.mean(0, keepdim=True)
        y = y.square().mean(0)
        y = y.add_(self.eps).sqrt()
        y = y.mean([1, 2, 3], keepdim=True)
        y = y.repeat(groups, 1, H, W)

        return torch.cat([x, y], dim=1)

    def _check_group_size(self, batch_size):
        if batch_size % self.group_size == 0: return self.group_size
        else:                                 return batch_size

class DiscBlock(nn.Module):
    def __init__(self,
        in_channels, out_channels,
        filter_size=4, act_name='lrelu', sample=True
    ) -> None:
        super().__init__()
        self.main = nn.Sequential(
            Conv2d(in_channels, out_channels, 3, 1, 1),
            get_activation(act_name),
            Conv2d(out_channels, out_channels, 3, 1, 1))
        self.skip = Conv2d(in_channels, out_channels, 1, bias=False)
        if not sample:
            self.down = Downsample(filter_size)

    def forward(self, x):
        h = self.main(x)
        x = self.skip(x)
        if hasattr(self, 'down'):
            h = self.down(h)
            x = self.down(x)
        else:
            h = h[:, :, ::2, ::2]
            x = x[:, :, ::2, ::2]
        return h + x

class DiscEpilogue(nn.Module):
    def __init__(self,
        mbsd_group_size, channels, bottom,
        gap, act_name='lrelu'
    ) -> None:
        super().__init__()

        layers = [
            MiniBatchStdDev(mbsd_group_size),
            Conv2d(channels+1, channels, 3, 1, 1),
            get_activation(act_name)]
        if gap:
            layers.append(nn.AdaptiveAvgPool2d(bottom))
        layers.extend([
            nn.Flatten(),
            Linear(channels*bottom**2, channels),
            get_activation(act_name),
            Linear(channels, 1)
        ])

        self.epilogue = nn.Sequential(*layers)

    def forward(self, x):
        return self.epilogue(x)

class Discriminator(nn.Module):
    def __init__(self,
        image_size=128, bottom=2, in_channels=3,
        channels=32, max_channels=512,
        mbsd_groups=4, gap=True, filter_size=4, act_name='lrelu'
    ) -> None:
        super().__init__()
        num_downs = int(math.log2(image_size) - math.log2(bottom))

        ochannels = channels
        self.input = nn.Sequential(
            Conv2d(in_channels, ochannels, 3, 1, 1),
            get_activation(act_name))

        layers = []
        for _ in range(num_downs):
            channels *= 2
            ichannels, ochannels = ochannels, min(max_channels, channels)
            layers.append(
                DiscBlock(ichannels, ochannels, filter_size, act_name))
        self.body = nn.Sequential(*layers)
        self.output = DiscEpilogue(
            mbsd_groups, ochannels, bottom, gap, act_name)

    def forward(self, x):
        x = self.input(x)
        x = self.body(x)
        x = self.output(x)
        return x

    def init_weight(self, func):
        self.apply(func)

'''init weights function'''
def init_weight_N01(m, lr=1):
    '''init weight with N(0, 1/lr)'''
    if isinstance(m, (nn.Linear, nn.Conv2d, ModulatedConv2d)):
        m.weight.data.normal_(0., 1/lr)
        if m.bias != None:
            m.bias.data.fill_(0.)

if __name__=='__main__':
    g = Generator(spe=True, no_pad=True)
    d = Discriminator()
    g.scale_image(1.5)
    z = torch.randn(3, 2, 512)
    image = g(z)
    prob = d(image)
    print(image.size(), prob.size())
    print(image.max(), image.min())
