'''
StyleGAN2 using custom CUDA kernel from official implementation
'''

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from thirdparty.stylegan2_ops.ops import (
    upfirdn2d,
    bias_act,
    conv2d_resample
)

class EqualizedLinearAct(nn.Module):
    '''linear -> bias -> activate
    '''
    def __init__(self,
        in_features, out_features, act_name='lrelu', gain=1.
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        # no option for disabling bias
        # doesn't matter much
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.scale = gain / (self.weight[0].numel() ** 0.5)
        self.act_name = act_name

    def forward(self, x):
        # act(x * W * scale + bias)
        weight = self.weight * self.scale
        x = F.linear(x, weight)
        if x.ndim ==3: dim = 2
        else: dim = 1
        x = bias_act.bias_act(x, self.bias.to(x.dtype), dim, self.act_name)
        return x

class EqualizedConv2dAct(nn.Module):
    '''convolution -> (down-sample) -> bias -> activate
    act_gain = 1. on residual output and skip
    '''
    def __init__(self,
        in_channels, out_channels, kernel_size, bias=True,
        down=1, filter_size=4, act_name='lrelu',
        gain=1., act_gain=None
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, 'assuming kernel size is odd'
        self.down = down
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        # elr scale
        self.scale = gain / (self.weight[0].numel() ** 0.5)

        self.act_name = act_name
        if act_gain:
            self.act_gain = act_gain
        else:
            self.act_gain = bias_act.activation_funcs[act_name].def_gain

        # FIR filter if down-sample
        if down > 1:
            filter = torch.tensor(_binomial_filter(filter_size), dtype=torch.float32)
            kernel = torch.einsum('i,j->ij', filter, filter)
            kernel /= kernel.sum()
            self.register_buffer('kernel', kernel)
        else:
            self.kernel = None

    def forward(self, x):
        # act(down(x * W * scale) + b)
        # {weight,b}.to(x.dtype) for AMP
        weight = self.weight * self.scale
        x = conv2d_resample.conv2d_resample(
            x, weight.type(x.dtype), self.kernel,
            down=self.down, padding=self.padding)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        x = bias_act.bias_act(x, b, 1, self.act_name, gain=self.act_gain)
        return x

class ModulatedConv2d(nn.Module):
    '''        ->   conv  ->
        -> modulate - ↑
    '''
    def __init__(self,
        in_channels, style_dim, out_channels, kernel_size,
        up=1, filter_size=4, demod=True, gain=1.
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1
        self.up = up
        self.out_channels = out_channels
        self.demod = demod
        self.padding = kernel_size // 2

        # FIR filter if up-sample
        if up > 1:
            filter = torch.tensor(_binomial_filter(filter_size), dtype=torch.float32)
            kernel = torch.einsum('i,j->ij', filter, filter)
            kernel /= kernel.sum()
            self.register_buffer('kernel', kernel)
        else:
            self.kernel = None

        # learnable affine transform for style code
        self.affine = EqualizedLinearAct(
            style_dim, in_channels, 'linear', gain)

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        # bias addition will be performed outside
        # because of pixel-wise noise input

        # elr scale
        self.scale = gain / (self.weight[0].numel() ** 0.5)

    def forward(self, x, w):
        # up(x) * demod(mod(W, a(w)) * scale)
        B, _, H, W = x.size()
        # affine
        w = self.affine(w) + 1.

        # modulate
        weight = self.weight[None, :, :, :, :] \
            * w[:, None, :, None, None] \
            * self.scale

        # demodulate
        if self.demod:
            d = torch.rsqrt(weight.pow(2).sum([2, 3, 4], keepdim=True).add(1e-8))
            weight = weight * d

        # reshaping for conv input
        x = x.reshape(1, -1, H, W)
        _, _, *ws = weight.size()
        weight = weight.reshape(B*self.out_channels, *ws)

        # conv with optional up-sample
        flip_weight = (self.up == 1) # slightly faster
        x = conv2d_resample.conv2d_resample(
            x, weight.to(x.dtype), self.kernel,
            self.up, padding=self.padding, groups=B,
            flip_weight=flip_weight)

        # return with bias
        return x.reshape(B, self.out_channels, H*self.up, W*self.up)

class InjectNoise(nn.Module):
    '''-> + noise ->
    '''
    def __init__(self, resolution) -> None:
        super().__init__()
        self.resolution = resolution
        self.register_buffer('noise', torch.randn(1, 1, resolution, resolution))
        self.scale = nn.Parameter(torch.tensor([0.]))

    def forward(self, x, noise='random'):
        # x + noise * scale
        if isinstance(noise, torch.Tensor):
            return x + noise * self.scale
        elif noise == 'random':
            B, _, H, W = x.size()
            noise = torch.randn(B, 1, H, W, device=x.device)
            return x + noise * self.scale
        elif noise == 'const':
            B = x.size(0)
            return x + self.noise.expand(B, -1, -1, -1) * self.scale

    def make_noise(self, batch_size, device):
        '''make noise which fits the resolution'''
        return torch.randn(batch_size, 1, self.resolution, self.resolution, device=device)

def _binomial_filter(filter_size):
    '''return binomial filter from size.'''
    def c(n,k):
        if(k<=0 or n<=k): return 1
        else: return c(n-1, k-1) + c(n-1, k)
    return [c(filter_size-1, j) for j in range(filter_size)]

class StyleLayer(nn.Module):
    '''-> modconv -> noise -> bias -> act ->
    '''
    def __init__(self,
        in_channels, style_dim, out_channels, kernel_size, resolution,
        up=False, filter_size=4, act_name='lrelu', gain=1.
    ) -> None:
        super().__init__()
        self.act_name = act_name

        if up:
            up = 2
            resolution *= 2
        else:
            up = 1

        self.conv = ModulatedConv2d(
            in_channels, style_dim, out_channels, kernel_size,
            up, filter_size, True, gain)
        self.add_noise = InjectNoise(resolution)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.make_noise = self.add_noise.make_noise

    def forward(self, x, w, noise='random'):
        # act(conv(x, w) + noise + b)
        #  conv(x, w): up(x) * demod(mod(W, a(w)) * scale)
        x = self.conv(x, w)
        x = self.add_noise(x, noise)
        x = bias_act.bias_act(x, self.bias, 1, self.act_name)
        return x

class ToRGB(nn.Module):
    def __init__(self,
        in_channels, style_dim, out_channels,
        kernel_size=1, gain=1.
    ) -> None:
        super().__init__()
        self.conv = ModulatedConv2d(
            in_channels, style_dim, out_channels, kernel_size,
            1, None, False, gain)

    def forward(self, x, w):
        # conv(x, w): up(x) * demod(mod(W, a(w)) * scale)
        return self.conv(x, w)

class PixelNorm(nn.Module):
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True).add(1e-8))

class MappingNetwork(nn.Module):
    def __init__(self,
        in_dim, style_dim=512, num_layers=8,
        pixel_norm=True, act_name='lrelu', gain=1.
    ) -> None:
        super().__init__()
        self.style_dim = style_dim

        if pixel_norm:
            self.norm = PixelNorm()

        layers = [EqualizedLinearAct(in_dim, style_dim, act_name, gain)]
        for _ in range(num_layers-1):
            layers.append(
                EqualizedLinearAct(style_dim, style_dim, act_name, gain))
        self.map = nn.Sequential(*layers)

    def forward(self, z):
        if hasattr(self, 'norm'):
            z = self.norm(z)
        return self.map(z)

    def avg_w(self, num_samples, device):
        noise = torch.randn(num_samples, self.style_dim, device=device)
        avg_w = self.forward(noise).mean(dim=0, keepdim=True)
        return avg_w

    def truncate_w(self, w, truncation_psi=0.7, num_samples=2000):
        return self.avg_w(num_samples, w.device).lerp(w, truncation_psi)

class ConstInput(nn.Module):
    def __init__(self, size) -> None:
        super().__init__()
        self.input = nn.Parameter(torch.randn(size))
    def forward(self, x):
        B = x.size(0)
        return self.input.expand(B, -1, -1, -1)

class Upsample(nn.Module):
    def __init__(self, filter_size) -> None:
        super().__init__()
        filter = torch.tensor(_binomial_filter(filter_size), dtype=torch.float32)
        kernel = torch.einsum('i,j->ij', filter, filter)
        kernel /= kernel.sum()
        self.register_buffer('kernel', kernel)
    def forward(self, x):
        # up(x)
        return upfirdn2d.upsample2d(x, self.kernel)

class Synthesis(nn.Module):
    def __init__(self,
        image_size, style_dim,
        in_channels=512, out_channels=3,
        channels=32, max_channels=512,
        bottom=4, kernel_size=3, filter_size=4,
        act_name='lrelu', gain=1.
    ) -> None:
        super().__init__()
        num_ups = int(math.log2(image_size) - math.log2(bottom))

        self.const_input = ConstInput((1, in_channels, bottom, bottom))
        channels = channels * 2 ** num_ups
        ochannels = min(channels, max_channels)
        self.input = StyleLayer(
            in_channels, style_dim, ochannels, kernel_size,
            bottom, False, filter_size, act_name, gain)
        self.input_torgb = ToRGB(
            ochannels, style_dim, out_channels,
            1, gain)
        self.num_layers = 1

        self.style_layers1 = nn.ModuleList()
        self.style_layers2 = nn.ModuleList()
        self.torgbs        = nn.ModuleList()

        self.up = Upsample(filter_size)
        resl = bottom
        for i in range(num_ups):
            channels = channels // 2
            ichannels, ochannels = ochannels, min(max_channels, channels)
            self.style_layers1.append(
                StyleLayer(
                    ichannels, style_dim, ochannels, kernel_size,
                    resl, False, filter_size, act_name, gain))
            self.style_layers2.append(
                StyleLayer(
                    ochannels, style_dim, ochannels, kernel_size,
                    resl, True, filter_size, act_name, gain))
            self.torgbs.append(
                ToRGB(ochannels, style_dim, out_channels, 1, gain))
            resl *= 2
            self.num_layers += 2

    def forward(self,
        w: list,
        noise: list
    ):
        assert len(w) == self.num_layers
        assert len(noise) == self.num_layers

        x = self.const_input(w[0])
        x = self.input(x, w[0], noise[0])
        image = self.input_torgb(x, w[0])

        for i, (style1, style2, torgb) in enumerate(
            zip(self.style_layers1, self.style_layers2, self.torgbs)):
            x = style1(x, w[1 + i*2], noise[1 + i*2])
            x = style2(x, w[2 + i*2], noise[2 + i*2])
            image = self.up(image) + torgb(x, w[2 + i*2])

        return image

    def make_noise(self, batch_size, device):
        noise = [self.input.make_noise(batch_size, device)]
        for style1, style2 in zip(self.style_layers1, self.style_layers2):
            noise.extend([
                style1.make_noise(batch_size, device),
                style2.make_noise(batch_size, device)
            ])
        return noise

class Generator(nn.Module):
    def __init__(self,
        image_size, style_dim,
        const_channels=512, out_channels=3,
        channels=32, max_channels=512,
        bottom=4, kernel_size=3, filter_size=4,
        latent_dim=512, map_num_layers=8,
        pixel_norm=True,
        act_name='lrelu', gain=1.
    ) -> None:
        super().__init__()

        self.mapping = MappingNetwork(
            latent_dim, style_dim, map_num_layers,
            pixel_norm, act_name, gain)
        self.synthesis = Synthesis(
            image_size, style_dim, const_channels, out_channels,
            channels, max_channels, bottom,
            kernel_size, filter_size, act_name, gain)
        self.num_layers = self.synthesis.num_layers
        self.make_noise = self.synthesis.make_noise

    def forward(self, z, noise='random', truncation_psi=1., mix_indices=None):

        w = self.mapping(z)

        if truncation_psi != 1.:
            w = self.mapping.truncate_w(w, truncation_psi)

        # make a list of style codes
        # corresponding to the given indices (if multiple style input)
        #   if indices=None, randomly select indices
        ws = self.to_syn_input(w, mix_indices)
        # process noise input
        #   if 'random' or 'const' make a list
        #   else, it should be list[Tensor]
        if isinstance(noise, str):
            noise = [noise for _ in range(self.num_layers)]

        image = self.synthesis(ws, noise)
        return image, w

    def to_syn_input(self,
        w,
        indices: list=None
    ):
        if w.ndim == 3:
            w = w.unbind(dim=1)
            if indices is None:
                _perm = torch.randperm(self.num_layers-2) + 1
                indices = _perm[:len(w)-1].tolist()
            indices = [0] + sorted(indices) + [self.num_layers]
            ws = []
            for i, index in enumerate(indices[:-1]):
                ws.extend([w[i].clone() for _ in range(index, indices[i+1])])
            return ws
        else:
            return [w.clone() for _ in range(self.num_layers)]

class ResBlock(nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel_size,
        filter_size=4, act_name='lrelu', gain=1.
    ) -> None:
        super().__init__()
        # act gain is based on official implementation
        self.main = nn.Sequential(
            EqualizedConv2dAct(
                in_channels, out_channels, kernel_size, True,
                1, None, act_name, gain),
            EqualizedConv2dAct(
                out_channels, out_channels, kernel_size, True,
                2, filter_size, act_name, gain, act_gain=1.))
        self.skip = EqualizedConv2dAct(
            in_channels, out_channels, 1, False,
            2, filter_size, 'linear', gain, act_gain=1.)

    def forward(self, x):
        # conv(conv(x)) + skip(x)
        # conv(x): act(down(x * W * scale) + b)
        # skip(x): x * W * scale + b
        h = self.main(x)
        x = self.skip(x)
        return h + x

class MinibatchStdDev(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        G = self.group_size if N % self.group_size == 0 else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
        return x

class DiscEpilogue(nn.Module):
    def __init__(self,
        mbsd_group_size, mbsd_channels,
        channels, bottom, act_name='lrelu', gain=1.
    ) -> None:
        super().__init__()
        self.epilogue = nn.Sequential(
            MinibatchStdDev(mbsd_group_size, mbsd_channels),
            EqualizedConv2dAct(
                channels+mbsd_channels, channels, 3, True,
                1, None, act_name, gain),
            nn.Flatten(),
            EqualizedLinearAct(
                channels*bottom**2, channels, act_name, gain),
            EqualizedLinearAct(
                channels, 1, 'linear', gain)
        )
    def forward(self, x):
        # linear(linear(conv(mbsd(x))))
        # conv(x): act(down(x * W * scale) + b)
        # linear(x): act(down(x * W * scale) + b)
        return self.epilogue(x)

class Discriminator(nn.Module):
    def __init__(self,
        image_size, in_channels=3, channels=64, max_channels=512, kernel_size=3,
        mbsd_group_size=4, mbsd_channels=1,
        bottom=4, filter_size=4, act_name='lrelu', gain=1.
    ) -> None:
        super().__init__()
        num_downs = int(math.log2(image_size) - math.log2(bottom))

        ochannels = channels
        self.from_rgb = EqualizedConv2dAct(
            in_channels, ochannels, 1,
            True, 1, None, act_name, gain)

        resblocks = []
        for _ in range(num_downs):
            channels *= 2
            ichannels, ochannels = ochannels, min(max_channels, channels)
            resblocks.append(
                ResBlock(
                    ichannels, ochannels, kernel_size,
                    filter_size, act_name, gain))
        self.resblocks = nn.Sequential(*resblocks)
        self.epilogue = DiscEpilogue(
            mbsd_group_size, mbsd_channels, ochannels,
            bottom, act_name, gain)

    def forward(self, x):
        x = self.from_rgb(x)
        x = self.resblocks(x)
        x = self.epilogue(x)
        return x
