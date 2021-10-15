
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.special
import scipy.signal

from thirdparty.stylegan3_ops.ops import bias_act
from thirdparty.stylegan3_ops.ops import filtered_lrelu
from thirdparty.stylegan3_ops.ops import conv2d_gradfix
from thirdparty.stylegan3_ops.ops import conv2d_resample

class Linear(nn.Module):
    '''linear'''
    def __init__(self,
        in_features, out_features, bias, act_name='linear', gain=1.
    ) -> None:
        super().__init__()
        self.act_name = act_name
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.scale = gain / (self.weight[0].numel() ** 0.5)

    def forward(self, x):
        x = F.linear(x, self.weight * self.scale)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.act_name)
        return x

class ModulatedConv(nn.Module):
    '''modulated convolution'''
    def __init__(self,
        in_channels, out_channels, kernel_size=3,
        padding=1, demod=True
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.padding = padding
        self.demod = demod

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.scale = 1 / (self.weight[0].numel() ** 0.5)

    def forward(self, x, s, input_gain=None):
        B, _, H, W = x.size()
        weight = self.weight

        # normalize
        # if self.demod:
        #     weight = weight * weight.square().mean([1, 2, 3], keepdim=True).rsqrt()
        #     s      = s * s.square().mean().rsqrt()

        # modulate (+ ELR)
        weight = weight[None, :, :, :, :] \
            * s[:, None, :, None, None] * self.scale

        # demod
        if self.demod:
            d = weight.square().sum([2, 3, 4]).add(1e-8).rsqrt()
            weight = weight * d[:, :, None, None, None]

        # scale with ema
        if input_gain is not None:
            input_gain = input_gain.expand(B, self.in_channels)
            weight = weight * input_gain[:, None, :, None, None]

        # conv
        x = x.reshape(1, -1, H, W)
        weight = weight.reshape(-1, self.in_channels, *weight.shape[-2:])
        x = conv2d_gradfix.conv2d(x, weight, None, 1, self.padding, 1, groups=B)
        x = x.reshape(B, -1, *x.shape[-2:])
        return x

def design_filter(numtaps, cutoff, width, fs, radial=False):
    '''make filter for sampling'''
    assert numtaps >= 1
    if numtaps == 1:
            return None

    if not radial:
        f = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)
        return torch.as_tensor(f, dtype=torch.float32)

    x = (np.arange(numtaps) - (numtaps - 1) / 2) / fs
    r = np.hypot(*np.meshgrid(x, x))
    f = scipy.special.j1(2 * cutoff * (np.pi * r)) / (np.pi * r)
    beta = scipy.signal.kaiser_beta(scipy.signal.kaiser_atten(numtaps, width / (fs / 2)))
    w = np.kaiser(numtaps, beta)
    f *= np.outer(w, w)
    f /= np.sum(f)
    return torch.as_tensor(f, dtype=torch.float32)

def get_layer_params(
    image_size, num_layers, channels,
    max_channels=512, image_channels=3, margin_size=10,
    first_cutoff=2, first_stopband=2**2.1, last_stopband_rel=2**0.3, num_critical=2):
    '''claculate layer parameters'''

    # Geometric progression of layer cutoffs and min. stopbands.
    last_cutoff = image_size / 2 # f_{c,N}
    last_stopband = last_cutoff * last_stopband_rel # f_{t,N}
    exponents = np.minimum(np.arange(num_layers + 1) / (num_layers - num_critical), 1)
    cutoffs = first_cutoff * (last_cutoff / first_cutoff) ** exponents # f_c[i]
    stopbands = first_stopband * (last_stopband / first_stopband) ** exponents # f_t[i]

    # Compute remaining layer parameters.
    sampling_rates = np.exp2(np.ceil(np.log2(np.minimum(stopbands * 2, image_size)))) # s[i]
    half_widths = np.maximum(stopbands, sampling_rates / 2) - cutoffs # f_h[i]
    sizes = sampling_rates + margin_size * 2
    sizes[-2:] = image_size
    channels = np.rint(np.minimum((channels / 2) / cutoffs, max_channels))
    channels[-1] = image_channels
    return channels, sizes, sampling_rates, cutoffs, half_widths

class StyleLayer(nn.Module):
    '''style layer'''
    def __init__(self,
        in_channels, style_dim, out_channels, kernel_size,
        in_size, out_size,
        in_sampling_rate, out_sampling_rate,
        in_cutoff, out_cutoff,
        in_half_width, out_half_width,
        is_rgb, is_critical_sampled,
        lrelu_sampling=2, filter_size=6, conv_clamp=256, ema_decay=0.999
    ) -> None:
        super().__init__()
        self.conv_clamp = conv_clamp
        self.ema_decay = ema_decay
        self.is_rgb = is_rgb
        self.gain = 1. if is_rgb else 2 ** 0.5
        self.negative_slope = 1. if is_rgb else 0.2

        self.affine = Linear(style_dim, in_channels, True)
        self.affine.bias.data.fill_(1.)
        self.register_buffer('ema', torch.ones([]))

        # design filters
        tmp_srate = max(in_sampling_rate, out_sampling_rate) * (1 if is_rgb else lrelu_sampling)
        # upsampling filter
        self.up_factor = int(np.rint(tmp_srate / in_sampling_rate))
        assert in_sampling_rate * self.up_factor == tmp_srate
        up_taps = filter_size * self.up_factor if self.up_factor > 1 and not is_rgb else 1
        self.register_buffer('up_filter', design_filter(
            up_taps, in_cutoff, in_half_width*2, tmp_srate))
        # downsampling filter
        self.down_factor = int(np.rint(tmp_srate / out_sampling_rate))
        assert out_sampling_rate * self.down_factor == tmp_srate
        down_taps = filter_size * self.down_factor if self.down_factor > 1 and not is_rgb else 1
        self.register_buffer('down_filter', design_filter(
            down_taps, out_cutoff, out_half_width*2, tmp_srate, not is_critical_sampled))

        # calc padding
        in_size = np.broadcast_to(np.asarray(in_size), [2])
        out_size = np.broadcast_to(np.asarray(out_size), [2])
        pad_total = (out_size - 1) * self.down_factor + 1
        pad_total -= (in_size + kernel_size - 1) * self.up_factor
        pad_total += up_taps + down_taps - 2
        pad_lo = (pad_total + self.up_factor) // 2
        pad_hi = pad_total - pad_lo
        self.padding = [int(pad_lo[0]), int(pad_hi[0]), int(pad_lo[1]), int(pad_hi[1])]

        # convolution layer
        self.conv = ModulatedConv(
            in_channels, out_channels, kernel_size, kernel_size-1, not is_rgb)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, w):

        # update ema of magnitude on training
        # TODO: this strickly requires calling .eval() before test.
        #       Not good.
        if self.training:
            stats = x.detach().to(torch.float32).square().mean()
            self.ema.copy_(stats.lerp_(self.ema, self.ema_decay))
        # scale weight
        input_gain = self.ema.rsqrt()

        # affine transform style
        s = self.affine(w)

        # conv
        x = self.conv(x, s, input_gain)
        # act
        x = filtered_lrelu.filtered_lrelu(
            x, self.up_filter, self.down_filter, self.bias.to(x.dtype),
            self.up_factor, self.down_factor, self.padding, self.gain,
            self.negative_slope, self.conv_clamp)

        return x

class SynthesisInput(nn.Module):
    '''fourier feature input'''
    def __init__(self,
        style_dim, channels, size, sampling_rate, bandwidth
    ) -> None:
        super().__init__()
        self.channels = channels
        self.bandwidth = bandwidth
        self.sampling_rate = sampling_rate
        self.size = list(map(int, (np.broadcast_to(np.asarray(size), [2]))))

        freqs = torch.randn(channels, 2)
        radii = freqs.square().sum(1, keepdim=True).sqrt()
        freqs /= radii * radii.square().exp().pow(0.25)
        freqs *= bandwidth
        phases = torch.rand(channels) - 0.5

        self.weight = nn.Parameter(torch.randn(channels, channels))
        self.scale = 1 / (channels ** 0.5)
        self.affine = Linear(style_dim, 4, True)
        # default to no translation
        self.affine.weight.data.fill_(0.)
        self.affine.bias.data.copy_(torch.tensor([1, 0, 0, 0], dtype=torch.float32))

        self.register_buffer('transform', torch.eye(3, 3))
        self.register_buffer('freqs', freqs)
        self.register_buffer('phases', phases)

    def forward(self, w):
        B, device = w.size(0), w.device

        t = self.affine(w)
        t = t / t[:, :2].norm(dim=1, keepdim=True) # norm with translation params
        # rotation matrix
        m_r = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
        m_r[:, 0, 0] =   t[:, 0]
        m_r[:, 0, 1] = - t[:, 1]
        m_r[:, 1, 0] =   t[:, 1]
        m_r[:, 1, 1] =   t[:, 0]
        # translation matrix
        m_t = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
        m_t[:, 0, 2] = - t[:, 2]
        m_t[:, 1, 2] = - t[:, 3]
        # transformation matrix
        transforms = m_r @ m_t @ self.transform.unsqueeze(0)

        phases = self.phases.unsqueeze(0)
        freqs  = self.freqs.unsqueeze(0)

        # transform frequencies
        phases = phases + (freqs @ transforms[:, :2, 2:]).squeeze(2)
        freqs = freqs @ transforms[:, :2, :2]

        # dampen out-of-band freqs
        amp = (1 - (freqs.norm(dim=2) - self.bandwidth) \
            / (self.sampling_rate / 2 - self.bandwidth)) \
            .clamp(0, 1)

        # sampling grid
        theta = torch.eye(2, 3, device=device)
        theta[0, 0] = 0.5 * self.size[0] / self.sampling_rate
        theta[1, 1] = 0.5 * self.size[1] / self.sampling_rate
        grids = F.affine_grid(theta.unsqueeze(0),
            [1, 1, self.size[1], self.size[0]], align_corners=False)

        # fourier features
        x = (grids.unsqueeze(3) @ freqs.permute(0, 2, 1).unsqueeze(1).unsqueeze(2)).squeeze(3)
        x = x + phases.unsqueeze(1).unsqueeze(2)
        x = torch.sin(x * (np.pi * 2))
        x = x * amp.unsqueeze(1).unsqueeze(2)

        # trainable mapping
        x = F.linear(x, self.weight * self.scale)
        x = x.permute(0, 3, 1, 2)
        return x

class PixelNorm(nn.Module):
    '''pixel normalization'''
    def forward(self, x):
        x = x / x.pow(2).mean(dim=1, keepdim=True).sqrt().add(1e-8)
        return x

class Mapping(nn.Module):
    '''mapping network'''
    def __init__(self,
        latent_dim, style_dim, num_layers=2,
        pixel_norm=True, ema_decay=0.998
    ) -> None:
        super().__init__()
        self.ema_decay = ema_decay
        act_name = 'lrelu'

        if pixel_norm: self.norm = PixelNorm()
        layers = [Linear(latent_dim, style_dim, True, act_name)]
        for _ in range(num_layers-1):
            layers.append(
                Linear(style_dim, style_dim, True, act_name))
        self.net = nn.Sequential(*layers)
        self.register_buffer('w_avg', torch.zeros(style_dim))

    def forward(self, z, truncation_psi=1.):

        if hasattr(self, 'norm'):
            z = self.norm(z)

        w = self.net(z)

        if self.training:
            stats = w.detach().to(torch.float32).mean(dim=0)
            self.w_avg.copy_(stats.lerp(self.w_avg, self.ema_decay))

        if truncation_psi != 1:
            w = self.w_avg.lerp(w, truncation_psi)
        return w

class Synthesis(nn.Module):
    '''synthesis network'''
    def __init__(self,
        image_size, num_layers=14, channels=32, max_channels=512, style_dim=512,
        image_channels=3, output_scale=0.25, margin_size=10,
        first_cutoff=2, first_stopband=2**2.1, last_stopband_rel=2**0.3,
        kernel_size=3
    ) -> None:
        super().__init__()
        self.num_ws = num_layers + 2 # +input+ToRGB

        # Matching my code practice in animeface
        # which calcs top channel size from the smallest size.
        # StyleGAN3 defaults to channels=2**15(=32768)
        # which final channel becomes 64 on 512x512 pix image
        log_resl_diff = int(math.log2(512)-math.log2(image_size))
        min_c_scale = channels / 64
        channels = int(2 ** (15 - log_resl_diff) * min_c_scale)
        # calc style layer args
        channels, sizes, sampling_rates, cutoffs, half_widths = get_layer_params(
            image_size, num_layers, channels, max_channels, image_channels,
            margin_size, first_cutoff, first_stopband, last_stopband_rel,
            num_critical=2)

        self.input = SynthesisInput(
            style_dim, int(channels[0]), sizes[0], sampling_rates[0], cutoffs[0])

        layers = []
        for i in range(num_layers+1):
            prev = max(i-1, 0)
            is_rgb = i == num_layers
            is_critically_sampled = (i >= num_layers - 2)
            layers.append(
                StyleLayer(
                    int(channels[prev]), style_dim, int(channels[i]), 1 if is_rgb else kernel_size,
                    int(sizes[prev]), int(sizes[i]), sampling_rates[prev], sampling_rates[i],
                    cutoffs[prev], cutoffs[i], half_widths[prev], half_widths[i],
                    is_rgb, is_critically_sampled))
        self.net = nn.ModuleList(layers)

        self.register_buffer('output_scale', torch.tensor([output_scale]))

    def forward(self, w):
        if w.ndim == 2: # repeat if no style mix
            w = w.unsqueeze(1).repeat(1, self.num_ws, 1)
        ws = w.unbind(dim=1)

        x = self.input(ws[0])
        for module, w in zip(self.net, ws[1:]):
            x = module(x, w)

        return x * self.output_scale

class Generator(nn.Module):
    def __init__(self,
        image_size, latent_dim, num_layers=14, map_num_layers=2,
        channels=32, max_channels=512, style_dim=512, pixel_norm=True,
        image_channels=3, output_scale=0.25, margin_size=10,
        first_cutoff=2, first_stopband=2**2.1, last_stopband_rel=2**0.3,
        kernel_size=3
    ) -> None:
        super().__init__()
        self.map = Mapping(
            latent_dim, style_dim, map_num_layers, pixel_norm)
        self.synthesis = Synthesis(
            image_size, num_layers, channels, max_channels, style_dim,
            image_channels, output_scale, margin_size, first_cutoff,
            first_stopband, last_stopband_rel, kernel_size)

    def forward(self, z, truncation_psi=1.):
        w = self.map(z, truncation_psi)
        image = self.synthesis(w)
        return image

def binomial_filter(filter_size):
    '''return binomial filter from size.'''
    def c(n,k):
        if(k<=0 or n<=k): return 1
        else: return c(n-1, k-1) + c(n-1, k)
    return [c(filter_size-1, j) for j in range(filter_size)]

class ConvAct(nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel_size, bias=True,
        down=1, filter_size=4, act_name='linear', gain=1., act_gain=None
    ) -> None:
        super().__init__()
        self.down = down
        self.act_name = act_name
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.scale = gain / (self.weight[0].numel() ** 0.5)
        self.act_gain = bias_act.activation_funcs[act_name].def_gain if act_gain is None else act_gain
        if down > 1:
            filter = torch.tensor(binomial_filter(filter_size), dtype=torch.float32)
            kernel = torch.outer(filter, filter)
            kernel /= kernel.sum()
            self.register_buffer('down_filter', kernel)
        else:
            self.down_filter = None

    def forward(self, x):
        weight = self.weight * self.scale
        x = conv2d_resample.conv2d_resample(
            x, weight.to(x.dtype), self.down_filter,
            1, self.down, self.padding)
        b = self.bias.to(x.dtype) if self.bias is not None else self.bias
        x = bias_act.bias_act(x, b, act=self.act_name, gain=self.act_gain)
        return x

class ResBlock(nn.Module):
    def __init__(self,
        in_channels, out_channels, filter_size=4,
        act_name='lrelu', gain=1.
    ) -> None:
        super().__init__()

        self.conv1 = ConvAct(
            in_channels, out_channels, 3, True, 1,
            filter_size, act_name, gain)
        self.conv2 = ConvAct(
            out_channels, out_channels, 3, True, 2,
            filter_size, act_name, gain, 0.5**0.5)
        self.skip = ConvAct(
            in_channels, out_channels, 1, False, 2,
            filter_size, 'linear', gain, 0.5**0.5)

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
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

        y = x.reshape(G, -1, F, c, H, W)
        y = y - y.mean(dim=0)
        y = y.square().mean(dim=0)
        y = (y + 1e-8).sqrt()
        y = y.mean(dim=[2,3,4])
        y = y.reshape(-1, F, 1, 1)
        y = y.repeat(G, 1, H, W)
        x = torch.cat([x, y], dim=1)
        return x

class DiscEpilogue(nn.Module):
    def __init__(self,
        mbsd_group_size, mbsd_channels,
        channels, bottom, act_name='lrelu', gain=1.
    ) -> None:
        super().__init__()
        self.epilogue = nn.Sequential(
            MinibatchStdDev(mbsd_group_size, mbsd_channels),
            ConvAct(
                channels+mbsd_channels, channels, 3, True, 1,
                None, act_name, gain),
            nn.Flatten(),
            Linear(channels*bottom**2, channels, True, act_name, gain),
            Linear(channels, 1, True, 'linear', gain))
    def forward(self, x):
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
        self.from_rgb = ConvAct(
            in_channels, ochannels, 1,
            True, 1, None, act_name, gain)

        resblocks = []
        for _ in range(num_downs):
            channels *= 2
            ichannels, ochannels = ochannels, min(max_channels, channels)
            resblocks.append(
                ResBlock(ichannels, ochannels, filter_size, act_name, gain))
        self.resblocks = nn.Sequential(*resblocks)
        self.epilogue = DiscEpilogue(
            mbsd_group_size, mbsd_channels, ochannels,
            bottom, act_name, gain)

    def forward(self, x):
        x = self.from_rgb(x)
        x = self.resblocks(x)
        x = self.epilogue(x)
        return x
