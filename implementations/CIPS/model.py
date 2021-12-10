
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from thirdparty.stylegan3_ops.ops import conv2d_resample
from thirdparty.stylegan3_ops.ops import bias_act

class Linear(nn.Module):
    '''Linear with Equalized Learning Rate'''
    def __init__(self,
        in_features, out_features, bias=True, act_name='lrelu', gain=1.
    ) -> None:
        super().__init__()
        self.act_name = act_name
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.scale  = gain * self.weight[0].numel() ** -0.5

    def forward(self, x):
        weight = self.weight * self.scale
        x = F.linear(x, weight)
        dim = 2 if x.ndim == 3 else 1
        b = self.bias.to(x.dtype) if self.bias is not None else None
        out = bias_act.bias_act(x, b, dim, self.act_name)
        return out

class ModulatedFC(nn.Module):
    '''Modulated FC Layer
    an efficient implementation presented in CIPS3D
    '''
    def __init__(self,
        in_features, style_dim, out_features, demod=True, gain=1.
    ) -> None:
        super().__init__()
        self.demod = demod

        self.affine = Linear(style_dim, in_features)
        self.affine.bias.data.fill_(1.)

        self.weight = nn.Parameter(torch.randn(1, in_features, out_features))
        self.scale = gain * self.weight[0].numel() ** -0.5

    def forward(self, x, style):

        # affine
        style = self.affine(style)

        # modulation
        if style.ndim == 2:
            style = style[:, :, None]
        weight = self.weight * style * self.scale

        # demodulation
        if self.demod:
            d = (weight.pow(2).sum(dim=1, keepdims=True) + 1e-8).rsqrt()
            weight = weight * d

        # linear
        out = torch.bmm(x, weight)
        return out

class StyleLayer(nn.Module):
    '''Style Layer (without noise input)'''
    def __init__(self,
        in_features, style_dim, out_features, act_name='lrelu'
    ) -> None:
        super().__init__()
        self.act_name = act_name
        self.modfc = ModulatedFC(
            in_features, style_dim, out_features, True)
        self.modfc_bias = nn.Parameter(torch.zeros(out_features))
    def forward(self, x, style):
        x = self.modfc(x, style)
        dim = 2 if x.ndim == 3 else 1
        b = self.modfc_bias.to(x.dtype) if self.modfc_bias is not None else None
        x = bias_act.bias_act(x, b, dim, self.act_name)
        return x

class PixelNorm(nn.Module):
    '''pixel normalization'''
    def forward(self, x):
        x = x / x.pow(2).mean(dim=1, keepdim=True).sqrt().add(1e-8)
        return x

class MappingNetwork(nn.Module):
    '''Mapping Network'''
    def __init__(self,
        latent_dim, style_dim, num_layers,
        pixel_norm=True, ema_decay=0.998, gain=1.
    ) -> None:
        super().__init__()
        self.ema_decay = ema_decay

        if pixel_norm: self.norm = PixelNorm()
        else:          self.norm = lambda x: x

        layers = [Linear(latent_dim, style_dim, gain=gain)]
        for _ in range(num_layers-1):
            layers.extend([Linear(style_dim, style_dim, gain=gain)])
        self.map = nn.Sequential(*layers)

        self.register_buffer('w_avg', torch.zeros(style_dim))

    def forward(self, z, truncation_psi=1.):

        # pixel norm
        z = self.norm(z)

        # map
        w = self.map(z)

        # avg w
        if self.training:
            stats = w.detach().to(torch.float32).mean(dim=0)
            self.w_avg.copy_(stats.lerp(self.w_avg, self.ema_decay))

        # truncatation trick
        if truncation_psi != 1:
            w = self.w_avg.lerp(w, truncation_psi)

        return w

class FourierFeatureInput(nn.Module):
    '''Fourier Feature Input'''
    def __init__(self,
        channels, size
    ) -> None:
        super().__init__()
        self.size = size
        self.channels = channels
        self.b = Linear(2, channels, bias=False)

    def forward(self, w):
        B, device = w.size(0), w.device
        coord = F.affine_grid(
            torch.eye(2, 3, device=device).unsqueeze(0),
            [1, 1, self.size, self.size], align_corners=False)
        x = self.b(coord.view(1, -1, 2).repeat(B, 1, 1))
        x = torch.sin(x)
        return x

class ConstantInput(nn.Module):
    '''Constant Input'''
    def __init__(self,
        channels, size
    ) -> None:
        super().__init__()
        self.constant = nn.Parameter(torch.randn(1, size**2, channels))
    def forward(self, w):
        B = w.size(0)
        return self.constant.repeat(B, 1, 1)

class SynthesisInput(nn.Module):
    '''Fourier Feature + Constant Input'''
    def __init__(self,
        channels, size
    ) -> None:
        super().__init__()
        self.ffi = FourierFeatureInput(channels, size)
        self.consti = ConstantInput(channels, size)
    def forward(self, w):
        ffi = self.ffi(w)
        consti = self.consti(w)
        return torch.cat([ffi, consti], dim=-1)

class ToRGB(nn.Module):
    '''ToRGB'''
    def __init__(self,
        in_features, style_dim, image_channels=3
    ) -> None:
        super().__init__()
        self.modfc = ModulatedFC(in_features, style_dim, image_channels, False)

    def forward(self, x, style):
        x = self.modfc(x, style)
        B, N, _ = x.size()
        S = int(N ** 0.5)
        # [BNC] -> [BCHW]
        x = x.permute(0, 2, 1).reshape(B, -1, S, S)
        return x

class Synthesis(nn.Module):
    '''Synthesis Network'''
    def __init__(self,
        image_size=128, image_channels=3, style_dim=512,
        channels=32, max_channels=512, num_layers=14
    ) -> None:
        super().__init__()
        assert num_layers % 2 == 0
        self.image_size = image_size
        self.image_channels = image_channels

        channels = channels * 2 ** num_layers
        ochannels = min(max_channels, channels)

        self.input = SynthesisInput(ochannels, image_size)
        self.input_fc = StyleLayer(ochannels*2, style_dim, ochannels)

        self.style_layers = nn.ModuleList([])
        self.to_rgbs      = nn.ModuleList([])

        for _ in range(num_layers//2):
            channels //= 2
            ichannels, ochannels = ochannels, min(max_channels, channels)
            self.style_layers.append(StyleLayer(ichannels, style_dim, ochannels))
            self.style_layers.append(StyleLayer(ochannels, style_dim, ochannels))
            self.to_rgbs.append(ToRGB(ochannels, style_dim, image_channels))
        self.num_style = num_layers + 1

        self.register_buffer('init_image',
            torch.zeros(1, image_channels, image_size, image_size))

    def forward(self, ws):
        if isinstance(ws, torch.Tensor):
            if ws.ndim == 2:
                ws = ws[:, None, :].repeat(1, self.num_style, 1)
            ws = ws.unbind(1)
        assert len(ws) == self.num_style

        x = self.input(ws[0])
        x = self.input_fc(x, ws[0])

        B = ws[0].size(0)
        image = self.init_image.repeat(B, 1, 1, 1)
        for i, (style_layer, w) in enumerate(zip(self.style_layers, ws[1:])):
            x = style_layer(x, w)
            if i % 2 == 1:
                image = image + self.to_rgbs[i//2](x, w)

        return image

class Generator(nn.Module):
    '''Generator'''
    def __init__(self,
        image_size=128, latent_dim=512, style_dim=512, num_layers=14,
        channels=32, max_channels=512, image_channels=3,
        map_num_layers=4, pixel_norm=True
    ) -> None:
        super().__init__()

        self.mapping = MappingNetwork(
            latent_dim, style_dim, map_num_layers, pixel_norm)
        self.synthesis = Synthesis(
            image_size, image_channels, style_dim,
            channels, max_channels, num_layers)

    def forward(self, x, truncation_psi=1.):
        w = self.mapping(x, truncation_psi)
        image = self.synthesis(w)
        return image


'''StyleGAN2 Discriminator'''

def binomial_filter(filter_size):
    '''return binomial filter from size.'''
    def c(n,k):
        if(k<=0 or n<=k): return 1
        else: return c(n-1, k-1) + c(n-1, k)
    return [c(filter_size-1, j) for j in range(filter_size)]

class ConvAct(nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel_size, bias=True,
        down=1, filter_size=4, act_name='lrelu', gain=1., act_gain=None
    ) -> None:
        super().__init__()
        self.down = down
        self.act_name = act_name
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias   = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.scale  = gain * self.weight[0].numel() ** -0.5
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
        b = self.bias.to(x.dtype) if self.bias is not None else None
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
        image_size, in_channels=3, channels=64, max_channels=512,
        mbsd_group_size=4, mbsd_channels=1,
        bottom=4, filter_size=4, act_name='lrelu', gain=1.
    ) -> None:
        super().__init__()
        num_downs = int(math.log2(image_size) - math.log2(bottom))

        ochannels = channels
        self.from_rgb = ConvAct(
            in_channels, ochannels, 1, True,
            1, None, act_name, gain)

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
