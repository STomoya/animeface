'''
References

AdaIN:
    Xun Huang, Serge Belongie,
    "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization",
    https://arxiv.org/abs/1703.06868
LIN, AdaLIN:
    Junho Kim, Minjae Kim, Hyeonwoo Kang, Kwanghee Lee,
    "U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation",
    https://arxiv.org/abs/1907.10830
PoLIN, AdaPoLIN:
    Bing Li, Yuanlue Zhu, Yitong Wang, Chia-Wen Lin, Bernard Ghanem, Linlin Shen,
    "AniGAN: Style-Guided Generative Adversarial Networks for Unsupervised Anime Face Generation",
    https://arxiv.org/abs/2102.12593
PONO:
    Boyi Li, Felix Wu, Kilian Q. Weinberger, Serge Belongie,
    "Positional Normalization",
    https://arxiv.org/abs/1907.04312
'''

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

'''normalizations'''

class LIN(nn.Module):
    def __init__(self,
        channels, affine=True
    ) -> None:
        super().__init__()
        self._channels = channels
        self._affine = affine

        self.layernorm = nn.GroupNorm(1, channels, affine=False)
        self.instancenorm = nn.InstanceNorm2d(channels, affine=False)

        if affine:
            self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
            self.beta  = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.rho = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        norml = self.layernorm(x)
        normi = self.instancenorm(x)
        out = norml * self.rho + normi * (1 - self.rho)
        if self._affine:
            out = self.gamma * out + self.beta
        return out

    @torch.no_grad()
    def post_step(self):
        self.rho.clamp_(0., 1.)

class PoLIN(nn.Module):
    def __init__(self,
        channels, affine=True
    ) -> None:
        super().__init__()
        self._channels = channels
        self._affine = affine

        self.layernorm = nn.GroupNorm(1, channels, affine=False)
        self.instancenorm = nn.InstanceNorm2d(channels, affine=False)

        if affine:
            self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
            self.beta  = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.pointwise_conv = nn.Conv2d(channels*2, channels, 1, bias=False)

    def forward(self, x):
        norml = self.layernorm(x)
        normi = self.instancenorm(x)
        norm = torch.cat([norml, normi], dim=1)
        out = self.pointwise_conv(norm)
        if self._affine:
            out = self.gamma * out + self.beta
        return out

class PONO(nn.Module):
    def __init__(self,
        size, affine=True, eps=1e-5
    ) -> None:
        super().__init__()
        self._size = size
        self._affine = affine
        self.eps = eps
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, *size))
            self.beta  = nn.Parameter(torch.zeros(1, 1, *size))

    def forward(self, x):
        mean = x.mean(dim=1, keepdims=True)
        std  = x.std(dim=1, keepdims=True) + self.eps
        out = (x - mean) / std
        if self._affine:
            out = self.gamma * out + self.beta
        return out

def get_normalization(name, CHW, affine=True):
    if   name == 'bn': return nn.BatchNorm2d(CHW[0], affine)
    elif name == 'in': return nn.InstanceNorm2d(CHW[0], affine)
    elif name == 'ln': return nn.LayerNorm(CHW, affine)
    elif name == 'lin': return LIN(CHW[0], affine)
    elif name == 'polin': return PoLIN(CHW[0], affine)
    elif name == 'pono': return PONO(CHW[1:], affine)
    raise Exception(f'Normalization: {name}')

class AdaptiveNormalization(nn.Module):
    def __init__(self,
        norm_name, CHW, style_dim
    ) -> None:
        super().__init__()
        self.norm = get_normalization(norm_name, CHW, False)
        self.affine = nn.Linear(style_dim, CHW[0]*2, bias=False)
        self.affine_bias = nn.Parameter(torch.zeros(1, CHW[0]*2))
        self.affine_bias.data[:, :CHW[0]] = 1.

    def forward(self, x, style):
        B = x.size(0)
        norm = self.norm(x)
        style = self.affine(style) + self.affine_bias
        gamma, beta = style.view(B, -1, 1, 1).chunk(2, dim=1)
        return gamma * norm + beta


'''simple reference based I2I model'''

class ContentEncoder(nn.Module):
    def __init__(self,
        in_channels, out_channels, channels, max_channels,
        image_size, bottom, norm_name
    ) -> None:
        super().__init__()

        ochannels = channels
        self.input = nn.Sequential(
            nn.Conv2d(in_channels, ochannels, 1),
            nn.ReLU(True))

        resl = image_size
        layers = []
        while resl > bottom:
            resl = resl // 2
            channels *= 2
            ichannels, ochannels = ochannels, min(max_channels, channels)
            layers.extend([
                nn.Conv2d(ichannels, ochannels, 3, 2, 1),
                get_normalization(norm_name, (ochannels, resl, resl)),
                nn.ReLU(True)])
        self.body = nn.Sequential(*layers)
        self.output = nn.Sequential(
            nn.Conv2d(ochannels, out_channels, 3, 1, 1),
            get_normalization(norm_name, (out_channels, bottom, bottom)),
            nn.ReLU(True))

    def forward(self, x):
        x = self.input(x)
        x = self.body(x)
        x = self.output(x)
        return x

class ResBlock(nn.Module):
    def __init__(self,
        in_channels, out_channels, resolution, norm_name, stride=1
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.norm1 = get_normalization(
            norm_name, (out_channels, resolution, resolution))
        self.act   = nn.ReLU(True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.norm2 = get_normalization(
            norm_name, (out_channels, resolution, resolution))

        if in_channels != out_channels or stride == 2:
            self.skip = nn.Conv2d(
                in_channels, out_channels, 1, stride, bias=False)

    def forward(self, x):
        hid = self.conv1(x)
        hid = self.norm1(hid)
        hid = self.act(hid)
        hid = self.conv2(hid)
        hid = self.norm2(hid)
        if hasattr(self, 'skip'):
            x = self.skip(x)
        return (hid + x) / (2 ** 0.5)

class StyleEncoder(nn.Module):
    def __init__(self,
        in_channels, style_dim, channels, max_channels,
        image_size, bottom, norm_name
    ) -> None:
        super().__init__()

        ochannels = channels
        self.input = nn.Sequential(
            nn.Conv2d(in_channels, ochannels, 1),
            nn.ReLU(True))

        resl = image_size
        layers = []
        while resl > bottom:
            resl = resl // 2
            channels *= 2
            ichannels, ochannels = ochannels, min(max_channels, channels)
            layers.append(ResBlock(
                ichannels, ochannels, resl, norm_name, 2))
        self.body = nn.Sequential(*layers)
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ochannels*bottom**2, style_dim),
            nn.ReLU(True))

    def forward(self, x):
        x = self.input(x)
        x = self.body(x)
        x = self.output(x)
        return x

class ResBlocks(nn.Sequential):
    def __init__(self,
        channels, resolution, norm_name, num_blocks
    ) -> None:
        layers = [
            ResBlock(channels, channels, resolution, norm_name)
            for _ in range(num_blocks)]
        super().__init__(*layers)

class Decoder(nn.Module):
    def __init__(self,
        in_channels, out_channels, style_dim, channels, max_channels,
        image_size, bottom, norm_name
    ) -> None:
        super().__init__()
        up = int(math.log2(image_size)-math.log2(bottom))
        channels = channels * 2 ** up
        ochannels = min(channels, max_channels)

        resl = bottom * 2
        self.up = nn.ModuleList([
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, ochannels, 3, 1, 1),
            AdaptiveNormalization(norm_name, (ochannels, resl, resl), style_dim),
            nn.LeakyReLU(0.2, True)])
        while resl < image_size:
            resl *= 2
            channels = channels // 2
            ichannels, ochannels = ochannels, min(max_channels, channels)
            self.up.extend([
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ichannels, ochannels, 3, 1, 1),
                AdaptiveNormalization(norm_name, (ochannels, resl, resl), style_dim),
                nn.LeakyReLU(0.2, True)])
        self.output = nn.Sequential(
            nn.Conv2d(ochannels, out_channels, 3, 1, 1),
            nn.Tanh())

    def forward(self, x, style):
        for module in self.up:
            if isinstance(module, AdaptiveNormalization):
                x = module(x, style)
            else:
                x = module(x)
        x = self.output(x)
        return x

class Generator(nn.Module):
    def __init__(self,
        norm_name, content_channels=1, style_channels=3,
        content_dim=512, style_dim=512, num_resblocks=4,
        channels=32, max_channels=512, image_size=128, bottom=16
    ) -> None:
        super().__init__()
        self.CE = ContentEncoder(
            content_channels, content_dim, channels, max_channels,
            image_size, bottom, norm_name)
        self.SE = StyleEncoder(
            style_channels, style_dim, channels, max_channels,
            image_size, bottom, norm_name)
        self.resblocks = ResBlocks(
            content_dim, bottom, norm_name, num_resblocks)
        self.D = Decoder(
            content_dim, style_channels, style_dim, channels, max_channels,
            image_size, bottom, norm_name)

    def forward(self, content, reference):
        style = self.SE(reference)
        c_feat = self.CE(content)
        feat = self.resblocks(c_feat)
        img = self.D(feat, style)
        return img

    def post_step(self):
        for module in self.modules():
            if isinstance(module, LIN):
                module.post_step()

class Discriminator(nn.Sequential):
    def __init__(self,
        in_channels, channels, max_channels,
        num_layers=4, image_size=128, norm_name='bn'
    ):
        resl = image_size // 2
        ochannels = channels
        layers = [
            nn.Conv2d(in_channels, channels, 4, 2, 1, padding_mode='reflect'),
            nn.ReLU(True)]
        for _ in range(num_layers-1):
            resl = resl // 2
            channels *= 2
            ichannels, ochannels = ochannels, min(max_channels, channels)
            layers.extend([
                nn.Conv2d(ichannels, ochannels, 4, 2, 1, padding_mode='reflect'),
                get_normalization(norm_name, (ochannels, resl, resl)),
                nn.ReLU(True)])
        layers.extend([
            nn.Conv2d(ochannels, ochannels, 4, 1, 1, padding_mode='reflect'),
            get_normalization(norm_name, (ochannels, resl, resl)),
            nn.ReLU(True),
            nn.Conv2d(ochannels, 1, 4, 1, 1, padding_mode='reflect')])
        super().__init__(*layers)

class MultiScale(nn.Module):
    def __init__(self,
        num_discs, in_channels, channels, max_channels,
        num_layers=4, image_size=128
    ) -> None:
        super().__init__()
        self.discs = nn.ModuleList([
            Discriminator(
                in_channels, channels, max_channels,
                num_layers, image_size, norm_name='bn')
            for _ in range(num_discs)])
        self.downsample = nn.AvgPool2d(2)

    def forward(self, x):
        output = []
        for disc in self.discs:
            output.append(disc(x))
            x = self.downsample(x)
        return output

if __name__=='__main__':
    g = Generator('ln')
    x = torch.randn(3, 1, 128, 128)
    y = torch.randn(3, 3, 128, 128)
    print(g(x, y).size())
