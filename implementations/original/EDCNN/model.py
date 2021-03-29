
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_activation(name, inplace=True):
    if name == 'relu': return nn.ReLU(inplace)
    if name == 'lrelu': return nn.LeakyReLU(0.2, inplace=inplace)
    if name == 'sigmoid': return nn.Sigmoid()
    if name == 'tanh': return nn.Tanh()

def get_normalization(name, channels):
    if name == 'in': return nn.InstanceNorm2d(channels)
    if name == 'bn': return nn.BatchNorm2d(channels)

class SobelConv2d(nn.Module):
    def __init__(self,
        in_channels, out_channels,
        stride=1, bias=True, learnable=True
    ):
        super().__init__()
        assert out_channels % 4 == 0, 'out channels must be times of 4'

        scale = torch.ones(out_channels, in_channels, 1, 1)
        if learnable: self.scale = nn.Parameter(scale)
        else: self.register_buffer('scale', scale)
        
        if bias: self.bias = nn.Parameter(torch.zeros(out_channels))
        else: self.bias = None

        groups = out_channels // 4
        _sobel_kernel = torch.tensor([[[[-1., -2., -1.],
                                        [ 0.,  0.,  0.],
                                        [ 1.,  2.,  1.]]],
                                      [[[-1.,  0.,  1.],
                                        [-2.,  0.,  2.],
                                        [-1.,  0.,  1.]]],
                                      [[[-2., -1.,  0.],
                                        [-1.,  0.,  1.],
                                        [ 0.,  1.,  2.]]],
                                      [[[ 0.,  1.,  2.],
                                        [-1.,  0.,  1.],
                                        [-2., -1.,  0.]]]])
        kernel = _sobel_kernel.repeat(groups, in_channels, 1, 1)
        self.register_buffer('kernel', kernel)
    def forward(self, x):
        weight = self.scale * self.kernel
        out = F.conv2d(
            x, weight, self.bias, padding=1)
        return torch.cat([x, out], dim=1)

def _support_sn(sn, layer):
    if sn: return nn.utils.spectral_norm(layer)
    return layer

def Conv2d(sn, *args, **kwargs):
    layer = nn.Conv2d(*args, **kwargs)
    return _support_sn(sn, layer)
def Linear(sn, *args, **kwargs):
    layer = nn.Linear(*args, **kwargs)
    return _support_sn(sn, layer)
def ConvTranspose2d(sn, *args, **kwargs):
    layer = nn.ConvTranspose2d(*args, **kwargs)
    return _support_sn(sn, layer)

class ResBlock(nn.Module):
    def __init__(self,
        in_channels, out_channels, stride=1,
        sn=True, bias=True, norm_name='in', act_name='lrelu'
    ):
        super().__init__()

        self.conv = nn.Sequential(
            get_normalization(norm_name, in_channels),
            get_activation(act_name),
            Conv2d(sn, in_channels, out_channels, 3, stride, 1, bias=bias),
            get_normalization(norm_name, out_channels),
            get_activation(act_name),
            Conv2d(sn, out_channels, out_channels, 3, 1, 1, bias=bias)
        )
        if stride > 1 or in_channels != out_channels:
            self.skip = Conv2d(sn, in_channels, out_channels, 1, stride, bias=bias)
        else: self.skip = None
    
    def forward(self, x):
        h = self.conv(x)
        if self.skip is not None:
            x = self.skip(x)
        return (h + x) / np.sqrt(2)

class StyleEncoder(nn.Module):
    def __init__(self,
        in_channels, style_dim, image_size, bottom_width=8,
        channels=32, blocks_per_resl=1,
        sn=True, bias=True, norm_name='in', act_name='lrelu'
    ):
        super().__init__()
        num_downs = int(np.log2(image_size)-np.log2(bottom_width))

        ochannels = channels
        self.input = nn.Sequential(
            Conv2d(sn, in_channels, ochannels, 7, 1, 3, bias=bias)
        )

        layers = []
        for _ in range(num_downs):
            channels *= 2
            ichannels, ochannels = ochannels, channels
            for i in range(blocks_per_resl):
                stride = 2 if i==0 else 1
                layers.append(
                    ResBlock(
                        ichannels, ochannels, stride,
                        sn, bias, norm_name, act_name
                    )
                )
                ichannels = ochannels
        self.res = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.output = nn.Sequential(
            nn.Flatten(),
            Linear(sn, ochannels, style_dim, bias=bias)
        )
    def forward(self, ref):
        x = self.input(ref)
        x = self.res(x)
        x = self.avgpool(x)
        style = self.output(x)
        return style

def ConvBlock(
    in_channels, out_channels, stride,
    sn=True, bias=True, norm_name='in', act_name='lrelu'
):
    return nn.Sequential(
        Conv2d(sn,
            in_channels, out_channels,
            3, stride, 1, bias=bias),
        get_normalization(norm_name, out_channels),
        get_activation(act_name)
    )

class Encoder(nn.Module):
    def __init__(self,
        in_channels, image_size, bottom_width=8, channels=32,
        sobel=True, learnable=True, conv_per_resl=1,
        sn=True, bias=True, norm_name='in', act_name='lrelu'
    ):
        super().__init__()
        num_downs = int(np.log2(image_size)-np.log2(bottom_width))

        input = []
        if sobel:
            input = [SobelConv2d(in_channels, channels, 1, bias, learnable)]
            ichannels, ochannels = channels+1, channels
        else:
            ichannels, ochannels = in_channels, channels
        input.extend([
            Conv2d(
                sn, ichannels, ochannels,
                7, 1, 3, bias=bias),
            get_activation(act_name)
        ])
        self.input = nn.Sequential(*input)

        layers = []
        for _ in range(num_downs):
            channels *= 2
            ichannels, ochannels = ochannels, channels
            for i in range(conv_per_resl):
                stride = 2 if i==0 else 1
                layers.append(
                    ConvBlock(
                        ichannels, ochannels, stride,
                        sn, bias, norm_name, act_name
                    )
                )
                ichannels = ochannels
        self.convs = nn.ModuleList(layers)
        self.out_channels = ochannels
    
    def forward(self, x):
        x = self.input(x)
        feats = [x]
        for module in self.convs:
            x = module(x)
            feats.append(x)
        return x, feats

class AdaIN(nn.Module):
    def __init__(self,
        channels, style_dim, affine=True,
        sn=True
    ):
        super().__init__()
        if affine:
            self.affine = Linear(sn, style_dim, channels*2, bias=False)
            self.affine_bias = nn.Parameter(torch.zeros(channels*2))
            self.affine_bias.data[:channels] = 1
        else: self.affine = None
        self.norm = get_normalization('in', channels)
    
    def forward(self, x, style):
        if self.affine is not None:
            style = self.affine(style) + self.affine_bias
        else:
            assert x.size(1)*2 == style.size(1), 'if not affine in AdaIN, style should be x.size(1)*2'
        if style.dim() != 4:
            style = style.view(style.size(0), -1, 1, 1)
        scale, bias = style.chunk(2, dim=1)
        norm = self.norm(x)
        return scale * norm + bias

class ConvAdaINBlock(nn.Module):
    def __init__(self,
        in_channels, out_channels, style_dim, stride,
        affine=True,
        sn=True, bias=True, act_name='lrelu'
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            Conv2d(sn, in_channels, out_channels, 3, stride, 1, bias=bias),
            AdaIN(out_channels, style_dim, affine, sn),
            get_activation(act_name)
        ])
    
    def forward(self, x, style):
        for module in self.layers:
            if isinstance(module, AdaIN):
                x = module(x, style)
            else:
                x = module(x)
        return x

class Decoder(nn.Module):
    def __init__(self,
        image_size, out_channels, style_dim, bottom_width=8, channels=32,
        conv_per_resl=1,
        sn=True, bias=True, norm_name='in', act_name='lrelu'
    ):
        super().__init__()
        num_downs = int(np.log2(image_size)-np.log2(bottom_width))
        channels = channels * 2 ** num_downs
        ochannels = channels

        layers = []
        for _ in range(num_downs):
            channels = channels // 2
            for i in range(conv_per_resl):
                _temp = []
                if i==0:
                    ichannels, ochannels = ochannels*2, channels
                elif i==1:
                    ichannels = ichannels//2 + ochannels
                    
                _temp.append(
                    ConvAdaINBlock(
                        ichannels, ochannels, style_dim, 1,
                        True, sn, bias, act_name
                    )
                )

                if i==conv_per_resl-1:
                    _temp.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
                
                layers.extend(_temp)
        self.convs = nn.ModuleList(layers)
        self.output = Conv2d(
            sn, ochannels, out_channels, 7, 1, 3, bias=bias
        )
    def forward(self, x, feats, style):
        feats = feats[::-1]
        index = 0
        for module in self.convs:
            if isinstance(module, nn.Upsample):
                x = module(x)
            else:
                x = module(torch.cat([x, feats[index]], dim=1), style)
                index += 1
        x = self.output(x)
        return x

def ResBlocks(
    channels, num_blocks,
    sn=True, bias=True, norm_name='in', act_name='lrelu'
):
    blocks = []
    for _ in range(num_blocks):
        blocks.append(
            ResBlock(
                channels, channels, 1,
                sn, bias, norm_name, act_name
            )
        )
    return nn.Sequential(*blocks)

class Generator(nn.Module):
    def __init__(self,
        image_size, in_channels=1, ref_channels=3,
        channels=32, style_dim=128, bottom_width=8,
        se_blocks_per_resl=1, num_res_blocks=5, 
        sobel=True, learnable_sobel=True, e_conv_per_resl=2,
        sn=True, bias=True, norm_name='in', act_name='lrelu'
    ):
        super().__init__()
        self.style_encoder = StyleEncoder(
            ref_channels, style_dim, image_size,
            bottom_width, channels, se_blocks_per_resl,
            sn, bias, norm_name, act_name
        )
        self.encoder = Encoder(
            in_channels, image_size, bottom_width,
            channels, sobel, learnable_sobel, e_conv_per_resl,
            sn, bias, norm_name, act_name
        )
        self.resblocks = ResBlocks(
            self.encoder.out_channels, num_res_blocks,
            sn, bias, norm_name, act_name
        )
        self.decoder = Decoder(
            image_size, ref_channels, style_dim,
            bottom_width, channels, e_conv_per_resl,
            sn, bias, norm_name, act_name
        )
    def forward(self, x, ref):
        style = self.style_encoder(ref)
        x, feats = self.encoder(x)
        x = self.resblocks(x)
        out = self.decoder(x, feats, style)
        return out

class Discriminator(nn.Module):
    def __init__(self,
        image_size, in_channels=3, num_layers=3, channels=32,
        sn=True, bias=True, norm_name='in', act_name='lrelu'
    ):
        super().__init__()

        ochannels = channels
        layers = [
            nn.Sequential(
                Conv2d(sn, in_channels, ochannels, 4, 2, bias=bias),
                get_activation(act_name)
            )
        ]
        for _ in range(num_layers):
            layers.append(
                nn.Sequential(
                    Conv2d(sn, channels, channels*2, 4, 2, bias=bias),
                    get_normalization(norm_name, channels*2),
                    get_activation(act_name)
                )
            )
            channels *= 2
        layers.append(
            Conv2d(sn, channels, 1, 4, bias=bias)
        )
        self.disc = nn.ModuleList(layers)
    def forward(self, x):
        out = []
        for module in self.disc:
            x = module(x)
            out.append(x)
        return out[-1], out[:-1]

def init_weight_N002(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        m.weight.data.normal_(0, 0.02)
        if not m.bias == None:
            m.bias.data.fill_(0.)
    elif isinstance(m, nn.BatchNorm2d):
        if not m.weight == None:
            m.weight.data.fill_(1.)
        if not m.bias == None:
            m.bias.data.fill_(0.)

def init_weight_xavier(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        if not m.bias == None:
            m.bias.data.fill_(0.)
    elif isinstance(m, nn.BatchNorm2d):
        if not m.weight == None:
            m.weight.data.fill_(1.)
        if not m.bias == None:
            m.bias.data.fill_(0.)

def init_weight_kaiming(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        if not m.bias == None:
            m.bias.data.fill_(0.)
    elif isinstance(m, nn.BatchNorm2d):
        if not m.weight == None:
            m.weight.data.sill_(1.)
        if not m.bias == None:
            m.bias.data.fill_(0.)

if __name__=='__main__':
    image_size = 256
    x = torch.randn(10, 1, image_size, image_size)
    y = torch.randn(10, 3, image_size, image_size)
    g = Generator(image_size, sobel=False)
    d = Discriminator(image_size)
    image = g(x, y)
    prob, _ = d(image)
    print(image.size(), prob.size())