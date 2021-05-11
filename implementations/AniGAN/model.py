
import math
import torch
import torch.nn as nn

def get_normalization(name, channels):
    if name == 'in': return nn.InstanceNorm2d(channels)
    if name == 'bn': return nn.BatchNorm2d(channels)
    raise Exception(f'no normalization as {name}')

def get_activation(name, inplace=True):
    if name == 'lrelu': return nn.LeakyReLU(0.2, inplace=inplace)
    if name == 'relu': return nn.ReLU(inplace)
    if name == 'tanh': return nn.Tanh()
    raise Exception(f'no activation as {name}')

def Conv2d(*args, **kwargs):
    return nn.utils.spectral_norm(nn.Conv2d(*args, **kwargs))
def Linear(*args, **kwargs):
    return nn.utils.spectral_norm(nn.Linear(*args, **kwargs))

class PoLIN_lazy(nn.Module):
    '''lazy construct normalization layers'''
    def __init__(self,
        channels, resolution
    ):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.layer_norm    = nn.LayerNorm((channels, resolution, resolution))
        self.conv = Conv2d(channels*2, channels, 1, bias=False)
    
    def forward(self, x):
        IN = self.instance_norm(x)
        LN = self.layer_norm(x)
        LIN = torch.cat([IN, LN], dim=1)
        x = self.conv(LIN)
        return x

class AdaPoLIN(nn.Module):
    def __init__(self,
        channels, resolution, affine=False, style_dim=None
    ):
        super().__init__()
        self.polin = PoLIN_lazy(channels, resolution)
        if affine:
            assert style_dim is not None, 'if affine transform style in PoLIN, provide "style_dim"'
            self.affine = nn.Linear(style_dim, channels*2, bias=False)
            self.affine_bias = nn.Parameter(torch.zeros(channels*2))
            self.affine_bias.data[:channels] = 1.
    
    def forward(self, x, style):
        if hasattr(self, 'affine'):
            style = self.affine(style) + self.affine_bias
        
        style = style.unsqueeze(-1).unsqueeze(-1)
        gamma, beta = style.chunk(2, dim=1)
        norm = self.polin(x)
        return gamma * norm + beta

class AST(nn.Module):
    def __init__(self,
        channels, resolution, num_convs=5, affine=False, style_dim=None,
        bias=True, act_name='lrelu'
    ):
        super().__init__()
        layers = []
        for _ in range(num_convs):
            layers.extend([
                Conv2d(channels, channels, 3, 1, 1, bias=bias),
                AdaPoLIN(channels, resolution, affine, style_dim),
                get_activation(act_name)
            ])
        self.ast = nn.ModuleList(layers)

    def forward(self, x, style):
        for module in self.ast:
            if isinstance(module, AdaPoLIN):
                x = module(x, style)
            else:
                x = module(x)
        return x

class FST(nn.Module):
    def __init__(self,
        in_channels, out_channels, resolution, affine=False, style_dim=None,
        bias=True, act_name='lrelu'
    ):
        super().__init__()
        self.fst = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias),
            PoLIN_lazy(out_channels, resolution),
            get_activation(act_name),
            Conv2d(out_channels, out_channels, 3, 1, 1, bias=bias),
            AdaPoLIN(out_channels, resolution, affine, style_dim),
            get_activation(act_name)
        ])
    def forward(self, x, style):
        for module in self.fst:
            if isinstance(module, AdaPoLIN):
                x = module(x, style)
            else:
                x = module(x)
        return x

class ConetentEncoder(nn.Module):
    def __init__(self,
        image_size, bottom_width, in_channels, channels,
        bias=True, norm_name='in', act_name='lrelu'
    ):
        super().__init__()
        num_downs = int(math.log2(image_size) - math.log2(bottom_width))

        self.input = nn.Sequential(
            Conv2d(in_channels, channels, 7, 1, 3, bias=bias),
            get_activation(act_name)
        )

        layers = []
        for _ in range(num_downs):
            layers.extend([
                Conv2d(channels, channels*2, 3, 2, 1, bias=bias),
                get_normalization(norm_name, channels*2),
                get_activation(act_name)
            ])
            channels *= 2
        self.out_channels = channels
        self.extract = nn.Sequential(*layers)
        self.output = Conv2d(channels, channels, 3, 1, 1, bias=bias)
    
    def forward(self, x):
        x = self.input(x)
        x = self.extract(x)
        x = self.output(x)
        return x

class StyleEncoder(ConetentEncoder):
    def __init__(self,
        image_size, bottom_width, in_channels, channels, affine=False, style_dim=None,
        bias=True, norm_name='in', act_name='lrelu'
    ):
        super().__init__(
            image_size, bottom_width, in_channels,
            channels, bias, norm_name, act_name)

        self.output = nn.Sequential(
            nn.Flatten(),
            Linear(self.out_channels*bottom_width**2, self.out_channels*2, bias=bias),
            get_activation(act_name),
            Linear(self.out_channels*2, style_dim if affine else self.out_channels*2, bias=bias)
        )
    
    def forward(self, x):
        x = self.input(x)
        x = self.extract(x)
        x = self.output(x)
        return x

class Decoder(nn.Module):
    def __init__(self,
        image_size, bottom_width, out_channels, channels,
        affine=False, style_dim=None, bias=True, act_name='lrelu'
    ):
        super().__init__()
        num_ups = int(math.log2(image_size) - math.log2(bottom_width))
        ichannels, ochannels = channels, channels
        self.ast = AST(channels, bottom_width, affine=affine, style_dim=style_dim, bias=bias, act_name=act_name)
        fsts = []
        resl = bottom_width
        for _ in range(num_ups):
            if affine:
                channels = channels // 2
                ichannels, ochannels = ochannels, channels
            resl *= 2
            fsts.append(
                FST(
                    ichannels, ochannels, resl, affine, style_dim,
                    bias, act_name))
        self.fsts = nn.ModuleList(fsts)
        self.output = nn.Sequential(
            Conv2d(ochannels, out_channels, 7, 1, 3, bias=bias),
            get_activation('tanh')
        )
    
    def forward(self, x, style):
        x = self.ast(x, style)
        for fst in self.fsts:
            x = fst(x, style)
        x = self.output(x)
        return x


class Generator(nn.Module):
    def __init__(self,
        image_size, in_channels=3, out_channels=3, bottom_width=8,
        channels=32, affine=False, style_dim=None,
        bias=True, norm_name='in', act_name='lrelu'
    ):
        super().__init__()

        self.contect_encoder = ConetentEncoder(
            image_size, bottom_width, in_channels,
            channels, bias, norm_name, act_name)
        self.style_encoder   = StyleEncoder(
            image_size, bottom_width, in_channels,
            channels, affine, style_dim, bias, norm_name, act_name)
        self.decoder = Decoder(
            image_size, bottom_width, out_channels,
            self.contect_encoder.out_channels,
            affine, style_dim, bias, act_name)
    
    def forward(self, x, ref):
        x = self.contect_encoder(x)
        style = self.style_encoder(ref)
        out = self.decoder(x, style)
        return out

class DiscHead(nn.Module):
    def __init__(self,
        branch_width, channels, max_channels=512, bias=True, norm_name='in', act_name='lrelu'
    ):
        super().__init__()

        layers = []
        ochannels = channels
        for _ in range(int(math.log2(branch_width)-1)):
            channels *= 2
            ichannels, ochannels = ochannels, min(max_channels, channels)
            layers.append(
                nn.Sequential(
                    Conv2d(ichannels, ochannels, 3, 2, 1, bias=bias),
                    get_normalization(norm_name, ochannels),
                    get_activation(act_name)))
        self.disc_head = nn.ModuleList(layers)
        self.output = Conv2d(ochannels, 1, 3, 2, 1, bias=bias)

    def forward(self, x):
        feats = []
        for module in self.disc_head:
            x = module(x)
            feats.append(x)
        
        return self.output(x), feats

class Discriminator(nn.Module):
    def __init__(self,
        image_size, branch_width, in_channels=3, channels=32, max_channels=512,
        bias=True, norm_name='in', act_name='lrelu'
    ):
        super().__init__()
        shallow_downs = int(math.log2(image_size)-math.log2(branch_width))

        self.input = nn.Sequential(
            Conv2d(in_channels, channels, 7, 1, 3, bias=bias),
            get_activation(act_name)
        )

        shared = []
        ochannels = channels
        for _ in range(shallow_downs):
            channels *= 2
            ichannels, ochannels = ochannels, min(max_channels, channels)
            shared.append(
                nn.Sequential(
                    Conv2d(ichannels, ochannels, 3, 2, 1, bias=bias),
                    get_normalization(norm_name, ochannels),
                    get_activation(act_name)))
        self.shared = nn.ModuleList(shared)
        self.A_head = DiscHead(branch_width, channels, max_channels, bias, norm_name, act_name)
        self.B_head = DiscHead(branch_width, channels, max_channels, bias, norm_name, act_name)

    def forward(self, x, return_features=True):
        x = self.input(x)
        shallow_feats = []
        for module in self.shared:
            x = module(x)
            shallow_feats.append(x)
        
        a_prob, a_feats = self.A_head(x)
        b_prob, b_feats = self.B_head(x)

        if return_features:
            return a_prob, b_prob, shallow_feats, a_feats, b_feats
        
        return a_prob, b_prob

if __name__=='__main__':
    x = torch.randn(10, 3, 128, 128)
    y = torch.randn(10, 3, 128, 128)
    g = Generator(128)
    image = g(x, y)
    d = Discriminator(128, 32)
    a_prob, b_prob, shared_feats, a_feats, b_feats = d(image)
    print(image.size())
    print(a_prob.size(), b_prob.size())
    print(len(shared_feats), len(a_feats), len(b_feats))
    print(g)