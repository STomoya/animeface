
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_activation(name, inplace=True):
    if name == 'lrelu': return nn.LeakyReLU(0.2, inplace=inplace)
    if name == 'relu': return nn.ReLU(inplace)
    if name == 'tanh': return nn.Tanh()
    if name == 'sigmoid': return nn.Sigmoid()

def get_normalization(name, channels):
    if name == 'bn': return nn.BatchNorm2d(channels)
    if name == 'in': return nn.InstanceNorm2d(channels)

SN = nn.utils.spectral_norm
def _support_sn(sn, layer):
    if sn: return SN(layer)
    return layer

def Conv2d(sn, *args, **kwargs):
    layer = nn.Conv2d(*args, **kwargs)
    return _support_sn(sn, layer)
def ConvTranspose2d(sn, *args, **kwargs):
    layer = nn.ConvTranspose2d(*args, **kwargs)
    return _support_sn(sn, layer)
def Linear(sn, *args, **kwargs):
    layer = nn.Linear(*args, **kwargs)
    return _support_sn(sn, layer)

class Subspace(nn.Module):
    def __init__(self, latent_dim, channels, resolution):
        super().__init__()
        self.U = nn.Parameter(torch.empty(latent_dim, channels, resolution, resolution))
        nn.init.orthogonal_(self.U)
        l_init = [[3.*i for i in range(latent_dim, 0, -1)]]
        self.L = nn.Parameter(torch.tensor(l_init))
        self.mu = nn.Parameter(torch.zeros(1, channels, resolution, resolution))
    
    def forward(self, z):

        # [1N] * [BN] = [BN]
        # [BN] -> [BN111]
        x = (self.L * z)[:, :, None, None, None]
        # [NCHW] -> [1NCHW]
        # [1NCHW] * [BN111] = [BNCHW]
        x = self.U[None, ...] * x
        # sum over latent input dim
        # [BNCHW] -> [BCHW]
        x = x.sum(1)

        x = x + self.mu
        return x

    def gram_schimdt(self, vector):
        '''this doesn't work.
            It stops by OOM.
        '''
        basis = vector[0:1] / vector[0:1].norm()
        for i in range(1, vector.size(0)):
            v = vector[i:i+1]
            w = v - torch.mm(torch.mm(v, basis.T), basis)
            w = w / w.norm()
            basis = torch.cat([basis, w], dim=0)
        return basis

class Layer(nn.Module):
    def __init__(self,
        in_channels, out_channels, latent_dim, resolution,
        sn=True, bias=True, norm_name='in', act_name='lrelu'
    ):
        super().__init__()

        self.subspace = Subspace(latent_dim, in_channels, resolution)
        self.subspace_dconv1 = ConvTranspose2d(sn,
            in_channels, in_channels, 1, 1, bias=bias)
        self.subspace_dconv2 = ConvTranspose2d(sn,
            in_channels, out_channels, 3, 2, 1, bias=bias, output_padding=1)

        self.dconv1 = nn.Sequential(
            get_normalization(norm_name, in_channels),
            get_activation(act_name),
            ConvTranspose2d(sn,
                in_channels, out_channels, 3, 2, 1,
                bias=bias, output_padding=1)
        )
        self.dconv2 = nn.Sequential(
            get_normalization(norm_name, out_channels),
            get_activation(act_name),
            ConvTranspose2d(sn,
                out_channels, out_channels, 3, 1, 1,
                bias=bias)
        )
    
    def forward(self, x, z):
        w = self.subspace(z)

        w_ = self.subspace_dconv1(w)
        x = self.dconv1(x + w_)
        w_ = self.subspace_dconv2(w)
        x = self.dconv2(x + w_)

        return x

class Generator(nn.Module):
    def __init__(self,
        image_size, in_dim=512, z_dim=6, image_channels=3, bottom_width=4,
        channels=32, max_channels=512,
        sn=True, bias=True, norm_name='in', act_name='lrelu'
    ):
        super().__init__()
        num_ups = int(np.log2(image_size) - np.log2(bottom_width))
        resl = bottom_width

        channels = channels * 2 ** num_ups
        ochannels = min(channels, max_channels)
        self.input = ConvTranspose2d(sn,
            in_dim, ochannels, 4, 2, bias=bias)

        layers = []
        for _ in range(num_ups):
            channels = channels // 2
            ichannels, ochannels = ochannels, min(max_channels, channels)
            layers.append(
                Layer(
                    ichannels, ochannels, z_dim, resl,
                    sn, bias, norm_name, act_name)
            )
            resl *= 2
        self.num_layers = len(layers)
        self.layers = nn.ModuleList(layers)
        self.output = nn.Sequential(
            get_activation(act_name),
            nn.Conv2d(ochannels, image_channels, 7, 1, 3, bias=bias),
            get_activation('tanh')
        )

    def forward(self, eps, zs):
        if eps.ndim != 4:
            eps = eps.unsqueeze(-1).unsqueeze(-1)
        x = self.input(eps)
        for layer, z in zip(self.layers, zs):
            x = layer(x, z)
        x = self.output(x)
        return x

def DiscConvBlock(
    in_channels, out_channels,
    sn=True, bias=True, norm_name='in', act_name='lrelu'
):
    return nn.Sequential(
        Conv2d(sn,
            in_channels, in_channels, 3, 1, 1,
            bias=bias),
        get_normalization(norm_name, in_channels),
        get_activation(act_name),
        Conv2d(sn,
            in_channels, out_channels, 3, 2, 1,
            bias=bias),
        get_normalization(norm_name, out_channels),
        get_activation(act_name)
    )

class Discriminator(nn.Module):
    def __init__(self,
        image_size, image_channels=3, bottom_width=4,
        channels=16, max_channels=512,
        sn=True, bias=True, norm_name='in', act_name='lrelu'
    ):
        super().__init__()
        num_downs = int(np.log2(image_size) - np.log2(bottom_width))

        ochannels = min(channels, max_channels)
        self.input = nn.Sequential(
            Conv2d(sn, image_channels, ochannels, 7, 1, 3, bias=bias),
            get_activation(act_name)
        )

        layers = []
        for _ in range(num_downs):
            channels *= 2
            ichannels, ochannels = ochannels, min(max_channels, channels)
            layers.append(
                DiscConvBlock(
                    ichannels, ochannels,
                    sn, bias, norm_name, act_name)
            )
        self.layers = nn.Sequential(*layers)

        self.output = nn.Sequential(
            nn.Flatten(),
            Linear(sn,
                bottom_width**2*ochannels, ochannels, bias=bias),
            get_activation(act_name),
            Linear(sn,
                ochannels, 1, bias=bias)
        )

    def forward(self, x):
        x = self.input(x)
        x = self.layers(x)
        x = self.output(x)
        return x

if __name__=='__main__':
    g = Generator(256)
    d = Discriminator(256)
    x = torch.randn(10, 512)
    z = [torch.randn(10, 6) for _ in range(g.num_layers)]
    out = g(x, z)
    prob = d(out)
    print(out.size(), prob.size())

    @torch.enable_grad()
    def orthogonal_regularizer(model):
        loss = 0
        for name, param in model.named_parameters():
            if 'U' in name:
                flat = param.view(param.size(0), -1)
                sym = torch.mm(flat, flat.T)
                eye = torch.eye(sym.size(-1), device=sym.device)
                loss = loss + (sym - eye).pow(2).sum() * 0.5
        return loss

    print(orthogonal_regularizer(g))