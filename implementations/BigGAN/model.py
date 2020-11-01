

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation(name, **kwargs):
    if name == 'relu':
        return nn.ReLU(**kwargs)
    elif name == 'lrelu':
        return nn.LeakyReLU(**kwargs)

'''alias for spectral normalization'''
SN = nn.utils.spectral_norm

'''global sum pooling'''
class GlobalSumPool2d(nn.Module):
    def forward(self, x):
        return torch.sum(x, [2, 3])

'''layer getters'''
def Linear(in_features, out_features, use_sn=True, sn_eps=1.e-12, **kwargs):
    layer = nn.Linear(in_features, out_features, **kwargs)
    if use_sn:
        return SN(layer, eps=sn_eps)
    return layer

def Conv2d(in_channels, out_channels, kernel_size, use_sn=True, sn_eps=1.e-12, **kwargs):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
    if use_sn:
        return SN(layer, eps=sn_eps)
    return layer

'''conditional normalizations'''
class ConditionalNorm2d(nn.Module):
    def __init__(self,
        z_dim, out_channels, norm='bn', use_sn=True, mlp=False, **kwargs
    ):
        '''
        mlp option is from codes for U-net based discriminator for GANs
        original is a single linear
        '''
        super().__init__()
        
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels, affine=False, **kwargs)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels, affine=False, **kwargs)
        elif norm == 'gn':
            self.norm = nn.GroupNorm(16, out_channels, affine=False, **kwargs)

        if mlp:
            self.gain = nn.Sequential(
                Linear(z_dim, z_dim, use_sn, bias=True),
                nn.ReLU(),
                Linear(z_dim, out_channels, use_sn, bias=False)
            )
            self.bias = nn.Sequential(
                Linear(z_dim, z_dim, use_sn, bias=True),
                nn.ReLU(),
                Linear(z_dim, out_channels, use_sn, bias=False)
            )
        else:
            self.gain = Linear(z_dim, out_channels, use_sn)
            self.bias = Linear(z_dim, out_channels, use_sn)
    
    def forward(self, x, y):
        gain = (1 - self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        
        norm = self.norm(x)

        out = norm * gain + bias

        return out

'''blocks'''

class GBlock(nn.Module):
    '''Generator block'''
    def __init__(self,
        in_channels, out_channels, z_dim, use_sn, activation, upsample,
        sn_eps=1.e-12, norm='bn', mlp=False, cn_eps=1e-5, channel_ratio=1
    ):
        super().__init__()

        mid_channels = in_channels // channel_ratio
        self.main_top = nn.ModuleList([
            ConditionalNorm2d(z_dim, in_channels, norm, use_sn, mlp, eps=cn_eps),
            activation
        ])
        self.upsample = upsample
        self.main_tail = nn.ModuleList([
            Conv2d(in_channels, mid_channels, 3, use_sn, sn_eps, padding=1),
            ConditionalNorm2d(z_dim, mid_channels, norm, use_sn, mlp, eps=cn_eps),
            Conv2d(mid_channels, out_channels, 3, use_sn, sn_eps, padding=1)
        ])

        if upsample or not in_channels == out_channels:
            self.skip_connect = Conv2d(in_channels, out_channels, 1, use_sn, sn_eps)
        else:
            self.skip_connect = None

    def forward(self, x, y):
        h = x
        for module in self.main_top:
            if isinstance(module, ConditionalNorm2d):
                h = module(h, y)
            else:
                h = module(h)
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        if self.skip_connect:
            x = self.skip_connect(x)
        for module in self.main_tail:
            if isinstance(module, ConditionalNorm2d):
                h = module(h, y)
            else:
                h = module(h)
        return h + x

class GBlockdeep(nn.Module):
    '''Generator block for BigGAN-deep'''
    def __init__(self,
        in_channels, out_channels, z_dim, use_sn, activation, upsample,
        sn_eps=1.e-12, norm='bn', mlp=False, cn_eps=1e-5, channel_ratio=4
    ):
        super().__init__()

        mid_channels = in_channels // channel_ratio
        self.main_top = nn.ModuleList([
            ConditionalNorm2d(z_dim, in_channels, norm, use_sn, mlp, eps=cn_eps),
            activation,
            Conv2d(in_channels, mid_channels, 1, use_sn, sn_eps),
            ConditionalNorm2d(z_dim, mid_channels, norm, use_sn, mlp, eps=cn_eps),
            activation
        ])
        self.upsample = upsample
        self.main_tail = nn.ModuleList([
            Conv2d(mid_channels, mid_channels, 3, use_sn, sn_eps, padding=1),
            ConditionalNorm2d(z_dim, mid_channels, norm, use_sn, mlp, eps=cn_eps),
            activation,
            Conv2d(mid_channels, mid_channels, 3, use_sn, sn_eps, padding=1),
            ConditionalNorm2d(z_dim, mid_channels, norm, use_sn, mlp, eps=cn_eps),
            activation,
            Conv2d(mid_channels, out_channels, 1, use_sn, sn_eps)
        ])
        
        self.drop_channel = not in_channels == out_channels
        self.out_channels = out_channels

    def forward(self, x, y):
        h = x
        for module in self.main_top:
            if isinstance(module, ConditionalNorm2d):
                h = module(h, y)
            else:
                h = module(h)
        if self.drop_channel:
            x = x[:, :self.out_channels]
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        for module in self.main_tail:
            if isinstance(module, ConditionalNorm2d):
                h = module(h, y)
            else:
                h = module(h)
        return h + x


def gblock(deep=False, *args, **kwargs):
    if not deep:
        return GBlock(*args, **kwargs)
    else:
        return GBlockdeep(*args, **kwargs)


class DBlock(nn.Module):
    '''Discriminator block'''
    def __init__(self,
        in_channels, out_channels, use_sn, activation, downsample,
        sn_eps=1.e-12, wide=True
    ):
        super().__init__()

        mid_channels = out_channels if wide else in_channels

        self.main = nn.Sequential(
            activation,
            Conv2d(in_channels, mid_channels, 3, use_sn, sn_eps, padding=1),
            activation,
            Conv2d(mid_channels, out_channels, 3, use_sn, sn_eps, padding=1)
        )
        self.downsample = downsample
        if downsample or not in_channels == out_channels:
            self.skip_connect = Conv2d(in_channels, out_channels, 1, use_sn, sn_eps)
        else:
            self.skip_connect = None
    
    def forward(self, x):
        h = self.main(x)
        if self.skip_connect:
            x = self.skip_connect(x)
        if self.downsample:
            h = self.downsample(h)
            x = self.downsample(x)
        return h + x

class DBlockdeep(nn.Module):
    '''Discriminator block for BigGAN-deep'''
    def __init__(self,
        in_channels, out_channels, use_sn, activation, downsample,
        sn_eps=1.e-12, wide=True, channel_ratio=4
    ):
        super().__init__()

        mid_channels = out_channels // channel_ratio

        self.main = nn.Sequential(
            activation,
            Conv2d(in_channels, mid_channels, 1, use_sn, sn_eps),
            activation,
            Conv2d(mid_channels, mid_channels, 3, use_sn, sn_eps, padding=1),
            activation,
            Conv2d(mid_channels, mid_channels, 3, use_sn, sn_eps, padding=1),
            activation
        )

        self.downsample = downsample

        self.tail = Conv2d(mid_channels, out_channels, 1, use_sn, sn_eps)

        if not in_channels == out_channels:
            self.skip_connect = Conv2d(in_channels, out_channels-in_channels, 1, use_sn, sn_eps)
        else:
            self.skip_connect = None

    def forward(self, x):
        
        h = self.main(x)
        if self.downsample:
            h = self.downsample(h)
            x = self.downsample(x)
        h = self.tail(h)
        if self.skip_connect:
            x = torch.cat([x, self.skip_connect(x)], 1)
        return h + x


def dblock(deep=False, *args, **kwargs):
    if not deep:
        return DBlock(*args, **kwargs)
    else:
        return DBlockdeep(*args, **kwargs)


class SelfAttention(nn.Module):
    '''
    non-local layer
    self attention from SAGAN
    '''
    def __init__(self,
        channels, use_sn, sn_eps=1e-12
    ):
        super().__init__()
        self.theta = Conv2d(channels, channels // 8, 1, use_sn, sn_eps, bias=False)
        self.phi   = Conv2d(channels, channels // 8, 1, use_sn, sn_eps, bias=False)
        self.g     = Conv2d(channels, channels // 2, 1, use_sn, sn_eps, bias=False)
        self.o     = Conv2d(channels // 2, channels, 1, use_sn, sn_eps, bias=False)
        self.gamma = nn.Parameter(torch.Tensor([0.]), requires_grad=True)

        self.max_pool = nn.MaxPool2d(2)
        self.softmax  = nn.Softmax(-1)

    def forward(self, x, y=None):
        B, C, H, W = x.size()
        theta = self.theta(x)
        phi   = self.max_pool(self.phi(x))
        g     = self.max_pool(self.g(x))

        theta = theta.view(-1, C//8, H*W)
        phi   = phi.view(-1, C//8, H*W//4)
        g     = g.view(-1, C//2, H*W//4)

        beta  = self.softmax(torch.bmm(theta.transpose(1, 2), phi))
        out   = self.o(torch.bmm(g, beta.transpose(1, 2)).view(B, C//2, H, W))
        return  self.gamma * out + x

class LambdaLayer(nn.Module):
    '''
    lambda layer from LambdaNetworks
    using this instead of self-attention is experimental
    '''
    def __init__(self,
        channels, use_sn, sn_eps=1e-12, n=64*64, keys=16, heads=4, u=1
    ):
        super().__init__()
        self.keys  = keys
        self.heads = heads
        self.us    = u
        self.vs    = channels // heads

        self.q = Conv2d(channels, keys * heads, 1, use_sn, sn_eps, bias=False)
        self.k = Conv2d(channels, keys * u,     1, use_sn, sn_eps, bias=False)
        self.v = Conv2d(channels, self.vs * u,  1, use_sn, sn_eps, bias=False)

        self.norm_q = nn.BatchNorm2d(keys * heads)
        self.norm_v = nn.BatchNorm2d(self.vs * u)

        nn.init.xavier_normal_(self.norm_q.weight)
        self.norm_q.bias.fill_(0.)
        nn.init.xavier_normal_(self.norm_v.weight)
        self.norm_v.bias.fill_(0.)

        self.pos_emb = nn.Parameter(torch.randn(n, n, keys, u))
    
    def forward(self, x, y=None):
        B, C, H, W = x.size()

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = self.norm_q(q)
        v = self.norm_v(v)

        q = q.view(B, self.heads, self.keys, H*W)
        k = k.view(B, self.us,    self.keys, H*W)
        v = v.view(B, self.us,    self.vs,   H*W)

        k = k.softmax(-1)

        lambdac = torch.einsum('bukm,buvm->bkv', k, v)
        Yc      = torch.einsum('bhkn,bkv->bhvn', q, lambdac)

        lambdap = torch.einsum('nmku,buvm->bnkv', self.pos_emb, v)
        Yp      = torch.einsum('bhkn,bnkv->bhvn', q, lambdap)

        Y = Yc + Yp
        Y = Y.contiguous().view(B, C, H, W)
        return Y

def Attention(channels, use_sn, sn_eps=1e-12, name='sa', resl=64):
    if name == 'sa':
        return SelfAttention(channels, use_sn, sn_eps)
    elif name == 'll':
        return LambdaLayer(channels, use_sn, sn_eps, resl**2)

'''Generator'''

GEN_ARCH = {
    128 : {
        'in'   : [16, 16, 8, 4, 2],
        'out'  : [16,  8, 4, 2, 1],
        'up'   : [True] * 5,
        'resl' : [8, 16, 32, 64, 128],
        'att'  : 64
    },
    256 : {
        'in'   : [16, 16, 8, 8, 4, 2],
        'out'  : [16,  8, 8, 4, 2, 1],
        'up'   : [True] * 6,
        'resl' : [8, 16, 32, 64, 128, 256],
        'att'  : 128
    },
    512 : {
        'in'   : [16, 16, 8, 8, 4, 2, 1],
        'out'  : [16,  8, 8, 4, 2, 1, 1],
        'up'   : [True] * 7,
        'resl' : [8, 16, 32, 64, 128, 256, 512],
        'att'  : 64
    }
}

GEN_ARCH_DEEP = {
    128 : {
        'in'   : [16, 16, 16, 16, 8, 8, 4, 4, 2, 2],
        'out'  : [16, 16, 16,  8, 8, 4, 4, 2, 2, 1],
        'up'   : [False, True] * 5,
        'resl' : [4, 8, 8, 16, 16, 32, 32, 64, 64, 128],
        'att'  : 64
    },
    256 : {
        'in'   : [16, 16, 16, 16, 8, 8, 8, 8, 4, 4, 2, 2],
        'out'  : [16, 16, 16,  8, 8, 8, 8, 4, 4, 2, 2, 1],
        'up'   : [False, True] * 6,
        'resl' : [4, 8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256],
        'att'  : 64
    },
    512 : {
        'in'   : [16, 16, 16, 16, 8, 8, 8, 8, 4, 4, 2, 2, 1, 1],
        'out'  : [16, 16, 16,  8, 8, 8, 8, 4, 4, 2, 2, 1, 1, 1],
        'up'   : [False, True] * 7,
        'resl' : [4, 8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512],
        'att'  : 64
    }
}

class Generator(nn.Module):
    '''UNCONDITIONAL BigGAN/BigGAN-deep Generator
    z is not split and passed to every conditional norm layer
    '''
    def __init__(self,
        image_size, z_dim, deep, channels=64, use_sn=True,
        att_name='sa', act_name='relu', norm_name='bn',
        mlp=False, amp=False
    ):
        super().__init__()
        assert image_size in list(GEN_ARCH.keys()), f'specify a image size in {GEN_ARCH.keys()}'

        sn_eps = 1e-4 if amp else 1e-12
        cn_eps = 1e-4 if amp else 1e-5

        arch = GEN_ARCH[image_size] if not deep else GEN_ARCH_DEEP[image_size]

        self.linear = Linear(z_dim, 4*4*16*channels, use_sn)
        self.linear_view_size = (-1, 16*channels, 4, 4)

        activation = get_activation(act_name)
        upsample   = nn.Upsample(scale_factor=2)

        self.blocks = nn.ModuleList()
        att = True # attention only appears once
        for index, resl in enumerate(arch['resl']):
            self.blocks.append(
                gblock(deep, arch['in'][index]*channels, arch['out'][index]*channels, z_dim,
                    use_sn, activation, upsample if arch['up'][index] else None,
                    sn_eps, norm_name, mlp, cn_eps)
            )
            if resl == arch['att'] and att:
                att = False
                self.blocks.append(
                    Attention(arch['out'][index]*channels, use_sn, sn_eps, att_name, resl)
                )
        self.out_layer = nn.Sequential(
            nn.BatchNorm2d(arch['out'][-1]*channels, eps=cn_eps),
            activation,
            Conv2d(arch['out'][-1]*channels, 3, 3, use_sn, sn_eps, padding=1),
            nn.Tanh()
        )
    def forward(self, x, y=None):
        if y == None:
            y = x
        x = self.linear(x)
        x = x.view(*self.linear_view_size)

        for block in self.blocks:
            x = block(x, y)

        x = self.out_layer(x)

        return x

'''Discriminator'''

DIS_ARCH = {
    128 : {
        'in'   : [0, 2, 4,  8, 16, 16],
        'out'  : [2, 4, 8, 16, 16, 16],
        'down' : [True] * 5 + [False],
        'resl' : [64, 32, 16, 8, 4, 4],
        'att'  : 64
    },
    256 : {
        'in'   : [0, 2, 4, 8,  8, 16, 16],
        'out'  : [2, 4, 8, 8, 16, 16, 16],
        'down' : [True] * 6 + [False],
        'resl' : [128, 64, 32, 16, 8, 4, 4],
        'att'  : 64
    },
    512 : {
        'in'   : [0, 1, 2, 4, 8,  8, 16, 16],
        'out'  : [1, 2, 4, 8, 8, 16, 16, 16],
        'down' : [True] * 7 + [False],
        'resl' : [256, 128, 64, 32, 16, 8, 4, 4],
        'att'  : 64
    },
}

DIS_ARCH_DEEP = {
    128 : {
        'in'   : [0, 1, 2, 2, 4, 4, 8,  8, 16, 16, 16],
        'out'  : [1, 2, 2, 4, 4, 8, 8, 16, 16, 16, 16],
        'down' : [False, True] * 5 + [False],
        'resl' : [128, 64, 64, 32, 32, 16, 16, 8, 8, 4, 4],
        'att'  : 64
    },
    256 : {
        'in'   : [0, 1, 2, 2, 4, 4, 8, 8, 8,  8, 16, 16, 16],
        'out'  : [1, 2, 2, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16],
        'down' : [False, True] * 6 + [False],
        'resl' : [256, 128, 128, 64, 64, 32, 32, 16, 16, 8, 8, 4, 4],
        'att'  : 64
    },
    512 : {
        'in'   : [0, 1, 1, 1, 2, 2, 4, 4, 8, 8, 8,  8, 16, 16, 16],
        'out'  : [1, 1, 1, 2, 2, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16],
        'down' : [False, True] * 7 + [False],
        'resl' : [512, 256, 256, 128, 128, 64, 64, 32, 32, 16, 16, 8, 8, 4, 4],
        'att'  : 64
    },
}

class Discriminator(nn.Module):
    '''UNCONTIONAL BigGAN/BigGAN-deep Discriminator
    almost the same but does not have condition projection.
    '''
    def __init__(self,
        image_size, deep, channels=64, use_sn=True, att_name='sa', act_name='relu', amp=False
    ):
        super().__init__()
        assert image_size in list(DIS_ARCH.keys()), f'specify a image size in {DIS_ARCH.keys()}'

        sn_eps = 1e-4 if amp else 1e-12

        arch = DIS_ARCH[image_size] if not deep else DIS_ARCH_DEEP[image_size]

        activation = get_activation(act_name)
        downsample = nn.AvgPool2d(2)

        layers = []
        for index, resl in enumerate(arch['resl']):
            inc = 3 if arch['in'][index] == 0 else arch['in'][index]*channels
            if deep and inc == 3: # deep model input is conv2d3x3 not resblock
                layers.append(
                    Conv2d(inc, arch['out'][index]*channels, 3, use_sn, sn_eps, padding=1)
                )
            else:
                layers.append(
                    dblock(deep, inc, arch['out'][index]*channels, use_sn, activation,
                        downsample if arch['down'][index] else None, sn_eps)
                )
            if resl == arch['att'] and not arch['resl'][index+1] == arch['att']:
                layers.append(
                    Attention(arch['out'][index]*channels, use_sn, sn_eps, att_name, resl)
                )
        self.blocks = nn.Sequential(
            *layers,
            activation,
            GlobalSumPool2d(),
            Linear(arch['out'][-1]*channels, 1, use_sn, sn_eps)
        )
    def forward(self, x):
        return self.blocks(x)


def init_weight_ortho(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(module.weight)
        if not module.bias == None:
            module.bias.data.fill_(0.)

if __name__ == "__main__":
    x = torch.randn(3, 120)
    image_size = 512
    deep = True
    g = Generator(image_size, 120, deep)
    d = Discriminator(image_size, deep)
    print(d(g(x)).size())