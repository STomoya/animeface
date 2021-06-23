
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract

def to3d(tensor):
    '''B(HW)C -> BCHW'''
    B, N, C = tensor.size()
    H = W = int(N**0.5)
    return tensor.reshape(B, H, W, C).permute(0, 3, 1, 2)

class To3d(nn.Module):
    def forward(self, x):
        return to3d(x)

def to2d(tensor):
    '''BCHW -> B(HW)C'''
    B, C, H, W = tensor.size()
    return tensor.reshape(B, C, H*W).transpose(-1, -2)

class To2d(nn.Module):
    def forward(self, x):
        return to2d(x)

def block(tensor, patch_size=8):
    '''split tensor to blocks'''
    B, C, H, W = tensor.size()
    patchs = patch_size**2
    tensor = tensor.reshape(B, C, patch_size, H//patch_size, patch_size, W//patch_size)
    tensor = tensor.permute(0, 3, 5, 2, 4, 1).reshape(B, H*W//patchs, patchs, C)
    return tensor

class Block(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
    def forward(self, x):
        return block(x, self.patch_size)

def unblock(tensor):
    '''blocked tensor back to normal'''
    B, M, N, C = tensor.size()
    H = W = int(M**0.5)
    patch_size = int(N**0.5)
    tensor = tensor.reshape(B, H, W, patch_size, patch_size, C)
    tensor = tensor.permute(0, 5, 3, 1, 4, 2).reshape(B, C, H*patch_size, W*patch_size)
    return tensor

class UnBlock(nn.Module):
    def forward(self, x):
        return unblock(x)

class MultiAxisAttention(nn.Module):
    '''Multi Axis Attention'''
    def __init__(self,
        dim, num_heads
    ):
        super().__init__()
        self.q = nn.Parameter(torch.randn(num_heads, dim, dim))
        self.k = nn.Parameter(torch.randn(dim, dim))
        self.v = nn.Parameter(torch.randn(dim, dim))
        self.o = nn.Parameter(torch.randn(num_heads, dim, dim))

    def forward(self, x):
        # QKV
        Q = contract('bmnd,hdk->bhmnk', x, self.q)
        Q1, Q2 = Q.chunk(2, dim=1)
        K = contract('bmnd,dk->bmnk', x, self.k)
        V = contract('bmnd,dv->bmnv', x, self.v)

        # attn map second axis
        logits = contract('bhxyk,bzyk->bhyxz', Q1, K)
        scores = logits.softmax(-1)
        O1 = contract('bhyxz,bzyv->bhxyv', scores, V)

        # attn map third axis
        logits = contract('bhxyk,bxzk->bhxyz', Q2, K)
        scores = logits.softmax(-1)
        O2 = contract('bhxyz,bxzv->bhxyv', scores, V)

        O = torch.cat([O1, O2], dim=1)
        Z = contract('bhmnd,hdv->bmnd', O, self.o)
        return Z

class MultiQueryAttention(nn.Module):
    def __init__(self,
        dim, latent_dim, num_heads
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(latent_dim, dim*2, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.scale = (dim // self.num_heads) ** -0.5

    def forward(self, x, z):
        B, xN, _ = x.size()
        _, zN, _ = z.size()

        Q = self.q(x) \
            .reshape(B, xN, self.num_heads, self.dim//self.num_heads) \
            .transpose(1, 2)
        KV = self.kv(z) \
            .reshape(B, zN, 2, self.num_heads, self.dim//self.num_heads) \
            .permute(2, 0, 3, 1, 4)
        K, V = KV.unbind(dim=0)

        attn = Q @ K.transpose(-1, -2) * self.scale
        attn = attn.softmax(-1)

        O = (attn @ V).permute(0, 2, 1, 3).reshape(B, xN, self.dim)
        Z = self.o(O)

        return Z

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim))
    def forward(self, x):
        return self.mlp(x)

class MQABlock(nn.Module):
    def __init__(self,
        dim, latent_dim, num_heads
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiQueryAttention(dim, latent_dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x, z):
        x = x + self.attn(self.norm1(x), z)
        x = x + self.mlp(self.norm2(x))
        return x

class MAABlock(nn.Module):
    def __init__(self,
        dim, num_heads, patch_size
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiAxisAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x):
        attn_in = block(to3d(self.norm1(x)), self.patch_size)
        attn_out = to2d(unblock(self.attn(attn_in)))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class MLPBlock(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim)
    def forward(self, x):
        x = x + self.mlp(self.norm(x))
        return x

class LowResolutionStage(nn.Module):
    def __init__(self,
        in_dim, out_dim, latent_dim,
        num_heads, num_attn, patch_size
    ):
        super().__init__()

        self.multiquery = MQABlock(in_dim, latent_dim, num_heads)
        self.multiaxis = nn.Sequential(*[
            MAABlock(in_dim, num_heads, patch_size)
            for _ in range(num_attn)
        ])

        self.output = nn.Sequential(
            To3d(),
            nn.PixelShuffle(2),
            To2d(),
            nn.Linear(in_dim//4, out_dim)
        )

    def forward(self, x, z):
        assert x.ndim == 3

        x = self.multiquery(x, z)
        x = self.multiaxis(x)

        x = self.output(x)

        return x

class HighResolutionStage(nn.Module):
    def __init__(self,
        in_dim, out_dim, latent_dim,
        num_heads, num_mlps, is_last=False
    ):
        super().__init__()

        self.multiquery = MQABlock(in_dim, latent_dim, num_heads)
        self.mlps = nn.Sequential(*[
            MLPBlock(in_dim)
            for _ in range(num_mlps)
        ])

        if not is_last:
            self.output = nn.Sequential(
                To3d(),
                nn.PixelShuffle(2),
                To2d(),
                nn.Linear(in_dim//4, out_dim)
            )

        self.rgb = nn.Sequential(nn.Linear(in_dim, 3), To3d())

    def forward(self, x, z):
        assert x.ndim == 3

        x = self.multiquery(x, z)
        x = self.mlps(x)

        image = self.rgb(x)

        if hasattr(self, 'output'):
            x = self.output(x)

        return x, image

class PositionEmbed(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(*size))
    def forward(self, x):
        return x + self.pe

class Generator(nn.Module):
    def __init__(self,
        latent_dim,                        # input latent dim
        dims=[512, 512, 256, 128, 64, 64], # dims of each layer
        bottom=8,                          # bottom width
        low_stages=4,                      # number of low resolution stages
        num_heads=[16, 8, 4, 4, 4, 4],     # number of heads for MHA
        num_blocks=[2, 2, 2, 2, 2, 2],     # number of MAA/MLP in each stage
        patch_sizes=[4, 4, 8, 8]           # patch sizes for MAA
    ):
        super().__init__()
        assert len(num_heads) == len(dims)
        assert len(num_blocks) == len(dims)
        assert len(patch_sizes) == low_stages

        self.bottom = bottom

        # input
        self.z_input = nn.Linear(latent_dim, latent_dim*bottom**2)
        self.z_pe = PositionEmbed((1, bottom**2, latent_dim))
        self.input = nn.Linear(latent_dim, dims[0]*bottom**2)

        # stages
        layers = []
        self.init_img_size = None
        for index, dim in enumerate(dims[:-1]):
            resl = int(2 ** (math.log2(bottom) + index))
            if index < low_stages:
                # low resolutions stages
                layers.extend([
                    PositionEmbed((1, resl**2, dim)),
                    LowResolutionStage(
                        dim, dims[index+1], latent_dim,
                        num_heads[index], num_blocks[index], patch_sizes[index])])
            else:
                # high resolution stages
                layers.extend([
                    PositionEmbed((1, resl**2, dim)),
                    HighResolutionStage(
                        dim, dims[index+1], latent_dim,
                        num_heads[index], num_blocks[index])])
                if self.init_img_size is None:
                    self.init_img_size = (resl//2, resl//2)
        # output
        resl = int(2 ** (math.log2(bottom) + index + 1))
        if self.init_img_size is None:
            self.init_img_size = (resl//2, resl//2)
        layers.extend([
            PositionEmbed((1, resl**2, dims[-1])),
            HighResolutionStage(
                dims[-1], None, latent_dim, num_heads[-1],
                num_blocks[-1], is_last=True)])
        self.generator = nn.ModuleList(layers)

        # upsample for images
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, z):

        x = self.input(z)
        x = x.reshape(x.size(0), self.bottom**2, -1)
        z = self.z_input(z)
        z = z.reshape(z.size(0), self.bottom**2, -1)
        z = self.z_pe(z)

        image = torch.zeros((z.size(0), 3, *self.init_img_size), device=z.device)
        for module in self.generator:
            if isinstance(module, LowResolutionStage):
                x = module(x, z)
            elif isinstance(module, HighResolutionStage):
                x, img = module(x, z)
                image = self.upsample(image) + img
            else:
                x = module(x)

        return image

def get_activation(name):
    if name == 'lrelu': return nn.LeakyReLU(0.2)
    if name == 'relu': return nn.ReLU()

def Conv2d(*args, **kwargs):
    return nn.utils.spectral_norm(nn.Conv2d(*args, **kwargs))
def Linear(*args, **kwargs):
    return nn.utils.spectral_norm(nn.Linear(*args, **kwargs))

class Blur2d(nn.Module):
    '''Gaussian filtering'''
    def __init__(self, filter=[1, 3, 3, 1]):
        super().__init__()
        kernel = torch.tensor(filter, dtype=torch.float32)
        kernel = torch.einsum('i,j->ij', kernel, kernel)
        kernel /= kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.register_buffer('kernel', kernel)

        kernel_size = len(filter)
        if kernel_size % 2 == 1:
            pad1, pad2 = kernel_size//2, kernel_size//2
        else:
            pad1, pad2 = kernel_size//2, (kernel_size-1)//2
        self.padding = (pad1, pad2, pad1, pad2)

    def forward(self, x):
        C = x.size(1)
        x = F.pad(x, self.padding)
        weight = self.kernel.expand(C, -1, -1, -1)
        x = F.conv2d(x, weight, groups=C)
        return x

class ResBlock(nn.Module):
    def __init__(self,
        in_channels, out_channels, down=True,
        filter=[1, 3, 3, 1], act_name='lrelu'
    ) -> None:
        super().__init__()

        self.main = nn.Sequential(
            Conv2d(in_channels, out_channels, 3, 1, 1),
            get_activation(act_name),
            Conv2d(out_channels, out_channels, 3, 1, 1))

        if down:
            self.down = nn.Sequential(
                Blur2d(filter), nn.AvgPool2d(2))
        if in_channels != out_channels or down:
            self.skip = Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        h = self.main(x)

        if hasattr(self, 'skip'):
            x = self.skip(x)
        if hasattr(self, 'down'):
            x = self.down(x)
            h = self.down(h)

        return h + x

class MiniBatchStdDev(nn.Module):
    def __init__(self, group_size=4, eps=1e-4):
        super().__init__()
        self.group_size = group_size
        self.eps = eps

    def forward(self, x):
        B, C, H, W = x.size()
        y = x
        groups = self._check_group_size(B)
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

class Discriminator(nn.Module):
    def __init__(self,
        image_size, channels=32, max_channels=512, bottom=8,
        act_name='lrelu', mbsd_groups=4
    ) -> None:
        super().__init__()
        num_downs = int(math.log2(image_size) - math.log2(bottom))

        ochannels = channels
        self.input = nn.Sequential(
            Conv2d(3, ochannels, 3, 1, 1),
            get_activation(act_name))

        blocks = []
        for _ in range(num_downs):
            ichannels, ochannels = ochannels, min(channels, max_channels)
            blocks.append(
                ResBlock(ichannels, ochannels, act_name=act_name))
        self.resblocks = nn.Sequential(*blocks)
        self.output = nn.Sequential(
            MiniBatchStdDev(mbsd_groups),
            Conv2d(ochannels+1, ochannels, 3, 1, 1),
            nn.Flatten(),
            Linear(ochannels*bottom**2, ochannels),
            get_activation(act_name),
            Linear(ochannels, 1))

    def forward(self, x):
        x = self.input(x)
        x = self.resblocks(x)
        x = self.output(x)
        return x
