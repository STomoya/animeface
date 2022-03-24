
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation(name):
    if   name == 'relu':  return nn.ReLU(True)
    elif name == 'lrelu': return nn.LeakyReLU(0.2, True)
    elif name == 'gelu':  return nn.GELU()
    elif name == 'swish': return nn.SiLU()
    raise Exception(f'{name}')

def get_normalization(name, channels, **kwargs):
    if   name == 'bn': return nn.BatchNorm2d(channels, **kwargs)
    elif name == 'in': return nn.InstanceNorm2d(channels, **kwargs)
    elif name == 'ln': return nn.GroupNorm(1, channels, **kwargs)
    elif name == 'gn': return nn.GroupNorm(16, channels, **kwargs)
    raise Exception(f'{name}')

def disable_bias(norm_layer):
    if norm_layer.bias is not None:
        bias_shape = norm_layer.bias.size()
        del norm_layer.bias
        norm_layer.register_buffer('bias', torch.zeros(bias_shape))


class GDFN(nn.Module):
    def __init__(self,
        channels, expansion=2.66, act_name='gelu'
    ) -> None:
        super().__init__()
        mid_channels = int(channels*expansion)
        self.pwconv12 = nn.Conv2d(channels, mid_channels*2, 1, bias=False)
        self.dwconv12 = nn.Conv2d(mid_channels*2, mid_channels*2, 3, 1, 1, bias=False, groups=mid_channels*2)
        self.act      = get_activation(act_name)
        self.pwconvo  = nn.Conv2d(mid_channels, channels, 1, bias=False)

    def forward(self, x):
        x = self.pwconv12(x)
        x1, x2 = self.dwconv12(x).chunk(2, dim=1)
        x = self.act(x1) * x2
        x = self.pwconvo(x)
        return x


class MDTA(nn.Module):
    def __init__(self,
        channels, num_heads
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        _temperature_init_scale = (channels / num_heads) ** -0.5

        self.pwqkv = nn.Conv2d(channels, channels*3, 1, bias=False)
        self.dwqkv = nn.Conv2d(channels*3, channels*3, 3, 1, 1, bias=False, groups=channels*3)
        self.proj_out = nn.Conv2d(channels, channels, 1, bias=False)
        self.temperature = nn.Parameter(torch.ones([]) * _temperature_init_scale)

    def forward(self, x):
        B, C, H, W = x.size()
        qkv_size = (B, self.num_heads, C//self.num_heads, H*W)

        qkv = self.pwqkv(x)
        qkv = self.dwqkv(qkv)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(*qkv_size)
        k = k.view(*qkv_size)
        v = v.view(*qkv_size)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = q @ k.transpose(-1, -2) * self.temperature
        attn = attn.softmax(-1)

        out = attn @ v
        out = out.view(B, C, H, W)

        out = self.proj_out(out)
        return out


class Block(nn.Module):
    def __init__(self,
        channels, num_heads, expansion=2.66, norm_name='ln', act_name='gelu', biasfree_norm=False
    ) -> None:
        super().__init__()

        self.norm1 = get_normalization(norm_name, channels)
        self.attn  = MDTA(channels, num_heads)
        self.norm2 = get_normalization(norm_name, channels)
        self.ff    = GDFN(channels, expansion, act_name)
        self.attn_scale = nn.Parameter(torch.ones([]) * 1e-2)
        self.ff_scale   = nn.Parameter(torch.ones([]) * 1e-2)

        if biasfree_norm:
            disable_bias(self.norm1)
            disable_bias(self.norm2)

    def forward(self, x):
        x = x + self.attn_scale * self.attn(self.norm1(x))
        x = x + self.ff_scale   * self.ff(self.norm2(x))
        return x


class Downsample(nn.Sequential):
    def __init__(self, channels) -> None:
        super().__init__(
            # C -> C // 2
            nn.Conv2d(channels, channels//2, 3, 1, 1, bias=False),
            # C//2 -> C*2
            nn.PixelUnshuffle(2))

class Upsample(nn.Sequential):
    def __init__(self, channels) -> None:
        super().__init__(
            # C -> C*2
            nn.Conv2d(channels, channels*2, 3, 1, 1, bias=False),
            # C*2 -> C//2
            nn.PixelShuffle(2))



class Generator(nn.Module):
    '''Restormer
    Modification:
        - Default to no skip connection of input to output.
            Restormer is originally not for I2I, but for image restoration, which is natural to sum the input to the output.
            On the other hand I2I most of the time doesn't want this behavior.
            Pass input_skip=True to enable input-to-output skip connection.
    '''
    def __init__(self,
        num_blocks=[2, 6, 6, 8], num_heads=[1, 2, 4, 8], num_refinement_blocks=4, ff_expansion=2.66, channels=48,
        norm_name='ln', act_name='gelu', biasfree_norm=False, io_channels=3, input_skip=True
    ) -> None:
        super().__init__()
        assert len(num_blocks) == len(num_heads)
        self.input_skip = input_skip
        base_channels = channels

        self.patch_embed = nn.Conv2d(io_channels, channels, 3, 1, 1, bias=False)

        self.downs = nn.ModuleList()
        for num_block, num_head in zip(num_blocks[:-1], num_heads[:-1]):
            self.downs.append(nn.ModuleList([
                # transformer blocks
                nn.Sequential(
                    *[Block(channels, num_head, ff_expansion, norm_name, act_name, biasfree_norm)
                    for _ in range(num_block)]),
                # downsample
                Downsample(channels)]))
            channels *= 2

        self.innermost = nn.Sequential(
            # transformer blocks
            *[Block(channels, num_heads[-1], ff_expansion, norm_name, act_name, biasfree_norm)
            for _ in range(num_blocks[-1])])

        self.ups = nn.ModuleList()
        for num_block, num_head in zip(reversed(num_blocks[:-1]), reversed(num_heads[:-1])):
            in_channels = cat_channels = channels
            # we don't reduce channels in last level
            use_reduce = (channels // 2) != base_channels
            reduced_channels = channels // 2 if use_reduce else channels
            reduce = nn.Conv2d(cat_channels, reduced_channels, 1, bias=False) if use_reduce else nn.Identity()
            self.ups.append(nn.ModuleList([
                # upsample
                Upsample(in_channels),
                # (cat) reduce
                reduce,
                # transformer blocks
                nn.Sequential(
                    *[Block(reduced_channels, num_head, ff_expansion, norm_name, act_name, biasfree_norm)
                    for _ in range(num_block)])]))
            channels //= 2

        self.refinement = nn.Sequential(
            # transformer blocks
            *[Block(reduced_channels, num_heads[0], ff_expansion, norm_name, act_name, biasfree_norm)
            for _ in range(num_refinement_blocks)])

        self.output = nn.Conv2d(reduced_channels, io_channels, 3, 1, 1, bias=False)

    def forward(self, x):
        skip = x

        x = self.patch_embed(x)

        dfeats = []
        for block, down in self.downs:
            x = block(x)
            dfeats.append(x)
            x = down(x)

        x = self.innermost(x)

        dfeats = dfeats[::-1]
        for i, (up, reduce, block) in enumerate(self.ups):
            x = up(x)
            x = reduce(torch.cat([x, dfeats[i]], dim=1))
            x = block(x)

        x = self.refinement(x)
        x = self.output(x)

        if self.input_skip:
            x = x + skip

        return x


class Discriminator(nn.Sequential):
    def __init__(self,
        num_layers=3, channels=64, max_channels=512, norm_name='bn', act_name='lrelu',
        in_channels=3
    ) -> None:

        ochannels = channels
        layers = [
            nn.Conv2d(in_channels, ochannels, 4, 2, 1),
            get_activation(act_name)]
        for _ in range(num_layers-1):
            channels *= 2
            ichannels, ochannels = ochannels, min(max_channels, channels)
            layers.extend([
                nn.Conv2d(ichannels, ochannels, 4, 2, 1),
                get_normalization(norm_name, ochannels),
                get_activation(act_name)])
        layers.extend([
            nn.Conv2d(ochannels, ochannels*2, 4, 1, 1),
            get_normalization(norm_name, ochannels*2),
            get_activation(act_name),
            nn.Conv2d(ochannels*2, 1, 4, 1, 1)])
        super().__init__(*layers)
