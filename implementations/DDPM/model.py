'''
Denoising Diffusion Probabilistic Model
code reference :
    #1 [lucidrains/denoising-diffusion-pytorch]
        - https://github.com/lucidrains/denoising-diffusion-pytorch/blob/master/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
    #2 [rosinality/denoising-diffusion-pytorch]
        - https://github.com/rosinality/denoising-diffusion-pytorch/blob/master/diffusion.py
'''

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract

from tqdm import tqdm

def cosine_beta_schedule(timesteps, s=0.008):
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    alpha = ((steps / timesteps) + s) / (1 + s) * math.pi * 0.5
    alpha = alpha.cos().pow(2)
    alpha = alpha / alpha[0]
    betas = 1 - alpha[1:] / alpha[:-1]
    return betas.clamp(max=0.999)

def extract(input, t, shape):
    out = input.gather(0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(reshape)
    return out

def noise_like(shape, device, repeat):
    if repeat:
        resid = [1] * (len(shape) - 1)
        shape_one = (1, *shape[1:])
        return torch.randn(shape_one, device=device).repeat(shape[0], *resid)
    else:
        return torch.randn(shape, device=device)

class GaussianDiffusion(nn.Module):
    def __init__(self,
        timesteps: int=1000
    ) -> None:
        super().__init__()
        _type = torch.float64
        _register = lambda name, tensor: self.register_buffer(name, tensor.type(torch.float32))

        betas = cosine_beta_schedule(timesteps)
        self.timesteps = timesteps

        alpha = 1 - betas
        alpha_cumprod = torch.cumprod(alpha, 0)
        alpha_cumprod_prev = torch.cat(
            [torch.tensor([1.], dtype=_type), alpha_cumprod[:-1]], 0)
        posterior_variance = betas * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod)

        _register('beta', betas)
        _register('alpha_cumprod', alpha_cumprod)
        _register('alpha_cumprod_prev', alpha_cumprod_prev)

        _register('sqrt_alpha_cumprod', torch.sqrt(alpha_cumprod))
        _register('sqrt_one_minus_alpha_cumprod', torch.sqrt(1 - alpha_cumprod))
        _register('log_one_minus_alpha_cumprod', torch.log(1 - alpha_cumprod))
        _register('sqrt_recip_alpha_cumprod', torch.rsqrt(alpha_cumprod))
        _register('sqrt_recipm1_alpha_cumprod', torch.sqrt(1 / alpha_cumprod - 1))
        _register('posterior_variance', posterior_variance)
        _register('posterior_log_variance_clipped',
            torch.log(posterior_variance.clamp(min=1e-20)))
        _register('posterior_mean_coef1',
            betas * torch.sqrt(alpha_cumprod_prev) / (1 - alpha_cumprod))
        _register('posterior_mean_coef2',
            (1 - alpha_cumprod_prev) * torch.sqrt(alpha) / (1 - alpha_cumprod))

    def q_sample(self, x_0, t, noise=None):
        '''image -> noise(t)'''
        if noise is None:
            noise = torch.randn_like(x_0)
        shape = x_0.shape
        x_noisy = (
            extract(self.sqrt_alpha_cumprod, t, shape) * x_0 +
            extract(self.sqrt_one_minus_alpha_cumprod, t, shape) * noise)
        return x_noisy, noise

    def predict_start_from_noise(self, x_t, t, noise):
        '''x_t -> x_t-1'''
        shape = x_t.shape
        return (
            extract(self.sqrt_recip_alpha_cumprod, t, shape) * x_t -
            extract(self.sqrt_recipm1_alpha_cumprod, t, shape) * noise)

    def q_posterior(self, x_0, x_t, t):
        '''mean, var'''
        shape = x_t.shape
        mean = (
            extract(self.posterior_mean_coef1, t, shape) * x_0 +
            extract(self.posterior_mean_coef2, t, shape) * x_t)
        var = extract(self.posterior_variance, t, shape)
        log_var = extract(self.posterior_log_variance_clipped, t, shape)
        return mean, var, log_var

    def p_mean_variance(self, model, x, t, clip_denoised: bool=True):
        '''mean, var'''
        x_recon = self.predict_start_from_noise(x, t, model(x, t))
        if clip_denoised:
            x_recon = x_recon.clamp(min=-1., max=1.)
        mean, var, log_var = self.q_posterior(x_recon, x, t)
        return mean, var, log_var

    def p_sample(self,
        model, x, t,
        clip_denoised: bool=True,
        repeat_noise: bool=False):
        '''x_t -> x_t-1'''
        mean, _, log_var = self.p_mean_variance(model, x, t, clip_denoised)
        noise = noise_like(x.shape, x.device, repeat_noise)
        shape = [x.size(0)] + [1] * (x.ndim - 1)
        nonzero_mask = (1 - (t == 0).type(torch.float32)).view(*shape)

        return mean + nonzero_mask * torch.exp(0.5 * log_var) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, noise=None):
        '''x_T -> x_0'''
        device = self.beta.device
        if noise is None:
            image = torch.randn(shape, device=device)
        else: image = noise

        for i in tqdm(reversed(range(self.timesteps)), total=self.timesteps, desc='Sampling'):
            image = self.p_sample(
                model, image,
                torch.full((shape[0], ), i, device=device))
        return image


'''U-Net'''

class Swish(nn.Module):
    '''swish activation'''
    def forward(self, x):
        return x * torch.sigmoid(x)
class Mish(nn.Module):
    '''mish activation'''
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

def get_activation(name):
    if name == 'lrelu': return nn.LeakyReLU(0.2)
    if name == 'swish': return Swish()
    if name == 'mish':  return Mish()
    raise Exception(f'ACT : {name}')

def get_normalization(name, channels, **kwargs):
    if name == 'bn': return nn.BatchNorm2d(channels, **kwargs)
    if name == 'in': return nn.InstanceNorm2d(channels, **kwargs)
    if name == 'gn': return nn.GroupNorm(32, channels, **kwargs)
    raise Exception(f'NORM : {name}')

@torch.no_grad()
def scaling_init(tensor, scale=1, dist='u'):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    scale /= (fan_in + fan_out) / 2
    if dist == 'n':
        std = math.sqrt(scale)
        return tensor.normal_(0., std)
    elif dist == 'u':
        bound = math.sqrt(3*scale)
        return tensor.uniform_(-bound, bound)

def conv(
    in_channels, out_channels, kernel_size,
    stride=1, padding=0, bias=True,
    scale=1.):
    _conv = nn.Conv2d(
        in_channels, out_channels, kernel_size,
        stride, padding, bias=bias)
    scaling_init(_conv.weight, scale)
    if _conv.bias is not None:
        nn.init.zeros_(_conv.bias)
    return _conv
def linear(
    in_features, out_features,
    scale=1.
):
    _linear = nn.Linear(in_features, out_features)
    scaling_init(_linear.weight, scale)
    nn.init.zeros_(_linear.bias)
    return _linear

class Upsample(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = conv(channels, channels, 3, 1, 1)
    def forward(self, x):
        return self.conv(self.up(x))
class Downsample(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.down = conv(channels, channels, 3, 2, 1)
    def forward(self, x):
        return self.down(x)

class ResBlock(nn.Module):
    def __init__(self,
        in_channels, out_channels,
        time_dim, time_affine=False,
        norm_name='gn', act_name='swish', dropout=0.
    ) -> None:
        super().__init__()

        self.time_affine = time_affine
        time_out_dim = out_channels
        time_scale = 1
        norm_affine = True
        if time_affine:
            time_out_dim *= 2
            time_scale = 1e-10
            norm_affine = False

        self.norm1 = get_normalization(norm_name, in_channels)
        self.act1  = get_activation(act_name)
        self.conv1 = conv(in_channels, out_channels, 3, 1, 1)

        self.time = nn.Sequential(
            get_activation(act_name),
            linear(time_dim, time_out_dim, time_scale))

        self.norm2 = get_normalization(norm_name, out_channels, affine=norm_affine)
        self.act2  = get_activation(act_name)
        self.drop  = nn.Dropout(dropout)
        self.conv2 = conv(out_channels, out_channels, 3, 1, 1, scale=1e-10)

        if in_channels != out_channels:
            self.skip = conv(in_channels, out_channels, 1, bias=False)

    def forward(self, x, t):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        if self.time_affine:
            t = self.time(t)[:, :, None, None]
            gamma, beta = t.chunk(2, dim=1)
            h = (1 + gamma) * self.norm2(h) + beta
        else:
            h  = h + self.time(t)[:, :, None, None]
            h = self.norm2(h)
        h = self.act2(h)
        h = self.drop(h)
        h = self.conv2(h)

        if hasattr(self, 'skip'):
            x = self.skip(x)
        return h + x

class SelfAttention(nn.Module):
    def __init__(self,
        channels, num_head=4, norm_name='gn'
    ) -> None:
        super().__init__()
        self.num_head = num_head

        self.norm = get_normalization(norm_name, channels)
        self.qkv = conv(channels, channels*3, 1, bias=False)
        self.o = conv(channels, channels, 1, scale=1e-10)

    def forward(self, x):
        B, C, H, W = x.size()
        head_dim = C // self.num_head

        norm = self.norm(x)
        qkv = self.qkv(norm).reshape(B, self.num_head, head_dim*3, H, W)
        q, k, v = qkv.chunk(3, dim=2)

        attn = contract('bnchw,bncyx->bnhwyx', q, k) / math.sqrt(C)
        attn = attn.view(B, self.num_head, H, W, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(B, self.num_head, H, W, H, W)

        out = contract('bnhwyx,bncyx->bnchw', attn, v)
        out = self.o(out.view(B, C, H, W))

        return out + x

class TimeEmbedding(nn.Module):
    '''sinusoidal position embedding'''
    def __init__(self, time_dim) -> None:
        super().__init__()
        self.dim = time_dim
        inv_freq = torch.exp(
            torch.arange(0, time_dim, 2, dtype=torch.float32) \
            * (- math.log(10000) / time_dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        shape = x.size()
        sinusoid_in = torch.ger(x.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb

class ResSABlock(nn.Module):
    ''' -> ResBlock x num_blocks -> SA -> '''
    def __init__(self,
        in_channels, out_channels, time_dim, time_affine=False,
        attn=True, num_blocks=1,
        attn_head=8, norm_name='gn', act_name='swish', dropout=0.
    ) -> None:
        super().__init__()

        self.res_blocks = nn.ModuleList([ResBlock(
            in_channels, out_channels, time_dim, time_affine,
            norm_name, act_name, dropout)])
        for _ in range(num_blocks-1):
            self.res_blocks.append(
                ResBlock(
                    out_channels, out_channels, time_dim, time_affine,
                    norm_name, act_name, dropout))
        if attn:
            self.attn = SelfAttention(
                out_channels, attn_head, norm_name)

    def forward(self, x, t):
        for block in self.res_blocks:
            x = block(x, t)
        if hasattr(self, 'attn'):
            x = self.attn(x)
        return x

class UNet(nn.Module):
    def __init__(self,
        image_size, bottom, in_channels=3, channels=32,
        attn_resls=[16], attn_head=8, time_affine=False,
        dropout=0., num_res=1,
        norm_name='gn', act_name='swish'
    ) -> None:
        super().__init__()
        num_sampling = int(math.log2(image_size)-math.log2(bottom))

        self.time_embed = nn.Sequential(
            TimeEmbedding(channels),
            linear(channels, channels*4),
            get_activation(act_name),
            linear(channels*4, channels*4))
        time_dim = channels*4

        ochannels = channels
        self.input = conv(in_channels, ochannels, 3, 1, 1)

        resl = image_size
        self.downs = nn.ModuleList()
        for i in range(num_sampling):
            resl /= 2
            channels *= 2
            ichannels, ochannels = ochannels, channels
            self.downs.append(
                ResSABlock(
                    ichannels, ochannels, time_dim, time_affine, resl in attn_resls,
                    num_res, attn_head, norm_name, act_name, dropout))
            if i != num_sampling-1:
                self.downs.append(Downsample(ochannels))

        self.mid_res1 = ResSABlock(
            ochannels, ochannels, time_dim, time_affine, True,
            num_res, attn_head, norm_name, act_name, dropout)
        self.mid_res2 = ResSABlock(
            ochannels, ochannels, time_dim, time_affine, False,
            num_res, attn_head, norm_name, act_name, dropout)

        self.ups = nn.ModuleList()
        for i in range(num_sampling):
            resl *= 2
            channels = channels // 2
            ichannels, ochannels = ochannels, channels
            self.ups.append(
                ResSABlock(
                    ichannels*2, ochannels, time_dim, time_affine, resl in attn_resls,
                    num_res, attn_head, norm_name, act_name, dropout))
            if i != num_sampling-1:
                self.ups.append(Upsample(ochannels))

        self.output = nn.Sequential(
            get_activation(act_name),
            conv(ochannels, 3, 3, 1, 1, 1e-10))

    def forward(self, x, t):
        time = self.time_embed(t)
        x = self.input(x)

        feats = []
        for module in self.downs:
            if isinstance(module, ResSABlock):
                x = module(x, time)
                feats.append(x)
            else:
                x = module(x)

        x = self.mid_res1(x, time)
        x = self.mid_res2(x, time)

        feat_index = 1
        for module in self.ups:
            if isinstance(module, ResSABlock):
                x = module(torch.cat([x, feats[-feat_index]], dim=1), time)
                feat_index += 1
            else:
                x = module(x)
        x = self.output(x)
        return x

if __name__=='__main__':
    unet = UNet(128, 16, time_affine=True)
    unet.to('cuda')
    b = 16
    image = torch.randn(b, 3, 128, 128).cuda()
    t = torch.randint(0, 1000, (b, )).cuda()
    noise = unet(image, t)
    print(noise.size())
