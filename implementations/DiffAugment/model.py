
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
modules
'''

class EqualizedLR(nn.Module):
    '''
    equalized learning rate
    '''
    def __init__(self, layer, gain=2):
        super(EqualizedLR, self).__init__()

        self.wscale = (gain / layer.weight[0].numel()) ** 0.5
        self.layer = layer

    def forward(self, x, gain=2):
        x = self.layer(x * self.wscale)
        return x

class Blur(nn.Module):
    '''
    low pass filter
    
    Does nothing for now
    '''
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class ScaleNoise(nn.Module):
    '''
    scale noise with learnable weight
    '''
    def __init__(self):
        super().__init__()
        self.scale = nn.Conv2d(1, 1, 1, bias=False)
        self.scale.weight.data.fill_(0)
    def forward(self, x):
        x = self.scale(x)
        return x

class AdaptiveInstanceNorm(nn.Module):
    '''
    AdaIN
    '''
    def __init__(self,
        channels, style_dim
    ):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, eps=1.e-8)
        
        self.linear = EqualizedLinear(style_dim, channels*2)
        self.linear.linear.layer.bias.data[:channels] = 1.

    def forward(self, x, style):
        norm = self.norm(x)
        style = self.linear(style).unsqueeze(2).unsqueeze(3)
        ys, yb = style.chunk(2, 1)
        x = ys * norm + yb
        return x

class MiniBatchStd(nn.Module):
    '''
    minibatch standard deviation
    '''
    def forward(self, x):
        std = torch.std(x).expand(x.shape[0], 1, *x.shape[2:])
        return torch.cat([x, std], dim=1)

class EqualizedLinear(nn.Module):
    '''
    equalized fully connected layer
    '''
    def __init__(self,
        in_channels, out_channels
    ):
        super().__init__()

        linear = nn.Linear(in_channels, out_channels)
        linear.weight.data.normal_(0, 1)
        linear.bias.data.fill_(0)
        self.linear = EqualizedLR(linear)
    def forward(self, x):
        x = self.linear(x)
        return x

class EqualizedConv2d(nn.Module):
    '''
    equalized convolutional layer
    '''
    def __init__(self,
        in_channels, out_channels, kernel_size, **kwargs
    ):
        super().__init__()

        conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, **kwargs
        )

        conv.weight.data.normal_(0, 1)
        conv.bias.data.fill_(0.)
        self.conv = EqualizedLR(conv)

    def forward(self, x):
        x = self.conv(x)
        return x

class LayerEpilogue(nn.Module):
    '''
    things to do on the end of each layer
    '''
    def __init__(self,
        channels, style_dim
    ):
        super().__init__()
        self.scale_noise = ScaleNoise()
        self.activation = nn.LeakyReLU(0.2)
        self.norm_layer = AdaptiveInstanceNorm(channels, style_dim)

    def forward(self, style, synthesis_out, noise):
        noise = self.scale_noise(noise)
        x = synthesis_out + noise
        x = self.activation(x)
        x = self.norm_layer(x, style)

        return x

class UpsampleBlur(nn.Module):
    '''
    upsample -> blur

    (the office implementation is upsample -> conv2d -> blur [1]
     but the paper says they implemented the blur "after each upsampling layer"[2]?

    [1] https://github.com/NVlabs/stylegan/blob/master/training/networks_stylegan.py#L520
    [2] https://arxiv.org/pdf/1812.04948.pdf )
    '''
    def __init__(self, scale_factor, mode='bilinear'):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        self.blur = Blur()
    def forward(self, x):
        x = self.upsample(x)
        x = self.blur(x)
        return x

class BlurDownsample(nn.Module):
    '''
    blur -> downsample (avg_pool)

    '''
    def __init__(self, scale_factor):
        super().__init__()
        self.blur = Blur()
        self.downsample = nn.AvgPool2d(2)
    def forward(self, x):
        x = self.blur(x)
        x = self.downsample(x)
        return x

class ToRGB(nn.Module):
    '''
    to rgb
    '''
    def __init__(self,
        in_channels, out_channels=3
    ):
        super(ToRGB, self).__init__()

        self.to_rgb = nn.Sequential(
            EqualizedConv2d(in_channels, out_channels, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.to_rgb(x)
        return x

class FromRGB(nn.Module):
    '''
    from rgb
    '''
    def __init__(self,
        out_channels, in_channels=3
    ):
        super(FromRGB, self).__init__()

        self.from_rgb = EqualizedConv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        x = self.from_rgb(x)
        return x

'''
blocks
'''

class Mapping(nn.Module):
    '''
    mapping latent vector to space W
    '''
    def __init__(self,
        style_dim,
        n_layers=8
    ):
        super().__init__()

        self.linear = nn.ModuleList([EqualizedLinear(style_dim, style_dim) for _ in range(n_layers)])
        self.activation = nn.LeakyReLU(0.2)
    def forward(self, x):
        for layer in self.linear:
            x = layer(x)
            x = self.activation(x)
        return x

class GeneratorBlock(nn.Module):
    '''
    generator block for a resolution
    '''
    def __init__(self,
        in_channels, out_channels, style_dim, is_first=False
    ):
        super().__init__()
        self.is_first = is_first

        if is_first:
            self.le0      = LayerEpilogue(style_dim, style_dim)
            self.conv1    = EqualizedConv2d(style_dim, style_dim, 3, padding=1)
            self.le1      = LayerEpilogue(style_dim, style_dim)
        else:
            self.upsample = UpsampleBlur(2)
            self.conv0    = EqualizedConv2d(in_channels, out_channels, 3, padding=1)
            self.le0      = LayerEpilogue(out_channels, style_dim)
            self.conv1    = EqualizedConv2d(out_channels, out_channels, 3, padding=1)
            self.le1      = LayerEpilogue(out_channels, style_dim)
        
    def forward(self, style, x):
        if not self.is_first:
            x = self.upsample(x)
            x = self.conv0(x)

        B, _, H, W = x.size()
        noise = torch.randn((B, 1, H, W), device=x.device)
        x = self.le0(style, x, noise)
        x = self.conv1(x)
        noise = torch.randn((B, 1, H, W), device=x.device)
        x = self.le1(style, x, noise)

        return x

class DiscriminatorBlock(nn.Module):
    '''
    discriminator block for a resolution
    '''
    def __init__(self,
        in_channels, out_channels, is_last=False
    ):
        super(DiscriminatorBlock, self).__init__()

        if is_last:
            self.block = nn.Sequential(
                MiniBatchStd(),
                EqualizedConv2d(in_channels+1, out_channels, 3, padding=1),
                nn.LeakyReLU(0.2),
                EqualizedConv2d(out_channels, out_channels, 4),
                nn.LeakyReLU(0.2),
                EqualizedConv2d(out_channels, 1, 1)
            )
        else:
            self.block = nn.Sequential(
                EqualizedConv2d(in_channels, out_channels, 3, padding=1),
                nn.LeakyReLU(0.2),
                EqualizedConv2d(out_channels, out_channels, 3, padding=1),
                BlurDownsample(2)
            )

    def forward(self, x):
        x = self.block(x)
        return x        

'''
Generator
'''

class Generator(nn.Module):
    def __init__(self,
        style_dim, mapping_layers=8
    ):
        super().__init__()

        self.mapping = Mapping(style_dim=style_dim, n_layers=mapping_layers)

        resl2param = {
              4: [512, 512, True],
              8: [512, 512, False],
             16: [512, 256, False],
             32: [256, 128, False],
             64: [128,  64, False],
            128: [ 64,  32, False]
        }

        self.resl_blocks = nn.ModuleList()
        self.rgb_layers  = nn.ModuleList()

        for resl, param in resl2param.items():
            self.resl_blocks.append(
                GeneratorBlock(
                    param[0], param[1], style_dim=style_dim, is_first=param[2])
            )
            self.rgb_layers.append(ToRGB(param[1]))
        self.upsample = UpsampleBlur(2)

        self.train_depth = 0
        self.alpha = 0
        self.synthesis_input = nn.Parameter(torch.ones(1, style_dim, 4, 4), requires_grad=True)

    def grow(self):
        self.train_depth += 1
        self.alpha = 0

    def forward(self, x, phase):
        style = self.mapping(x)
        if phase == 't':
            return self.transition_forward(style)
        else:
            return self.stablization_forward(style)
    
    def transition_forward(self, style):
        x = self.synthesis_input.expand(style.size(0), -1, -1, -1)
        for index, block in enumerate(self.resl_blocks):
            x = block(style, x)
            if index == self.train_depth-1:
                x_pre = self.upsample(x)
            if index == self.train_depth:
                break
        
        rgb_pre = self.rgb_layers[index-1](x_pre)
        rgb_cur = self.rgb_layers[index](x)
        return (1 - self.alpha) * rgb_pre + self.alpha * rgb_cur

    def stablization_forward(self, style):
        x = self.synthesis_input.expand(style.size(0), -1, -1, -1)
        for index, block in enumerate(self.resl_blocks):
            x = block(style, x)
            if index == self.train_depth:
                break
        
        rgb = self.rgb_layers[index](x)
        return rgb

    def update_alpha(self, delta, phase):
        if phase == 't':
            self.alpha = min(1, self.alpha+delta)

'''
Discriminator
'''

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.train_depth = 0

        self.resl2param = {
            4   : (512, 512, True),
            8   : (512, 512, False),
            16  : (256, 512, False),
            32  : (128, 256, False),
            64  : ( 64, 128, False),
            128 : ( 32,  64, False)
        }

        self.resolution_blocks = nn.ModuleList()
        self.rgb_layers        = nn.ModuleList()
        for resl in self.resl2param:
            param = self.resl2param[resl]
            self.resolution_blocks.append(
                DiscriminatorBlock(
                    param[0], param[1], param[2]
                )
            )
            self.rgb_layers.append(FromRGB(out_channels=param[0]))
        self.downsample = BlurDownsample(2)

        self.alpha = 0

    def grow(self):
        self.train_depth += 1
        self.alpha = 0

    def forward(self, x, phase):
        if phase == 't':
            return self.transition_forward(x)
        else:
            return self.stablization_forward(x)
    
    def transition_forward(self, x):
        size = x.size(2)

        x_down = self.downsample(x)
        x_pre = self.rgb_layers[self.train_depth-1](x_down)

        x = self.rgb_layers[self.train_depth](x)
        x_cur = self.resolution_blocks[self.train_depth](x)

        x = (1 - self.alpha) * x_pre + self.alpha * x_cur
        for block in self.resolution_blocks[self.train_depth-1::-1]:
            x = block(x)

        return x.view(x.size(0), -1)

    def stablization_forward(self, x):
        
        x = self.rgb_layers[self.train_depth](x)
        for block in self.resolution_blocks[self.train_depth::-1]:
            x = block(x)

        return x.view(x.size(0), -1)

    def update_alpha(self, delta, phase):
        if phase == 't':
            self.alpha = min(1, self.alpha+delta)

if __name__ == "__main__":
    # test
    G = Generator(512)
    D = Discriminator()

    for _ in range(4):
        G.grow()
        D.grow()

        style = torch.randn(10, 512)
        out = G(style, 't')
        out = D(out, 't')
        print(out.size())
