
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
layers
'''

class Conv2d(nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel_size, **kwargs
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, **kwargs
        )
        self.conv.weight.data.normal_(0., 1.)
        if not self.conv.bias == None:
            self.conv.bias.data.fill_(0.)
    def forward(self, x):
        x = self.conv(x)
        return x

class SNConv2d(nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel_size, **kwargs
    ):
        super().__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        conv.weight.data.normal_(0., 1.)
        if conv.bias.data == None:
            conv.bias.data.fill_(0.)
        self.conv = nn.utils.spectral_norm(conv)
    def forward(self, x):
        x = self.conv(x)
        return x

class Linear(nn.Module):
    def __init__(self,
        in_features, out_features, **kwargs
    ):
        super().__init__()
        self.linear = nn.Linear(
            in_features, out_features, **kwargs
        )
        self.linear.weight.data.normal_(0., 1.)
    def forward(self, x):
        x = self.linear(x)
        return x

class ILN(nn.Module):
    def __init__(self,
        channels, resl, eps=1.e-8
    ):
        super().__init__()
        
        self.rho = nn.Parameter(torch.Tensor(1, channels, 1, 1))
        self.rho.data.fill_(0.)

        self.instance_norm = nn.InstanceNorm2d(channels, eps=eps, affine=False)
        self.layer_norm    = nn.LayerNorm((channels, resl, resl), eps=eps, elementwise_affine=False)

        self.gamma = nn.Parameter(torch.Tensor(1, channels, 1, 1))
        self.beta  = nn.Parameter(torch.Tensor(1, channels, 1, 1))
        self.gamma.data.fill_(1.)
        self.beta.data.fill_(0.)

    def forward(self, x):
        i_norm = self.instance_norm(x)
        l_norm = self.layer_norm(x)
        out = i_norm * self.rho.expand(x.size(0), -1, -1, -1) + l_norm * (1 - self.rho.expand(x.size(0), -1, -1, -1))
        out = out * self.gamma.expand(x.size(0), -1, -1, -1) + self.beta.expand(x.size(0), -1, -1, -1)
        return out

class AdaILN(nn.Module):
    def __init__(self,
        channels, resl, eps=1.e-8
    ):
        super().__init__()
        
        self.rho = nn.Parameter(torch.Tensor(1, channels, 1, 1))
        self.rho.data.fill_(1.)

        self.instance_norm = nn.InstanceNorm2d(channels, eps=eps, affine=False)
        self.layer_norm    = nn.LayerNorm((channels, resl, resl), eps=eps, elementwise_affine=False)

    def forward(self, x, gamma, beta):
        i_norm = self.instance_norm(x)
        l_norm = self.layer_norm(x)
        out = i_norm * self.rho.expand(x.size(0), -1, -1, -1) + l_norm * (1 - self.rho.expand(x.size(0), -1, -1, -1))
        out = out * gamma.view(out.size(0), -1, 1, 1) + beta.view(out.size(0), -1, 1, 1)
        return out

class CAM(nn.Module):
    def __init__(self,
        channels
    ):
        super().__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.global_avg_pool_fc = Linear(channels, 1, bias=False)
        self.global_max_pool_fc = Linear(channels, 1, bias=False)

        self.conv = Conv2d(channels*2, channels, 1, bias=True)

    def forward(self, x):

        gap = self.global_avg_pool(x)
        gap_logit = self.global_avg_pool_fc(gap.view(x.size(0), -1))
        gap_weight = self.global_avg_pool_fc.linear.weight.data.clone()
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = self.global_max_pool(x)
        gmp_logit = self.global_max_pool_fc(gmp.view(x.size(0), -1))
        gmp_weight = self.global_max_pool_fc.linear.weight.data.clone()
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap, gmp], dim=1)
        x = self.conv(cam_logit)
        return x, cam_logit

class GammaBeta(nn.Module):
    def __init__(self,
        channels, resl
    ):
        super().__init__()

        self.fc = nn.Sequential(
            Linear(channels*resl*resl, channels, bias=False),
            nn.ReLU(),
            Linear(channels, channels, bias=False),
            nn.ReLU()
        )
        self.gamma = Linear(channels, channels, bias=False)
        self.beta  = Linear(channels, channels, bias=False)

    def forward(self, x):
        x = self.fc(x.view(x.size(0), -1))
        gamma = self.gamma(x)
        beta = self.beta(x)
        return gamma, beta

'''
blocks
'''


class DownsampleBlock(nn.Module):
    def __init__(self,
        channels,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(3),
            Conv2d(3, channels, 7, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            Conv2d(channels, channels*2, 3, stride=2, bias=False),
            nn.InstanceNorm2d(channels*2),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            Conv2d(channels*2, channels*4, 3, stride=2, bias=False),
            nn.InstanceNorm2d(channels*2),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.block(x)
        return x

class UpsampleBlock(nn.Module):
    def __init__(self,
        channels, resl,
        upscale_mode='nearest'
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=upscale_mode),
            nn.ReflectionPad2d(1),
            Conv2d(channels, channels//2, 3, bias=False),
            ILN(channels//2, resl*2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode=upscale_mode),
            nn.ReflectionPad2d(1),
            Conv2d(channels//2, channels//4, 3, bias=False),
            ILN(channels//4, resl*4),
            nn.ReLU(),
            nn.ReflectionPad2d(3),
            Conv2d(channels//4, 3, 7, bias=False),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.block(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self,
        channels, bias
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            Conv2d(channels, channels, 3, bias=bias),
            nn.InstanceNorm2d(channels),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            Conv2d(channels, channels, 3, bias=bias),
            nn.InstanceNorm2d(channels)
        )
    def forward(self, x):
        x = x + self.block(x)
        return F.relu(x)

class ResnetAdaILNBlock(nn.Module):
    def __init__(self,
        channels, resl, bias
    ):
        super().__init__()

        self.pad0  = nn.ReflectionPad2d(1)
        self.conv0 = Conv2d(channels, channels, 3, bias=bias)
        self.norm0 = AdaILN(channels, resl)
        self.pad1  = nn.ReflectionPad2d(1)
        self.conv1 = Conv2d(channels, channels, 3, bias=bias)
        self.norm1 = AdaILN(channels, resl)

    def forward(self, x, gamma, beta):

        x_ = self.pad0(x)
        x_ = self.conv0(x_)
        x_ = self.norm0(x_, gamma, beta)
        x_ = F.relu(x_)

        x_ = self.pad1(x_)
        x_ = self.conv1(x_)
        x_ = self.norm1(x_, gamma, beta)

        x = x + x_
        return F.relu(x)

'''
Generator
'''

class Generator(nn.Module):
    def __init__(self,
        image_size=128,
        channels=64,
        n_blocks=6,
        upscale_mode='nearest'
    ):
        super().__init__()

        # image_size -> image_size // 4
        # channel    -> channel * 4
        self.downsmaple = DownsampleBlock(channels)
        self.encoder = nn.ModuleList([ResnetBlock(channels*4, False) for _ in range(n_blocks)])

        self.cam = CAM(channels*4)
        self.gamma_beta = GammaBeta(channels*4, image_size//4)

        self.decoder = nn.ModuleList([ResnetAdaILNBlock(channels*4, image_size//4, False) for _ in range(n_blocks)])
        # image_size -> image_size * 4
        # channel    -> 3
        self.upsample = UpsampleBlock(channels*4, image_size//4, upscale_mode)

    def forward(self, x):

        x = self.downsmaple(x)

        for layer in self.encoder:
            x = layer(x)
        
        x, cam_logit = self.cam(x)
        x = F.relu(x)
        
        gamma, beta = self.gamma_beta(x)

        for layer in self.decoder:
            x = layer(x, gamma, beta)

        x = self.upsample(x)

        return x, cam_logit


'''
Discriminator
'''

class Discriminator(nn.Module):
    def __init__(self,
        channels=64,
        n_layers=3
    ):
        super().__init__()

        head = [
            nn.ReflectionPad2d(1),
            SNConv2d(3, channels, 4, stride=2),
            nn.LeakyReLU(0.2)
        ]

        times = 1
        for _ in range(1, n_layers-1):
            head += [
                nn.ReflectionPad2d(1), 
                SNConv2d(channels*times, channels*times*2, 4, stride=2),
                nn.LeakyReLU(0.2)
            ]
            times *= 2

        head += [
            nn.ReflectionPad2d(1),
            SNConv2d(channels*times, channels*times*2, 4),
            nn.LeakyReLU(0.2)
        ]

        self.head = nn.Sequential(*head)
        self.cam = CAM(channels*times*2)
        self.tale = nn.Sequential(
            nn.ReflectionPad2d(1),
            SNConv2d(channels*times*2, 1, 4)
        )
    
    def forward(self, x):

        x = self.head(x)
        x, cam_logit = self.cam(x)
        x = F.leaky_relu(x, 0.2)
        x = self.tale(x)
        return x, cam_logit

if __name__ == "__main__":
    G = Generator()
    D = Discriminator(n_layers=5)

    x = torch.Tensor(10, 3, 128, 128).normal_(0.5, 0.5)
    image, g_cam_logit = G(x)
    print(image.size())

    logit, d_cam_logit = D(image)
    print(logit.size())