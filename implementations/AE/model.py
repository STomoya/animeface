
import functools
import math

import torch
import torch.nn as nn

def normalization(name, *args, **kwargs):
    if name == 'bn': return nn.BatchNorm2d(*args, **kwargs)
    elif name == 'in': return nn.InstanceNorm2d(*args, **kwargs)
    else: return nn.Identity()

def activation(name, *args, **kwargs):
    if name == 'relu': return nn.ReLU(*args, **kwargs)
    elif name == 'sigmoid': return nn.Sigmoid()
    elif name == 'tanh': return nn.Tanh()
    else: raise Exception(f'{name} not supported')

class ConvBlock(nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel_size,
        norm_name='bn', act_name='relu', **kwargs
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs),
            normalization(norm_name, out_channels),
            activation(act_name)
        )
    def forward(self, x):
        return self.block(x)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1).contiguous()

class View(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
    def forward(self, x):
        return x.view(x.size(0), *self.size).contiguous()

class Encoder(nn.Module):
    def __init__(self,
        num_layers, in_channels, channels, out_dim, fin_size,
        downsample=False, norm_name='bn', act_name='relu'
    ):
        super().__init__()
        convb_func = functools.partial(ConvBlock, norm_name=norm_name, act_name=act_name)

        stride = 2 if downsample else 1
        layers = [convb_func(in_channels, channels, 3, padding=1)] \
            + [convb_func(channels, channels, 3, stride=stride, padding=1) for _ in range(num_layers)] \
            + [Flatten(), nn.Linear(channels*fin_size**2, out_dim)]
        self.encode = nn.Sequential(*layers)
    def forward(self, x):
        return self.encode(x)
        
class Decoder(nn.Module):
    def __init__(self,
        num_layers, in_dim, channels, out_channels, init_size,
        upsample=False, norm_name='bn', act_name='relu', up_mode='bilinear', output_act='sigmoid'
    ):
        super().__init__()
        convb_func = functools.partial(ConvBlock, norm_name=norm_name, act_name=act_name)

        upsample = nn.Upsample(scale_factor=2, mode=up_mode, align_corners=False) if upsample else nn.Identity()
        layers = [nn.Linear(in_dim, channels*init_size**2), activation(act_name),
                  View((channels, init_size, init_size))]
        for _ in range(num_layers):
            layers += [convb_func(channels, channels, 3, padding=1), upsample]
        layers += [nn.Conv2d(channels, out_channels, 3, padding=1),
                   activation(output_act) if output_act is not 'sigmoid' else nn.Identity()] # nn.Identity for AMP. use BCEWithLogitsLoss, not nn.Sigmoid + BCELoss
        self.decode = nn.Sequential(*layers)
    def forward(self, x):
        return self.decode(x)

class AE(nn.Module):
    def __init__(self,
        enc_dim, image_size, min_size=8, num_layers=None, img_channels=3, channels=64,
        norm_name='bn', act_name='relu', up_mode='bilinear', output_act='sigmoid'
    ):
        super().__init__()
        if num_layers:
            assert math.log2(image_size) > num_layers
            # calculate minimun by from num_layers
            min_size = 2 ** int(math.log2(image_size) - num_layers +1)
        elif min_size:
            # calculate number of layers by min_size
            num_layers = int(math.log2(image_size) - math.log2(min_size))
        else: raise Exception('at least value needed in either "min_size" or "num_layers"')
        
        self.encoder = Encoder(num_layers, img_channels, channels, enc_dim, min_size,
                                True, norm_name, act_name)
        self.decoder = Decoder(num_layers, enc_dim, channels, img_channels,
                                min_size, True, norm_name, act_name, up_mode, output_act)
    def forward(self, x, ret_feature=False):
        feat  = self.encoder(x)
        recon = self.decoder(feat)

        if ret_feature: return recon, feat
        else:           return recon

if __name__ == "__main__":
    from torchsummary import summary
    ae = AE(128, 512, channels=64)
    summary(ae, torch.randn(10, 3, 512, 512))
