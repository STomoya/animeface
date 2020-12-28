
import torch
import torch.nn as nn

def get_normalization(name, channels, **kwargs):
    if name == 'bn': return nn.BatchNorm2d(channels, **kwargs)
    elif name == 'in': return nn.InstanceNorm2d(channels, **kwargs)
    else: return nn.Identity()

def get_activation(name, inplace=True):
    if name == 'relu': return nn.ReLU(inplace=inplace)
    elif name == 'lrelu': return nn.LeakyReLU(0.2, inplace=inplace)
    else: return nn.Identity()

class Conv2dBlock(nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel_size, stride, padding,
        norm_name, act_name, padding_mode='reflect'
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode=padding_mode),
            get_normalization(norm_name, out_channels),
            get_activation(act_name, True)
        )
    def forward(self, x):
        return self.block(x)

class ConvTranspose2dBlock(nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel_size, stride, padding, output_padding,
        norm_name, act_name
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=output_padding),
            get_normalization(norm_name, out_channels),
            get_activation(act_name, True)
        )
    def forward(self, x):
        return self.block(x)

class ResBlock(nn.Module):
    def __init__(self,
        channels, norm_name, act_name
    ):
        super().__init__()
        self.block = nn.Sequential(
            Conv2dBlock(channels, channels, 3, 1, 1, norm_name, act_name),
            Conv2dBlock(channels, channels, 3, 1, 1, norm_name, None)
        )
    def forward(self, x):
        h = self.block(x)
        return h + x

class GlobalG(nn.Module):
    def __init__(self,
        in_channels, out_channels, channels, num_downs=3, num_blocks=9,
        norm_name='in', act_name='relu'
    ):
        super().__init__()

        layers = [Conv2dBlock(in_channels, channels, 7, 1, 3, norm_name, act_name)]
        for _ in range(num_downs):
            layers.append(
                Conv2dBlock(channels, channels*2, 3, 2, 1, norm_name, act_name, 'zeros')
            )
            channels = channels * 2
        for _ in range(num_blocks):
            layers.append(
                ResBlock(channels, norm_name, act_name)
            )
        for _ in range(num_downs):
            layers.append(
                ConvTranspose2dBlock(channels, channels//2, 3, 2, 1, 1, norm_name, act_name)
            )
            channels = channels // 2
        self.extract = nn.Sequential(*layers)
        self.output = nn.Sequential(
            nn.Conv2d(channels, out_channels, 7, 1, 3, padding_mode='reflect'),
            nn.Tanh()
        )
    def forward(self, x):
        feat = self.extract(x)
        out = self.output(feat)
        return feat, out

class LocalG(nn.Module):
    def __init__(self,
        in_channels, out_channels, channels, num_blocks=3,
        norm_name='in', act_name='relu'
    ):
        super().__init__()

        self.down = nn.Sequential(
            Conv2dBlock(in_channels, channels, 7, 1, 3, norm_name, act_name),
            Conv2dBlock(channels, channels*2, 3, 2, 1, norm_name, act_name, 'zeros')
        )
        channels = channels * 2
        layers = []
        for _ in range(num_blocks):
            layers.append(
                ResBlock(channels, norm_name, act_name)
            )
        layers.extend([
            ConvTranspose2dBlock(channels, channels//2, 3, 2, 1, 1, norm_name, act_name),
            nn.Conv2d(channels//2, out_channels, 7, 1, 3, padding_mode='reflect'),
            nn.Tanh()
        ])
        self.output = nn.Sequential(*layers)

    def forward(self, x, global_feat):
        feat = self.down(x)
        out = self.output(feat + global_feat)
        return out

class Generator(nn.Module):
    def __init__(self,
        in_channels, out_channels, channels,
        local_num_blocks=3, global_num_blocks=9, global_num_downs=4,
        norm_name='in', act_name='relu'
    ):
        super().__init__()
        self.downsample = nn.AvgPool2d(2)
        self.local_G = LocalG(
            in_channels, out_channels, channels, local_num_blocks,
            norm_name, act_name
        )
        self.global_G = GlobalG(
            in_channels, out_channels, channels*2, global_num_downs, global_num_blocks,
            norm_name, act_name
        )

    def forward(self, x):
        low_x = self.downsample(x)
        g_feat, g_image = self.global_G(low_x)
        l_image = self.local_G(x, g_feat)
        return l_image, g_image

class SingleScaleDiscriminator(nn.Module):
    def __init__(self,
        in_channels, channels,
        norm_name='in', act_name='lrelu'        
    ):
        super().__init__()
        layers = [Conv2dBlock(in_channels, channels, 4, 2, 1, '', act_name, 'zeros')]
        for _ in range(3):
            layers.append(
                Conv2dBlock(channels, channels*2, 4, 2, 1, norm_name, act_name, 'zeros')
            )
            channels = channels * 2
        layers.append(nn.Conv2d(channels, 1, 3, padding=1))
        self.discriminate = nn.ModuleList(layers)
        
    def forward(self, x):
        out = []
        for module in self.discriminate:
            x = module(x)
            out.append(x)
        return out[-1], out[:-1]
            

class Discriminator(nn.Module):
    def __init__(self,
        in_channels, channels, num_scales,
        norm_name='in', act_name='lrelu'
    ):
        super().__init__()
        self.num_scales = num_scales

        self.downsample = nn.AvgPool2d(2)
        discs = []
        for _ in range(num_scales):
            discs.append(SingleScaleDiscriminator(in_channels, channels, norm_name, act_name))
        self.discriminates = nn.ModuleList(discs)

    def forward(self, x):
        '''
        returns a list of tuples which contains
        1) real/fake probability 2) list of features of each layers
        for each scale
        (e.g. output[0][1][0] will be the feature of the first layer for the top scale)
        '''
        outs = []
        for module in self.discriminates:
            outs.append(module(x))
            x = self.downsample(x)
        return outs

def init_weight_normal(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        m.weight.data.normal_(0., 0.02)
        if not m.bias == None:
            m.bias.data.fill_(0.)
    if isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1, 0.02)
        m.bias.data.fill_(0.)

if __name__ == "__main__":
    g = Generator(1, 3, 32)
    image = torch.randn(4, 1, 128, 128)
    out, _ = g(image)
    d = Discriminator(3, 32, 3)
    dout = d(out)
    print(dout[0][0].size(), dout[0][1][0].size())
