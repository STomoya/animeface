
import torch
import torch.nn as nn
import numpy as np

def get_activation(name, inplace=True):
    if name == 'relu': return nn.ReLU(inplace)
    if name == 'lrelu': return nn.LeakyReLU(0.2, inplace=inplace)
    if name == 'tanh': return nn.Tanh()

def get_normalization(name, channels):
    if name == 'in': return nn.InstanceNorm2d(channels)
    if name == 'bn': return nn.BatchNorm2d(channels)

def _support_sn(sn, layer):
    if sn: return nn.utils.spectral_norm(layer)
    return layer

def Conv2d(sn, *args, **kwargs):
    layer = nn.Conv2d(*args, **kwargs)
    return _support_sn(sn, layer)
def Linear(sn, *args, **kwargs):
    layer = nn.Linear(*args, **kwargs)
    return _support_sn(sn, layer)
def ConvTranspose2d(sn, *args, **kwargs):
    layer = nn.ConvTranspose2d(*args, **kwargs)
    return _support_sn(sn, layer)

class Block(nn.Module):
    def __init__(self,
        in_channels, out_channels, stride=1,
        sn=True, bias=False, norm_name='in', act_name='lrelu'
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            Conv2d(sn, in_channels, out_channels, 3, stride, bias=bias),
            get_normalization(norm_name, out_channels),
            nn.ReflectionPad2d(1),
            Conv2d(sn, out_channels, out_channels, 3, bias=bias),
            get_normalization(norm_name, out_channels)
        )
        self.tail = nn.Sequential(
            nn.ReflectionPad2d(1),
            Conv2d(sn, out_channels*2, out_channels, 3, bias=bias),
            get_normalization(norm_name, out_channels),
            get_activation(act_name)
        )
        if stride < 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                Conv2d(sn, in_channels, out_channels, 1, stride, bias=bias),
                get_normalization(norm_name, out_channels)
            )
    
    def forward(self, x):
        h = self.head(x)
        if hasattr(self, 'skip'):
            x = self.skip(x)
        x = self.tail(torch.cat([h, x], dim=1))
        return x

class Layer(nn.Module):
    def __init__(self,
        in_channels, out_channels, stride, num_block=2,
        sn=True, bias=False, norm_name='in', act_name='lrelu'
    ):
        super().__init__()
        layers = []
        for i in range(num_block):
            stride = stride if i == 0 else 1
            layers.append(
                Block(
                    in_channels, out_channels, stride,
                    sn, bias, norm_name, act_name))
            in_channels = out_channels
        self.layer = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layer(x)

class Encoder(nn.Module):
    def __init__(self,
        image_size, in_channels=3, bottom_width=8, num_downs=None,
        num_ret_feats=3, channels=32, layer_num_blocks=2,
        sn=True, bias=True, norm_name='in', act_name='lrelu'
    ):
        super().__init__()
        self.num_ret_feats = num_ret_feats
        if num_downs is None:
            num_downs = int(np.log2(image_size) - np.log2(bottom_width))
        assert num_ret_feats < num_downs

        # input
        self.input = nn.Sequential(
            nn.ReflectionPad2d(3),
            Conv2d(sn, in_channels, channels, 7, bias=bias),
            get_normalization(norm_name, channels),
            get_activation(act_name),
            nn.ReflectionPad2d(1),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        ochannels = channels
        layers = []
        # GANILLA layers
        for i in range(num_downs):
            channels *= 2
            ichannels, ochannels = ochannels, channels
            stride = 1 if i == 0 else 2
            layers.append(
                Layer(
                    ichannels, ochannels, stride, layer_num_blocks,
                    sn, bias, norm_name, act_name))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = self.input(x)
        feats = [x]
        for module in self.layers:
            feats.append(module(feats[-1]))
        return feats[-1], feats[-(self.num_ret_feats+1):-1]

class Decoder(nn.Module):
    def __init__(self,
        image_size, out_channels=3, bottom_width=8, num_ups=None,
        num_feats=3, channels=32, layer_num_blocks=2,
        sn=True, bias=True, norm_name='in', act_name='lrelu'
    ):
        super().__init__()
        if num_ups is None:
            num_ups = int(np.log2(image_size) - np.log2(bottom_width))
        channels = channels * 2 ** num_ups

        # input
        self.input = nn.Sequential(
            nn.ReflectionPad2d(1),
            Conv2d(sn, channels, channels//2, 3, bias=bias)
        )

        convs = []
        # convs for feats from encoder
        for _ in range(num_feats):
            channels = channels // 2
            convs.append(
                nn.Sequential(
                    nn.ReflectionPad2d(1),
                    Conv2d(sn, channels, channels//2, 3, bias=bias)))
        self.convs = nn.ModuleList(convs)
        # upsampling
        self.upsample = nn.Upsample(scale_factor=2)

        # additional upsamples
        if num_ups != num_feats:
            add_ups = num_ups - num_feats - 1
            ups = []
            for _ in range(add_ups):
                channels = channels // 2
                ups.append(
                    nn.Sequential(
                        nn.ReflectionPad2d(1),
                        Conv2d(sn, channels, channels//2, 3, bias=bias)))
            self.ups = nn.ModuleList(ups)

        channels = channels // 2
        # output
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            Conv2d(sn, channels, out_channels, 7, bias=bias),
            get_activation('tanh')
        )

    def forward(self, x, feats):
        x = self.input(x)
        x = self.upsample(x)
        for module, feat in zip(self.convs, feats[::-1]):
            x = x + feat
            x = module(x)
            x = self.upsample(x)
        if hasattr(self, 'ups'):
            for module in self.ups:
                x = module(x)
                x = self.upsample(x)
        return self.output(x)

class Generator(nn.Module):
    def __init__(self,
        image_size, image_channels=3, bottom_width=8, num_downs=None,
        num_feats=3, channels=32, layer_num_blocks=2,
        sn=True, bias=True, norm_name='in', act_name='lrelu'
    ):
        super().__init__()

        self.encoder = Encoder(
            image_size, image_channels, bottom_width,
            num_downs, num_feats, channels, layer_num_blocks,
            sn, bias, norm_name, act_name
        )
        self.decoder = Decoder(
            image_size, image_channels, bottom_width,
            num_downs, num_feats, channels, layer_num_blocks,
            sn, bias, norm_name, act_name
        )

    def forward(self, x):
        enc_out = self.encoder(x)
        img = self.decoder(*enc_out)
        return img

class Discriminator(nn.Module):
    def __init__(self,
        image_size, in_channels=3, num_layers=3, channels=32,
        sn=True, bias=True, norm_name='in', act_name='lrelu'
    ):
        super().__init__()

        ochannels = channels
        layers = [
            nn.Sequential(
                Conv2d(sn, in_channels, ochannels, 4, 2, bias=bias),
                get_activation(act_name)
            )
        ]
        for _ in range(num_layers):
            layers.append(
                nn.Sequential(
                    Conv2d(sn, channels, channels*2, 4, 2, bias=bias),
                    get_normalization(norm_name, channels*2),
                    get_activation(act_name)
                )
            )
            channels *= 2
        layers.append(
            Conv2d(sn, channels, 1, 4, bias=bias)
        )
        self.disc = nn.ModuleList(layers)
    def forward(self, x):
        out = []
        for module in self.disc:
            x = module(x)
            out.append(x)
        return out[-1], out[:-1]

if __name__=='__main__':
    g = Generator(128)
    d = Discriminator(128)
    x = torch.randn(10, 3, 128, 128)
    img = g(x)
    prob = d(img)
    print(img.size())
    print(prob[0].size())
