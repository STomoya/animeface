
import torch
import torch.nn as nn
import numpy as np

def get_activation(name, inplace=True):
    if name == 'relu': return nn.ReLU(inplace)
    if name == 'lrelu': return nn.LeakyReLU(0.2, inplace=inplace)
    if name == 'tanh': return nn.Tanh()
    if name == 'sigmoid': return nn.Sigmoid()

def get_normalization(name, channels):
    if name == 'bn': return nn.BatchNorm2d(channels)
    if name == 'in': return nn.InstanceNorm2d(channels)

SN = nn.utils.spectral_norm

def _support_sn(sn: bool, layer: nn.Module) -> nn.Module:
    if sn: return SN(layer)
    return layer

def Conv2d(sn: bool, *args, **kwargs):
    layer = nn.Conv2d(*args, **kwargs)
    return _support_sn(sn, layer)

def Linear(sn: bool, *args, **kwargs):
    layer = nn.Linear(*args, **kwargs)
    return _support_sn(sn, layer)

def ConvTranspose2d(sn: bool, *args, **kwargs):
    layer = nn.ConvTranspose2d(*args, **kwargs)
    return _support_sn(sn, layer)

class ConvBlock(nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel_size, stride, padding, bias,
        sn=True, norm_name='in', act_name='lrelu'
    ):
        super().__init__()
        self.conv = nn.Sequential(
            Conv2d(sn,
                in_channels, out_channels, kernel_size,
                stride, padding, bias=bias),
            get_normalization(norm_name, out_channels),
            get_activation(act_name)
        )
        self.out_channels = out_channels
    def forward(self, x):
        return self.conv(x)

class ResBlock(nn.Module):
    def __init__(self,
        channels, num_conv=2,
        sn=True, bias=True, norm_name='in', act_name='lrelu'
    ):
        super().__init__()

        layers = []
        for _ in range(num_conv-1):
            layers.extend([
                Conv2d(sn, channels, channels, 3, 1, 1, bias=bias),
                get_normalization(norm_name, channels),
                get_activation(act_name)
            ])
        layers.extend([
            Conv2d(sn, channels, channels, 3, 1, 1, bias=bias),
            get_normalization(norm_name, channels)
        ])
        self.conv = nn.Sequential(*layers)
        self.act = get_activation(act_name, False)
    
    def forward(self, x):
        h = x
        h = self.conv(h)
        return self.act(x + h) / np.sqrt(2)

def ResBlocks(
    num_blocks, channels, num_conv=2,
    sn=True, bias=True, norm_name='in', act_name='lrelu'
):
    layers = []
    for _ in range(num_blocks):
        layers.append(
            ResBlock(
                channels, num_conv,
                sn, bias, norm_name, act_name)
        )
    return nn.Sequential(*layers)

class Encoder(nn.Module):
    def __init__(self,
        in_channels, image_size, bottom_width=8, channels=16, layer_per_resl=2,
        sn=True, bias=True, norm_name='in', act_name='lrelu'
    ):
        super().__init__()
        num_down = int(np.log2(image_size)-np.log2(bottom_width))

        def _make_resl_block(ichannels, ochannels, stride) -> list:
            _layers = []
            for i in range(layer_per_resl):
                if i == 0:
                    _layers.append(
                        ConvBlock(
                            ichannels, ochannels, 3, stride, 1,
                            bias, sn, norm_name, act_name)
                    )
                else:
                    _layers.append(
                        ConvBlock(
                            ochannels, ochannels, 3, 1, 1,
                            bias, sn, norm_name, act_name
                        )
                    )
            return _layers

        ichannels, ochannels = in_channels, channels
        layers = _make_resl_block(ichannels, ochannels, 1)
        for i in range(num_down):
            channels *= 2
            ichannels, ochannels = ochannels, channels
            layers.extend(
                _make_resl_block(ichannels, ochannels, 2)
            )
        
        self.encoder = nn.ModuleList(layers)
        self.avgpool = nn.AdaptiveAvgPool2d(bottom_width)
        self.out_channels = sum([conv.out_channels for conv in self.encoder])
    
    def forward(self, x):

        feats = []
        for module in self.encoder:
            x = module(x)
            feats.append(x)
        
        out = torch.cat([self.avgpool(feat) for feat in feats], dim=1)

        return out, feats

class Decoder(nn.Module):
    def __init__(self,
        in_channels, image_size, out_channels,
        bottom_width=8, channels=16, layer_per_resl=2,
        sn=True, bias=True, norm_name='in', act_name='lrelu'
    ):
        super().__init__()
        num_ups = int(np.log2(image_size)-np.log2(bottom_width))

        def _make_resl_block(ichannels, ochannels, last_act_name=act_name):
            _layers = []
            for i in range(layer_per_resl):
                if i == layer_per_resl-1:
                    _layers.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                            ConvBlock(
                                ichannels//2+ochannels, ochannels, 3, 1, 1,
                                bias, sn, norm_name, last_act_name
                            )
                        )
                    )
                elif i == 0:
                    _layers.append(
                        ConvBlock(
                            ichannels, ochannels, 3, 1, 1,
                            bias, sn, norm_name, act_name
                        )
                    )
                else:
                    _layers.append(
                        ConvBlock(
                            ichannels//2+ochannels, ochannels, 3, 1, 1,
                            bias, sn, norm_name, act_name
                        )
                    )
            return _layers
        
        channels = channels * 2 ** num_ups
        ichannels, ochannels = in_channels*2, channels
        layers = [
            ConvBlock(ichannels, ochannels, 3, 1, 1, bias, sn, norm_name, act_name)
        ]
        for _ in range(num_ups):
            channels = channels // 2
            ichannels, ochannels = ochannels*2, channels
            layers.extend(
                _make_resl_block(ichannels, ochannels)
            )
        layers.extend([
            ConvBlock(ochannels*2, channels, 3, 1, 1, bias, sn, norm_name, act_name),
            nn.Sequential(
                Conv2d(sn, channels*2, out_channels, 3, 1, 1, bias=bias),
                get_activation('tanh')
            )
        ])
        self.decoder = nn.ModuleList(layers)
    
    def forward(self, x, res, feats):
        feats.append(res)
        feats = feats[::-1]

        for module, feat in zip(self.decoder, feats):
            x = module(torch.cat([x, feat], dim=1))

        return x

class SCFT(nn.Module):
    def __init__(self,
        channels,
        sn=True, bias=False
    ):
        super().__init__()
        self.kv = Linear(sn, channels, channels*2, bias=bias)
        self.q = Linear(sn, channels, channels, bias=bias)
        self.scale = np.sqrt(channels)

    def forward(self, ref, sketch):
        if ref.dim() == 4:
            # [B,HW,C]
            ref = ref.reshape(ref.size(0), ref.size(1), -1).permute(0, 2, 1)
            sketch = sketch.reshape(sketch.size(0), sketch.size(1), -1).permute(0, 2, 1)
        B, HW, C = ref.size()

        # [3,B,HW,C]
        kv = self.kv(ref).reshape(B, HW, 2, C).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]
        q = self.q(sketch)

        attn = q @ k.permute(0, 2, 1)
        attn = attn.softmax(-1) / self.scale

        heat = attn @ v
        out = sketch + heat

        H = int(np.sqrt(HW))
        out = out.permute(0, 2, 1).reshape(B, C, H, H)

        return out, [q, k]

class Generator(nn.Module):
    def __init__(self,
        image_size, in_channels=1, ref_channels=3, bottom_width=8,
        enc_channels=16, layer_per_resl=2, num_res_blocks=7,
        sn=True, bias=True, scft_bias=False, norm_name='in', act_name='lrelu'
    ):
        super().__init__()
        self.ref_encoder = Encoder(
            ref_channels, image_size, bottom_width,
            enc_channels, layer_per_resl, sn, bias,
            norm_name, act_name
        )
        self.sketch_encoder = Encoder(
            in_channels, image_size, bottom_width,
            enc_channels, layer_per_resl,
            sn, bias, norm_name, act_name
        )
        self.scft = SCFT(
            self.ref_encoder.out_channels, sn, scft_bias
        )
        self.resblocks = ResBlocks(
            num_res_blocks, self.ref_encoder.out_channels,
            2, sn, bias, norm_name, act_name
        )
        self.decoder = Decoder(
            self.sketch_encoder.out_channels, image_size, ref_channels,
            bottom_width, enc_channels, layer_per_resl,
            sn, bias, norm_name, act_name
        )
    def forward(self, sketch, ref, return_qk=False):
        sketch, feats = self.sketch_encoder(sketch)
        ref, _ = self.ref_encoder(ref)
        scft, qk = self.scft(ref, sketch)
        res = self.resblocks(scft)
        image = self.decoder(sketch, res, feats)
        if return_qk:
            return image, qk
        return image


class Discriminator(nn.Module):
    def __init__(self,
        image_size, in_channels, num_layers=3, channels=32,
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

def init_weight_N002(m):
    if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
        m.weight.data.normal_(0, 0.02)
        if not m.bias == None:
            m.bias.data.fill_(0.)
    if isinstance(m, nn.BatchNorm2d):
        if not m.weight == None:
            m.weight.fill_(0)
        if not m.bias == None:
            m.bias.data.fill_(0.)

def init_weight_xavier(m):
    if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
        nn.init.xavier_normal_(m.weight)
        if not m.bias == None:
            m.bias.data.fill_(0.)
    elif isinstance(m, nn.BatchNorm2d):
        if not m.weight == None:
            m.weight.data.fill_(1.)
        if not m.bias == None:
            m.bias.data.fill_(0.)

def init_weight_kaiming(m):
    if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight)
        if not m.bias == None:
            m.bias.data.fill_(0.)
    elif isinstance(m, nn.BatchNorm2d):
        if not m.weight == None:
            m.weight.data.sill_(1.)
        if not m.bias == None:
            m.bias.data.fill_(0.)

if __name__=='__main__':
    x = torch.randn(10, 3, 128, 128)
    y = torch.randn(10, 1, 128, 128)
    g = Generator(128)
    d = Discriminator(128, 3)

    img = g(y, x)
    prob = d(img)
    print(prob[0].size())
    print(img.size())