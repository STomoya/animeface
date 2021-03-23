
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_activation(name, inplace=True):
    if name == 'relu': return nn.ReLU(inplace=inplace)
    if name == 'lrelu': return nn.LeakyReLU(0.1, inplace=inplace)
    if name == 'tanh': return nn.Tanh()
    if name == 'sigmoid': return nn.Sigmoid()

def get_normalization(name, channels):
    if name == 'bn': return nn.BatchNorm2d(channels)
    if name == 'in': return nn.InstanceNorm2d(channels)

SN = nn.utils.spectral_norm
class ELR(nn.Module):
    def __init__(self, layer, gain=1.):
        super().__init__()
        self.coef = gain / (layer.weight[0].numel() ** 0.5)
        self.layer = layer
    def forward(self, x):
        # x*w* coef + bias
        x = x * self.coef
        return self.layer(x)

def Linear(norm, *args, **kwargs):
    layer = nn.Linear(*args, **kwargs)
    if norm == 'sn': return SN(layer)
    if norm == 'elr': return ELR(layer)
    return layer

def Conv2d(norm, *args, **kwargs):
    layer = nn.Conv2d(*args, **kwargs)
    if norm == 'sn': return SN(layer)
    if norm == 'elr': return ELR(layer)
    return layer

def ConvTranspose2d(norm, *args, **kwargs):
    layer = nn.ConvTranspose2d(*args, **kwargs)
    if norm == 'sn': return SN(layer)
    if norm == 'elr': return ELR(layer)
    return layer

class BasicBlock(nn.Module):
    def __init__(self,
        in_channels, out_channels, stride=1,
        weight_norm='sn', bias=True, norm_name='in', act_name='lrelu'
    ):
        super().__init__()
        self.conv = nn.Sequential(
            get_normalization(norm_name, in_channels),
            get_activation(act_name),
            Conv2d(weight_norm, in_channels, out_channels, 3, stride, 1, bias=bias),
            get_normalization(norm_name, out_channels),
            get_activation(act_name),
            Conv2d(weight_norm, out_channels, out_channels, 3, 1, 1, bias=bias)
        )
        if in_channels != out_channels or stride > 1:
            self.skip = Conv2d(weight_norm, in_channels, out_channels, 1, stride, bias=bias)
        else: self.skip = None
    
    def forward(self, x):
        h = x
        h = self.conv(h)
        if self.skip is not None:
            x = self.skip(x)
        return (h + x) / np.sqrt(2)

class ResNet(nn.Module):
    '''resnet extractor (not used)'''
    def __init__(self,
        blocks, in_channels=3, channels=64,
        weight_norm='sn', bias=True, norm_name='in', act_name='lrelu'
    ):
        super().__init__()
        assert len(blocks) == 4

        self.out_channels = channels
        self.input = nn.Sequential(
            Conv2d(weight_norm, in_channels, self.out_channels, 3, 1, 1, bias=bias),
            nn.MaxPool2d((2, 2))
        )
    
        def _make_layer(num_block, channels, stride):
            strides = [stride] + [1]*(num_block-1)
            layers = []
            for stride in strides:
                layers.append(
                    BasicBlock(
                        self.out_channels, channels, stride,
                        weight_norm, bias, norm_name, act_name
                    )
                )
                self.out_channels = channels
            return nn.Sequential(*layers)

        self.layer1 = _make_layer(blocks[0], channels*2**0, 1)
        self.layer2 = _make_layer(blocks[1], channels*2**1, 2)
        self.layer3 = _make_layer(blocks[2], channels*2**2, 2)
        self.layer4 = _make_layer(blocks[3], channels*2**3, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.input(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)

class Projector(nn.Module):
    def __init__(self,
        in_channels, projection_features=128, hidden_features=128,
        weight_norm='sn', bias=True, act_name='lrelu'
    ):
        super().__init__()

        def _mlp(out_channels):
            return nn.Sequential(
                Linear(weight_norm, in_channels, hidden_features, bias=bias),
                get_activation(act_name),
                Linear(weight_norm, hidden_features, out_channels, bias=bias)
            )

        self.adv = _mlp(1)
        self.project_con = _mlp(projection_features)
        self.project_supcon = _mlp(projection_features)
    
    def forward(self, x, stop_grad: bool):
        if stop_grad: x_ = x.detach()
        else: x_ = x

        adv = self.adv(x_)
        project_con = self.project_con(x)
        project_supcon = self.project_supcon(x)

        _temp = (project_con.mean() + project_supcon.mean()) * 0.
        adv = adv + _temp

        return adv, project_con, project_supcon

class Discriminator(nn.Module):
    def __init__(self,
        extractor: nn.Module, projection_features=128, hidden_features=128,
        weight_norm='sn', bias=True, act_name='lrelu'
    ):
        super().__init__()
        self.extractor = extractor
        self.projection = Projector(
            self.extractor.out_channels, projection_features,
            hidden_features, weight_norm, bias, act_name
        )
    
    def forward(self, x, stop_grad: bool=False):
        feature = self.extractor(x)
        return self.projection(feature, stop_grad)

