
import functools

import torch
import torch.nn as nn
import numpy as np

def get_activation(name, inplace=True):
    if name == 'relu': return nn.ReLU(inplace=inplace)
    elif name == 'lrelu': return nn.LeakyReLU(0.2, inplace=inplace)

def get_normalization(name, channels, **kwargs):
    if name == 'bn': return nn.BatchNorm2d(channels, **kwargs)
    elif name == 'in': return nn.InstanceNorm2d(channels, **kwargs)

def ConvBlock(
    in_channels, out_channels, use_bias=True,
    norm_name='bn', act_name='relu'
):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=use_bias),
        get_normalization(norm_name, out_channels),
        get_activation(act_name)
    )

def ConvTransposeBlock(
    in_channels, out_channels, use_bias=True,
    norm_name='bn', act_name='relu'
):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1, output_padding=1, bias=use_bias),
        get_normalization(norm_name, out_channels),
        get_activation(act_name)
    )

class Encoder(nn.Module):
    def __init__(self,
        image_size, z_dim, in_channels=3, target_resl=4, channels=32, max_channels=512,
        use_bias=True, norm_name='in', act_name='lrelu'
    ):
        super().__init__()
        convb_func = functools.partial(
            ConvBlock, use_bias=use_bias, norm_name=norm_name, act_name=act_name
        )
        
        ochannels = channels
        layers = [convb_func(in_channels, ochannels)]
        image_size = image_size // 2
        while image_size > target_resl:
            image_size = image_size // 2
            channels *= 2
            ichannels, ochannels = ochannels, min(max_channels, channels)
            layers.append(convb_func(ichannels, ochannels))
        layers.extend([
            nn.Flatten()
        ])
        self.extract = nn.Sequential(*layers)
        num_features = ochannels * (target_resl ** 2)
        self.mu  = nn.Linear(num_features, z_dim, bias=use_bias)
        self.var = nn.Linear(num_features, z_dim, bias=use_bias)

    def forward(self, x):
        assert x.size(2) is x.size(3)

        feat = self.extract(x)

        mu, var = self.mu(feat), self.var(feat)

        return mu, var

class Decoder(nn.ModuleList):
    def __init__(self,
        image_size, z_dim, out_channels=3, channels=32, max_channels=2**10,
        use_bias=True, norm_name='in', act_name='relu'
    ):
        super().__init__()
        convb_func = functools.partial(
            ConvTransposeBlock, use_bias=use_bias, norm_name=norm_name, act_name=act_name
        )

        num_layers = int(np.log2(image_size) - 2)
        channels = channels * 2 ** num_layers
        ochannels = min(max_channels, channels)
        layers = [
            nn.ConvTranspose2d(z_dim, ochannels, 4, 2, 0, bias=use_bias),
            get_normalization(norm_name, ochannels),
            get_activation(act_name)
        ]
        for _ in range(num_layers):
            channels = channels // 2
            ichannels, ochannels = ochannels, min(max_channels, channels)
            layers.append(convb_func(ichannels, ochannels))
        layers.extend([
            nn.Conv2d(ochannels, out_channels, 3, padding=1, bias=use_bias),
            nn.Tanh()
        ])
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1, 1, 1)
        return self.decode(x)

class VAE(nn.Module):
    def __init__(self,
        image_size, z_dim, image_channels, channels=32, max_channels=2**10,
        enc_target_resl=4,
        use_bias=True, norm_name='in', act_name='relu'
    ):
        super().__init__()
        self.encoder = Encoder(
            image_size, z_dim, image_channels, enc_target_resl,
            channels, max_channels,
            use_bias, norm_name, act_name
        )
        self.decoder = Decoder(
            image_size, z_dim, image_channels, channels, max_channels,
            use_bias, norm_name, act_name
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recons = self.decoder(z)
        return recons, z, mu, logvar

def init_weight(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        m.weight.data.normal_(0., 1.)
        if not m.bias == None:
            m.bias.data.fill_(0.)
    if isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.)
        m.bias.data.fill_(0.)

if __name__ == "__main__":
    vae = VAE(256, 256, 3)
    x = torch.randn(32, 3, 256, 256)

    recons, z, mu, logvar = vae(x)

    print(recons.size(), z.size(), mu.size(), logvar.size())