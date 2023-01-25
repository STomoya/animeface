
import torch
import torch.nn as nn

def get_normalization(name, *args, **kwargs):
    if name == 'bn': return nn.BatchNorm2d(*args, **kwargs)
    elif name == 'in': return nn.InstanceNorm2d(*args, **kwargs)

def get_activation(name, *args, **kwargs):
    if name == 'relu': return nn.ReLU(*args, **kwargs)
    elif name == 'lrelu': return nn.LeakyReLU(0.2)

class Encoder(nn.Module):
    def __init__(self,
        image_size, num_layers, embed_dim, image_channels=3, channels=32, norm_name='bn', act_name='relu'
    ) -> None:
        super().__init__()

        self.input = nn.Sequential(
            nn.Conv2d(image_channels, channels, 3, 1, 1),
            get_activation(act_name)
        )

        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Conv2d(channels, channels*2, 3, 2, 1),
                get_normalization(norm_name, channels*2),
                get_activation(act_name)
            ])
            image_size //= 2
            channels *= 2
        layers += [nn.Flatten()]
        self.hid_layers = nn.Sequential(*layers)

        self.mu = nn.Linear(channels*image_size**2, embed_dim)
        self.logvar = nn.Linear(channels*image_size**2, embed_dim)

        self.bottom_size = image_size
        self.bottom_channels = channels

    def forward(self, x):
        x = self.input(x)
        x = self.hid_layers(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return (mu, logvar)


class Decoder(nn.Module):
    def __init__(self,
        image_size, embed_dim, bottom_width, channels, image_channels=3, norm_name='bn', act_name='relu'
    ) -> None:
        super().__init__()

        self.input = nn.Sequential(
            nn.Linear(embed_dim, channels*bottom_width**2),
            get_activation(act_name)
        )
        self.input_shape = (-1, channels, bottom_width, bottom_width)

        layers = []
        resl = bottom_width
        while resl < image_size:
            layers.extend([
                nn.Conv2d(channels, channels//2, 3, 1, 1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                get_normalization(norm_name, channels//2),
                get_activation(act_name)
            ])
            resl *= 2
            channels //= 2
        self.hid_layers = nn.Sequential(*layers)

        self.output = nn.Sequential(
            nn.Conv2d(channels, image_channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.input(x)
        x = x.view(*self.input_shape)
        x = self.hid_layers(x)
        x = self.output(x)
        return x


class VAE(nn.Module):
    def __init__(self,
        image_size, embed_dim, num_hid_layers=4, image_channels=3, channels=32, norm_name='bn', act_name='relu'
    ) -> None:
        super().__init__()
        self.encoder = Encoder(image_size, num_hid_layers, embed_dim, image_channels, channels, norm_name, act_name)
        self.decoder = Decoder(image_size, embed_dim, self.encoder.bottom_size, self.encoder.bottom_channels,
            image_channels, norm_name, act_name)

    def encode(self, x): return self.encoder(x)
    def decode(self, x): return self.decoder(x)
    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z
    def forward(self, x, return_muvar=False):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)

        if return_muvar:
            return reconstructed, (mu, logvar)

        return reconstructed
