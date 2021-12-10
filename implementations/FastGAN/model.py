
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_normalization(name, channels):
    if   name == 'bn': return nn.BatchNorm2d(channels)
    elif name == 'in': return nn.InstanceNorm2d(channels)
    raise Exception(f'Normalization: {name}')

def Conv2d(*args, **kwargs):
    return nn.utils.spectral_norm(nn.Conv2d(*args, **kwargs))
def ConvTranspose2d(*args, **kwargs):
    return nn.utils.spectral_norm(nn.ConvTranspose2d(*args, **kwargs))
def Linear(*args, **kwargs):
    return nn.utils.spectral_norm(nn.Linear(*args, **kwargs))

class UpBlock(nn.Module):
    def __init__(self,
        in_channels, out_channels, transposed=False,
        norm_name='bn'
    ) -> None:
        super().__init__()
        bias = norm_name != 'bn'

        if transposed:
            self.conv = ConvTranspose2d(
                in_channels, out_channels*2, 4, 2, 1, bias=bias)
        else:
            self.up = nn.Upsample(scale_factor=2)
            self.conv = Conv2d(
                in_channels, out_channels*2, 3, 1, 1, bias=bias)

        self.norm = get_normalization(norm_name, out_channels*2)
        self.act  = nn.GLU(dim=1)

    def forward(self, x):
        if hasattr(self, 'up'):
            x = self.up(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class SkipLayerExcitation(nn.Module):
    def __init__(self,
        in_channels, out_channels, interp_size=4
    ) -> None:
        super().__init__()

        self.global_content = nn.Sequential(
            nn.AdaptiveAvgPool2d(interp_size),
            Conv2d(in_channels, in_channels, interp_size),
            nn.LeakyReLU(0.2, True),
            Conv2d(in_channels, out_channels, 1))

    def forward(self, high, low):
        low = self.global_content(low)
        attn = torch.sigmoid(low)
        return high * attn

class View(nn.Module):
    def __init__(self, size) -> None:
        super().__init__()
        self._size = size
    def forward(self, x):
        return x.view(-1, *self._size)

class Generator(nn.Module):
    def __init__(self,
        latent_dim, image_size, channels, max_channels, interp_size=4,
        image_channels=3, bottom=4, norm_name='bn', transposed=False, num_sle=None
    ) -> None:
        super().__init__()
        num_ups = int(math.log2(image_size)-math.log2(bottom))
        channels = channels * 2 ** num_ups
        ochannels = min(max_channels, channels)
        bias = norm_name != 'bn'

        if transposed:
            self.input = nn.Sequential(
                View((latent_dim, 1, 1)),
                ConvTranspose2d(latent_dim, ochannels*2, 4, 2, bias=bias),
                get_normalization(norm_name, ochannels*2),
                nn.GLU(dim=1))
        else:
            self.input = nn.Sequential(
                Linear(latent_dim, ochannels*2*bottom**2, bias=bias),
                View((ochannels*2, bottom, bottom)),
                get_normalization(norm_name, ochannels*2),
                nn.GLU(dim=1))

        self.ups = nn.ModuleList()
        _channels = []
        for _ in range(num_ups):
            channels = channels // 2
            ichannels, ochannels = ochannels, min(max_channels, channels)
            self.ups.append(UpBlock(ichannels, ochannels, transposed, norm_name))
            _channels.append(ochannels)

        if num_sle is None:
            num_sle = len(_channels[:-1]) // 2
        self.collect_feats = []
        self.apply_sle     = []
        self.sle = nn.ModuleList()
        for i in range(num_sle):
            self.collect_feats.append(i)
            self.apply_sle.append(len(_channels)+i-num_sle-1)
            self.sle.append(
                SkipLayerExcitation(
                    _channels[i], _channels[i-num_sle-1], interp_size))

        self.output = nn.Sequential(
            Conv2d(ochannels, image_channels, 3, 1, 1),
            nn.Tanh())
    def forward(self, x):
        x = self.input(x)

        sle_index = 0
        feats = []
        for i, up in enumerate(self.ups):
            x = up(x)
            if i in self.collect_feats:
                feats.append(x)
            if i in self.apply_sle:
                x = self.sle[sle_index](x, feats[sle_index])
                sle_index += 1
        x = self.output(x)
        return x

class ResBlock(nn.Module):
    def __init__(self,
        in_channels, out_channels,
        norm_name='bn'
    ) -> None:
        super().__init__()
        bias = norm_name != 'bn'

        self.main = nn.Sequential(
            Conv2d(in_channels, out_channels, 4, 2, 1, bias=bias),
            get_normalization(norm_name, out_channels),
            nn.LeakyReLU(0.2, True),
            Conv2d(out_channels, out_channels, 3, 1, 1, bias=bias),
            get_normalization(norm_name, out_channels),
            nn.LeakyReLU(0.2, True))
        self.skip = nn.Sequential(
            nn.AvgPool2d(2),
            Conv2d(in_channels, out_channels, 1, 1, bias=bias),
            nn.LeakyReLU(0.2, True))
    def forward(self, x):
        hid = self.main(x)
        x = self.skip(x)
        return hid + x

class SimpleDecoder(nn.Sequential):
    def __init__(self,
        in_channels, bottom, image_size=128, image_channels=3,
        transposed=False, norm_name='bn'
    ) -> None:
        num_ups = int(math.log2(image_size) - math.log2(bottom))
        bias = norm_name != 'bn'

        def _make_layers(_inc, _outc):
            if transposed:
                layers = [ConvTranspose2d(_inc, _outc*2, 4, 2, 1, bias=bias)]
            else:
                layers = [
                    nn.Upsample(scale_factor=2),
                    Conv2d(_inc, _outc*2, 3, 1, 1, bias=bias)]
            layers.extend([
                get_normalization(norm_name, _outc*2),
                nn.GLU(dim=1)])
            return layers

        layers = []
        ochannels = channels = in_channels
        for _ in range(num_ups):
            channels = channels // 2
            ichannels, ochannels = ochannels, channels
            layers.extend(_make_layers(ichannels, ochannels))
        layers.extend([
            Conv2d(ochannels, image_channels, 3, 1, 1, bias=True),
            nn.Tanh()])
        super().__init__(*layers)

class Discriminator(nn.Module):
    def __init__(self,
        image_size, init_down_size=256, image_channels=3,
        channels=32, max_channels=1024,
        norm_name='bn', bottom=8, decoder_image_size=128
    ) -> None:
        super().__init__()
        self.part_size = decoder_image_size

        init_downs = int(math.log2(image_size) - math.log2(init_down_size))
        num_downs  = int(math.log2(init_down_size) - math.log2(bottom))
        bias = norm_name != 'bn'

        ochannels = channels
        if init_downs == 0:
            self.input = nn.Sequential(
                Conv2d(image_channels, ochannels, 3, 1, 1, bias=bias),
                nn.LeakyReLU(0.2, True))
        else:
            layers = [
                Conv2d(image_channels, ochannels, 4, 2, 1, bias=bias),
                nn.LeakyReLU(0.2, True)]
            for _ in range(init_downs-1):
                channels *= 2
                ichannels, ochannels = ochannels, min(max_channels, channels)
                layers.extend([
                    Conv2d(ichannels, ochannels, 4, 2, 1, bias=bias),
                    get_normalization(norm_name, ochannels),
                    nn.LeakyReLU(0.2, True)])
            self.input = nn.Sequential(*layers)

        self.resblocks = nn.ModuleList()
        self.decoders  = nn.ModuleList()
        resl = init_down_size
        for i in range(num_downs):
            resl = resl // 2
            channels *= 2
            ichannels, ochannels = ochannels, min(max_channels, channels)
            self.resblocks.append(ResBlock(ichannels, ochannels, norm_name))
            if resl in [16, 8]:
                self.decoders.append(
                    SimpleDecoder(
                        ochannels, 8, decoder_image_size, image_channels,
                        norm_name=norm_name))
        self.to_logits = nn.Sequential(
            Conv2d(ochannels, ochannels*2, 1, 1, bias=bias),
            get_normalization(norm_name, ochannels*2),
            nn.LeakyReLU(0.2, True),
            Conv2d(ochannels*2, 1, 4, 1, 0, bias=True))

    def forward(self, x, return_recon=True):
        org_img = x
        x = self.input(x)

        decoder_inputs = []
        for resblock in self.resblocks:
            x = resblock(x)
            if x.size(-1) in [16, 8]:
                decoder_inputs.append(x)
        # logits
        logits = self.to_logits(x)

        '''reconstruction
        always returns the "loss"
        '''

        # small reconstruction
        recon = self.decoders[-1](decoder_inputs[-1])
        img_small = F.interpolate(org_img, self.part_size)
        recon_full_loss = F.mse_loss(recon, img_small)

        # part reconstruction
        # crop
        part_id = torch.randint(4, (1, )).item()
        img_half = org_img.size(-1) // 2
        if part_id == 0:
            part_feat = decoder_inputs[-2][:, :, :8, :8]
            img_part = org_img[:, :, :img_half, :img_half]
        elif part_id == 1:
            part_feat = decoder_inputs[-2][:, :, 8:, :8]
            img_part = org_img[:, :, img_half:, :img_half]
        elif part_id == 2:
            part_feat = decoder_inputs[-2][:, :, :8, 8:]
            img_part = org_img[:, :, :img_half, img_half:]
        elif part_id == 3:
            part_feat = decoder_inputs[-2][:, :, 8:, 8:]
            img_part = org_img[:, :, img_half:, img_half:]
        recon_part = self.decoders[-2](part_feat)
        img_part = F.interpolate(img_part, self.part_size)
        recon_part_loss = F.mse_loss(recon_part, img_part)

        if return_recon:
            return logits, recon_full_loss+recon_part_loss, [recon, img_small, recon_part, img_part]

        return logits, recon_full_loss+recon_part_loss

if __name__=='__main__':
    g = Generator(128, 1024, 32, 1024)
    d = Discriminator(1024)
    x = torch.randn(3, 128)
    image = g(x)
    out = d(image, False)
    print(image.size())
    print(list(map(lambda x:x.size(), out)))
