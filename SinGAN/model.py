
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    '''
    Convolution Block
    '''
    def __init__(self,
        in_channels, out_channels, kernel_size,
        norm_layer='bn', **kwargs
    ):
        super().__init__()

        # assign layers
        layers = []
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        if norm_layer == 'bn':
            layers += [conv, nn.BatchNorm2d(out_channels)]
        elif norm_layer == 'in':
            layers += [conv, nn.InstanceNorm2d(out_channels)]
        elif norm_layer == 'sn':
            layers.append(nn.utils.spectral_norm(conv))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.conv = nn.Sequential(*layers)

        # param init
        self.conv[0].weight.data.normal_(0, 0.02)
        if not self.conv[0].bias == None:
            self.conv[0].bias.data.fill_(0)
        if norm_layer == 'bn':
            self.conv[1].weight.data.normal_(1, 0.02)
            self.conv[1].bias.data.fill_(0.)

    def forward(self, x):
        x = self.conv(x)
        return x

class SingleScaleGenerator(nn.Module):
    '''
    generator for each scale
    '''
    def __init__(self,
        in_channels, channels, kernel_size=3, norm_layer='bn', num_layers=5, img_out=True, **kwargs
    ):
        super().__init__()

        pad = ((kernel_size - 1) * num_layers) // 2
        self.pad_noised = nn.ZeroPad2d(pad)
        
        layers = [ConvBlock(in_channels, channels, kernel_size, norm_layer, **kwargs)]
        for _ in range(num_layers - 2):
            layers.append(ConvBlock(channels, channels, kernel_size, norm_layer, **kwargs))
        layers.append(nn.Conv2d(channels, in_channels, kernel_size, **kwargs))
        if img_out:
            layers.append(nn.Tanh())
        self.conv = nn.Sequential(*layers)
    
    def forward(self, noised_img, pre_img):
        # pad to fit out size when after conv
        pad_img = self.pad_noised(noised_img)
        # conv
        x = self.conv(pad_img)
        # skip connect
        out = x + pre_img
        return out

class SingleScaleDiscriminator(nn.Module):
    '''
    discriminator for each scale
    '''
    def __init__(self,
        in_channels, channels, kernel_size=3, norm_layer='bn', num_layers=5, **kwargs
    ):
        super().__init__()

        layers = [ConvBlock(in_channels, channels, kernel_size, norm_layer, **kwargs)]
        for _ in range(num_layers - 2):
            layers.append(ConvBlock(channels, channels, kernel_size, norm_layer, **kwargs))
        layers.append(nn.Conv2d(channels, 1, kernel_size, **kwargs))
        self.conv = nn.Sequential(*layers)
    
    def forward(self, img):
        x = self.conv(img)
        return x



class Generator:
    '''
    Generator class with all scales
    NOT a inherited class of nn.Module
    '''
    def __init__(self,
        img_sizes, device,
        img_channels=3, channels=32, kernel_size=3, norm_layer='bn', num_layers=5, img_out=True, **kwargs
    ):
        self.device = device
        self.img_sizes = img_sizes
        self.num_scale = len(img_sizes)
        self.cur_scale = 0
        self.rec_noise = self.gnoise((1, *img_sizes[0]))
        self.noise_amp = [1]

        self.generators = []
        for scale in range(1, self.num_scale+1):
            self.generators.append(SingleScaleGenerator(
                img_channels, channels, kernel_size, norm_layer,
                num_layers, img_out, **kwargs
            ))
            if scale % 4 == 0:
                channels *= 2
    
    def forward(self, sizes=None, rec=False):
        sizes = sizes if sizes else self.img_sizes
        for scale, G in enumerate(self.generators):
            if scale == 0:
                # empty image
                pre = torch.zeros((1, 3, *sizes[0]), device=self.device)
                # noise
                noise = self.gnoise((1, *sizes[0])) if not rec else self.rec_noise
                noise = noise.expand(1, 3, *sizes[0])
            else:
                # upsample pre image
                pre = self.upsample(image, sizes[scale])
                # noise
                noise = self.gnoise((3, *sizes[scale])) if not rec else torch.zeros((1, 3, *sizes[scale]), device=self.device)
            noise = noise * self.noise_amp[scale] + pre
            # gen image
            image = G(noise.detach(), pre)
            
            if scale == self.cur_scale:
                break
        return image

    def gnoise(self, size):
        noise = torch.randn(1, size[0], size[1]//2, size[2]//2, device=self.device)
        noise = self.upsample(noise, size[1:])
        return noise
    
    def upsample(self, tensor, size):
        up = nn.Upsample(size=(size[0], size[1]), mode='bilinear', align_corners=True)
        return up(tensor)

    def update_amp(self, real, rec_fake):
        mse = nn.MSELoss()
        rmse = torch.sqrt(mse(real, rec_fake))
        self.noise_amp.append(0.1 * rmse)

    def progress(self, real, rec_fake):
        self.update_amp(real, rec_fake)
        self.eval()
        self.cur_scale += 1
        # init G with pre G's params if possible
        if not self.cur_scale % 4 == 0:
            state_dict = copy.deepcopy(self.generators[self.cur_scale-1].state_dict())
            self.generators[self.cur_scale].load_state_dict(state_dict)

    def to(self, device=None):
        for m in self.generators:
            m.to(self.device if not device else device)

    def cpu(self):
        self.to(torch.device('cpu'))
    
    def eval(self, all=False):
        for scale, m in enumerate(self.generators):
            m.eval()
            if scale == self.cur_scale and not all:
                break
    
    def parameters(self):
        return self.generators[self.cur_scale].parameters()

    def __str__(self):
        return '\n'.join(['Scale {}\n'.format(scale) + g.__str__() for scale, g in enumerate(self.generators, 1)])

class Discriminator:
    '''
    Discriminator class with all scales
    NOT a inherited class of nn.Module
    '''
    def __init__(self,
        img_sizes, device,
        img_channels=3, channels=32, kernel_size=3, norm_layer='bn', num_layers=5, **kwargs
    ):
        self.device = device
        self.num_scale = len(img_sizes)
        self.cur_scale = 0

        self.discriminators = []
        for scale in range(1, self.num_scale+1):
            if scale % 4 == 0:
                channels *= 2
            self.discriminators.append(SingleScaleDiscriminator(
                img_channels, channels, kernel_size,
                norm_layer, num_layers, **kwargs
            ))

    def forward(self, img):
        x = self.discriminators[self.cur_scale](img)
        return x
    
    def progress(self):
        self.cpu()
        self.cur_scale += 1
        if not self.cur_scale == self.num_scale:
            self.to()

    def to(self, device=None):
        self.discriminators[self.cur_scale].to(self.device if not device else device)
    
    def cpu(self):
        self.to(torch.device('cpu'))

    def parameters(self):
        return self.discriminators[self.cur_scale].parameters()

    def __str__(self):
        return '\n'.join(['Scale {}\n'.format(scale) + d.__str__() for scale, d in enumerate(self.discriminators, 1)])

if __name__ == "__main__":
    '''test single scale'''
    # for norm in ['bn', 'sn', 'in']:
    #     g = SingleScaleGenerator(3, 32, norm_layer=norm,bias=False)
    #     d = SingleScaleDiscriminator(3, 32, norm_layer=norm, bias=False)

    #     x = torch.randn(1, 3, 32, 25)

    #     img = g(x, x)
    #     prob = d(img)

    #     print(img.size(), prob.size())

    '''test'''
    img_sizes = []
    test = []
    for times in range(1000):
        size = round(250 * (0.80 ** times)), round(230 * (0.75 ** times))
        test_size = round(500 * (0.80 ** times)), round(460 * (0.75 ** times))
        if size[0] > 25 and size[1] > 25:
            img_sizes.append(size)
            test.append(test_size)
        else:
            break
    img_sizes = sorted(img_sizes)
    test = sorted(test)
    print(img_sizes)
    device = torch.device('cuda:0')
    Gs = Generator(img_sizes, device)
    Ds = Discriminator(img_sizes, device)
    Gs.to()
    Ds.to()
    for _ in img_sizes:
        img = Gs.forward(rec=True)
        prob = Ds.forward(img)
        print(img.size(), prob.size())
        Gs.progress(None, None)
        Ds.progress()
    img = Gs.forward(test)
    print(img.size())