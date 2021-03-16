
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class View(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
    def forward(self, x):
        return x.view(x.size(0), *self.size)

def get_normalization(name, channels):
    if name == 'in': return nn.InstanceNorm2d(channels)
    if name == 'bn': return nn.BatchNorm2d(channels)

def get_activation(name, inplace=True):
    if name == 'relu': return nn.ReLU(inplace=inplace)
    if name == 'lrelu': return nn.LeakyReLU(0.2, inplace=inplace)
    if name == 'sigmoid': return nn.Sigmoid()
    if name == 'tanh': return nn.Tanh()

class AdaIN(nn.Module):
    '''Adaptive Instance Norm'''
    def __init__(self,
        channels, style_channels, affine=True
    ):
        super().__init__()
        if affine:
            self.affine = nn.Linear(style_channels, channels*2, bias=False)
            self.affine_bias = nn.Parameter(torch.zeros(channels*2))
            self.affine_bias.data[:channels] = 1.
        else:
            self.affine = None
        self.norm = get_normalization('in', channels)

    def forward(self, x, y):
        if self.affine is not None:
            y = self.affine(y) + self.affine_bias
        y = y.view(*y.size(), 1, 1)
        scale, bias = y.chunk(2, dim=1) 
        norm = self.norm(x)
        return scale * norm + bias

class ResBlock(nn.Module):
    '''Residual block'''
    def __init__(self,
        in_channels, out_channels, norm_name, act_name,
        bias=True, down=False, up=False
    ):
        super().__init__()
        assert not down or not up, 'down and up set at the same time.'
        # main layers
        # norm -> act -> conv
        layers = [
            get_normalization(norm_name, in_channels),
            get_activation(act_name),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)
        ]
        # up or down sampling
        if down:
            layers.append(nn.AvgPool2d(2))
        if up:
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        # norm -> act -> conv
        layers.extend([
            get_normalization(norm_name, out_channels),
            get_activation(act_name),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=bias)
        ])
        self.block = nn.Sequential(*layers)

        # conv for skip connection
        if in_channels != out_channels or down or up:
            layers = [nn.Conv2d(in_channels, out_channels, 1, bias=bias)]
            if down:
                layers.append(nn.AvgPool2d(2))
            if up:
                layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            self.skip = nn.Sequential(*layers)
        else: self.skip = None
    
    def forward(self, x, y=None):
        h = x
        h = self.block(h)
        if self.skip is not None:
            x = self.skip(x)
        return h + x / np.sqrt(2)

class ResBlockAdaIN(nn.Module):
    '''Residual block with AdaIN'''
    def __init__(self,
        in_channels, style_dim, out_channels, act_name,
        bias=True, affine_each=False
    ):
        super().__init__()

        layers = [
            AdaIN(in_channels, style_dim, affine_each),
            get_activation(act_name),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias),
            AdaIN(in_channels, style_dim, affine_each),
            get_activation(act_name),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=bias)
        ]
        self.block = nn.ModuleList(layers)
    
    def forward(self, x, y=None):
        h = x
        for module in self.block:
            if isinstance(module, AdaIN):
                h = module(h, y)
            else:
                h = module(h)
        return h + x / np.sqrt(2)

class Encoder(nn.Module):
    '''Encoder'''
    def __init__(self,
        in_channels, channels=32, num_downs=2,
        norm_name='in', act_name='lrelu', bias=True
    ):
        super().__init__()
        ochannels = channels
        self.input = nn.Conv2d(in_channels, ochannels, 1, bias=bias)
        layers = []
        for _ in range(num_downs):
            channels *= 2
            ichannels, ochannels = ochannels, channels
            layers.append(
                ResBlock(
                    ichannels, ochannels, norm_name,
                    act_name, bias, down=True
                )
            )
        self.out_channels = ochannels
        self.downs = nn.Sequential(*layers)

    def forward(self, x):
        x = self.input(x)
        x = self.downs(x)
        return x

class Decoder(nn.Module):
    '''Decoder'''
    def __init__(self,
        in_channels, out_channels, num_ups=2,
        norm_name='in', act_name='lrelu', bias=True
    ):
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range(num_ups):
            ichannels, ochannels = channels, channels // 2
            layers.append(
                ResBlock(
                    ichannels, ochannels, norm_name,
                    act_name, bias, up=True
                )
            )
            channels = channels // 2
        self.ups = nn.Sequential(*layers)
        self.output = nn.Sequential(
            nn.Conv2d(ochannels, out_channels, 1, bias=bias),
            get_activation('tanh')
        )
    
    def forward(self, x):
        x = self.ups(x)
        x = self.output(x)
        return x

class PixelNorm(nn.Module):
    '''pixel normalization'''
    def forward(self, x):
        x = x / x.pow(2).mean(dim=1, keepdim=True).sqrt().add_(1e-4)
        return x

class Mapper(nn.Module):
    '''Mapper'''
    def __init__(self,
        latent_dim, num_tag, style_dim, mid_dim=256,
        act_name='relu', bias=True, num_shared_layers=3, num_tag_layers=3,
        normalize_latent=False, single_path=False
    ):
        super().__init__()
        self.single_path = single_path
        if single_path:
            num_tag = 1
        self.num_tag = num_tag

        layers = [
            nn.Linear(latent_dim, mid_dim, bias=bias),
            get_activation(act_name)
        ]
        for _ in range(num_shared_layers-1):
            layers.extend([
                nn.Linear(mid_dim, mid_dim, bias=bias),
                get_activation(act_name)
            ])
        self.shared = nn.Sequential(*layers)

        per_tag_module = []
        for _ in range(num_tag):
            layers = []
            for _ in range(num_tag_layers-1):
                layers.extend([
                    nn.Linear(mid_dim, mid_dim, bias=bias),
                    get_activation(act_name)
                ])
            layers.append(nn.Linear(mid_dim, style_dim, bias=bias))
            per_tag_module.append(nn.Sequential(*layers))
        self.style = nn.ModuleList(per_tag_module)

        if normalize_latent: self.norm = PixelNorm()
        else: self.norm = None

    def forward(self, z, j):
        if self.single_path:
            j = 0
        assert 0 <= j < self.num_tag
        
        if self.norm is not None:
            z = self.norm(z)
        z = self.shared(z)
        return self.style[j](z)

class Extractor(nn.Module):
    '''Extractor'''
    def __init__(self,
        in_channels, num_tag, style_dim,
        image_size, channels=32, bottom_width=8,
        norm_name='in', act_name='lrelu', bias=True,
        single_path=False
    ):
        super().__init__()
        self.single_path = single_path
        if single_path:
            num_tag = 1
        self.num_tag = num_tag

        ochannels = channels
        layers = [nn.Conv2d(in_channels, ochannels, 1, bias=bias)]
        num_downs = int(np.log2(image_size)-np.log2(bottom_width))
        for _ in range(num_downs):
            channels *= 2
            ichannels, ochannels = ochannels, channels
            layers.append(
                ResBlock(
                    ichannels, ochannels,
                    norm_name, act_name, bias, down=True
                )
            )
        layers.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ochannels, style_dim*num_tag, bias=bias),
            View((num_tag, style_dim))
        ])
        self.extract = nn.Sequential(*layers)

    def forward(self, x, j):
        if self.single_path:
            j = 0
        assert 0 <= j < self.num_tag

        x = self.extract(x)
        return x[:, j]

class Translator(nn.Module):
    '''Translator module'''
    def __init__(self,
        in_channels, style_dim, num_blocks=7,
        act_name='lrelu', bias=True,
        affine_each=False
    ):
        super().__init__()
        
        self.input = nn.Conv2d(in_channels, in_channels, 1, bias=bias)
        if not affine_each:
            self.affine = nn.Linear(style_dim, in_channels*2, bias=False)
            self.affine_bias = nn.Parameter(torch.zeros(in_channels*2))
            self.affine_bias.data[:in_channels] = 1.
        else:
            self.affine = None
        layers = []
        for _ in range(num_blocks):
            layers.append(
                ResBlockAdaIN(
                    in_channels, style_dim, in_channels,
                    act_name, bias, affine_each
                )
            )
        self.blocks = nn.ModuleList(layers)
        self.feat = nn.Conv2d(in_channels, in_channels, 1, bias=bias)
        self.mask = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=bias),
            get_activation('sigmoid')
        )
    
    def forward(self, x, y):
        h = x
        h = self.input(h)

        if self.affine is not None:
            y = self.affine(y) + self.affine_bias
        
        for module in self.blocks:
            h = module(h, y)

        mask = self.mask(h)
        h = self.feat(h)

        return h * mask + x * (1 - mask)

class CategoryModule(nn.Module):
    '''Module of Modules for a category'''
    def __init__(self,
        image_size, num_tag, image_channels, style_dim, latent_dim, enc_channels,
        map_mid_dim=256, map_num_shared_layers=3, map_num_tag_layers=3,
        channels=32, ex_bottom_width=8,
        trans_num_blocks=7,
        norm_name='in', act_name='lrelu', bias=True,
        normalize_latent=False, single_path=False, affine_each=False
    ):
        super().__init__()
        self.map = Mapper(
            latent_dim, num_tag, style_dim,
            map_mid_dim, act_name, bias,
            map_num_shared_layers, map_num_tag_layers,
            normalize_latent, single_path
        )
        self.extract = Extractor(
            image_channels, num_tag, style_dim,
            image_size, channels, ex_bottom_width,
            norm_name, act_name, bias,
            single_path
        )

        self.translate = Translator(
            enc_channels, style_dim, trans_num_blocks,
            act_name, bias,
            affine_each
        )
    
    def forward(self, x, y, j):
        '''extract style code and translate'''
        if y.dim() == 2:
            code = self.map(y, j)
        if y.dim() == 4:
            code = self.extract(y, j)
        return self.translate(x, code)

class Generator(nn.Module):
    '''Generator'''
    def __init__(self,
        # params
        tags: list,             # list of int containing number of tags in each category
        image_size: int,        # input/output image size
        image_channels: int,    # input image channel size
        out_channels: int,      # output image channel size
        style_dim: int,         # dimension of style code
        latent_dim: int,        # dimension of noise input
        # encoder/decoder params
        enc_num_downs=2,
        # mapper params
        map_mid_dim=256, map_num_shared_layers=3, map_num_tag_layers=3,
        # extractor params
        channels=32, ex_bottom_width=8,
        # translator params
        trans_num_blocks=7,
        # common params
        norm_name='in', act_name='lrelu', bias=True,
        # experimantal
        normalize_latent=False, # use pixel normalization to input latent
        single_path=False,      # one layer for all tags
        affine_each=False       # affine input each AdaIN layer
    ):
        super().__init__()
        self.tags = tags

        # encoder
        self.encode = Encoder(
            image_channels, channels, enc_num_downs,
            norm_name, act_name, bias
        )
        # decoder
        self.decode = Decoder(
            self.encode.out_channels, out_channels, enc_num_downs,
            norm_name, act_name, bias
        )

        # category modules
        self.category_modules = nn.ModuleList()
        for num_tag in tags:
            self.category_modules.append(
                CategoryModule(
                    image_size, num_tag, image_channels, style_dim, latent_dim, self.encode.out_channels,
                    map_mid_dim, map_num_shared_layers, map_num_tag_layers,
                    channels, ex_bottom_width,
                    trans_num_blocks,
                    norm_name, act_name, bias,
                    normalize_latent, single_path, affine_each
                )
            )
    
    @staticmethod
    def assert_refs(tags, refs):
        '''check input and warn'''
        assert len(refs) == len(tags), f'input list of length {len(tags)}'
        refs_ = [ref for ref in refs if ref is not None]
        tags_ = [tag for i, tag in enumerate(tags) if refs[i] is not None]
        assert all([True if len(ref) == 2 else False for ref in refs_]), 'check input format for reference input'
        input_j = [j for _, j in refs_]
        assert all([isinstance(j, int) for j in input_j]), 'j must be a integer for slicing'
        assert all([0 <= j < num_tag for j, num_tag in zip(input_j, tags_)]), 'j must be smaller than number of tags'
        input_data = [data for data, _ in refs_]
        assert all([data.dim() == 2 or data.dim() == 4 for data in input_data]), 'check input data'

    def forward(self, x, refs: list=None):
        x = self.encode(x)
        if refs is not None:
            self.assert_refs(self.tags, refs)
            for module, input in zip(self.category_modules, refs):
                if input is not None:
                    data, j = input
                    x = module(x, data, j)
        x = self.decode(x)
        return x

class Discriminator(nn.Module):
    '''Discriminator
        Architecture is completely different from the official implementation
        Here uses PatchGAN with receptive field of 70
        but has different last layers for every category
    
    things that are the same:
        - conditions are concatenated before the last layers (pix2pix concatenates them before the input to D)
        - output channels are [(number of tags) * 2 for (number of tags) in tags]
            x2 are each for cycle and fake images, similar to CycleGAN but in one model.
    '''
    def __init__(self,
        tags: list,          # input same object as "tags" in G
        image_size: int,     # image size
        image_channels: int, # channel of input image
        num_layers=3, channels=32,
        norm_name='in', act_name='lrelu', bias=True,
        # other params
        ret_feat=False,      # return features (for feature matching loss in pix2pixhd)
        single_path=False    # one layer for all tags
    ):
        super().__init__()
        self.single_path = single_path
        self.ret_feat = ret_feat
        if single_path:
            tags = [1]
        self.num_category = len(tags)

        ochannels = channels
        layers = [
            nn.Sequential(
                nn.Conv2d(image_channels, ochannels, 4, 2, bias=bias),
                get_activation(act_name)
            )
        ]
        for _ in range(num_layers-1):
            channels *= 2
            ichannels, ochannels = ochannels, channels
            layers.append(
                nn.Sequential(
                    nn.Conv2d(ichannels, ochannels, 4, 2, bias=bias),
                    get_normalization(norm_name, ochannels),
                    get_activation(act_name)
                )
            )
        self.shared = nn.Sequential(*layers)

        self.tail = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(ochannels+num_tag, ochannels*2, 4, bias=bias),
                    get_normalization(norm_name, ochannels*2),
                    get_activation(act_name),
                ),
                nn.Conv2d(ochannels*2, num_tag*2, 4, bias=bias) # each for A2B and B2A like CycleGAN
            ])
            for num_tag in tags
        ])
        self.tags = tags
    
    def forward(self, x, i, j):
        if self.single_path:
            i, j = 0, 0
        assert 0 <= i < self.num_category

        outs = []
        for module in self.shared:
            x = module(x)
            outs.append(x)

        B, _, H, W = x.size()
        condition = F.one_hot(torch.tensor([j], device=x.device), self.tags[i])
        condition = condition.reshape(1, -1, 1, 1).expand(B, -1, H, W)
        x = torch.cat([x, condition], dim=1)

        for module in self.tail[i]:
            x = module(x)
            outs.append(x)

        B, C, H, W = x.size()
        x = x.view(B, 2, C//2, H, W)
        outs[-1] = x[:, :, j]
        
        if self.ret_feat:
            return outs[-1], outs[:-1]
        return outs[-1]

def init_weight_xavier(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        if not m.bias == None:
            m.bias.data.fill_(0.)
    elif isinstance(m, nn.BatchNorm2d):
        if not m.weight == None:
            m.weight.data.fill_(1.)
        if not m.bias == None:
            m.bias.data.fill_(0.)

def init_weight_kaiming(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        if not m.bias == None:
            m.bias.data.fill_(0.)
    elif isinstance(m, nn.BatchNorm2d):
        if not m.weight == None:
            m.weight.data.sill_(1.)
        if not m.bias == None:
            m.bias.data.fill_(0.)

if __name__=='__main__':
    g = Generator([10, 10, 2], 128, 3, 3, 256, 128, normalize_latent=True, single_path=True, affine_each=True)
    d = Discriminator([10, 10, 2], 128, 3, ret_feat=True)
    x = torch.randn(10, 3, 128, 128)
    y = torch.randn(10, 128)
    refs = [(x, 5), (x, 3), (y, 0)]
    img = g(x, refs)
    prob, feats = d(img, 1)
    print(img.size())
    print(prob.size())
    for i, feat in enumerate(feats):
        print(f'{i:02}:\t{feat.size()}')