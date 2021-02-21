
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_activation(name, inplace=True):
    if name == 'gelu': return nn.GELU()

def get_normalization(name, channels):
    if name == 'ln': return nn.LayerNorm(channels)

class MLP(nn.Module):
    def __init__(self,
        in_features, hidden_features=None, out_features=None,
        act_name='gelu', dropout=0.
    ):
        super().__init__()
        if out_features is None: out_features = in_features
        if hidden_features is None: hidden_features = in_features

        layers = [
            nn.Linear(in_features, hidden_features),
            get_activation(act_name),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(dropout)
        ]
        self.fc = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.fc(x)

class Attention(nn.Module):
    '''Multi-Head Attention
    [code from] : https://github.com/VITA-Group/TransGAN/blob/master/models/ViT_8_8.py#L49-L75
    [modification]
        - no matmul class
    '''
    def __init__(self,
        dim, num_heads=8, bias=False, attn_dropout=0., proj_dropout=0.
    ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x):
        B, N, C = x.size()
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x

class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    def forward(self, x):
        return x + self.module(x)

class EncoderBlock(nn.Module):
    def __init__(self,
        dim, num_heads=8, qkv_bias=False, attn_dropout=0., dropout=0.,
        mlp_ratio=4, act_name='gelu', norm_name='ln'
    ):
        super().__init__()

        self.encode = nn.Sequential(
            Residual(nn.Sequential(
                get_normalization(norm_name, dim),
                Attention(dim, num_heads, qkv_bias, attn_dropout, dropout)
            )),
            Residual(nn.Sequential(
                get_normalization(norm_name, dim),
                MLP(dim, dim*mlp_ratio, dim, act_name, dropout)
            ))
        )
    
    def forward(self, x):
        return self.encode(x)

class Upsample(nn.Module):
    def forward(self, x):
        B, N, C = x.size()
        H, W = (int(np.sqrt(N)), ) * 2
        assert N == H*W
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = F.pixel_shuffle(x, 2)
        B, C, H, W = x.size()
        x = x.reshape(B, C, H*W).permute(0, 2, 1)
        return x

class AddPositionEmbed(nn.Module):
    def __init__(self, size, init_func=partial(nn.init.normal_, std=0.02)):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(size))
        init_func(self.pe)
    def forward(self, x):
        return x + self.pe

class Generator(nn.Module):
    def __init__(self,
        depths: list, latent_dim, image_channels=3, bottom_width=8, embed_dim=384,
        num_heads=4, mlp_ratio=4, qkv_bias=False, dropout=0., attn_dropout=0.,
        act_name='gelu', norm_name='ln'
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        resl = bottom_width
        num_ups = len(depths) - 1

        # input linear
        self.input = nn.Linear(latent_dim, resl**2 * embed_dim)
        # positional embedding and encoder blocks
        self.add_pes = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for index, depth in enumerate(depths):
            self.add_pes.append(AddPositionEmbed((1, resl**2, embed_dim)))

            blocks = []
            for i in range(depth):
                blocks.append(
                    EncoderBlock(
                        embed_dim, num_heads, qkv_bias, attn_dropout, dropout,
                        mlp_ratio, act_name, norm_name
                    )
                )
            if index < num_ups:
                blocks.append(Upsample())
                resl *= 2
                embed_dim = embed_dim // 4

            self.blocks.append(nn.Sequential(*blocks))            
        self.out_resl = resl

        # output linear
        self.output = nn.Sequential(
            nn.Conv2d(embed_dim, image_channels, 1, 1, 0),
            # nn.Tanh()
        )
    
    def forward(self, x):
        B = x.size(0)
        x = self.input(x)
        x = x.reshape(B, -1, self.embed_dim)
        for add_pe, block in zip(self.add_pes, self.blocks):
            x = add_pe(x)
            x = block(x)
        x = x.permute(0, 2, 1).reshape(B, -1, self.out_resl, self.out_resl)
        x = self.output(x)
        return x

    @staticmethod
    def depths_len_from_target_width(target_width, bottom_width=8):
        return int(np.log2(target_width) - np.log2(bottom_width)) + 1

class AppendClsToken(nn.Module):
    def __init__(self, embed_dim, init_func=partial(nn.init.normal_, std=0.02)):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        init_func(self.cls_token)
    def forward(self, x):
        B = x.size(0)
        cls_token = self.cls_token.expand(B, -1, -1)
        return torch.cat([x, cls_token], dim=1)

class Flatten(nn.Module):
    def forward(self, x):
        x = torch.flatten(x, 2)
        x = x.permute(0, 2, 1)
        return x

class Discriminator(nn.Module):
    def __init__(self,
        depth: int, image_size: int, patch_size=8, image_channels=3, embed_dim=384,
        num_heads=4, mlp_ratio=4, qkv_bias=False, dropout=0., attn_dropout=0.,
        act_name='gelu', norm_name='ln'
    ):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2

        # input + cls token + position embed
        self.input = nn.Sequential(
            nn.Conv2d(image_channels, embed_dim, patch_size, stride=patch_size),
            Flatten(),
            AppendClsToken(embed_dim),
            AddPositionEmbed((1, num_patches+1, embed_dim)),
            nn.Dropout(dropout)
        )
        # encoder blocks
        blocks = []
        for i in range(depth):
            blocks.append(
                EncoderBlock(
                    embed_dim, num_heads, qkv_bias, attn_dropout, dropout,
                    mlp_ratio, act_name, norm_name
                )
            )
        blocks.append(get_normalization(norm_name, embed_dim))
        self.blocks = nn.Sequential(*blocks)

        # head
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.input(x)
        x = self.blocks(x)
        x = x[:, -1, :]
        x = self.head(x)
        return x

def init_weight_normal(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(0, 0.02)
        if not m.bias == None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.)
        m.bias.data.fill_(0)

if __name__=='__main__':
    g = Generator([5, 2, 2], 128, embed_dim=1024)
    d = Discriminator(7, 32)
    x = torch.randn(32, 128)
    img = g(x)
    prob = d(img)
    print(img.size(), prob.size())