
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name, inplace, negative_slope=0.2):
    if name == 'lrelu':
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
    elif name == 'relu':
        return nn.ReLU(inplace=inplace)

'''
layers
'''

class SNConv2d(nn.Module):
    '''
    conv2d with spectral normalization
    '''
    def __init__(self,
        in_channels, out_channels, kernel_size, **kwargs
    ):
        super().__init__()
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        )
    def forward(self, x):
        return self.conv(x)


'''
modules for G
'''

class AdaIN3d(nn.Module):
    '''
    adaptive instance normalization 3d
    '''
    def __init__(self,
        noise_channels, channels, activation='relu'
    ):
        super().__init__()

        self.norm = nn.InstanceNorm3d(channels)

        self.noise_linear = nn.Sequential(
            nn.Linear(noise_channels, channels*2),
            get_activation(activation, True, 0.2)
        )

        self.noise_linear[0].weight.data.normal_(0., 0.02)
        self.noise_linear[0].bias.data.fill_(0.)

    def forward(self, x, noise):
        norm = self.norm(x)
        noise = self.noise_linear(noise)
        scale, bias = torch.split(noise, norm.size(1), dim=1)
        out = self.to_3d(scale) * norm + self.to_3d(bias)
        return out

    def to_3d(self, x):
        return x.view(-1, x.size(1), 1, 1, 1)

class AdaIN2d(nn.Module):
    '''
    adaptive instance normalization 2d
    '''
    def __init__(self,
        noise_channels, channels, activation='relu'
    ):
        super().__init__()

        self.norm = nn.InstanceNorm2d(channels)

        self.noise_linear = nn.Sequential(
            nn.Linear(noise_channels, channels*2),
            get_activation(activation, True, 0.2)
        )

        self.noise_linear[0].weight.data.normal_(0., 0.02)
        self.noise_linear[0].bias.data.fill_(0.)

    def forward(self, x, noise):
        norm = self.norm(x)
        noise = self.noise_linear(noise)
        scale, bias = torch.split(noise, norm.size(1), dim=1)
        out = self.to_2d(scale) * norm + self.to_2d(bias)
        return out

    def to_2d(self, x):
        return x.view(-1, x.size(1), 1, 1)

class GBlock3d(nn.Module):
    '''
    conv3d -> non-linear
    '''
    def __init__(self,
        in_channels, out_channels, kernel_size, padding, activation='lrelu', **kwargs
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, **kwargs),
            get_activation(activation, True, 0.2)
        )

        self.block[0].weight.data.normal_(0., 0.02)
        if not self.block[0].bias == None:
            self.block[0].bias.data.fill_(0.)
    def forward(self, x):
        x = self.block(x)
        return x

class Block2d(nn.Module):
    '''
    conv2d -> non-linear
    '''
    def __init__(self,
        in_channels, out_channels, kernel_size, padding, activation='lrelu', **kwargs
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, **kwargs),
            get_activation(activation, True, 0.2)
        )

        self.block[0].weight.data.normal_(0., 0.02)
        if not self.block[0].bias == None:
            self.block[0].bias.data.fill_(0.)
    def forward(self, x):
        x = self.block(x)
        return x


class GUpsampleBlock3d(nn.Module):
    '''
    convtranspose3d -> adain -> non-linear
    '''
    def __init__(self,
        in_channels, out_channels, kernel_size, stride, padding, noise_channels, activation='lrelu', norm_activation='relu', **kwargs
    ):
        super().__init__()

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, **kwargs)
        self.normalization = AdaIN3d(noise_channels, out_channels, norm_activation)
        self.activation = get_activation(activation, False, 0.2)

        self.conv.weight.data.normal_(0., 0.02)
        if not self.conv.bias == None:
            self.conv.bias.data.fill_(0.)

    def forward(self, x, noise):
        x = self.conv(x)
        x = self.normalization(x, noise)
        x = self.activation(x)
        return x

class GUpsampleBlock2d(nn.Module):
    '''
    convtranspose2d -> adain -> non-linear
    '''
    def __init__(self,
        in_channels, out_channels, kernel_size, stride, padding, noise_channels, activation='lrelu', norm_activation='relu', **kwargs
    ):
        super().__init__()

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, **kwargs)
        self.normalization = AdaIN2d(noise_channels, out_channels, norm_activation)
        self.activation = get_activation(activation, False, 0.2)

        self.conv.weight.data.normal_(0., 0.02)
        if not self.conv.bias == None:
            self.conv.bias.data.fill_(0.)
    
    def forward(self, x, noise):
        x = self.conv(x)
        x = self.normalization(x, noise)
        x = self.activation(x)
        return x

class Transform3d(nn.Module):
    '''
    transform 5d tensor using a rotation matrix
    '''
    def forward(self, x, theta):
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        out = F.grid_sample(x, grid, padding_mode='zeros', align_corners=True)
        return out

'''
modules for D
'''

class DBlock2d(nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel_size, stride, padding, activation='lrelu', **kwargs
    ):
        super().__init__()

        self.snconv = SNConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, **kwargs)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True)
        self.linear = nn.Linear(out_channels, 1)
        self.activation = get_activation(activation, False, 0.2)

        self.snconv.conv.weight.data.normal_(1, 0.02)
        if not self.snconv.conv.bias == None:
            self.snconv.conv.bias.data.fill_(0.)
        self.linear.weight.data.normal_(1., 0.02)
        self.linear.bias.data.fill_(0.)

    def forward(self, x):
        x = self.snconv(x)
        x = self.norm(x)
        B, C, *_ = x.size()
        tmp_x = x.view(B, C, -1)
        mean, var = tmp_x.mean(-1), tmp_x.var(-1)
        style = torch.cat([mean, var], dim=0)
        style_logit = self.linear(style)
        x = self.activation(x)
        return x, style_logit



class Generator(nn.Module):
    def __init__(self,
        channels=512, noise_channels=128, activation='lrelu', const_size=4
    ):

        super().__init__()

        self.const_noise = nn.Parameter(torch.randn((1, channels, const_size, const_size, const_size)))
        
        self.up_conv_3d = nn.ModuleList()
        for _ in range(2):
            self.up_conv_3d.append(
                GUpsampleBlock3d(channels, channels // 2, 3, 2, 1,
                                 noise_channels, activation, output_padding=1)
            )
            channels = channels // 2

        self.transform = Transform3d()

        self.conv_3d = nn.Sequential(
            GBlock3d(channels, channels//2, 3, 1, activation, padding_mode='replicate'),
            GBlock3d(channels//2, channels//2, 3, 1, activation, padding_mode='replicate')
        )
        channels = channels // 2
        
        # collapse depth here

        channels = channels * const_size * (2 ** 2)
        self.projection = Block2d(channels, channels//2, 1, 0, activation, padding_mode='replicate')

        channels = channels // 2
        self.up_conv_2d = nn.ModuleList()
        for _ in range(3):
            self.up_conv_2d.append(
                GUpsampleBlock2d(channels, channels//2, 4, 2, 1,
                                 noise_channels, activation)
            )
            channels = channels // 2

        self.out = nn.Sequential(
            nn.Conv2d(channels, 3, 3, stride=1, padding=1, padding_mode='replicate'),
            nn.Tanh()
        )

    def forward(self, z, theta):

        B = z.size(0)
        x = self.const_noise.expand(B, *self.const_noise.size()[1:])

        for block in self.up_conv_3d:
            x = block(x, z)

        x = self.transform(x, theta)

        x = self.conv_3d(x)

        # collapse depth
        # official implementation (https://github.com/thunguyenphuoc/HoloGAN/blob/master/model_HoloGAN.py#L507) is
        # (B, H, W, D, C) -> (B, C, H, W, D*C)
        # but pytorch is channel first so,
        # (B, C, H, W, D) -> (B, C, D, H, W) -> (B, C*D, H, W)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = x.view(B, x.size(1)*x.size(2), x.size(3), x.size(4))

        x = self.projection(x)

        for block in self.up_conv_2d:
            x = block(x, z)

        x = self.out(x)

        return x


class Discriminator(nn.Module):
    def __init__(self,
        channels=64, noise_channels=128, activation='lrelu', img_size=128
    ):
        super().__init__()

        self.head = Block2d(3, channels, 5, 2, activation, stride=2, padding_mode='replicate')
        img_size = img_size // 2

        self.clf_blocks = nn.ModuleList()
        for _ in range(4):
            self.clf_blocks.append(
                DBlock2d(channels, channels*2, 5, 2, 2, activation, padding_mode='replicate')
            )
            channels = channels * 2
            img_size = img_size // 2
        
        features = channels * img_size ** 2
        self.rf_linear = nn.Linear(features, 1)
        self.z_linear = nn.Sequential(
            nn.Linear(features, 128),
            get_activation(activation, True, 0.2),
            nn.Linear(128, noise_channels),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.head(x)

        logits = []
        for block in self.clf_blocks:
            x, logit = block(x)
            logits.append(logit)
        
        x = self.flatten(x)
        real_fake = self.rf_linear(x)
        z_reconstruct = self.z_linear(x)

        return real_fake, z_reconstruct, logits

    def flatten(self, x):
        return x.view(x.size(0), -1)

if __name__ == "__main__":
    import numpy as np
    angles = [0, 0, -45, 45, 0, 0]
    def radius(deg):
        return deg * (np.pi / 180)
    angles = {
        'x_min' : radius(angles[0]),
        'x_max' : radius(angles[1]),
        'y_min' : radius(angles[2]),
        'y_max' : radius(angles[3]),
        'z_min' : radius(angles[4]),
        'z_max' : radius(angles[5])
    }
    batch_size = 3
    samples = []
    for _ in range(batch_size):
        samples.append(
            [
                np.random.uniform(angles['x_min'], angles['x_max']),
                np.random.uniform(angles['y_min'], angles['y_max']),
                np.random.uniform(angles['z_min'], angles['z_max'])
            ]
        )
    samples = np.array(samples)
    theta = np.zeros((batch_size, 3, 4))
    def rot_x(angle):
        matrix = np.zeros((3, 3)).astype(np.float32)
        matrix[0, 0] =   1.
        matrix[1, 1] =   np.cos(angle)
        matrix[1, 2] = - np.sin(angle)
        matrix[2, 1] =   np.sin(angle)
        matrix[2, 2] =   np.cos(angle)
        # print(matrix)
        return matrix
    def rot_y(angle):
        matrix = np.zeros((3, 3)).astype(np.float32)
        matrix[0, 0] =   np.cos(angle)
        matrix[0, 2] =   np.sin(angle)
        matrix[1, 1] =   1.
        matrix[2, 1] = - np.sin(angle)
        matrix[2, 2] =   np.cos(angle)
        # print(matrix)
        return matrix
    def rot_z(angle):
        matrix = np.zeros((3, 3)).astype(np.float32)
        matrix[0, 0] =   np.cos(angle)
        matrix[0, 2] = - np.sin(angle)
        matrix[1, 1] =   np.sin(angle)
        matrix[2, 1] =   np.cos(angle)
        matrix[2, 2] =   1.
        # print(matrix)
        return matrix
    def pad_matrix(matrix):
        return np.hstack([matrix, np.zeros((3, 1))])
    samples_x = samples[:, 0]
    samples_y = samples[:, 1]
    samples_z = samples[:, 2]
    for i in range(batch_size):
        theta[i] = pad_matrix(np.dot(np.dot(rot_z(samples_z[i]), rot_y(samples_y[i])), rot_x(samples_x[i])))
    theta = torch.from_numpy(theta).float()
    z = torch.randn(batch_size, 128)
    G = Generator()
    img = G(z, theta)
    print(img.size())

    D = Discriminator()
    rf, z, logits = D(img)
    print(
        rf.size(), z.size(), logits[0].size(), logits[1].size(), logits[2].size(), logits[3].size()
    )

    from utils import style_loss
    print(style_loss(logits, logits))
