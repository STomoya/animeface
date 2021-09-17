'''
Frequently used layers
'''

import torch
import torch.nn as nn

def get_activation(
    name: str,
    inplace: bool=True
) -> nn.Module:
    if   name == 'relu':    return nn.ReLU(inplace)
    elif name == 'lrelu':   return nn.LeakyReLU(0.2, inplace)
    elif name == 'tanh':    return nn.Tanh()
    elif name == 'gelu':    return nn.GELU()
    elif name == 'swish':   return nn.SiLU()
    elif name == 'prelu':   return nn.PReLU()
    elif name == 'sigmoid':
        raise Exception(f'Activation: use torch.sigmoid()')
    raise Exception(f'Activation: {name} not found')

def get_normalization(
    name: str,
    channels: int
) -> nn.Module:
    if name == 'in': return nn.InstanceNorm2d(channels)
    if name == 'bn': return nn.BatchNorm2d(channels)
    raise Exception(f'Normalization: {name} not found')

class MiniBatchStdDev(nn.Module):
    '''Mini-Batch Standard Deviation'''
    def __init__(self,
        group_size: int=4,
        eps: float=1e-4
    ) -> None:
        super().__init__()
        self.group_size = group_size
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        y = x
        groups = self._check_group_size(B)
        # calc stddev and concatenate
        y = y.view(groups, -1, C, H, W)
        y = y - y.mean(0, keepdim=True)
        y = y.square().mean(0)
        y = y.add_(self.eps).sqrt()
        y = y.mean([1, 2, 3], keepdim=True)
        y = y.repeat(groups, 1, H, W)

        return torch.cat([x, y], dim=1)

    def _check_group_size(self, batch_size: int) -> int:
        if batch_size % self.group_size == 0: return self.group_size
        else:                                 return batch_size

def SNConv2d(*args, **kwargs) -> nn.Module:
    return nn.utils.spectral_norm(nn.Conv2d(*args, **kwargs))
def SNLinear(*args, **kwargs) -> nn.Module:
    return nn.utils.spectral_norm(nn.Linear(*args, **kwargs))
def SNConvTranspose2d(*args, **kwargs) -> nn.Module:
    return nn.utils.spectral_norm(nn.ConvTranspose2d(*args, **kwargs))
