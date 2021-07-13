
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.cuda.amp import autocast
import torchvision.transforms.functional as TF

from nnutils.loss._base import Loss

class VGG(nn.Module):
    '''VGG with only feature layers'''
    def __init__(self,
        layers: int=16,
        pretrained: bool=True
    ):
        super().__init__()
        assert layers in [16, 19], 'only supports VGG16 and VGG19'
        if layers == 16:
            vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
            slices = [4, 9, 16, 23, 30]
        if layers == 19:
            vgg_pretrained_features = models.vgg19(pretrained=pretrained).features
            slices = [4, 9, 18, 27, 36]

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(slices[0]):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(slices[0], slices[1]):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(slices[1], slices[2]):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(slices[2], slices[3]):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(slices[3], slices[4]):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        return h_relu1, h_relu2, h_relu3, h_relu4, h_relu5

def gram_matrix(x):
    B, C, H, W = x.size()
    feat = x.reshape(B, C, H*W)
    G = torch.bmm(feat, feat.permute(0, 2, 1))
    return G.div(C*H*W)

class VGGLoss(Loss):
    '''loss using vgg

    args:
        device: torch.device
            the device working on.
        vgg: int (default: 16)
            layers of VGG model. 16 or 19.
        p: int (default: 2)
            Lp. 1: L1, 2: L2
        normalized: bool (default: True)
            if the input is normalized or not.
        return_all: bool (default: False)
            return all intermediate results

    NOTE: all loss in one class to avoid loading VGG several times to device.
    '''
    def __init__(self,
        device,
        vgg: int=16,
        p: int=2,
        normalized: bool=True,
        return_all: bool=False
    ) -> None:
        super().__init__(return_all)
        assert p in [1, 2]
        self.p = p
        self.normalized = normalized
        self.vgg = VGG(vgg, pretrained=True)
        self.vgg.to(device)

    def _check_index(self,
        index: int) -> None:
        def assert_index(index):
            assert 0 <= index <= 4
        if isinstance(index, int):
            assert_index(index)
        if isinstance(index, (list, tuple)):
            for i in index:
                assert_index(i)

    def loss_fn(self,
        x: torch.Tensor,
        y: torch.Tensor,
        p: Optional[int]=None
    ) -> torch.Tensor:
        p_ = self.p
        if p is not None:
            p_ = p

        if p_ == 1:
            return F.l1_loss(x, y)
        elif p_ == 2:
            return F.mse_loss(x, y)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        '''normalize input tensor'''
        return TF.normalize(x, 0.5, 0.5)

    def style_loss(self,
        real: torch.Tensor,
        fake: torch.Tensor,
        block_indices: list[int]=[0, 1, 2, 3],
        p: Optional[int]=None
    ) -> torch.Tensor:
        '''style loss introduced in
        "Perceptual Losses for Real-Time Style Transfer and Super-Resolution",
        Justin Johnson, Alexandre Alahi, and Li Fei-Fei
        '''
        self._check_index(block_indices)
        if not self.normalized:
            real, fake = self.normalize(real), self.normalize(fake)
        loss = 0
        with autocast(False):
            real_acts = self.vgg(real.float())
            fake_acts = self.vgg(fake.float())
            for index in block_indices:
                loss = loss \
                    + self.loss_fn(
                        gram_matrix(fake_acts[index]),
                        gram_matrix(real_acts[index]),
                        p
                    )

        return loss

    def content_loss(self,
        real: torch.Tensor,
        fake: torch.Tensor,
        block_index: int=2,
        p: Optional[int]=None
    ) -> torch.Tensor:
        '''content loss intruduced in
        "Perceptual Losses for Real-Time Style Transfer and Super-Resolution",
        Justin Johnson, Alexandre Alahi, and Li Fei-Fei
        '''
        self._check_index(block_index)
        if not self.normalized:
            real, fake = self.normalize(real), self.normalize(fake)
        loss = 0
        real_acts = self.vgg(real)
        fake_acts = self.vgg(fake)

        loss = self.loss_fn(
            fake_acts[block_index],
            real_acts[block_index],
            p
        )

        return loss

    def vgg_loss(self,
        real: torch.Tensor,
        fake: torch.Tensor,
        block_indices: list[int]=[0, 1, 2, 3, 4],
        p: Optional[int]=None
    ) -> torch.Tensor:
        '''perceptual loss used in pix2pixHD.
        They seem to use the activations of all convolution blocks,
        and calculates the distance with L1,
        different from using only 4 blocks and L2 in style loss and content loss.
        '''
        self._check_index(block_indices)
        if not self.normalized:
            real, fake = self.normalize(real), self.normalize(fake)
        loss = 0
        real_acts = self.vgg(real)
        fake_acts = self.vgg(fake)

        for index in block_indices:
            loss = loss + self.loss_fn(
                real_acts[index], fake_acts[index], p
            )

        return loss
