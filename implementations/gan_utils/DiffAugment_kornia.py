
'''
kornia based Differetiable Augmentation

I took care so that it can be used by almost the same way as the official implementation,
but modified it so that I can set the parameters from other files.

Difference from official implementation
(Mostly because of using karnia)
- padding mode for translation
    official : zero
    here     : same
- brightness and contrast range
    official : brightness [-0.5,  0.5]
               contrast   [ 0.5,  1.5]
    here     : brightness [0.75, 1.25]
               contrast   [0.75, 1.25]
- denorm before augmentation
    official : not needed
    here     : karnia.augmentation requires data values between [0, 1]
               so if normalized, denorm -> augment -> norm
'''

import torch
import torch.nn as nn
import kornia.augmentation as aug
from kornia.constants import SamplePadding

class kDiffAugment:
    def __init__(self,
        brightness=(0.75, 1.25), contrast=(0.75, 1.25), saturation=(0., 2.), translate=(0.125, 0.125),
        normalized=True, mean=0.5, std=0.5, device=None
    ):
        if normalized:
            if isinstance(mean, (tuple, list)) and isinstance(std, (tuple, list)):
                if not device:
                    raise Exception('Please specify a torch.device() object when using mean and std for each channels')
                mean = torch.Tensor(mean).to(device)
                std = torch.Tensor(std).to(device)
            self.normalize = aug.Normalize(mean, std)
            self.denormalize = aug.Denormalize(mean, std)
        else:
            self.normalize, self.denormalize = None, None

        color_jitter = aug.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, p=1.) # rand_brightness, rand_contrast, rand_saturation
        affine = aug.RandomAffine(degrees=0, translate=translate, padding_mode=SamplePadding.BORDER, p=1.)    # rand_translate
        cutout = aug.RandomErasing(value=0.5, p=1.)                                                           # rand_cutout

        self.augmentations = {
            'color' : color_jitter,
            'translation' : affine,
            'cutout' : cutout
        }
        
    def __call__(self, x, policy):
        if self.denormalize:
            x = self.denormalize(x)
        policy = self.__encode_policy(policy)
        for p in policy:
            aug_func = self.augmentations[p]
            x = aug_func(x)
        if self.normalize:
            x = self.normalize(x)
        return x
        
    def __encode_policy(self, policy):
        if isinstance(policy, (tuple, list)):
            return policy
        return policy.split(',')