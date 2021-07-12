
from __future__ import annotations

from collections.abc import Callable
import glob, random
from typing import Optional
import torch
from dataset._base import (
    pilImage,
    Image,
    LRHR)
from dataset._base import make_default_transform

class Danbooru(Image):
    '''Danbooru Dataset
    '''
    def __init__(self,
        image_size: int,
        num_images: int=None,
        transform: Optional[Callable]=None
    ) -> None:
        self.num_images = num_images
        if transform is None:
            transform = make_default_transform(image_size, 1.2)
        super().__init__(transform)
    def _load(self) -> list[str]:
        images = glob.glob('/usr/src/data/danbooru/2020/*/*.jpg', recursive=True)
        if self.num_images is not None:
            random.shuffle(images)
            images = images[:self.num_images]
        return images

class DanbooruSR(LRHR):
    '''Danbooru super resolution dataset
    '''
    def __init__(self,
        image_size: int,
        scale: float=2,
        resize_ratio: float=1.1,
        num_images: Optional[int]=None,
        transform: Optional[Callable]=None
    ) -> None:
        self.num_images = num_images
        super().__init__(image_size, scale, resize_ratio)
        if isinstance(transform, Callable):
            self.transform = transform

    def _load(self) -> list[str]:
        image_paths = glob.glob('/usr/src/data/danbooru/2020/*/*.jpg', recursive=True)
        if self.num_images is not None:
            random.shuffle(image_paths)
            image_paths = image_paths[:self.num_images]
        return image_paths

class DanbooruAutoPair(Danbooru):
    '''Automatically generated pair images Danbooru Dataset
    '''
    def __init__(self,
        image_size: int,
        pair_transform: Callable,
        num_images: Optional[int]=None,
        transform: Optional[Callable]=None
    ) -> None:
        super().__init__(image_size, num_images, transform)
        self.pair_transform = pair_transform

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index]

        temp_image = pilImage.open(image).convert('RGB')
        image, pair_image = self._transform(temp_image)

        return image, pair_image

    def _transform(self, pil_image) -> tuple[torch.Tensor, torch.Tensor]:
        pair_image = self.pair_transform(pil_image)
        pair_image = self.transform(pair_image)
        image = self.transform(pil_image)
        return image, pair_image
