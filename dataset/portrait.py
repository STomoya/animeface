
from __future__ import annotations

import random
import glob
from collections.abc import Callable
from typing import Optional

from dataset._base import (
    Image,
    ImageImage,
    ImageXDoG,
    LRHR)
from dataset._base import make_default_transform

class DanbooruPortrait(Image):
    '''Danbooru Portrait Dataset
    '''
    def __init__(self,
        image_size: int,
        num_images: Optional[int]=None,
        transform: Optional[Callable]=None
    ) -> None:
        self.num_images = num_images
        if transform is None:
            transform = make_default_transform(image_size, 1.2)
        super().__init__(transform)

    def _load(self) -> list[str]:
        image_paths = glob.glob('/usr/src/data/danbooru/portraits/portraits/*')
        if self.num_images is not None:
            random.shuffle(image_paths)
            image_paths = image_paths[:self.num_images]
        return image_paths

class DanbooruPortraitCelebA(ImageImage):
    '''Danbooru Portraits + CelebA dataset
    '''
    def __init__(self,
        image_size: int,
        num_images: Optional[int]=None,
        transform: Optional[Callable]=None
    ) -> None:
        self.num_images = num_images
        if transform is None:
            transform = make_default_transform(image_size, 1.2)
        super().__init__(transform)

    def _load(self) -> tuple[list[str], list[str]]:
        images = glob.glob('/usr/src/data/danbooru/portraits/portraits/*')
        celeba = glob.glob('/usr/src/data/celeba/img_align_celeba/*')
        length = min(len(images), len(celeba))
        images, celeba = images[:length], celeba[:length]
        if self.num_images is not None and self.num_images < length:
            random.shuffle(images)
            random.shuffle(celeba)
            images, celeba = images[:self.num_images], celeba[:self.num_images]
        return images, celeba

class DanbooruPortraitSR(LRHR):
    '''Danbooru Portraits super resolution dataset
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
        image_paths = glob.glob('/usr/src/data/danbooru/portraits/portraits/*')
        if self.num_images is not None:
            random.shuffle(image_paths)
            image_paths = image_paths[:self.num_images]
        return image_paths

class DanbooruPortraitXDoG(ImageXDoG):
    '''Image + XDoG Danbooru Portrait Dataset
    '''
    def __init__(self,
        image_size: int,
        num_images: Optional[int]=None,
        transform: Optional[Callable]=None
    ) -> None:
        self.num_images = num_images
        if transform is None:
            transform = make_default_transform(image_size, 1.2, hflip=False)
        super().__init__(transform)

    def _load(self) -> tuple[list[str], list[str]]:
        image_paths = glob.glob('/usr/src/data/danbooru/portraits/portraits/*')
        if self.num_images is not None:
            random.shuffle(image_paths)
            image_paths = image_paths[:self.num_images]
        xdog_paths  = [path.replace('portraits/portraits', 'portraits/xdog') for path in image_paths]
        return image_paths, xdog_paths
