
from __future__ import annotations

import glob
import random

from collections.abc import Callable
from typing import Optional

from dataset._base import (
    Image, ImageImage, LRHR,
    make_default_transform)

class AAHQ(Image):
    def __init__(self,
        image_size: int,
        num_images: int=None,
        transform: Callable=None
    ) -> None:
        self.num_images = num_images
        if transform is None:
            transform = make_default_transform(image_size)
        super().__init__(transform)

    def _load(self):
        images = glob.glob('/usr/src/data/aahq/*')
        if self.num_images is not None and 0 < self.num_images < len(images):
            random.shuffle(images)
            images = images[:self.num_images]
        return images

class AAHQSR(LRHR):
    def __init__(self,
        image_size: int,
        scale: float=2,
        resize_scale: float=1.,
        num_images: int=None,
        transform: Optional[Callable]=None
    ) -> None:
        self.num_images = num_images
        super().__init__(image_size, scale, resize_scale)
        if callable(transform):
            self.transform = transform

    def _load(self) -> list[str]:
        image_paths = glob.glob('/usr/src/data/aahq/*')
        if self.num_images is not None and 0 < self.num_images < len(image_paths):
            random.shuffle(image_paths)
            image_paths = image_paths[:self.num_images]
        return image_paths

class AAHQCelebA(ImageImage):
    def __init__(self,
        image_size: int,
        num_images: int=None,
        transform: Callable=None
    ) -> None:
        self.num_images = num_images
        if transform is None:
            transform = make_default_transform(image_size)
        super().__init__(transform)

    def _load(self):
        images = glob.glob('/usr/src/data/aahq/*')
        celeba = glob.glob('/usr/src/data/celeba/img_align_celeba/*')
        length = min(len(images), len(celeba))
        images, celeba = images[:length], celeba[:length]
        if self.num_images is not None and 0 < self.num_images < length:
            random.shuffle(images)
            random.shuffle(celeba)
            images, celeba = images[:self.num_images], celeba[:self.num_images]
        return images, celeba
