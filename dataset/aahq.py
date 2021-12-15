
import glob
import random

from collections.abc import Callable
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
