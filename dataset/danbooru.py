
from collections.abc import Callable
import glob, random

from dataset._base import (
    pilImage,
    Image,
    LRHR)
from dataset._base import make_default_transform

class Danbooru(Image):
    '''Danbooru Dataset
    '''
    def __init__(self, image_size, num_images=None, transform=None):
        self.num_images = num_images
        if transform is None:
            transform = make_default_transform(image_size, 1.2)
        super().__init__(transform)
    def _load(self):
        images = glob.glob('/usr/src/data/danbooru/2020/*/*.jpg', recursive=True)
        if self.num_images is not None:
            random.shuffle(images)
            images = images[:self.num_images]
        return images

class DanbooruSR(LRHR):
    '''Danbooru super resolution dataset
    '''
    def __init__(self, image_size, scale=2, resize_ratio=1.1, num_images=None, transform=None):
        self.num_images = num_images
        super().__init__(image_size, scale, resize_ratio)
        if isinstance(transform, Callable):
            self.transform = transform

    def _load(self):
        image_paths = glob.glob('/usr/src/data/danbooru/2020/*/*.jpg', recursive=True)
        if self.num_images is not None:
            random.shuffle(image_paths)
            image_paths = image_paths[:self.num_images]
        return image_paths

class DanbooruAutoPair(Danbooru):
    '''Automatically generated pair images Danbooru Dataset
    '''
    def __init__(self, image_size, pair_transform, num_images=None, transform=None):
        super().__init__(image_size, num_images, transform)
        self.pair_transform = pair_transform

    def __getitem__(self, index):
        image = self.images[index]

        temp_image = pilImage.open(image).convert('RGB')
        image, pair_image = self._transform(temp_image)

        return image, pair_image

    def _transform(self, pil_image):
        pair_image = self.pair_transform(pil_image)
        pair_image = self.transform(pair_image)
        image = self.transform(pil_image)
        return image, pair_image
