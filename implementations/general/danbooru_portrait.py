
import random
import glob
from collections.abc import Callable

from .dataset_base import Image, ImageImage, ImageXDoG, LRHR, make_default_transform

class DanbooruPortraitDataset(Image):
    '''Danbooru Portrait Dataset
    '''
    def __init__(self, image_size, transform=None, num_images=None):
        if transform is None:
            transform = make_default_transform(image_size, 1.2)
        super().__init__(transform)
        if num_images is not None:
            assert 0 < num_images <= len(self.images) and isinstance(num_images, int)
            random.shuffle(self.images)
            self.images = self.images[:num_images]
            self.length = len(self.images)
    
    def _load(self):
        image_paths = glob.glob('/usr/src/data/danbooru/portraits/portraits/*')
        return image_paths

class DanbooruPortraitCelebADataset(ImageImage):
    def __init__(self, image_size, transform=None, num_images=None):
        if transform is None:
            transform = make_default_transform(image_size, 1.2)
        super().__init__(transform)
        if num_images < self.length:
            self.images1 = self.images1[:num_images]
            self.images2 = self.images2[:num_images]
            self.length  = num_images

    def _load(self):
        images = glob.glob('/usr/src/data/danbooru/portraits/portraits/*')
        celeba = glob.glob('/usr/src/data/celeba/img_align_celeba/*')
        length = min(len(images), len(celeba))
        return images[:length], celeba[:length]

class DanbooruPortraitSRDataset(LRHR):
    def __init__(self, image_size, scale=2, resize_ratio=1.1, transform=None, num_images=None):
        super().__init__(image_size, scale, resize_ratio)
        if isinstance(transform, Callable):
            self.transform = transform
        if num_images is not None:
            assert 0 < num_images <= len(self.images) and isinstance(num_images, int)
            random.shuffle(self.images)
            self.images = self.images[:num_images]
            self.length = len(self.images)
    
    def _load(self):
        image_paths = glob.glob('/usr/src/data/danbooru/portraits/portraits/*')
        return image_paths

class XDoGDanbooruPortraitDataset(ImageXDoG):
    '''Image + XDoG Danbooru Portrait Dataset
    '''
    def __init__(self, image_size, transform=None, num_images=None):
        if transform is None:
            transform = make_default_transform(image_size, 1.2, hflip=False)
        super().__init__(transform)
        if num_images is not None:
            assert 0 < num_images <= len(self.images) and isinstance(num_images, int)
            random.shuffle(self.images)
            self.images = self.images[:num_images]
            self.length = len(self.images)
            self.xdogs = [path.replace('portraits/portraits', 'portraits/xdog') for path in self.images]

    def _load(self):
        image_paths = glob.glob('/usr/src/data/danbooru/portraits/portraits/*')
        xdog_paths  = [path.replace('portraits/portraits', 'portraits/xdog') for path in image_paths]
        return image_paths, xdog_paths
