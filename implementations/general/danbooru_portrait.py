
import random
import glob

from .dataset_base import Image, ImageXDoG, make_default_transform

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
