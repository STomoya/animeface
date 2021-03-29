
from collections.abc import Callable
import glob

from .dataset_base import pilImage, Image, LRHR, make_default_transform

class DanbooruDataset(Image):
    '''Danbooru Dataset
    '''
    def __init__(self, image_size, transform=None):
        if transform is None:
            transform = make_default_transform(image_size, 1.2)
        super().__init__(transform)
    def _load(self):
        return glob.glob('/usr/src/data/danbooru/2020/*/*.jpg', recursive=True)

class DanbooruSRDataset(LRHR):
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
        image_paths = glob.glob('/usr/src/data/danbooru/2020/*/*.jpg', recursive=True)
        return image_paths

class GeneratePairImageDanbooruDataset(DanbooruDataset):
    '''Automatically generated pair images Danbooru Dataset
    '''
    def __init__(self, pair_transform, image_size, transform=None):
        super().__init__(image_size, transform)
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
