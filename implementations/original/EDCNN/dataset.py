'''
Dataset for a simpler gray-scale to rgb image task

sketch (XDoG) images are already edges so sketch to rgb image task
will not be a suitable task for edge enhancement module.
'''

import glob
import random

import torchvision.transforms.functional as TF
from dataset._base import WrappedDataset, pilImage

class _ImageGrayOTF(WrappedDataset):
    '''
    dataset with rgb and gray paired image.
    gray images are generated on the fly
    '''
    def __init__(self, image_size, resize_ratio=1.):
        self.image_size = image_size
        self.resize_ratio = resize_ratio
        self.images = self._load()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]

        image = pilImage.open(image).convert('RGB')
        rgb, gray = self.transform(image)

        return rgb, gray

    def transform(self, image):
        image = TF.resize(image, int(self.image_size*self.resize_ratio))
        image = TF.center_crop(image, self.image_size)
        gray = TF.rgb_to_grayscale(image)
        image = TF.adjust_hue(image, (random.random()-0.5)/5)
        image = TF.to_tensor(image)
        image = TF.normalize(image, 0.5, 0.5)
        gray = TF.to_tensor(gray)
        gray = TF.normalize(gray, 0.5, 0.5)
        return image, gray

class AnimeFace(_ImageGrayOTF):
    def _load(self):
        return glob.glob('/usr/src/data/animefacedataset/images/*')

class Danbooru(_ImageGrayOTF):
    def __init__(self, image_size, resize_ratio=1.125, num_images=None):
        self.num_images = num_images
        super().__init__(image_size, resize_ratio)
    def _load(self):
        images = glob.glob('/usr/src/data/danbooru/portraits/portraits/*')
        if self.num_images is not None:
            random.shuffle(images)
            return images[:self.num_images]
        return images
