
import glob

from .dataset_base import pilImage, Image, make_default_transform

class DanbooruDataset(Image):
    '''Danbooru Dataset
    '''
    def __init__(self, image_size, transform=None):
        if transform is None:
            transform = make_default_transform(image_size, 1.2)
        super().__init__(transform)
    def _load(self):
        return glob.glob('/usr/src/data/danbooru/danbooru-images/**/*.jpg', recursive=True)

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
