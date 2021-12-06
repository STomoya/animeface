
import glob
import random
from dataset._base import WrappedDataset, pilImage, TF

class GrayRGB(WrappedDataset):
    def __init__(self, image_size) -> None:
        super().__init__()
        self.image_size = image_size
        self.images = self._load()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]

        rgb = pilImage.open(image).convert('RGB')
        gray = pilImage.open(image).convert('L')
        gray, rgb = self.transform(gray, rgb)
        return gray, rgb

    def transform(self, gray, rgb):
        gray = TF.resize(gray, self.image_size)
        gray = TF.center_crop(gray, self.image_size)
        gray = TF.to_tensor(gray)
        gray = TF.normalize(gray, 0.5, 0.5)
        rgb  = TF.resize(rgb, self.image_size)
        rgb  = TF.center_crop(rgb, self.image_size)
        rgb  = TF.to_tensor(rgb)
        rgb  = TF.normalize(rgb, 0.5, 0.5)
        return gray, rgb

class AnimeFace(GrayRGB):
    def _load(self):
        return glob.glob('/usr/src/data/animefacedataset/images/*')

class DanbooruPortraitsTest(GrayRGB):
    def __init__(self, image_size, num_images=5) -> None:
        self.num_images = num_images
        super().__init__(image_size)

    def _load(self):
        images = glob.glob('/usr/src/data/danbooru/portraits/portraits/*')
        random.seed(0)
        random.shuffle(images)
        images = images[:self.num_images]
        return images
