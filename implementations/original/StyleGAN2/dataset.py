
import glob
import random
from dataset._base import Image, ImageImage, make_default_transform, pilImage

class AnimePhoto(Image):
    def __init__(self, image_size, anime='animeface', transform=None) -> None:
        self._anime_image_type = anime
        if transform is None:
            transform = make_default_transform(image_size, 1. if anime == 'animeface' else 1.2)
        super().__init__(transform)
    def _load(self):
        if self._anime_image_type == 'animeface':
            anime = glob.glob('/usr/src/data/animefacedataset/images/*')
        elif self._anime_image_type == 'danbooru':
            anime = glob.glob('/usr/src/data/danbooru/portraits/portraits/*')
        photo = glob.glob('/usr/src/data/celeba/img_align_celeba/*')
        length = min(len(anime), len(photo))
        anime, photo = anime[:length], photo[:length]
        return anime+photo

class XDoG(Image):
    def __init__(self, image_size, anime='animeface', num_images=None, transform=None) -> None:
        self._anime_image_type = anime
        self._num_images = num_images
        if transform is None:
            transform = make_default_transform(image_size, 1. if anime == 'animeface' else 1.2)
        super().__init__(transform)

    def _load(self):
        if self._anime_image_type == 'animeface':
            return glob.glob('/usr/src/data/animefacedataset/xdog/*')
        elif self._anime_image_type == 'danbooru':
            images = glob.glob('/usr/src/data/danbooru/portraits/xdog/*')
            if self._num_images is not None:
                random.shuffle(images)
                images = images[:self._num_images]
            return images

    def __getitem__(self, index):
        image = self.images[index]
        image = pilImage.open(image).convert('L')
        return self.transform(image)

class AnimePhotoSeparate(ImageImage):
    def __init__(self, image_size, anime='animeface', transform=None) -> None:
        self._anime_image_type = anime
        if transform is None:
            transform = make_default_transform(image_size, 1. if anime == 'animeface' else 1.2)
        super().__init__(transform)
    def _load(self):
        if self._anime_image_type == 'animeface':
            anime = glob.glob('/usr/src/data/animeface/images/*')
        elif self._anime_image_type == 'danbooru':
            anime = glob.glob('/usr/src/data/danbooru/portraits/portraits/*')
        photo = glob.glob('/usr/src/data/celeba/img_align_celeba/*')
        length = min(len(anime), len(photo))
        anime, photo = anime[:length], photo[:length]
        return anime, photo

    def shuffle(self):
        random.shuffle(self.images1)
