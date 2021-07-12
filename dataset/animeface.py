
from __future__ import annotations

import os, glob, csv
from collections.abc import Callable
from typing import Optional

from dataset._base import (
    Image,
    ImageImage,
    ImageXDoG,
    ImageLabel,
    ImageOnehot,
    LRHR)
from dataset._base import make_default_transform

# strip posted year from path
_year_from_path = lambda path: int(os.path.splitext(os.path.basename(path))[0].split('_')[-1])

class AnimeFace(Image):
    '''AnimeFace dataset
    '''
    def __init__(self,
        image_size: int,
        min_year: Optional[int]=2005,
        transform: Optional[Callable]=None
    ) -> None:
        self.min_year = min_year
        if transform is None:
            transform = make_default_transform(image_size)
        super().__init__(transform)

    def _load(self) -> list[str]:
        images = glob.glob('/usr/src/data/animefacedataset/images/*')
        if self.min_year is not None:
            images = [path for path in images if _year_from_path(path) >= self.min_year]
        return images

class AnimeFaceCelebA(ImageImage):
    '''AnimeFace + CelebA dataset
    For unpaired I2I
    '''
    def __init__(self,
        image_size: int,
        min_year: Optional[int]=2005,
        transform: Optional[Callable]=None
    ) -> None:
        self.min_year = min_year
        if transform is None:
            transform = make_default_transform(image_size)
        super().__init__(transform)

    def _load(self) -> tuple[list[str], list[str]]:
        images = glob.glob('/usr/src/data/animefacedataset/images/*')
        celeba = glob.glob('/usr/src/data/celeba/img_align_celeba/*')
        if self.min_year is not None:
            images = [path for path in images if _year_from_path(path) >= self.min_year]
        length = min(len(images), len(celeba))
        return images[:length], celeba[:length]

class AnimeFaceSR(LRHR):
    '''AimeFace super resolution dataset
    '''
    def __init__(self,
        image_size: int,
        scale: float=2,
        transform: Optional[Callable]=None
    ) -> None:
        if image_size > 128:
            import warnings
            warnings.warn('animeface dataset image size is small. you should use danbooru dataset for super-resolution tasks')
        super().__init__(image_size, scale)
        if isinstance(transform, Callable):
            self.transform = transform

    def _load(self) -> list[str]:
        return glob.glob('/usr/src/data/animefacedataset/images/*')

class AnimeFaceXDoG(ImageXDoG):
    '''Image + XDoG Anime Face Dataset
    '''
    def __init__(self,
        image_size: int,
        min_year: Optional[int]=2005,
        transform: Optional[Callable]=None
    ) -> None:
        self.min_year = min_year
        if transform is None:
            transform = make_default_transform(image_size, hflip=False)
        super().__init__(transform)

    def _load(self) -> tuple[list[str], list[str]]:
        images = glob.glob('/usr/src/data/animefacedataset/images/*')
        if self.min_year is not None:
            images = [path for path in images if _year_from_path(path) >= self.min_year]
        xdogs  = [path.replace('images', 'xdog') for path in images]
        return images, xdogs

class AnimeFaceLabel(ImageLabel):
    '''Labeled Anime Face Dataset
    images, illustration2vec tags
    '''
    def __init__(self,
        image_size: int,
        transform: Optional[Callable]=None
    ) -> None:
        if transform is None:
            transform = make_default_transform(image_size)
        super().__init__(transform)

    def _load(self) -> tuple[list[str], list[int]]:
        dataset_file_path = '/usr/src/data/animefacedataset/labels.csv'
        with open(dataset_file_path, 'r', encoding='utf-8') as fin:
            csv_reader = csv.reader(fin)
            data_list = [line for line in csv_reader]

        # file format
        # "path/to/file","i2vtag"

        image_paths = [line[0] for line in data_list]
        i2v_labels  = [line[1] for line in data_list]

        return image_paths, i2v_labels

class AnimeFaceOneHot(ImageOnehot):
    '''One-Hot Labeled Anime Face Dataset
    images, one-hot encoded illustration2vec tags
    '''
    def __init__(self,
        image_size: int,
        transform: Optional[Callable]=None
    ) -> None:
        if transform is None:
            transform = make_default_transform(image_size)
        super().__init__(transform)

    def _load(self) -> tuple[list[str], list[int]]:
        dataset_file_path = '/usr/src/data/animefacedataset/labels.csv'
        with open(dataset_file_path, 'r', encoding='utf-8') as fin:
            csv_reader = csv.reader(fin)
            data_list = [line for line in csv_reader]

        # file format
        # "path/to/file","i2vtag"

        image_paths = [line[0] for line in data_list]
        i2v_labels  = [line[1] for line in data_list]

        return image_paths, i2v_labels
