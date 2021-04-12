
import os
import csv
import glob
from collections.abc import Callable

from .dataset_base import Image, ImageImage, ImageXDoG, ImageLabel, ImageOnehot, LRHR
from .dataset_base import make_default_transform

class AnimeFaceDataset(Image):
    '''Anime Face Dataset
    '''
    def __init__(self, image_size, transform=None):
        if transform is None:
            transform = make_default_transform(image_size)
        super().__init__(transform)

    def _load(self):
        return glob.glob('/usr/src/data/animefacedataset/images/*')

class AnimeFaceCelebADataset(ImageImage):
    def __init__(self, image_size, min_year=2005, transform=None):
        self.min_year = min_year
        if transform is None:
            transform = make_default_transform(image_size)
        super().__init__(transform)
    
    def _load(self):
        images = glob.glob('/usr/src/data/animefacedataset/images/*')
        celeba = glob.glob('/usr/src/data/celeba/img_align_celeba/*')
        images = [path for path in images if self._year_from_path(path) >= self.min_year]
        length = min(len(images), len(celeba))
        return images[:length], celeba[:length]

    def _year_from_path(self, path):
        name, _ = os.path.splitext(os.path.basename(path))
        year = int(name.split('_')[-1])
        return year

class YearAnimeFaceDataset(AnimeFaceDataset):
    '''AnimeFaceDataset with minimum year option
    '''
    def __init__(self, image_size, min_year=2005, transform=None):
        super().__init__(image_size, transform)
        self.images = [path for path in self.images if self._year_from_path(path) >= min_year]
        self.length = len(self.images)
    
    def _year_from_path(self, path):
        name, _ = os.path.splitext(os.path.basename(path))
        year = int(name.split('_')[-1])
        return year

class AnimeFaceSRDataset(LRHR):
    def __init__(self, image_size, scale=2, transform=None):
        if image_size > 128:
            import warnings
            warnings.warn('animeface dataset image size is small. you should use danbooru dataset for super-resolution tasks')
        super().__init__(image_size, scale)
        if isinstance(transform, Callable):
            self.transform = transform
    
    def _load(self):
        return glob.glob('/usr/src/data/animefacedataset/images/*')

class XDoGAnimeFaceDataset(ImageXDoG):
    '''Image + XDoG Anime Face Dataset
    '''
    def __init__(self, image_size, min_year=2005, transform=None):
        if transform is None:
            transform = make_default_transform(image_size, hflip=False)
        super().__init__(transform)

    def _load(self):
        images = glob.glob('/usr/src/data/animefacedataset/images/*')
        xdogs  = [path.replace('images', 'xdog') for path in images]
        return images, xdogs

class LabeledAnimeFaceDataset(ImageLabel):
    '''Labeled Anime Face Dataset
    images, illustration2vec tags
    '''
    def __init__(self, image_size, transform=None):
        if transform is None:
            transform = make_default_transform(image_size)
        super().__init__(transform)

    def _load(self):
        dataset_file_path = '/usr/src/data/animefacedataset/labels.csv'
        with open(dataset_file_path, 'r', encoding='utf-8') as fin:
            csv_reader = csv.reader(fin)
            data_list = [line for line in csv_reader]
        
        # file format
        # "path/to/file","i2vtag"

        image_paths = [line[0] for line in data_list]
        i2v_labels  = [line[1] for line in data_list]

        return image_paths, i2v_labels

class OneHotLabeledAnimeFaceDataset(ImageOnehot):
    '''One-Hot Labeled Anime Face Dataset
    images, one-hot encoded illustration2vec tags
    '''
    def __init__(self, image_size, transform=None):
        if transform is None:
            transform = make_default_transform(image_size)
        super().__init__(transform)

    def _load(self):
        dataset_file_path = '/usr/src/data/animefacedataset/labels.csv'
        with open(dataset_file_path, 'r', encoding='utf-8') as fin:
            csv_reader = csv.reader(fin)
            data_list = [line for line in csv_reader]
        
        # file format
        # "path/to/file","i2vtag"

        image_paths = [line[0] for line in data_list]
        i2v_labels  = [line[1] for line in data_list]

        return image_paths, i2v_labels
