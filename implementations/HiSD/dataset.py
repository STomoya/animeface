'''
Dataset with multiple categories
'''

import os
import csv

from dataset._base import Image, make_default_transform
from dataset import to_loader

def _read_csv(filename):
    '''read csv file'''
    with open(filename, 'r', encoding='utf-8') as fin:
        csv_reader = csv.reader(fin)
        data_list = [line for line in csv_reader]
    return data_list

def _split_to_tags(label_file):
    data_list = _read_csv(label_file)
    labels = [line[1] for line in data_list]
    unique_labels = list(set(labels))
    image_paths = [[] for _, _ in enumerate(unique_labels)]
    for path, tag in data_list:
        image_paths[unique_labels.index(tag)].append(path)
    return image_paths, unique_labels

class Tag(Image):
    def __init__(self, image_paths, image_size, resize_ratio=1., transform=None):
        if transform is None:
            transform = make_default_transform(
                image_size, resize_ratio
            )
        super().__init__(transform)
        self.images = image_paths
    def _load(self):
        return []

class Category:
    def __init__(self,
        label_file, image_size, batch_size, num_workers=os.cpu_count(),
        resize_ratio=1., transform=None
    ):
        image_paths, unique_labels = _split_to_tags(label_file)
        self.loaders = [
            to_loader(
                Tag(images, image_size, resize_ratio, transform),
                batch_size, num_workers=num_workers, pin_memory=False)
            for images in image_paths
        ]
        self.iterable = [iter(loader) for loader in self.loaders]

        self.tags = unique_labels
        self.num_tags = len(unique_labels)
        self.length = sum([len(images) for images in image_paths])

    def sample(self, j):
        try:
            return next(self.iterable[j])
        except StopIteration as si:
            self.iterable[j] = None
            self.iterable[j] = iter(self.loaders[j])
            return next(self.iterable[j])
        else:
            raise

class Hair(Category):
    def __init__(self,
        root, image_size, batch_size, num_workers=os.cpu_count(),
        resize_ratio=1., transform=None
    ):
        label_file = os.path.join(root, 'hair_color_labels.csv')
        super().__init__(
            label_file, image_size, batch_size,
            num_workers, resize_ratio, transform
        )

class Eye(Category):
    def __init__(self,
        root, image_size, batch_size, num_workers=os.cpu_count(),
        resize_ratio=1., transform=None
    ):
        label_file = os.path.join(root, 'eye_color_labels.csv')
        super().__init__(
            label_file, image_size, batch_size,
            num_workers, resize_ratio, transform
        )

class Glass(Category):
    def __init__(self,
        root, image_size, batch_size, num_workers=os.cpu_count(),
        resize_ratio=1., transform=None
    ):
        label_file = os.path.join(root, 'glass_labels.csv')
        super().__init__(
            label_file, image_size, batch_size,
            num_workers, resize_ratio, transform
        )

class _CategoricalInfiniteLoader:
    '''contain multiple category datasets
    '''
    def __init__(self, root,
        image_size, resize_ratio, batch_size, num_workers=os.cpu_count(),
        transform=None, image_dataset_class=[Hair, Eye, Glass]
    ):
        self.category = [
            dataset(root, image_size, batch_size, num_workers, resize_ratio, transform)
            for dataset in image_dataset_class
        ]
        self.length = sum([cat.length for cat in self.category])
        self.tags = [cat.tags for cat in self.category]
        self.num_tags = [cat.num_tags for cat in self.category]
    def sample(self, i, j):
        return self.category[i].sample(j)

    def __len__(self):
        return self.length

class AnimeFace(_CategoricalInfiniteLoader):
    '''anime face dataset'''
    def __init__(self,
        image_size, batch_size, num_workers=os.cpu_count(),
        image_dataset_class: list=[Hair, Eye, Glass],
        transform=None
    ):
        root = '/usr/src/data/animefacedataset'
        super().__init__(
            root, image_size, 1., batch_size, num_workers,
            transform, image_dataset_class
        )

class DanbooruPortrait(_CategoricalInfiniteLoader):
    '''danbooru portrait dataset'''
    def __init__(self,
        image_size, batch_size, num_workers=os.cpu_count(),
        image_dataset_class: list=[Hair, Eye, Glass],
        transform=None
    ):
        root = '/usr/src/data/danbooru/portraits'
        super().__init__(
            root, image_size, 1.2, batch_size, num_workers,
            transform, image_dataset_class
        )
