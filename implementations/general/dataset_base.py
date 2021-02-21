
import random
import csv

import torch
import torchvision.transforms as T
import numpy as np
from torch.utils.data import Dataset
from PIL import Image as pilImage
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def make_default_transform(
    image_size: int,
    resize_ratio: float=1.,
    hflip: bool=True,
    normalize: bool=True
):
    '''default transforms
    '''
    t_list = [
        T.Resize(int(image_size*resize_ratio)),
        T.CenterCrop(image_size)
    ]
    if hflip:
        t_list.append(T.RandomHorizontalFlip())
    
    t_list.append(T.ToTensor())
    if normalize:
        t_list.append(T.Normalize(0.5, 0.5))

    return T.Compose(t_list)

class _common_attr:
    '''common attributes for dataset classes
    '''
    def _load(self):
        raise NotImplementedError()
    def __len__(self):
        return self.length

class Image(Dataset, _common_attr):
    '''Image dataset base class
    '''
    def __init__(self, transform):
        self.images = self._load()
        self.length = len(self.images)
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]

        image = pilImage.open(image).convert('RGB')
        image = self.transform(image)

        return image

class ImageXDoG(Dataset, _common_attr):
    '''Image + XDoG dataset base class
    '''
    def __init__(self, transform):
        self.images, self.xdogs = self._load()
        self.length = len(self.images)
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]
        xdog  = self.xdogs[index]

        image = pilImage.open(image).convert('RGB')
        xdog  = pilImage.open(xdog).convert('L')

        image = self.transform(image)
        xdog  = self.transform(xdog)

        return image, xdog

    def shuffle_xdog(self):
        random.shuffle(self.xdogs)

class ImageLabel(Dataset, _common_attr):
    '''Image + Label dataset base class
    '''
    def __init__(self, transform):
        self.images, labels = self._load()
        self.length = len(self.images)
        self.transform = transform

        self._make_label(labels)

    def _make_label(self, labels):
        self.encoder = LabelEncoder()
        labels = np.array(labels).reshape(-1)
        self.labels = self.encoder.fit_transform(labels)
        self.num_classes = len(self.encoder.classes_)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        image = pilImage.open(image).convert('RGB')
        image = self.transform(image)

        return image, label

    def inverse_transform(self, label):
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()
        if not isinstance(label, np.ndarray):
            label = np.array(label)
        label = label.reshape(-1)
        return self.encoder.inverse_transform(label)

class ImageOnehot(ImageLabel):
    '''Image + One-Hot label dataset base class
    '''
    def __init__(self, transform):
        super().__init__(transform)
    
    def _make_label(self, labels):
        self.encoder = OneHotEncoder()
        labels = np.array(labels).reshape(-1, 1)
        self.labels = self.encoder.fit_transform(labels).toarray()
        self.num_classes = len(self.encoder.categories_[0])

    def inverse_transform(self, label):
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()
        if not isinstance(label, np.ndarray):
            label = np.array(label)
        label = label.reshape(1, -1)
        return self.encoder.inverse_transform(label)
        