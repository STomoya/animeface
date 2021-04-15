
import random
import csv

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
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

class ImageImage(Dataset, _common_attr):
    def __init__(self, transform):
        self.images1, self.images2 = self._load()
        self.length = len(self.images1)
        self.transform = transform
    
    def __getitem__(self, index):
        image1 = self.images1[index]
        image2 = self.images2[index]

        image1 = pilImage.open(image1).convert('RGB')
        image2 = pilImage.open(image2).convert('RGB')
        image1 = self.transform(image1)
        image2 = self.transform(image2)

        return image1, image2

class LRHR(Dataset, _common_attr):
    '''Low Resolution, High Resolution dataset base class
    '''
    def __init__(self, image_size, scale=2, resize_ratio=1.):
        self.images = self._load()
        self.length = len(self.images)
        self.image_size = image_size
        self.resize_ratio = resize_ratio
        self.scale = scale
    
    def __getitem__(self, index):
        image = self.images[index]

        image = pilImage.open(image).convert('RGB')
        lr, sr = self.transform(image)

        return lr, sr

    def transform(self, img):
        return self._default_transform(img)

    def _default_transform(self, img):
        lr_size = self.image_size // self.scale
        sr = TF.resize(img, int(self.image_size*self.resize_ratio))
        sr = TF.center_crop(sr, self.image_size)
        lr = TF.resize(img, int(lr_size*self.resize_ratio))
        lr = TF.center_crop(lr, lr_size)

        if random.random() > 0.5:
            sr = TF.hflip(sr)
            lr = TF.hflip(lr)

        sr = TF.to_tensor(sr)
        sr = TF.normalize(sr, 0.5, 0.5)
        lr = TF.to_tensor(lr)
        lr = TF.normalize(lr, 0.5, 0.5)
        return lr, sr

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
        