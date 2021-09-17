
from __future__ import annotations

import os
import random
from collections.abc import Callable
from typing import Union

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image as pilImage
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def make_default_transform(
    image_size: int,
    resize_scale: float=1.,
    hflip: bool=True,
    normalize: bool=True
) -> Callable:
    '''default transforms
    '''
    t_list = [
        T.Resize(int(image_size*resize_scale)),
        T.CenterCrop(image_size)
    ]
    if hflip:
        t_list.append(T.RandomHorizontalFlip())

    t_list.append(T.ToTensor())
    if normalize:
        t_list.append(T.Normalize(0.5, 0.5))

    return T.Compose(t_list)

class WrappedDataset(Dataset):
    '''Wrapped Dataset class
    with some methods'''
    def _load(self):
        raise NotImplementedError()

    @classmethod
    def asloader(cls,
        batch_size: int,
        cls_args: tuple,
        cls_kwargs: dict={},
        shuffle: bool=True,
        num_workers: int=os.cpu_count(),
        pin_memory: bool=torch.cuda.is_available()
    ):
        dataset = cls(*cls_args, **cls_kwargs)
        return DataLoader(
            dataset, batch_size, shuffle,
            num_workers=num_workers, pin_memory=pin_memory)

class Image(WrappedDataset):
    '''Image dataset base class
    '''
    def __init__(self,
        transform: Callable
    ) -> None:
        self.images = self._load()
        self.transform = transform

    def __getitem__(self, index) -> torch.Tensor:
        image = self.images[index]

        image = pilImage.open(image).convert('RGB')
        image = self.transform(image)

        return image

    def __len__(self) -> int:
        return len(self.images)

class ImageImage(WrappedDataset):
    '''Image Image dataset base class
    '''
    def __init__(self,
        transform: Callable
    ) -> None:
        self.images1, self.images2 = self._load()
        self.transform = transform

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        image1 = self.images1[index]
        image2 = self.images2[index]

        image1 = pilImage.open(image1).convert('RGB')
        image2 = pilImage.open(image2).convert('RGB')
        image1 = self.transform(image1)
        image2 = self.transform(image2)

        return image1, image2

    def __len__(self) -> int:
        return len(self.images1)

class LRHR(WrappedDataset):
    '''Low Resolution, High Resolution dataset base class
    '''
    def __init__(self,
        image_size: int,
        scale: float=2.,
        resize_scale: float=1.
    ) -> None:
        self.images = self._load()
        self.image_size = image_size
        self.resize_scale = resize_scale
        self.scale = scale

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index]

        image = pilImage.open(image).convert('RGB')
        lr, sr = self.transform(image)

        return lr, sr

    def transform(self, img) -> tuple[torch.Tensor, torch.Tensor]:
        return self._default_transform(img)

    def _default_transform(self, img) -> tuple[torch.Tensor, torch.Tensor]:
        lr_size = self.image_size // self.scale
        sr = TF.resize(img, int(self.image_size*self.resize_scale))
        sr = TF.center_crop(sr, self.image_size)
        lr = TF.resize(img, int(lr_size*self.resize_scale))
        lr = TF.center_crop(lr, lr_size)

        if random.random() > 0.5:
            sr = TF.hflip(sr)
            lr = TF.hflip(lr)

        sr = TF.to_tensor(sr)
        sr = TF.normalize(sr, 0.5, 0.5)
        lr = TF.to_tensor(lr)
        lr = TF.normalize(lr, 0.5, 0.5)
        return lr, sr

    def __len__(self) -> int:
        return len(self.images)

class ImageXDoG(WrappedDataset):
    '''Image + XDoG dataset base class
    '''
    def __init__(self,
        transform: Callable
    ) -> None:
        self.images, self.xdogs = self._load()
        self.transform = transform

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index]
        xdog  = self.xdogs[index]

        image = pilImage.open(image).convert('RGB')
        xdog  = pilImage.open(xdog).convert('L')

        image = self.transform(image)
        xdog  = self.transform(xdog)

        return image, xdog

    def shuffle_xdog(self) -> None:
        random.shuffle(self.xdogs)

    def __len__(self) -> int:
        return len(self.images)

class ImageLabel(WrappedDataset):
    '''Image + Label dataset base class
    '''
    def __init__(self,
        transform: Callable
    ) -> None:
        self.images, labels = self._load()
        self.transform = transform

        self._make_label(labels)

    def _make_label(self, labels) -> None:
        self.encoder = LabelEncoder()
        labels = np.array(labels).reshape(-1)
        self.labels = self.encoder.fit_transform(labels)
        self.num_classes = len(self.encoder.classes_)

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        image = self.images[index]
        label = self.labels[index]

        image = pilImage.open(image).convert('RGB')
        image = self.transform(image)

        return image, label

    def inverse_transform(self,
        label: Union[torch.Tensor, np.ndarray, list]
    ) -> list[str]:
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()
        if not isinstance(label, np.ndarray):
            label = np.array(label)
        label = label.reshape(-1)
        return self.encoder.inverse_transform(label)

    def __len__(self) -> int:
        return len(self.images)

class ImageOnehot(ImageLabel):
    '''Image + One-Hot label dataset base class
    '''
    def __init__(self,
        transform: Callable
    ) -> None:
        super().__init__(transform)

    def _make_label(self, labels) -> None:
        self.encoder = OneHotEncoder()
        labels = np.array(labels).reshape(-1, 1)
        self.labels = self.encoder.fit_transform(labels).toarray()
        self.num_classes = len(self.encoder.categories_[0])

    def inverse_transform(self,
        label: Union[torch.Tensor, np.ndarray, list]
    ) -> list[str]:
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()
        if not isinstance(label, np.ndarray):
            label = np.array(label)
        label = label.reshape(1, -1)
        return self.encoder.inverse_transform(label)
