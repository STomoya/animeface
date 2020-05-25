
import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class DanbooruDataset(Dataset):
    def __init__(self, transform):
        self.image_paths = self._load()
        self.length = len(self.image_paths)

        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image

    def _load(self):
        base_path = Path('/usr/src/data/danbooru/danbooru-images')
        image_paths = base_path.glob('**/*.jpg')
        image_paths = [str(path) for path in image_paths]

        return image_paths

class GeneratePairImageDanbooruDataset(Dataset):
    def __init__(self, pair_transform, image_size):
        self.image_paths = self._load()
        self.length = len(self.image_paths)

        self.co_transform_head = transforms.RandomResizedCrop(image_size)
        self.pair_transform = pair_transform
        self.co_transform_tail = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        temp_image = Image.open(image_path).convert('RGB')

        image, pair_image = self._transform(temp_image)

        return image, pair_image

    def _transform(self, pil_image):

        pil_image  = self.co_transform_head(pil_image)

        pair_image = self.pair_transform(pil_image)

        pair_image = self.co_transform_tail(pair_image)
        image      = self.co_transform_tail(pil_image)

        return image, pair_image


    def _load(self):
        base_path = Path('/usr/src/data/danbooru/danbooru-images')
        image_paths = base_path.glob('**/*.jpg')
        image_paths = [str(path) for path in image_paths]

        return image_paths

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = DanbooruDataset(transform)
    for data in dataset:
        print(data.size())
        break

    pair_transform = transforms.Resize(128)
    
    dataset = GeneratePairImageDanbooruDataset(pair_transform, 256)
    for data, data2 in dataset:
        print(data.size())
        print(data2.size())
        break