
import csv
from pathlib import Path
import glob

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class AnimeFaceDataset(Dataset):
    '''
    Anime Face Dataset
    
    images
    '''
    def __init__(self, image_size):
        self.image_paths = self._load()
        self.length = len(self.image_paths)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image

    def _load(self):
        base = Path('/usr/src/data/animefacedataset/images')
        image_paths = base.glob('*')
        image_paths = [str(path) for path in image_paths]
        return image_paths

class YearAnimeFaceDataset(Dataset):
    def __init__(self, image_size, min_year=2005):
        self.image_paths = self._load(min_year)
        self.length = len(self.image_paths)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image

    def _load(self, min_year):
        image_paths = glob.glob('/usr/src/data/animefacedataset/images/*')
        year_from_path = lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1])
        image_paths = [path for path in image_paths if year_from_path(path) >= min_year]
        return image_paths

class XDoGAnimeFaceDataset(YearAnimeFaceDataset):
    def __init__(self, image_size, min_year=2005):
        super.__init__(image_size, min_year)
        self.xdog_paths = [path.replace('images', 'xdog'), for path in self.image_paths]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            # no random flip for pair images
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
    def __getitem__(self, index):
        rgb_image_path = self.image_paths[index]
        xdog_image_path = self.xdog_paths[index]

        rgb_image = Image.open(rgb_image_path).convert('RGB')
        xdog_image = Image.open(xdog_image_path).convert('L')

        rgb_image = self.transform(rgb_image)
        xdog_image = self.transform(xdog_image)

        return rgb_image, xdog_image

class LabeledAnimeFaceDataset(Dataset):
    '''
    Labeled Anime Face Dataset

    images, illustration2vec tags, and year
    '''
    def __init__(self, image_size):
        self.image_paths, i2v_labels, year_labels = self._load()
        self.length = len(self.image_paths)

        self.i2v_encoder = LabelEncoder()
        self.year_encoder = LabelEncoder()
        i2v_labels  = np.array(i2v_labels).reshape(-1, 1)
        year_labels = np.array(year_labels).reshape(-1, 1)
        self.i2v_labels = self.i2v_encoder.fit_transform(i2v_labels)
        self.year_labels = self.year_encoder.fit_transform(year_labels)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        i2v_label  = self.i2v_labels[index]
        year_label = self.year_labels[index]

        return image, {'i2v' : i2v_label, 'year' : year_label}

    def _load(self):
        dataset_file_path = '/usr/src/data/animefacedataset/labels.csv'
        with open(dataset_file_path, 'r', encoding='utf-8') as fin:
            csv_reader = csv.reader(fin)
            data_list = [line for line in csv_reader]
        
        # file format
        # "path/to/file","i2vtag","year"

        image_paths = [line[0] for line in data_list]
        i2v_labels  = [line[1] for line in data_list]
        year_labels = [line[2] for line in data_list]

        return image_paths, i2v_labels, year_labels

class OneHotLabeledAnimeFaceDataset(Dataset):
    '''
    One-Hot Labeled Anime Face Dataset

    images, one-hot encoded illustration2vec tags, one-hot encoded year

    One-hot encoding is done by sklearn.preprocessing.OneHotEncoder
    '''
    def __init__(self, image_size):
        self.image_paths, i2v_labels, year_labels = self._load()
        self.length = len(self.image_paths)

        # one-hot encode label info
        # encoders are class attr for inverse transform
        self.i2v_onehot_encoder  = OneHotEncoder()
        self.year_onehot_encoder = OneHotEncoder()
        i2v_labels  = np.array(i2v_labels).reshape(-1, 1)
        year_labels = np.array(year_labels).reshape(-1, 1)
        self.i2v_labels  = self.i2v_onehot_encoder.fit_transform(i2v_labels).toarray()
        self.year_labels = self.year_onehot_encoder.fit_transform(year_labels).toarray()

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        i2v_label  = self.i2v_labels[index]
        year_label = self.year_labels[index]

        return image, {'i2v' : i2v_label, 'year' : year_label}

    def i2v_inverse_transform(self, onehot):
        # if torch.Tensor -> chaneg to numpy array
        if hasattr(onehot, 'size'):
            onehot = onehot.cpu().numpy()
        return self.i2v_onehot_encoder.inverse_transform(onehot)
    
    def year_inverse_transform(self, onehot):
        # if torch.Tensor -> chaneg to numpy array
        if isinstance(onehot, torch.Tensor):
            onehot = onehot.cpu().numpy()
        return self.year_onehot_encoder.inverse_transform(onehot)

    def _load(self):
        dataset_file_path = '/usr/src/data/animefacedataset/labels.csv'
        with open(dataset_file_path, 'r', encoding='utf-8') as fin:
            csv_reader = csv.reader(fin)
            data_list = [line for line in csv_reader]
        
        # file format
        # "path/to/file","i2vtag","year"

        image_paths = [line[0] for line in data_list]
        i2v_labels  = [line[1] for line in data_list]
        year_labels = [line[2] for line in data_list]

        return image_paths, i2v_labels, year_labels


if __name__ == "__main__":
    def to_loader(
        dataset,
        batch_size,
        shuffle=True
    ):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader

    dataset = YearAnimeFaceDataset(100)

    dataset = AnimeFaceDataset(100)
    dataset = to_loader(dataset, 32)
    for data in dataset:
        print(data.size())
        break

    dataset = LabeledAnimeFaceDataset(100)
    dataset = to_loader(dataset, 32)
    for data, label in dataset:
        print(data.size())
        print(label['i2v'])
        print(label['year'])
        break

    dataset = OneHotLabeledAnimeFaceDataset(100)
    dataset = to_loader(dataset, 32)
    for data, label in dataset:
        print(data.size())
        print(label['i2v'].size())
        print(dataset.dataset.i2v_inverse_transform(label['i2v']))
        print(label['year'].size())
        print(dataset.dataset.year_inverse_transform(label['year']))
        break