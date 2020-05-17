
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

def to_loader(
    dataset,
    batch_size
):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

class AnimeFaceDataset(Dataset):
    def __init__(self, image_size):
        self.image_paths = self._load()
        self.length = len(self.image_paths)
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image

    def _load(self):
        base = Path('/usr/src/data/images')
        image_paths = base.glob('*')
        image_paths = [str(path) for path in image_paths]
        return image_paths

if __name__ == "__main__":
    dataset = AnimeFaceDataset(100)
    dataset = to_loader(dataset, 32)
    for data in dataset:
        print(data.size())
        break
