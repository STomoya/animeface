
import random
from pathlib import Path

from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

class DanbooruPortraitDataset(Dataset):
    '''
    Danbooru Portrait Dataset

    images
    '''
    def __init__(self, image_size, transform=None, num_images=None):
        self.image_paths = self._load()
        if num_images is not None:
            assert isinstance(num_images, int)
            random.shuffle(self.image_paths)
            self.image_paths = self.image_paths[:num_images]
        self.length = len(self.image_paths)
        if transform:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.Resize(int(image_size*1.2)),
                T.CenterCrop(image_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(0.5, 0.5)
            ])
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image
    
    def _load(self):
        base = Path('/usr/src/data/danbooru/portraits/portraits')
        image_paths = base.glob('*.jpg')
        image_paths = [str(path) for path in image_paths]
        return image_paths

class XDoGDanbooruPortraitDataset(DanbooruPortraitDataset):
    def __init__(self, image_size, transform=None, num_images=None):
        super().__init__(image_size, transform, num_images)
        self.xdog_paths = [path.replace('portraits/portraits', 'portraits/xdog') for path in self.image_paths]
        if transform is None:
            self.transform = T.Compose([
                T.Resize(int(image_size*1.2)),
                T.CenterCrop(image_size),
                # no random flip for pair images
                T.ToTensor(),
                T.Normalize(0.5, 0.5)
            ])
    def __getitem__(self, index):
        rgb_image_path = self.image_paths[index]
        xdog_image_path = self.xdog_paths[index]

        rgb_image = Image.open(rgb_image_path).convert('RGB')
        xdog_image = Image.open(xdog_image_path).convert('L')

        rgb_image = self.transform(rgb_image)
        xdog_image = self.transform(xdog_image)

        return rgb_image, xdog_image

    def shuffle_xdog(self):
        random.shuffle(self.xdog_paths)

if __name__ == "__main__":
    dataset = XDoGDanbooruPortraitDataset(128)
    from torch.utils.data import DataLoader
    print(len(dataset))
    dataset = DataLoader(dataset, batch_size=32, shuffle=True)
    print(len(dataset))
    # exit()

    from torchvision.utils import save_image
    for image, line in dataset:
        save_image(image, 'sample.png', nrow=3, normalize=True)
        save_image(line, 'line.png', nrow=3, normalize=True)
        break
