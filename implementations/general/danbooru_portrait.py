
from pathlib import Path

from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

class DanbooruPortraitDataset(Dataset):
    '''
    Danbooru Portrait Dataset

    images
    '''
    def __init__(self, image_size, transform=None):
        self.image_paths = self._load()
        self.length = len(self.image_paths)
        if transform:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.Resize(int(image_size*1.2)),
                T.CenterCrop(image_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
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

if __name__ == "__main__":
    dataset = DanbooruPortraitDataset(128)
    from torch.utils.data import DataLoader
    print(len(dataset))
    dataset = DataLoader(dataset, batch_size=32, shuffle=True)
    print(len(dataset))
    # exit()

    from torchvision.utils import save_image
    for image in dataset:
        save_image(image, 'sample.png', nrow=3, normalize=True)
        break
