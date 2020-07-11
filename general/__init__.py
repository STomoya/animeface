
from .anime_face import AnimeFaceDataset, LabeledAnimeFaceDataset, OneHotLabeledAnimeFaceDataset
from .danbooru import DanbooruDataset, GeneratePairImageDanbooruDataset

from .DiffAugment_pytorch import DiffAugment

from torch.utils.data import DataLoader

def to_loader(
    dataset,
    batch_size,
    shuffle=True
):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader