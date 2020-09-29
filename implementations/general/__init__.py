
from .anime_face import AnimeFaceDataset, LabeledAnimeFaceDataset, OneHotLabeledAnimeFaceDataset
from .danbooru import DanbooruDataset, GeneratePairImageDanbooruDataset
from .danbooru_portrait import DanbooruPortraitDataset

from .fp16 import network_to_half

from torch.utils.data import DataLoader

def to_loader(
    dataset,
    batch_size,
    shuffle=True,
    num_workers=8
):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader