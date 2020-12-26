
from .anime_face import AnimeFaceDataset, LabeledAnimeFaceDataset, OneHotLabeledAnimeFaceDataset, YearAnimeFaceDataset
from .danbooru import DanbooruDataset, GeneratePairImageDanbooruDataset
from .danbooru_portrait import DanbooruPortraitDataset
from .utils import get_device, Status

from .fp16 import network_to_half

from torch.utils.data import DataLoader

def to_loader(
    dataset,
    batch_size,
    shuffle=True,
    num_workers=8,
    use_gpu=True
):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=use_gpu)
    return loader