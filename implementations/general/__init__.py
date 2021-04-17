
import os

import torch
from torch.utils.data import DataLoader

from .anime_face import AnimeFaceDataset
from .anime_face import AnimeFaceCelebADataset
from .anime_face import LabeledAnimeFaceDataset
from .anime_face import OneHotLabeledAnimeFaceDataset
from .anime_face import YearAnimeFaceDataset
from .anime_face import XDoGAnimeFaceDataset
from .anime_face import AnimeFaceSRDataset

from .danbooru import DanbooruDataset
from .danbooru import GeneratePairImageDanbooruDataset
from .danbooru import DanbooruSRDataset

from .danbooru_portrait import DanbooruPortraitDataset
from .danbooru_portrait import DanbooruPortraitCelebADataset
from .danbooru_portrait import XDoGDanbooruPortraitDataset
from .danbooru_portrait import DanbooruPortraitSRDataset

from .utils import get_device, Status
from .arg_utils import save_args

def to_loader(
    dataset,
    batch_size,
    shuffle=True,
    num_workers=os.cpu_count(),
    use_gpu=torch.cuda.is_available()
):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=use_gpu)
    return loader