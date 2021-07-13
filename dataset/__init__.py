
from dataset.animeface import (
    AnimeFace,
    AnimeFaceCelebA,
    AnimeFaceSR,
    AnimeFaceXDoG,
    AnimeFaceLabel,
    AnimeFaceOneHot
)

from dataset.portrait import (
    DanbooruPortrait,
    DanbooruPortraitCelebA,
    DanbooruPortraitSR,
    DanbooruPortraitXDoG
)

from dataset.danbooru import (
    Danbooru,
    DanbooruSR,
    DanbooruAutoPair
)

import os
import torch
from torch.utils.data import DataLoader, Dataset
from collections.abc import Iterable

def cycle(iterable: Iterable):
    while True:
        for i in iterable:
            yield i

def to_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool=True,
    num_workers: int=os.cpu_count(),
    pin_memory: bool=torch.cuda.is_available()
) -> DataLoader:
    loader = DataLoader(
        dataset, batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory
    )
    return loader
