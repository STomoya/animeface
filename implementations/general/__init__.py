
from .anime_face import AnimeFaceDataset, LabeledAnimeFaceDataset, OneHotLabeledAnimeFaceDataset, YearAnimeFaceDataset, XDoGAnimeFaceDataset, AnimeFaceSRDataset
from .danbooru import DanbooruDataset, GeneratePairImageDanbooruDataset, DanbooruSRDataset
from .danbooru_portrait import DanbooruPortraitDataset, XDoGDanbooruPortraitDataset, DanbooruPortraitSRDataset
from .utils import get_device, Status
from .arg_utils import save_args

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