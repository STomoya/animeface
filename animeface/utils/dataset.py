
from functools import partial
import os
from storch.dataset import ImageFolder, make_transform_from_config

def build_image_dataset(config, worker_init_fn=None, generator=None):
    transform = make_transform_from_config(config.transforms)
    animeface_filtering = config.get('filter_by_year', None)
    filter_fn = partial(animeface_dataset_filter, min_year=animeface_filtering) if animeface_filtering is not None else None
    dataset = ImageFolder(config.data_root, transform, config.num_images, filter_fn=filter_fn)
    dataset = dataset.setup_loader(worker_init_fn=worker_init_fn, generator=generator, **config.loader).toloader()
    return dataset


def animeface_dataset_filter(path, min_year):
    # strip posted year from path
    year = int(os.path.splitext(os.path.basename(path))[0].split('_')[-1])
    return year >= min_year
