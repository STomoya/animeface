
from storch.dataset import ImageFolder, make_transform_from_config

def build_image_dataset(config, worker_init_fn=None, generator=None):
    transform = make_transform_from_config(config.transforms)
    dataset = ImageFolder(config.data_root, transform, config.num_images)
    dataset = dataset.setup_loader(worker_init_fn=worker_init_fn, generator=generator, **config.loader).toloader()
    return dataset
