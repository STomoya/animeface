
import torch
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image

import storch
from storch.imageops import make_image_grid
from storch.torchops import optimizer_step, set_seeds
from storch.status import Status
from storch.checkpoint import Checkpoint

from animeface import utils


def run(config, folder):
    cfg = config.config

    if config.reproduce.enabled:
        worker_init_fn, generator = set_seeds(**config.reproduce.params)
    else: worker_init_fn, generator = None, None

    # env
    device = torch.device(config.env.device)
    amp = config.env.amp

    # dataset
    dataset = utils.dataset.build_image_dataset(cfg.data, worker_init_fn, generator)

    # model
    model = storch.construct_class_by_name(**cfg.model.model)
    model.to(device)

    # optimizer
    optimizer = storch.construct_class_by_name(model.parameters(), **cfg.train.optimizer)

    # loss
    criterion = storch.construct_class_by_name(**cfg.train.loss)

    # status
    status = Status(len(dataset) * cfg.train.epochs, folder.root / cfg.run.log_file, False,
        cfg.run.log_interval, cfg.run.name
    )
    status.log_stuff(config, model, optimizer, dataset)

    # others
    scaler = GradScaler() if amp else None

    # checkpoint
    checkpoint = Checkpoint(folder.root, keep_last=1)
    checkpoint.register(model=model, optimizer=optimizer, status=status)
    if scaler is not None: checkpoint.register(scaler=scaler)
    checkpoint.load_latest()

    while not status.is_end():
        for image in dataset:
            image = image.to(device)

            with autocast(amp):
                recon = model(image)
                loss = criterion(recon, image)

            optimizer_step(loss, optimizer, scaler,
                zero_grad=True, set_to_none=True, update_scaler=True
            )

            kbatches = status.get_kbatches()
            if status.batches_done % cfg.train.snapshot_every == 0:
                images = make_image_grid(image, recon, num_images=8)
                save_image(images, folder.image / f'{kbatches}.png', normalize=True)
                torch.save(model.state_dict(), folder.model / f'{kbatches}.torch')
            if status.batches_done % cfg.train.checkpoint_frequency == 0:
                checkpoint.save()

            status.update(**{
                'Loss/MSE': loss
            })
