
import torch
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image

import storch
from storch.imageops import make_image_grid
from storch.torchops import optimizer_step, set_seeds
from storch.status import Status
from storch.checkpoint import Checkpoint

from animeface import utils
from animeface.implementations.VAE.utils import kl_divergence_loss_fn


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
    recon_loss_fn = storch.construct_class_by_name(**cfg.train.loss)

    # status
    status = Status(len(dataset) * cfg.train.epochs, folder.root / cfg.run.log_file, False,
        cfg.run.log_interval, cfg.run.name
    )
    status.log_stuff(config, model, optimizer, dataset)

    # others
    scaler = GradScaler() if amp else None
    const_z = torch.randn(16, cfg.model.model.embed_dim, device=device)

    # checkpoint
    checkpoint = Checkpoint(folder.root, keep_last=1)
    checkpoint.register(model=model, optimizer=optimizer, status=status)
    if scaler is not None: checkpoint.register(scaler=scaler)
    checkpoint.load_latest()

    while not status.is_end():
        for image in dataset:
            image = image.to(device)

            with autocast(amp):
                recon, (mu, logvar) = model(image, return_muvar=True)
                recon_loss = recon_loss_fn(recon, image)
                kl_div_loss = kl_divergence_loss_fn(mu, logvar)
                loss = recon_loss + kl_div_loss

            optimizer_step(loss, optimizer, scaler,
                zero_grad=True, set_to_none=True, update_scaler=True, clip_grad_norm=True, max_norm=5.0
            )

            kbatches = status.get_kbatches()
            if status.batches_done % cfg.train.snapshot_every == 0:
                recon_images = make_image_grid(image, recon, num_images=8)
                with torch.no_grad(), autocast(amp):
                    model.eval()
                    sampled_images = model.decode(const_z)
                    model.train()
                images = torch.cat([recon_images, sampled_images], dim=0)
                # | input | recon | x 4 cols (8 cols in total) x 2 rows
                # | sampled image | x 8 cols                   x 2 rows
                save_image(images, folder.image / f'{kbatches}.png', normalize=True, nrow=8)
                torch.save(model.state_dict(), folder.model / f'{kbatches}.torch')
            if status.batches_done % cfg.train.checkpoint_frequency == 0:
                checkpoint.save()

            status.update(**{
                'Loss/total': loss, 'Loss/MSE': recon_loss, 'Loss/KL': kl_div_loss
            })
