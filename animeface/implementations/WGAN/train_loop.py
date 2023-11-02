"""Simplified training loop of StyleGAN2.
This training loop can be used as default for training most GAN models.
"""


import copy
import torch
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image

import storch
from storch.torchops import freeze, optimizer_step, update_ema, set_seeds
from storch.status import Status
from storch.checkpoint import Checkpoint
from storch.loss import calc_grad

from animeface import utils


def clip_weights(model):
    for params in model.parameters():
        params.data.clamp_(-0.02, 0.02)


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
    G = storch.construct_class_by_name(**cfg.model.generator)
    test_model = G
    D = storch.construct_class_by_name(**cfg.model.discriminator)
    if cfg.model.ema:
        G_ema = copy.deepcopy(G)
        freeze(G_ema)
        G_ema.to(device)
        test_model = G_ema
    else:
        G_ema = None

    G.to(device)
    D.to(device)

    # optimizer
    optim_G = storch.construct_class_by_name(G.parameters(), **cfg.train.optimizer.generator)
    optim_D = storch.construct_class_by_name(D.parameters(), **cfg.train.optimizer.discriminator)

    # loss
    adv_fn = storch.construct_class_by_name(**cfg.train.loss.adversarial)
    gp_type = cfg.train.loss.gradient_penalty.type
    gp_every = cfg.train.loss.gradient_penalty.every

    # augmentation
    if cfg.train.augmentation is None:
        augment = lambda x:x
    else:
        augment = storch.construct_class_by_name(**cfg.train.augmentation)

    # status
    status = Status(len(dataset) * cfg.train.epochs, folder.root / cfg.run.log_file, False,
        cfg.run.log_interval, cfg.run.name)
    status.log_stuff(config, G, optim_G, D, optim_D, dataset)

    # others
    scaler = GradScaler() if amp else None
    const_z = torch.randn(16, cfg.model.generator.latent_dim, device=device)

    # checkpointer
    checkpoint = Checkpoint(folder.root, keep_last=1)
    checkpoint.register(G=G, D=D, optim_G=optim_G, optim_D=optim_D, status=status)
    if scaler is not None: checkpoint.register(scaler=scaler)
    if G_ema is not None: checkpoint.register(G_ema=G_ema)

    # resume training if any checkpoint is present.
    checkpoint.load_latest()

    while not status.is_end():
        for real in dataset:
            real = real.to(device)
            if gp_type is not None and status.batches_done % gp_every == 0:
                real = real.requires_grad_(True)
            z = torch.randn(real.size(0), cfg.model.generator.latent_dim, device=device)

            with autocast(amp):
                # G forward
                fake = G(z)

                # augment images
                real_aug = augment(real)
                fake_aug = augment(fake)

                # D forward
                real_logits = D(real_aug)
                fake_logits = D(fake_aug.detach())

                # loss
                adv_loss = adv_fn.d_loss(real_logits, fake_logits) * cfg.train.lambdas.adv
                penalty = 0.0
                if real.requires_grad == True:
                    gradients = calc_grad(real_logits, real, scaler)
                    if gp_type == 'r1':
                        gradients = gradients.reshape(gradients.size(0), -1)
                        penalty = gradients.norm(2, dim=1).pow(2).mean() / 2.
                    else:
                        raise
                gp_loss = penalty * cfg.train.lambdas.gp
                D_loss = adv_loss + gp_loss

            optimizer_step(D_loss, optim_D, scaler, zero_grad=True, set_to_none=True)
            clip_weights(D)

            z = torch.randn(real.size(0), cfg.model.generator.latent_dim, device=device)
            with autocast(amp):
                # G forward
                fake = G(z)

                # augment images
                fake_aug = augment(fake)

                # D forward
                fake_logits = D(fake_aug)

                # loss
                G_loss = adv_fn.g_loss(fake_logits) * cfg.train.lambdas.adv

            optimizer_step(G_loss, optim_G, scaler, zero_grad=True, set_to_none=True, update_scaler=True)
            if G_ema is not None:
                update_ema(G, G_ema, cfg.model.ema_decay, copy_buffers=True)

            if status.batches_done % cfg.train.snapshot_every == 0:
                with torch.no_grad(), autocast(amp):
                    G.eval()
                    images = test_model(const_z)
                    state_dict = test_model.state_dict()
                    G.train()
                kbatches = status.get_kbatches()
                save_image(images, folder.image / f'{kbatches}.png', normalize=True)
                torch.save(state_dict, folder.model / f'{kbatches}.torch')
            if status.batches_done % cfg.train.checkpoint_frequency == 0:
                checkpoint.save()

            status.update(**{
                'Loss/D/adv': adv_loss,
                'Loss/D/penalty': gp_loss,
                'Loss/D': D_loss,
                'Loss/G/adv': G_loss,
                'Loss/G': G_loss
            })
