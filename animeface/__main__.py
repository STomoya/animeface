
import os
import sys

import storch
from omegaconf import OmegaConf
from storch.hydra_utils import get_hydra_config, save_hydra_config
from storch.path import Folder, Path


def setup():
    command_args = sys.argv[1:]
    if len(command_args) == 1 and command_args[0].endswith('config.yaml') and os.path.exists(command_args[0]):
        config_file = command_args[0]
        config = OmegaConf.load(config_file)
        folder = Folder(Path(config_file).dirname())
        folder.add_children(model='models', image='images')
    else:
        config = get_hydra_config('config', 'config.yaml')
        folder = Folder(Path(config.config.run.folder) / config.config.run.name / storch.get_now_string())
        folder.add_children(model='models', image='images')
        folder.mkdir()
        save_hydra_config(config, folder.root / 'config.yaml')

    return config, folder


def launch():
    config, folder = setup()
    train_loop = storch.get_obj_by_name(config.config.train_loop).run
    train_loop(config, folder)


if __name__=='__main__':
    launch()
