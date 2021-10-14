
from __future__ import annotations

import os
import json
import datetime
from argparse import ArgumentParser
from utils.misc import EasyDict

def get_default_parser() -> ArgumentParser:
    parser = ArgumentParser()

    # implementation name
    parser.add_argument('name')
    default_args = EasyDict()
    default_args.image_size     = [128, 'Size of image.']
    default_args.batch_size     = [32, 'Batch size']
    default_args.dataset        = ['animeface', 'Dataset name']
    default_args.min_year       = [2005, 'Minimum of generated year. Ignored when dataset==danbooru']
    default_args.num_images     = [60000, 'Number of images to include in training set. Ignored when dataset==animeface']
    default_args.save           = [1000, 'Interval for saving the model. Also used for sampling images and saving them for qualitative eval']
    default_args.max_iters      = [-1, 'Maximum iterations to train the model. If < 0, it will be calculated using --default-epochs']
    default_args.default_epochs = [100, 'Used to calculate the max iteration if --max-iters < 0']
    default_args.disable_gpu    = [False, 'Disable GPU']
    default_args.disable_amp    = [False, 'Disable AMP']
    default_args.debug          = [False, 'Debug mode']

    parser = add_args(parser, default_args)
    return parser

def add_args(
    parser: ArgumentParser,
    arg_defaults: dict,
    prefix: str='--'
) -> ArgumentParser:
    for k, v in arg_defaults.items():
        option = prefix + k.replace('_', '-')

        if len(v) == 1:
            default, help = v[0], ''
        elif len(v) == 2:
            default, help = v
        else:
            raise Exception('args_defaults value must be a list of [default] or [default, help]')

        value_type = type(default)
        if value_type in [float, int, str]:
            parser.add_argument(option, default=default, type=value_type, help=help)
        elif value_type == bool:
            if default:
                raise Exception('Only supports store_true action')
            parser.add_argument(option, default=default, action='store_true', help=help)
        elif value_type in [list, tuple]:
            ele_type = type(default[0])
            parser.add_argument(option, default=default, type=ele_type, nargs='*', help=help)
        elif isinstance(value_type, type):
            parser.add_argument(option, default=None, type=default, help=help)

    return parser

def save_args(
    args: dict,
    identify: bool=True,
    id: str=None
) -> None:
    args_dict = vars(args)
    if identify:
        if id is None:
            id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        args_file = f'args-{id}.json'
    else: args_file = 'args.json'
    filename = os.path.join('/usr/src/implementations', args.name, 'result', args_file)
    with open(filename, 'w', encoding='utf-8') as fout:
        json.dump(args_dict, fout, indent=2)
