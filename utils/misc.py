
from __future__ import annotations

from functools import wraps
from typing import Any

import torch
from PIL import Image

class EasyDict(dict):
    '''
    dict that can access keys like attributes.
    '''
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

class print_for_repr:
    '''
    Decorator which prints positional and keyword arguments for reproduction

    Usage:
        @print_for_repr()
        def method(a, b, c):
            pass
        method(10, 20, c=30)

        >> Called: method(10, 20, c=30)

        # or

        class Net(nn.Module):
            @print_for_repr()
            def __init__(self, image_size, channels):
                pass
        Net(128, channels=32)

        >> Called: Net(128, channels=32)

        # use different printing function

        import logging
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)
        print_for_repr.print_func = logger.debug

        method(10, 20, c=30)
        >> DEBUG:__main__:Called: method(10, 20, c=30)
        Net(128, channels=32)
        >> DEBUG:__main__:Called: Net(128, channels=32)
    '''
    print_func = print

    def __call__(self, func):

        @wraps(func)
        def wrapper(*args, **kwargs):

            if func.__name__ == '__init__':
                name = args[0].__class__.__name__
                print_args = args[1:]
            else:
                name = func.__name__
                print_args = args

            message = 'Called: {}({}{}{})'.format(
                name,
                ', '.join(map(str, print_args)),
                ', ' if kwargs else '',
                ', '.join(f'{key}={value}' for key, value in kwargs.items()))
            self.print_func(message)
            return func(*args, **kwargs)

        return wrapper

def gif_from_files(
    image_paths: list[str],
    filename: str='out.gif',
    optimize: bool=False,
    duration: int=500,
    loop: int=0
) -> None:
    images = [Image.open(str(path)) for path in image_paths]
    images[0].save(filename,
        save_all=True,
        append_images=images[1:],
        optimize=optimize,
        duration=duration,
        loop=loop
    )

def make_image_grid(*image_tensors, num_images=None):
    ''''''
    _split = lambda x: x.chunk(x.size(0), 0)
    image_tensor_lists = map(_split, image_tensors)
    images = []
    for index, image_set in enumerate(zip(*image_tensor_lists)):
        images.extend(list(image_set))
        if num_images is not None and index == num_images-1:
            break
    return torch.cat(images, dim=0)
