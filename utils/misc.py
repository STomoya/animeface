
from __future__ import annotations
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
