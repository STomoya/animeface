
from __future__ import annotations

from typing import Any
from PIL import Image

def debug_mode():
    from utils import debug

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
