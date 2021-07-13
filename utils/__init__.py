
from __future__ import annotations

from typing import Any

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
