
from __future__ import annotations
from typing_extensions import runtime_checkable

import warnings
import gc
import functools
import inspect
from loguru import logger

import torch

Tensor = torch.Tensor
_checkio_num_indent = 0

class checkio:
    """Print in/output tensor stats to debug when coding models.

    NOTE: use checkio() only when debug.
    It can be used as a decorator or context manager.

    Examples::python

        @checkio()
        def method(tensor):
            # do something here
            return tensor

        # or

        tensor = torch.rand(2, 2)
        with checkio():
            tensor = tensor.sum()

    args:
        name: str
            name the block.
        input: bool (default: True)
            print input tensor stats
        output: bool (default: True)
            print output tensor stats
        funcs: list[str] (default: ['size'])
            list of func names in str,
            that can be called on tensor objects
    """
    def __init__(self,
        name: str=None,
        input: bool=True,
        output: bool=True,
        funcs: list[str]=['size']
    ) -> None:
        self._name = name
        self._input = input
        self._output = output
        self._check_invalid_funcs(funcs)
        self._funcs = funcs

    @staticmethod
    def _check_invalid_funcs(funcs):
        _dummy_tensor = torch.rand(2, 2)
        for func in funcs:
            assert hasattr(_dummy_tensor, func), f'torch.Tensor object does not have .{func}()'

    def __enter__(self):
        """context manager enter"""
        global _checkio_num_indent
        if self._input:
            keys, args = [], []
            for k, v in inspect.currentframe().f_back.f_locals.items():
                if isinstance(v, Tensor):
                    keys.append(k)
                    args.append(v)
            string = self._make_string_list(args)
            string = ', '.join([f'{k} : ' + s for k, s in zip(keys, string)])
            logger.debug('    ' * _checkio_num_indent + ' >> INPUT  : '+string)
        _checkio_num_indent += 1

    def __exit__(self, exc_type, exc_value, traceback):
        """context manager exit"""
        global _checkio_num_indent
        _checkio_num_indent -= 1
        if self._output:
            keys, args = [], []
            for k, v in inspect.currentframe().f_back.f_locals.items():
                if isinstance(v, Tensor):
                    keys.append(k)
                    args.append(v)
            string = self._make_string_list(args)
            string = ', '.join([f'{k} : ' + s for k, s in zip(keys, string)])
            logger.debug('    ' * _checkio_num_indent + ' << OUTPUT : '+string)

    def __call__(self, func):
        """decorator"""
        @functools.wraps(func)
        def inner(*args, **kwargs):
            global _checkio_num_indent

            signature = str(inspect.signature(func))

            if self._name:
                logger.debug('    ' * _checkio_num_indent + self._name + signature)
            else:
                logger.debug('    ' * _checkio_num_indent + func.__qualname__ + signature)

            if self._input:
                string = self._inspect_input(*args, **kwargs)
                logger.debug('    ' * _checkio_num_indent + ' >> INPUT  : '+string)
            _checkio_num_indent += 1

            output = func(*args, **kwargs)

            _checkio_num_indent -= 1
            if self._output:
                string = self._inspect_output(output)
                logger.debug('    ' * _checkio_num_indent + ' << OUTPUT : '+string)

            return output
        return inner

    def _make_string_list(self, iterable) -> list:
        string_list = []
        for el in iterable:
            if isinstance(el, Tensor):
                el = el.clone().detach()
                for func in self._funcs:
                    if hasattr(el, func):
                        string_list.append(f'{func}->' + str(getattr(el, func)()))
        return string_list

    def _inspect_input(self, *args, **kwargs):
        string = self._make_string_list(args)
        string.extend(self._make_string_list(kwargs.values()))
        return ', '.join(string)

    def _inspect_output(self, output):
        string = []
        if isinstance(output, tuple):
            string = self._make_string_list(output)
        else:
            for func in self._funcs:
                if hasattr(output, func):
                    string.append(f'{func}->' + str(getattr(output, func)()))
        return ', '.join(string)

    @classmethod
    def set_all_funcs(cls, funcs):
        '''collect all checkio obj and change funcs attr
        NOTE: This implementation is not good...
        '''
        cls._check_invalid_funcs(funcs)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            for o in gc.get_objects():
                if o.__class__ == checkio:
                    o._funcs = funcs
