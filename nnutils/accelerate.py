'''
Modified implementation of HuggingFace/Accelerate
[original] https://github.com/huggingface/accelerate

Modified by: STomoya - https://github.com/STomoya

Differences:
    NOT implemented:
        - DeepSpeed
        - TPU
        - Multi-device

    Changed:
        - grad_scaler.update() done manually.
            - Originally done right after optimizer.step()
            - For using mulitiple optimizers.
'''

from __future__ import annotations
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from nnutils import get_device

class MiniAcceleratedOptimizer(optim.Optimizer):
    '''Optimizer wrapper'''
    def __init__(self,
        optimizer: optim.Optimizer,
        scaler: GradScaler=None
    ) -> None:
        self._optimizer = optimizer
        self._scaler = scaler

    def step(self, closure=None):
        if self._scaler is not None:
            self._scaler.step(self._optimizer, closure)
            # huggingface/accelerate updates scaler here,
            # but we won't for multiple optimizers
            # self._scaler.update()
        else:
            self._optimizer.step(closure)
    @property
    def param_groups(self):
        return self._optimizer.param_groups
    @param_groups.setter
    def param_groups(self, param_groups):
        self._optimizer.param_groups = param_groups

    @property
    def defaults(self):
        return self._optimizer.defaults
    @defaults.setter
    def defaults(self, defaults):
        self._optimizer.defaults = defaults

    def add_param_group(self, param_group):
        self._optimizer.add_param_group(param_group)

    def load_state_dict(self, state_dict):
        self._optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self._optimizer.state_dict()

    def zero_grad(self, set_to_none=None):
        if set_to_none is None:
            set_to_none = False
        self._optimizer.zero_grad(set_to_none=set_to_none)

def is_tensor(data):
    '''check if data is torch.Tensor object'''
    return isinstance(data, torch.Tensor)

def recursive_apply(func, data, test_type=is_tensor):
    '''recursively apply func to data which satisfies test_type(data)

    Arguments:
        func: Callable
            function to apply
        data: Any
            data
        test_type: Callable -> bool
            test type
    '''
    if isinstance(data, (tuple, list)):
        return type(data)(recursive_apply(func, element) for element in data)
    elif isinstance(data, dict):
        return {key: recursive_apply(func, value) for key, value in data.items()}
    elif test_type(data):
        return func(data)
    return data

class DataLoaderWrapper:
    '''Wrapper for dataloader
    send batch to device

    Arguments:
        dataloader: DataLoader
            DataLoader object
        device: torch.device
            device to send the data to
    '''
    def __init__(self,
        dataloader: DataLoader,
        device
    ) -> None:
        self._dataloader = dataloader
        self._dataloader_iter = iter(self._dataloader)
        self._device = device
        self._to_device = lambda tensor: tensor.to(self._device)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self._dataloader_iter)
            batch = recursive_apply(self._to_device, batch)
            return batch
        except StopIteration as si:
            self._dataloader_iter = iter(self._dataloader)
            raise

class MiniAccelerator:
    '''
    Accelerator class.

    Usage:
        model = Model(...)
        optimizer = optim.Adam(model.parameters())
        dataset = DataLoader(Dataset(...), 32)

        # make accelerator object
        accelerator = MiniAccelerator()
        # prepare model optimizers, and dataloader
        model, optimizer, dataset = accelerator.prepare(model, optimizer, dataset)

        for data in dataset: # data is automatically set to device
            with accelerator.autocast(): # for AMP
                output = model(data)
                loss   = criterion(output)
            # use accelerator.backward(loss) instead of loss.backward()
            accelerator.backward(loss)
            # optimizer step
            optimizer.step()

    Arguments:
        amp: bool (default: True)
            use native AMP.
        device_placement: bool (default: True)
            automatically place tensors to device.
        device: torch.device (default: None)
            if None, device is automatically set by nnutils.get_device
    '''
    def __init__(self,
        amp: bool=True,
        device_placement: bool=True,
        device=None
    ) -> None:
        self._amp = amp
        self._device_placement = device_placement
        self._device = device if device is not None else get_device()
        self._scaler = GradScaler() if amp else None

    @property
    def scaler(self):
        return self._scaler
    @property
    def device(self):
        return self._device
    @device.setter
    def device(self, device):
        self._device = device

    def update(self):
        '''update scaler if using amp'''
        if self._scaler is not None:
            self._scaler.update()

    def backward(self, loss: torch.Tensor):
        '''backward function

        Arguments:
            loss: torch.Tensor
                loss to back propagate.
        '''
        if self._scaler is not None:
            self._scaler.scale(loss).backward()
        else:
            loss.backward()

    def prepare(self, *args):
        '''prepare nn.Module, DataLoader, optim.Optimizer'''
        if len(args) == 0:
            return
        prepared = []
        for arg in args:
            if isinstance(arg, nn.Module):
                prepared.append(self._prepair_model(arg))
            elif isinstance(arg, optim.Optimizer):
                prepared.append(self._prepare_optimizer(arg))
            elif isinstance(arg, DataLoader):
                prepared.append(self._prepair_dataloader(arg))
            else:
                prepared.append(arg)
        return tuple(prepared) if len(prepared) > 1 else prepared[0]

    def _prepare_optimizer(self,
        optimizer: optim.Optimizer
    ) -> optim.Optimizer:
        '''prepare optimizer'''
        if self._scaler is not None:
            optimizer = MiniAcceleratedOptimizer(optimizer, self._scaler)
        return optimizer

    def _prepair_model(self,
        model: nn.Module
    ) -> nn.Module:
        '''prepare model'''
        if self._device_placement:
            model.to(self._device)
        return model

    def _prepair_dataloader(self,
        dataloader: DataLoader
    ) -> DataLoader | DataLoaderWrapper:
        '''prepare dataloader'''
        if self._device_placement:
            dataloader = DataLoaderWrapper(
                dataloader, self._device)
        return dataloader

    @contextmanager
    def autocast(self):
        if self._amp:
            pytorch_autocast = autocast()
            pytorch_autocast.__enter__()
            yield
            pytorch_autocast.__exit__()
        else:
            yield
