
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.profiler import profile, record_function, ProfilerActivity

from nnutils.initialize import init
from nnutils.training import (
    sample_nnoise,
    sample_unoise,
    update_ema
)

def get_device(gpu: bool=torch.cuda.is_available()):
    if gpu and torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')

def freeze(model: torch.nn.Module) -> None:
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

def unfreeze(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True
    model.train()

def profile_once(
    fn: Callable,
    input: tuple[Any],
    sort_by: str='self_cpu_time_total',
    backward: bool=True
):

    if not isinstance(input, tuple):
        input = (input, )

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        with record_function('forward'):
            output = fn(*input)
        if backward:
            output = output.mean()
            with record_function('backward'):
                output.backward()
    print(prof.key_averages().table(sort_by=sort_by, row_limit=100))
