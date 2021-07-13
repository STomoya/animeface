
from collections.abc import Callable
from typing import Any

import torch
from torch.profiler import profile, record_function, ProfilerActivity

def get_device(gpu: bool=torch.cuda.is_available()):
    if gpu and torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')

def profile_once(
    fn: Callable,
    input: Any,
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
