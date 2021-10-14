
from __future__ import annotations

from typing import Union
import torch

def sample_nnoise(
    size: tuple[int],
    device: Union[torch.device, str],
    mean: float=0.,
    std: float=1.
) -> torch.Tensor:
    return torch.empty(size, device=device).normal_(mean, std)

def sample_unoise(
    size: tuple[int],
    device: Union[torch.device, str],
    start: float=0.,
    end: float=1.
) -> torch.Tensor:
    return torch.empty(size, device=device).uniform_(start, end)

@torch.no_grad()
def update_ema(
    model: torch.nn.Module,
    model_ema: torch.nn.Module,
    decay: float=0.999,
    copy_buffers: bool=False
) -> None:
    model.eval()
    param_ema = dict(model_ema.named_parameters())
    param     = dict(model.named_parameters())
    for key in param_ema.keys():
        param_ema[key].data.mul_(decay).add_(param[key].data, alpha=(1 - decay))
    if copy_buffers:
        buffer_ema = dict(model_ema.named_buffers())
        buffer     = dict(model.named_buffers())
        for key in buffer_ema.keys():
            buffer_ema[key].data.copy_(buffer[key].data)
    model.train()
