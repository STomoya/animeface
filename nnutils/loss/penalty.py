
from typing import Optional

import torch
import torch.nn as nn
from torch.autograd import grad, Variable
from torch.cuda.amp import autocast, GradScaler

from nnutils.loss._base import Loss

def calc_grad(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
    scaler: Optional[GradScaler]=None
) -> torch.Tensor:
    with autocast(False):
        if isinstance(scaler, GradScaler):
            outputs = scaler.scale(outputs)
        ones = torch.ones(outputs.size(), device=outputs.device)
        gradients = grad(
            outputs=outputs, inputs=inputs, grad_outputs=ones,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        if isinstance(scaler, GradScaler):
            gradients = gradients / scaler.get_scale()
    return gradients

class gradient_penalty(Loss):
    def __call__(self,
        real: torch.Tensor,
        fake: torch.Tensor,
        D: nn.Module,
        scaler: Optional[GradScaler]=None,
        center: float=1.
    ) -> torch.Tensor:

        assert center in [1., 0.]

        device = real.device

        alpha = torch.rand(1, device=device)
        x_hat = real * alpha + fake * (1 - alpha)
        x_hat = Variable(x_hat, requires_grad=True)

        d_x_hat = D(x_hat)

        gradients = calc_grad(d_x_hat, x_hat, scaler)

        gradients = gradients.reshape(gradients.size(0), -1)
        penalty = (gradients.norm(2, dim=1) - center).pow(2).mean()

        return penalty

class dragan_penalty(Loss):
    def __call__(self,
        real: torch.Tensor,
        D: nn.Module,
        scaler: Optional[GradScaler]=None,
        center: float=1.
    ) -> torch.Tensor:

        device = real.device

        alpha = torch.rand((real.size(0), 1, 1, 1), device=device).expand(real.size())
        beta = torch.rand(real.size(), device=device)
        x_hat = real * alpha + (1 - alpha) * (real + 0.5 * real.std() * beta)
        x_hat = Variable(x_hat, requires_grad=True)

        d_x_hat = D(x_hat)

        gradients = calc_grad(d_x_hat, x_hat, scaler)

        gradients = gradients.reshape(gradients.size(0), -1)
        penalty = (gradients.norm(2, dim=1) - center).pow(2).mean()

        return penalty

class r1_regularizer(Loss):
    def __call__(self,
        real: torch.Tensor,
        D: nn.Module,
        scaler: Optional[GradScaler]=None
    ) -> torch.Tensor:
        real_loc = Variable(real, requires_grad=True)
        d_real_loc = D(real_loc)

        gradients = calc_grad(d_real_loc, real_loc, scaler)

        gradients = gradients.reshape(gradients.size(0), -1)
        penalty = gradients.norm(2, dim=1).pow(2).mean() / 2.

        return penalty

class r2_regularizer(r1_regularizer):
    def __call__(self,
        fake: torch.Tensor,
        D: nn.Module,
        scaler: Optional[GradScaler]
    ) -> torch.Tensor:
        return super().__call__(fake, D, scaler=scaler)
