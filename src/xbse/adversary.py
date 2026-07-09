"""Optional adversary — strip a NAMED nuisance variable via gradient reversal.

LaBSE-style instances may want the embedding to carry no information about a nuisance (language,
time-period, surface register). Attach a classifier that predicts the nuisance through a gradient
reversal layer; the encoder is trained to make it fail. Purely optional per instance.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lambd)


class NuisanceHead(nn.Module):
    """Predicts a nuisance label from the (grad-reversed) embedding; encoder learns to defeat it."""

    def __init__(self, dim: int, n_classes: int, lambd: float = 1.0):
        super().__init__()
        self.lambd = lambd
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, n_classes))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(grad_reverse(z, self.lambd))
