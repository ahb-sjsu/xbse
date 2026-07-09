"""Domain-adversarial training + polar readout — the cross-dataset rehabilitation.

Single-dataset contrastive fine-tuning of BGE-M3 learns a corpus's *surface* (the easiest way
to separate its pairs), so a per-dimension encoder scored 0.75-0.955 within its own corpus yet
~0.50 (random) on a second corpus of the SAME dimension. The fix has two parts, both from
`geometric-methods`:

- **Gradient reversal (ch. 14, DANN).** A dataset-discriminator head reads the embedding through a
  gradient-reversal layer; the encoder is trained to make the *moral* distinction WHILE being
  unable to tell which corpus a text came from. This directly penalizes the dataset-identifiable
  surface the single-corpus encoders were exploiting.

- **Polar readout (severity vs. specifics).** The moral-scenario space is empirically bifactor:
  one dominant general "severity / moral-loading" factor + ~5 specifics (rank_test.py). In polar
  coordinates that is radius = severity, angles = which dimension. `severity_auroc` measures
  whether the *general severity* transfers cross-dataset even when the specific direction does not
  — an honest partial-win metric, reported alongside the headline structure AUROC.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    """Identity forward; on backward, negate and scale the gradient (DANN, ch. 14).

    Placed between the encoder and the domain classifier: minimizing the classifier's loss w.r.t.
    its own weights still teaches it to read domain, but the reversed gradient pushes the ENCODER
    toward representations from which domain cannot be read — i.e. corpus-invariant."""
    return _GradReverse.apply(x, lambd)


class DomainHead(nn.Module):
    """Small MLP that tries to name the source corpus of an embedding (the adversary)."""

    def __init__(self, dim: int, n_domains: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, n_domains))

    def forward(self, z: torch.Tensor, lambd: float) -> torch.Tensor:
        return self.net(grad_reverse(z, lambd))


def dann_lambda(step: int, total: int, max_lambda: float = 1.0) -> float:
    """The DANN schedule (Ganin & Lempitsky): ramp λ from 0 to max as 2/(1+e^-10p)-1.

    Starting at 0 lets the encoder first learn the task; the adversary bites only once there is a
    representation worth de-confounding — training it adversarially from step 0 is unstable."""
    p = min(max(step / max(total, 1), 0.0), 1.0)
    return max_lambda * (2.0 / (1.0 + torch.e ** (-10.0 * p)) - 1.0)


def polar_severity(z: torch.Tensor, mean_dir: torch.Tensor) -> torch.Tensor:
    """Scalar 'severity / moral-loading' coordinate of each row.

    z is L2-normalized (so its literal radius is 1); the informative magnitude is the projection
    onto the dataset's dominant moral axis `mean_dir` (the general factor). This is the polar
    'radius' of the bifactor decomposition — how far along the severity axis, independent of which
    specific dimension (the residual 'angles')."""
    return z @ F.normalize(mean_dir, dim=-1)
