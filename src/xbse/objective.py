"""Contrastive objective (shared across instances).

InfoNCE with in-batch negatives plus optional explicit hard negatives (the PairSource's
opposite-structure examples). Symmetric anchor<->positive form.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def info_nce(anchor: torch.Tensor, positive: torch.Tensor,
             hard_neg: torch.Tensor | None = None, temperature: float = 0.05) -> torch.Tensor:
    """anchor, positive: (B, d) L2-normalized. hard_neg: (B, d) optional.

    Positives are the matched row; in-batch other-rows are negatives; hard_neg adds one
    explicit different-structure negative per row (the sensitivity signal).
    """
    b = anchor.size(0)
    logits = anchor @ positive.t() / temperature          # (B, B)
    if hard_neg is not None:
        extra = (anchor * hard_neg).sum(-1, keepdim=True) / temperature   # (B, 1)
        logits = torch.cat([logits, extra], dim=1)         # (B, B+1)
    labels = torch.arange(b, device=anchor.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits[:, :b].t(), labels)    # symmetric on the square block
    return 0.5 * (loss_a + loss_b)
