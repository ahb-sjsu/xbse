"""Smoke tests — no GPU/model needed. Verify the teeth (circularity guard) and the objective."""
import pytest
import torch
import torch.nn.functional as F

from xbse.pairs import PairSource, Triplet, CircularityError
from xbse.objective import info_nce


class _Leaky(PairSource):
    name = "leaky"
    def train_triplets(self):
        yield Triplet("a shared rule", "a shared rule (para)", "the opposite rule")
    def heldout_eval(self):
        return {"structural_pairs": [("a shared rule", "other", True)],  # leaks train text
                "surface_pairs": [], "ood_texts": []}


class _Clean(PairSource):
    name = "clean"
    def train_triplets(self):
        yield Triplet("train anchor", "train anchor variant", "train negative")
    def heldout_eval(self):
        return {"structural_pairs": [("held x", "held y", True)],
                "surface_pairs": [("held x", "held x paraphrase")], "ood_texts": []}


def test_circularity_guard_throws():
    with pytest.raises(CircularityError):
        _Leaky().assert_heldout_disjoint()


def test_clean_source_passes():
    _Clean().assert_heldout_disjoint()   # must not raise


def test_info_nce_is_scalar_positive():
    a = F.normalize(torch.randn(4, 8), dim=-1)
    p = F.normalize(torch.randn(4, 8), dim=-1)
    n = F.normalize(torch.randn(4, 8), dim=-1)
    loss = info_nce(a, p, n)
    assert loss.ndim == 0 and loss.item() > 0
