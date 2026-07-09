"""Train/held-out splitting + the anti-circularity guard (design §2.1).

Splitting a domain by a stable KEY (paper id, case id, rule id) guarantees no leakage between
train and held-out. `assert_disjoint` is the teeth: it hashes training text and throws if any
held-out / reserved-boundary item reuses it. Comments get violated; this raises.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .pairs import CircularityError, stable_frac


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def split_by_key(items: Iterable, key_fn, holdout_frac: float = 0.1, seed: int = 0):
    """Deterministic split by hashing a stable key -> (train, heldout). No RNG state, resumable.

    Uses stable_frac (md5), NOT Python's salted hash() — otherwise the split silently varied
    per process and the 'deterministic/resumable' promise was false."""
    train, heldout = [], []
    for it in items:
        k = str(key_fn(it))
        bucket = stable_frac(f"{seed}|{k}")
        (heldout if bucket < holdout_frac else train).append(it)
    return train, heldout


def assert_disjoint(
    train_texts: Iterable[str], reserved_texts: Iterable[str], who: str = "instance"
):
    """Throw if any reserved (held-out / downstream-boundary) text appears in training."""
    train = {_norm(t) for t in train_texts}
    for t in reserved_texts:
        if _norm(t) in train:
            raise CircularityError(
                f"[{who}] reserved/held-out text leaked into training: {t[:70]!r}"
            )
