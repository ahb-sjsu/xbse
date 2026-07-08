"""ReaBSE — reasoning *-BSE, scoped to ARC transformation rules. Ties to latte/arc-equivariant-search.

This is where the equivariant cousin re-enters as a legitimate *-BSE: embed an ARC TASK by its
transformation RULE, invariant to the D4×color symmetry group (the same orbit latte votes over).

invariance axis  -> positives = same task under a D4 rotation/reflection + color permutation
                    (same rule, different surface grids)
sensitivity axis -> negatives = a DIFFERENT task (different rule)
independent label -> ARC task identity + solve-correctness oracle (admission passes only BECAUSE
                     it's scoped to a correctness oracle; unscoped "reasoning" would not)

Data: ARC-AGI task JSONs (public). On NRP the pod clones fchollet/ARC-AGI; locally point at a dir.
"""
from __future__ import annotations
import glob
import json
import os
from typing import Iterator

import numpy as np

from ..pairs import PairSource, Triplet
from ..admission import AdmissionCriteria

REABSE_CONFIG = {"base_model": "BAAI/bge-m3", "holdout_frac": 0.1}


def _d4(grid, k):
    g = np.array(grid, dtype=int)
    if k >= 4:
        g = np.fliplr(g); k -= 4
    return np.rot90(g, k).tolist()


def _recolor(grid, perm):
    return [[perm[c] for c in row] for row in grid]


def _serialize(train_examples) -> str:
    parts = []
    for ex in train_examples:
        parts.append("I:" + ";".join("".join(map(str, row)) for row in ex["input"]))
        parts.append("O:" + ";".join("".join(map(str, row)) for row in ex["output"]))
    return " ".join(parts)


def _augment(train_examples, seed: int):
    rng = np.random.default_rng(seed)
    k = int(rng.integers(0, 8))
    perm = list(rng.permutation(10))
    return [{"input": _recolor(_d4(ex["input"], k), perm),
             "output": _recolor(_d4(ex["output"], k), perm)} for ex in train_examples]


class ReaBSEPairSource(PairSource):
    name = "reabse"
    admission = AdmissionCriteria(
        invariant_structure="the ARC transformation rule",
        surface_class="grid orientation + color labeling (the D4×color symmetry group)",
        independent_label_source="ARC task identity + solve-correctness oracle (scoped to the oracle)",
    )

    def __init__(self, data_dir: str, holdout_frac: float = 0.1):
        self.data_dir = data_dir
        self.holdout_frac = holdout_frac
        self._tasks = None

    def _load(self):
        if self._tasks is None:
            self._tasks = []
            for fp in sorted(glob.glob(os.path.join(self.data_dir, "*.json"))):
                try:
                    t = json.load(open(fp))
                    if t.get("train"):
                        self._tasks.append((os.path.basename(fp), t["train"]))
                except Exception:
                    pass
        return self._tasks

    def _split(self):
        train, held = [], []
        for tid, tr in self._load():
            b = (hash((self.name, tid)) % 1000) / 1000.0
            (held if b < self.holdout_frac else train).append((tid, tr))
        return train, held

    AUG_PER_TASK = 16   # D4xcolor group is huge; sample many augmentations per task (fix data starvation)

    def train_triplets(self) -> Iterator[Triplet]:
        train, _ = self._split()
        n = len(train)
        for i, (_tid, tr) in enumerate(train):
            for a in range(self.AUG_PER_TASK):
                # augment BOTH anchor and positive -> forces group-invariance, not canonical->aug
                anchor = _serialize(_augment(tr, seed=1000 * i + a))
                positive = _serialize(_augment(tr, seed=1000 * i + a + 500))
                negative = _serialize(train[(i + 1 + a) % n][1])   # vary the negative task too
                yield Triplet(anchor=anchor, positive=positive, negative=negative)

    def heldout_eval(self) -> dict:
        _, held = self._split()
        m = min(len(held), 400)
        structural_pairs, surface_pairs = [], []
        for i in range(m - 1):
            base = _serialize(held[i][1])
            aug = _serialize(_augment(held[i][1], seed=10_000 + i))
            other = _serialize(held[i + 1][1])
            structural_pairs.append((base, aug, True))       # same rule (augmented) -> near
            structural_pairs.append((base, other, False))    # different rule -> far
            surface_pairs.append((base, aug))                # same rule, different surface
        return {"structural_pairs": structural_pairs, "surface_pairs": surface_pairs, "ood_texts": []}
