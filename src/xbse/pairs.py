"""PairSource — the ONLY thing that differs between *-BSE instances.

An instance defines what should be INVARIANT (positives: same structure, different surface) and
what should be SEPARATED (negatives: different structure). Everything else is shared.

Circularity discipline: a PairSource must expose a train/held-out split, and any boundary used
for downstream discontinuity testing must NOT appear in training positives/negatives — else the
test is rigged. See docs/MOBSE_PLAN.md.
"""
from __future__ import annotations
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator


@dataclass
class Triplet:
    anchor: str
    positive: str      # same structure, different surface  (invariance target)
    negative: str      # different structure                (sensitivity target)


class CircularityError(RuntimeError):
    """Raised when held-out evaluation text leaked into training — the test would be rigged."""


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


class PairSource(ABC):
    """Yields training triplets for one domain. Subclass per *-BSE instance."""

    name: str = "abstract"

    @abstractmethod
    def train_triplets(self) -> Iterator[Triplet]:
        """Triplets for the training split only."""

    @abstractmethod
    def heldout_eval(self) -> dict:
        """Held-out validation material for the shared gate. Must return:
        {
          "structural_pairs": [(a, b, same_structure: bool), ...],  # near iff same_structure
          "surface_pairs":    [(a, a_paraphrase), ...],             # should stay near (invariance)
          "ood_texts":        [str, ...],                           # out-of-domain control corpus
        }
        These items must be DISJOINT from anything in train_triplets() — enforced below.
        """

    def assert_heldout_disjoint(self) -> None:
        """TEETH for the circularity rule. Hashes every training-triplet text and fails LOUD if
        any held-out eval item (structural or surface anchor, ignoring paraphrase surface) reuses
        it. A comment gets violated the night someone's in a hurry; this throws."""
        train = set()
        for t in self.train_triplets():
            train.update((_norm(t.anchor), _norm(t.positive), _norm(t.negative)))
        ev = self.heldout_eval()
        for a, b, _ in ev["structural_pairs"]:
            for txt in (a, b):
                if _norm(txt) in train:
                    raise CircularityError(
                        f"[{self.name}] held-out structural text leaked into training: {txt[:70]!r}")
        for a, _para in ev["surface_pairs"]:          # the anchor of a surface pair, not its paraphrase
            if _norm(a) in train:
                raise CircularityError(
                    f"[{self.name}] held-out surface anchor leaked into training: {a[:70]!r}")
