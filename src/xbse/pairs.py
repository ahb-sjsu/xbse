"""PairSource — the ONLY thing that differs between *-BSE instances.

An instance defines what should be INVARIANT (positives: same structure, different surface) and
what should be SEPARATED (negatives: different structure). Everything else is shared.

Circularity discipline: a PairSource must expose a train/held-out split, and any boundary used
for downstream discontinuity testing must NOT appear in training positives/negatives — else the
test is rigged. See docs/MOBSE_PLAN.md.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

from .admission import AdmissionCriteria


@dataclass
class Triplet:
    anchor: str
    positive: str  # same structure, different surface  (invariance target)
    negative: str  # different structure                (sensitivity target)


class CircularityError(RuntimeError):
    """Raised when held-out evaluation text leaked into training — the test would be rigged."""


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def stable_frac(key: str) -> float:
    """Deterministic [0,1) bucket for the train/held split.

    Python's built-in hash() is salted per process (PYTHONHASHSEED), so a hash()-based
    split silently drew a DIFFERENT held-out set on every run — making AUROC numbers
    incomparable across runs (and, on an unlucky salt, degenerate). This md5-based bucket
    is stable across processes and machines, so @350 vs @2500 see the SAME held set."""
    return (int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) % 1000) / 1000.0


class PairSource(ABC):
    """Yields training triplets for one domain. Subclass per *-BSE instance.

    Every instance MUST declare `admission` (an AdmissionCriteria). check_admission() runs the
    executable filter — a domain without a definable invariant structure, surface class, AND an
    independent label source may not be built (design §3.1).

    Every instance SHOULD declare `bar` (a pre-registered xbse.bar.Bar derived from corpus
    properties — see scripts/estimate_noise_ceiling.py). An instance without one falls back to
    the legacy LeBSE bar with a loud warning; that fallback is a migration aid, not a policy."""

    name: str = "abstract"
    admission: AdmissionCriteria | None = None
    bar = None  # xbse.bar.Bar | None — pre-registered per-instance validation bar

    def check_admission(self) -> AdmissionCriteria:
        if self.admission is None:
            from .admission import AdmissionError

            raise AdmissionError(f"[{self.name}] declares no AdmissionCriteria — cannot be built.")
        return self.admission.validate(self.name)

    def resolve_bar(self):
        """Return this instance's pre-registered Bar; warn loudly on legacy fallback.

        The fallback exists so pre-Bar instances keep running during migration. It is
        deliberately noisy: a bar that nobody chose is not a pre-registration."""
        if self.bar is not None:
            return self.bar
        from .bar import LEBSE_LEGACY_BAR

        print(
            f"[{self.name}] WARNING: no per-instance Bar declared — falling back to the legacy "
            f"LeBSE bar (auroc>{LEBSE_LEGACY_BAR.auroc_min}). Derive and pre-register a bar "
            f"(scripts/estimate_noise_ceiling.py) before treating any PASS/FAIL as meaningful.",
            flush=True,
        )
        return LEBSE_LEGACY_BAR

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
                        f"[{self.name}] held-out structural text leaked into training: {txt[:70]!r}"
                    )
        for a, _para in ev["surface_pairs"]:  # the anchor of a surface pair, not its paraphrase
            if _norm(a) in train:
                raise CircularityError(
                    f"[{self.name}] held-out surface anchor leaked into training: {a[:70]!r}"
                )
