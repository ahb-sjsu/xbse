"""Bar — a pre-registered, per-instance validation bar with PROVENANCE.

The legacy gate imported LeBSE's citation-retrieval AUROC (0.971) as a universal bar. That number
is apt for retrieval-style structure (SciBSE, LeBSE) and unreachable for cross-corpus moral
valence built on noisy human labels — so the single bar made the gate simultaneously unpassable
(AUROC half) and, with the weak surface pairs, vacuous (fuzz half).

The fix is NOT "lower the number." A bar chosen after seeing scores is calibrated to bless them.
The fix is a bar that carries its own derivation:

  - `source`     — where the number comes from (e.g. "noise-ceiling(rot-agree) * 0.90").
  - `derivation` — one sentence, written BEFORE the training run, saying how it was computed.
  - `registered` — the date it was set. Bars live in git next to the code, so the commit
                   history IS the pre-registration record.

Discipline (write it down, then the code enforces what it can):
  A bar may be set or TIGHTENED before a training run. It may never be loosened after one.
  `scripts/estimate_noise_ceiling.py` derives ceilings from label-noise estimates that are
  properties of the CORPUS, not of any model's results — that is what makes the number honest.
"""

from __future__ import annotations

from dataclasses import dataclass


class BarError(ValueError):
    """Raised when a Bar is malformed (empty provenance, out-of-range thresholds)."""


@dataclass(frozen=True)
class Bar:
    auroc_min: float
    fuzz_min: float
    source: str  # where the number comes from, e.g. "noise-ceiling(rot-agree)*0.9"
    derivation: str  # one sentence: HOW auroc_min was computed, before training
    registered: str = ""  # ISO date the bar was pre-registered (git history is the record)

    def __post_init__(self):
        if not (0.5 <= self.auroc_min <= 1.0):
            raise BarError(f"auroc_min={self.auroc_min} outside (0.5, 1.0] — not a usable bar.")
        if self.fuzz_min <= 0:
            raise BarError(f"fuzz_min={self.fuzz_min} must be > 0.")
        for field in ("source", "derivation"):
            if not (getattr(self, field) or "").strip():
                raise BarError(
                    f"Bar.{field} is empty. A bar without provenance is a number chosen after "
                    f"the fact — derive it (see scripts/estimate_noise_ceiling.py) and say how."
                )

    def as_thresholds(self) -> dict:
        return {"auroc>": self.auroc_min, "fuzz>": self.fuzz_min}


# The legacy universal bar, kept for the instances where it is actually apt (retrieval-style
# structure: SciBSE, a future LeBSE refactor). It is NO LONGER the default for every instance;
# an instance that declares no Bar falls back to this WITH A LOUD WARNING (see PairSource.
# resolve_bar) until it pre-registers its own.
LEBSE_LEGACY_BAR = Bar(
    auroc_min=0.97,
    fuzz_min=1.0,
    source="LeBSE held-out citation-retrieval AUROC 0.971 (ahb-sjsu/lebse MODEL_CARD)",
    derivation=(
        "Legacy universal bar imported from LeBSE's citation-retrieval validation; apt for "
        "retrieval-style structure, NOT derived for noisy-label valence dimensions."
    ),
    registered="2026-07-07",
)
