"""Valence readout — turn a validated `*-BSE` embedding into a signed dimension score.

The contrastive training already places same-valence texts together and opposite-valence apart, so
the **signed valence axis** is just the direction from the negative centroid to the positive
centroid in embedding space. No extra training: `fit()` encodes a labeled sample, stores the axis
plus a calibration (center + scale of the projection), and `score()` returns `tanh` of the
standardized projection in [-1, 1] with a confidence.

Discipline: a scorer may only be built from a checkpoint that PASSED the gate. `from_pairsource`
takes the instance's `Report` and calls `require_pass` before constructing anything — the same rule
every downstream tool obeys (`report.py`). This keeps the "tools import a *validated* core"
invariant: you cannot score with an unvalidated encoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .report import Report, require_pass

DEAD_BAND = 0.05


@dataclass
class Valence:
    value: float  # signed valence in [-1, +1]  (+ upheld, − violated)
    confidence: float  # 0..1, grows with distance from the decision boundary
    direction: str  # "positive" | "negative" | "neutral"


class DimensionScorer:
    """Frozen encoder + a signed valence axis for ONE dimension. Build via `fit`/`from_pairsource`."""

    def __init__(self, encoder, axis: np.ndarray, center: float, scale: float, name: str = ""):
        self.encoder = encoder
        self.axis = axis / (np.linalg.norm(axis) + 1e-9)
        self.center = float(center)
        self.scale = float(scale) if scale > 1e-6 else 1.0
        self.name = name

    # ---- construction -------------------------------------------------------
    @classmethod
    def fit(cls, encoder, pos_texts, neg_texts, name: str = "") -> DimensionScorer:
        """Axis = normalize(mean(+) − mean(−)); center/scale calibrate the projection to [-1,1]."""
        zp = _np(encoder.encode(list(pos_texts)))
        zn = _np(encoder.encode(list(neg_texts)))
        axis = _unit(zp.mean(0)) - _unit(zn.mean(0))
        axis = axis / (np.linalg.norm(axis) + 1e-9)
        allz = np.concatenate([zp, zn], 0)
        proj = allz @ axis
        center = float(np.mean([proj[: len(zp)].mean(), proj[len(zp) :].mean()]))  # midpoint
        scale = float(proj.std() + 1e-6)
        return cls(encoder, axis, center, scale, name)

    @classmethod
    def from_pairsource(
        cls, encoder, source, report: Report, checkpoint_hash: str, max_per_sign: int = 400
    ) -> DimensionScorer:
        """Gated builder: refuses unless `report` is a PASS for `checkpoint_hash` (require_pass)."""
        require_pass(report, checkpoint_hash)
        pos, neg = _labeled_sample(source, max_per_sign)
        return cls.fit(encoder, pos, neg, name=getattr(source, "name", ""))

    # ---- inference ----------------------------------------------------------
    def score(self, text: str) -> Valence:
        return self.score_batch([text])[0]

    def score_batch(self, texts: list[str], db: float = DEAD_BAND) -> list[Valence]:
        z = _np(self.encoder.encode(list(texts)))
        p = (z @ self.axis - self.center) / self.scale
        out = []
        for pi in p:
            v = float(np.tanh(pi))
            conf = float(min(1.0, abs(pi) / 2.0))
            direction = "positive" if v > db else ("negative" if v < -db else "neutral")
            out.append(Valence(value=v, confidence=conf, direction=direction))
        return out


# ------------------------------------------------------------------ helpers
def _np(x):
    """Accept a torch tensor or ndarray; return float32 ndarray."""
    return (
        x.detach().cpu().numpy().astype("float32")
        if hasattr(x, "detach")
        else np.asarray(x, "float32")
    )


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def _labeled_sample(source, max_per_sign: int):
    """Pull (text, sign) rows a PairSource exposes and split into +/- text lists (capped)."""
    rows = source._rows()  # (…, text, sign) — text at [-2], sign at [-1] by convention
    pos, neg = [], []
    for r in rows:
        text, sign = r[-2], r[-1]
        if sign == "+" and len(pos) < max_per_sign:
            pos.append(text)
        elif sign == "-" and len(neg) < max_per_sign:
            neg.append(text)
    return pos, neg
