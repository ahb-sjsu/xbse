"""Cross-dimension specificity gate — closes XBSE_REVIEW_1 F2 (R1).

Cross-dataset transfer + the BoW control prove a feeder's signal is
non-lexical and non-corpus-bound. They do NOT prove the axis learned is the
dimension named: two corpora for the same dimension can share a distributed
confound (general valence, severity, "badness") invisible to bag-of-words —
and the bifactor readout (``docs/BIFACTOR_READOUT.md``) shows this is not
hypothetical: several axes are >0.9 predictable from the general-valence
channel alone.

This module promotes cross-dimension discrimination from an experiment to a
STANDING GATE:

  * :func:`discrimination_matrix` — the full D x D matrix ``M[f][d]`` =
    AUROC of feeder *f* scored on dimension *d*'s held-out pairs;
  * :class:`SpecificityBar` — feeder *d* passes iff, on *d*'s own held-out
    pairs, it beats every OTHER validated feeder by at least the registered
    ``specificity_margin``: ``M[d][d] - max_{f != d} M[f][d] >= margin``.

Registered margin (REVIEW_RESPONSE_1.md, before any full-matrix run):
``specificity_margin = 0.05`` — half the validation margin, because the
competitor here is a trained sibling rather than a null. Same bar law as
everywhere in this repo: may be tightened, never loosened.

A dimension that fails is not deleted; it is DEMOTED per the bifactor
prereg's rule — displayed as general-valence (G) rather than as its own
axis, and labeled so in the scorecard. One measurement in nine coats is a
finding, not a failure, but it must be labeled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

REGISTERED_SPECIFICITY_MARGIN = 0.05  # registered 2026-07-23, REVIEW_RESPONSE_1.md


def _auroc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Rank AUROC of positive vs negative scores; ties get average ranks."""
    pos = np.asarray(pos_scores, float)
    neg = np.asarray(neg_scores, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    x = np.concatenate([pos, neg])
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg_rank = (csum - counts + 1 + csum) / 2.0  # average 1-based rank per value
    ranks = avg_rank[inv]
    r_pos = float(ranks[: len(pos)].sum())
    n_p, n_n = len(pos), len(neg)
    return (r_pos - n_p * (n_p + 1) / 2.0) / (n_p * n_n)


def discrimination_matrix(scorers: dict, heldout_pairs: dict) -> dict:
    """``M[feeder][dim]`` = AUROC of ``scorers[feeder]`` on ``heldout_pairs[dim]``.

    ``scorers``: dim name -> callable(texts) -> scores (higher = more of the
    dimension's positive pole). ``heldout_pairs``: dim name ->
    ``(pos_texts, neg_texts)`` held-out pairs for that dimension.

    Returns nested dict; ``json.dump``-ready for the scorecard.
    """
    M: dict = {}
    for f_name, scorer in scorers.items():
        M[f_name] = {}
        for d_name, (pos, neg) in heldout_pairs.items():
            M[f_name][d_name] = _auroc(np.asarray(scorer(pos)), np.asarray(scorer(neg)))
    return M


@dataclass(frozen=True)
class SpecificityBar:
    """Feeder *d* must beat every other feeder on *d*'s own held-out pairs.

    ``margin`` defaults to the registered value and, per house law, may be
    passed HIGHER but never lower.
    """

    margin: float = REGISTERED_SPECIFICITY_MARGIN

    def __post_init__(self):
        if self.margin < REGISTERED_SPECIFICITY_MARGIN:
            raise ValueError(
                f"specificity margin {self.margin} below the registered "
                f"{REGISTERED_SPECIFICITY_MARGIN}: a bar may be tightened, never loosened"
            )

    def check(self, matrix: dict, dim: str) -> dict:
        """Gate verdict for one dimension from a :func:`discrimination_matrix`."""
        own = matrix[dim][dim]
        rivals = {f: matrix[f][dim] for f in matrix if f != dim}
        if not rivals:
            raise ValueError("specificity needs at least two feeders")
        best_rival = max(rivals, key=lambda f: rivals[f])
        gap = own - rivals[best_rival]
        return {
            "dim": dim,
            "own_auroc": float(own),
            "best_rival": best_rival,
            "best_rival_auroc": float(rivals[best_rival]),
            "specificity_gap": float(gap),
            "margin": self.margin,
            "specific": bool(gap >= self.margin),
            "disposition": "own-axis" if gap >= self.margin else "DEMOTE-to-G",
        }

    def check_all(self, matrix: dict) -> dict:
        """All dimensions; ``json.dump``-ready scorecard block."""
        return {d: self.check(matrix, d) for d in matrix}
