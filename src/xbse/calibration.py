"""Per-feeder score calibration — closes XBSE_REVIEW_1 F1 (R2).

AUROC is a *ranking* statistic; the downstream engine consumes *scores*. A
binary VALIDATED gate launders unequal reliabilities into equal authority:
a 0.62-AUROC feeder must not enter decisions with the same standing as a
0.85 one. This module makes the reliability explicit and machine-readable:

  * :func:`platt` / :func:`isotonic` — score -> probability maps fit on the
    held-out pairs (never on training pairs; caller enforces);
  * :func:`ece` + :func:`reliability_curve` — expected calibration error and
    the diagram behind it;
  * :func:`calibration_fields` — the exact dict that ships inside every
    ``Report``: raw and calibrated ECE, the method, the curve, and the
    per-dimension ``reliability_weight`` a consumer MUST multiply into any
    decision confidence (an unweighted consumer is a documented, deliberate
    exception — see REVIEW_RESPONSE_1.md R2 disposition).

The reliability weight is deliberately simple and monotone:
``max(0, 2*auroc - 1)`` (Gini/Somers' D rescale of AUROC to [0,1]) — 0.5
AUROC feeders contribute nothing, 1.0 AUROC feeders full weight. It is a
*floor* convention, registered here; a consumer may use the full calibrated
posterior instead, never less.
"""

from __future__ import annotations

import numpy as np


def platt(scores: np.ndarray, labels: np.ndarray):
    """Platt scaling: logistic map score -> P(label=1), fit on held-out pairs.

    Returns a callable ``probs = f(scores)``.
    """
    from sklearn.linear_model import LogisticRegression

    s = np.asarray(scores, dtype=float).reshape(-1, 1)
    y = np.asarray(labels, dtype=int)
    lr = LogisticRegression(C=1e6, solver="lbfgs")  # near-unregularized
    lr.fit(s, y)
    return lambda x: lr.predict_proba(np.asarray(x, float).reshape(-1, 1))[:, 1]


def isotonic(scores: np.ndarray, labels: np.ndarray):
    """Isotonic (monotone, nonparametric) calibration; callable like :func:`platt`."""
    from sklearn.isotonic import IsotonicRegression

    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(np.asarray(scores, float), np.asarray(labels, int))
    return lambda x: ir.predict(np.asarray(x, float))


def reliability_curve(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10):
    """Equal-width reliability diagram: per-bin (mean prob, empirical rate, count)."""
    p = np.asarray(probs, float)
    y = np.asarray(labels, int)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append(
                {
                    "bin": b,
                    "mean_prob": float(p[m].mean()),
                    "empirical_rate": float(y[m].mean()),
                    "count": int(m.sum()),
                }
            )
    return rows


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error: count-weighted |mean prob - empirical rate|."""
    rows = reliability_curve(probs, labels, n_bins)
    n = sum(r["count"] for r in rows)
    if n == 0:
        return float("nan")
    return float(sum(r["count"] * abs(r["mean_prob"] - r["empirical_rate"]) for r in rows) / n)


def reliability_weight(auroc: float) -> float:
    """Registered floor convention: Gini rescale ``max(0, 2*AUROC - 1)``."""
    return max(0.0, 2.0 * float(auroc) - 1.0)


def calibration_fields(
    scores: np.ndarray,
    labels: np.ndarray,
    auroc: float,
    method: str = "isotonic",
    n_bins: int = 10,
    eval_frac: float = 0.5,
    seed: int = 0,
) -> dict:
    """The calibration block for a ``Report`` (XBSE_REVIEW_1 F1 closure fields).

    ``scores``/``labels`` MUST be the held-out validation pairs, never training
    pairs — and the calibrator itself is honest about its own evaluation: the
    held-out pairs are split (seeded) into a CAL part that fits the map and an
    EVAL part that measures ``calibration_ece``, because an isotonic fit can
    always drive its in-sample ECE to ~0. The shipped map is refit on ALL
    pairs (best map forward; its honest error is the held-out number). Raw
    scores are min-max squashed for the raw-ECE reference (monotone, so
    ranking is untouched).
    """
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_eval = max(1, int(len(y) * eval_frac))
    ev, cal_idx = idx[:n_eval], idx[n_eval:]
    if len(cal_idx) < 2:  # degenerate tiny inputs: fall back to in-sample
        ev, cal_idx = idx, idx
    fit = isotonic if method == "isotonic" else platt
    lo, hi = float(s.min()), float(s.max())
    raw01 = (s - lo) / (hi - lo) if hi > lo else np.full_like(s, 0.5)
    cal_map = fit(s[cal_idx], y[cal_idx])
    p_ev = np.clip(cal_map(s[ev]), 0.0, 1.0)
    full_map = fit(s, y)  # shipped forward; measured by the split number above
    p_full = np.clip(full_map(s), 0.0, 1.0)
    return {
        "calibration_method": method,
        "calibration_ece": ece(p_ev, y[ev], n_bins),  # held-out within held-out
        "calibration_ece_insample": ece(p_full, y, n_bins),
        "raw_ece": ece(raw01, y, n_bins),
        "reliability_curve": reliability_curve(p_full, y, n_bins),
        "reliability_weight": reliability_weight(auroc),
        "n_calibration_pairs": int(len(y)),
        "n_ece_eval_pairs": int(len(ev)),
    }
