"""The shared validation gate — the whole point of the framework.

Every *-BSE clears the SAME gate MECHANISM; the bar it clears is a per-instance, PRE-REGISTERED
`Bar` that carries its own derivation (see `xbse.bar`). "Same bar for everyone" was the right
instinct against self-grading, but a single number imported from citation retrieval was the wrong
implementation: it made the AUROC half unpassable for noisy-label valence dimensions while the
weak surface pairs made the fuzz half vacuous. What is shared and non-negotiable now is:
  - the METRICS (structure AUROC, fuzz ratio, surface invariance diagnostic),
  - the REQUIREMENT that the bar be derived from corpus properties (label-noise ceiling,
    baseline lift) and registered before training — never loosened after a run,
  - the HARD STOP semantics.

A good embedding must:
  1. keep surface-only variants NEAR       (surface_invariance -> 1)
  2. push different-structure items APART   (structural separation large)
  3. therefore move MORE for structure than for surface   (fuzz_ratio > 1)
  4. separate same- vs different-structure by distance     (structure_auroc high)

The historical failure this guards against: an earlier moral encoder had fuzz_ratio ~0.20 —
surface perturbations moved the embedding MORE than moral ones. That must fail the gate.

HARD STOP: if gate()["passed"] is False, do not build downstream tools and do not make claims.
"""

from __future__ import annotations

import numpy as np

from .bar import LEBSE_LEGACY_BAR, Bar

# Backward-compat alias: LEBSE_BAR used to be a plain dict and the universal default. It is now
# the legacy Bar object, apt only for retrieval-style structure (SciBSE / LeBSE). gate() accepts
# either a Bar or the old {"fuzz_min", "auroc_min"} dict.
LEBSE_BAR = LEBSE_LEGACY_BAR


def _coerce_bar(bar) -> Bar:
    """Accept a Bar, a legacy dict, or None (-> legacy LeBSE bar)."""
    if bar is None:
        return LEBSE_LEGACY_BAR
    if isinstance(bar, Bar):
        return bar
    return Bar(
        auroc_min=float(bar["auroc_min"]),
        fuzz_min=float(bar["fuzz_min"]),
        source=str(bar.get("source", "ad-hoc dict bar")),
        derivation=str(bar.get("derivation", "legacy dict-style bar; no derivation recorded")),
    )


def _to_np(x) -> np.ndarray:
    """Torch tensor or ndarray -> float32 ndarray (lets the gate run on torch-free encoders)."""
    return (
        x.detach().cpu().numpy().astype("float32")
        if hasattr(x, "detach")
        else np.asarray(x, dtype="float32")
    )


def _sim(encoder, a: list[str], b: list[str]) -> np.ndarray:
    za, zb = _to_np(encoder.encode(a)), _to_np(encoder.encode(b))
    return (za * zb).sum(-1)  # cosine sim (embeddings are L2-normed)


def surface_invariance(encoder, surface_pairs) -> float:
    """Mean cosine similarity of (text, paraphrase). Want ~1 — surface must not move z."""
    a = [p[0] for p in surface_pairs]
    b = [p[1] for p in surface_pairs]
    return float(np.mean(_sim(encoder, a, b)))


def structural_separation(encoder, structural_pairs) -> float:
    """Mean cosine DISTANCE for different-structure pairs. Want large — structure must move z."""
    diff = [(a, b) for a, b, same in structural_pairs if not same]
    if not diff:
        return 0.0
    sim = _sim(encoder, [a for a, _ in diff], [b for _, b in diff])
    return float(np.mean(1.0 - sim))


def fuzz_ratio(encoder, structural_pairs, surface_pairs) -> float:
    """How much more the embedding moves for STRUCTURE than for SURFACE. Want > 1."""
    struct_move = structural_separation(encoder, structural_pairs)
    surface_move = max(1.0 - surface_invariance(encoder, surface_pairs), 1e-6)
    return struct_move / surface_move


def structure_vs_surface_auroc(encoder, structural_pairs) -> float:
    """AUROC of similarity predicting same-structure. Want high (distance separates structure)."""
    from sklearn.metrics import roc_auc_score

    sim = _sim(encoder, [a for a, _, _ in structural_pairs], [b for _, b, _ in structural_pairs])
    y = [1 if same else 0 for _, _, same in structural_pairs]
    return float(roc_auc_score(y, sim))


def gate(encoder, eval_data: dict, bar: Bar | dict | None = None) -> dict:
    """Run the full gate on a PairSource.heldout_eval() dict against a pre-registered Bar.

    Pass the instance's own Bar (PairSource.resolve_bar()). A dict is accepted for backward
    compatibility; None falls back to the legacy LeBSE bar (apt for retrieval-style structure
    only). Returns metrics + pass/fail + the bar's provenance, so a Report never contains a
    threshold without its derivation.
    """
    b = _coerce_bar(bar)
    sp, fp = eval_data["structural_pairs"], eval_data["surface_pairs"]
    inv = surface_invariance(encoder, fp)  # reported diagnostic, NOT a gate
    fr = fuzz_ratio(encoder, sp, fp)
    au = structure_vs_surface_auroc(encoder, sp)
    passed = (fr > b.fuzz_min) and (au > b.auroc_min)
    return {
        "structure_auroc": au,
        "fuzz_ratio": fr,
        "surface_invariance": inv,
        "thresholds": b.as_thresholds(),
        "bar_source": b.source,
        "bar_derivation": b.derivation,
        "bar_registered": b.registered,
        "passed": bool(passed),
        "verdict": (
            f"VALIDATED — clears the pre-registered bar [{b.source}]; earn the next tool"
            if passed
            else "FAILED GATE — no downstream tools, no claims; iterate the encoder"
        ),
    }
