"""The shared validation gate — the whole point of the framework.

Every *-BSE clears the SAME bar, so no instance can grade itself easier than another. A good
embedding must:
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

# The bar every instance must clear is the SAME one LeBSE already cleared — not a bar each
# instance sets for itself. Populate these from ahb-sjsu/lebse's validation report so MoBSE is
# held to a literal, identical standard. (Defaults are conservative placeholders until filled.)
LEBSE_BAR = {
    "fuzz_min": 1.0,      # structure must move z more than surface (LeBSE cleared > 1)
    "auroc_min": 0.70,    # TODO: set to LeBSE's held-out structure-vs-surface AUROC
    "inv_min": 0.90,      # TODO: set to LeBSE's surface-invariance (paraphrase cosine)
}


def _sim(encoder, a: list[str], b: list[str]) -> np.ndarray:
    za, zb = encoder.encode(a), encoder.encode(b)
    return (za * zb).sum(-1).float().cpu().numpy()          # cosine sim (embeddings are L2-normed)


def surface_invariance(encoder, surface_pairs) -> float:
    """Mean cosine similarity of (text, paraphrase). Want ~1 — surface must not move z."""
    a = [p[0] for p in surface_pairs]; b = [p[1] for p in surface_pairs]
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


def gate(encoder, eval_data: dict, bar: dict | None = None) -> dict:
    """Run the full gate on a PairSource.heldout_eval() dict against the LeBSE bar (the SAME bar
    every instance clears — pass a different `bar` only to tighten it). Returns metrics + pass/fail."""
    bar = bar or LEBSE_BAR
    fuzz_min, auroc_min, inv_min = bar["fuzz_min"], bar["auroc_min"], bar["inv_min"]
    sp, fp = eval_data["structural_pairs"], eval_data["surface_pairs"]
    inv = surface_invariance(encoder, fp)
    fr = fuzz_ratio(encoder, sp, fp)
    au = structure_vs_surface_auroc(encoder, sp)
    passed = (fr > fuzz_min) and (au > auroc_min) and (inv > inv_min)
    return {
        "surface_invariance": inv,
        "fuzz_ratio": fr,
        "structure_auroc": au,
        "thresholds": {"fuzz>": fuzz_min, "auroc>": auroc_min, "surface_inv>": inv_min},
        "passed": bool(passed),
        "verdict": ("VALIDATED — a real instrument; earn the next tool"
                    if passed else
                    "FAILED GATE — no downstream tools, no claims; iterate the encoder"),
    }
