# Pre-registration — PolarQuant shared-mode-removal probe for presence transfer

**Registered 2026-07-13, before running. Script: `scripts/presence_polar_probe.py`.**

## Motivation
Foundation-*presence* (identity) transfer RoT↔Reddit fails for 3/5 foundations, and attempt-2b showed
**domain-adversarial training makes it worse** (degraded all five; loyalty went pass→fail). Hypothesis:
the register confound lives in a small number of dominant *shared* angular modes; removing those
geometrically (and renormalizing to the unit sphere — the PolarQuant move; magnitude is already gone
since BGE-M3 embeddings are L2-normed) may reduce the register confound **without** the adversary's
collateral stripping of identity.

## Method (frozen, no fine-tuning — isolates the geometry)
Frozen BGE-M3. For each foundation, fit the removal basis on **TRAIN rows only** (heldout excluded by
the JointPairSource split → no leakage; basis uses corpus-id + unsupervised PCs, **never** the presence
label → no target leakage). Removal variants (project the direction(s) out of each L2-normed embedding,
renormalize): `k0_raw` (none), `mean`, `domain` (corpus-mean-difference axis), `pc1`, `pc1_2`, `pc1_3`
(top centred PCs), `domain+pc1_2`. Each scored by the SAME `gate()` cross-domain `structure_auroc` on
the heldout pairs. Also report `|PC_i · domain_dir|` (does the register axis coincide with a top PC?).

## Pre-registered predictions
1. **Primary:** removing the register/domain axis (and/or the top PC if it aligns with domain) lifts
   cross-domain `structure_auroc` over `k0_raw` for at least purity + loyalty. Directional prediction:
   `lift_over_raw ≥ 0.05` counts as the geometry doing real work.
2. **Rival / null:** removal does not move (or degrades) `structure_auroc` ⇒ register and identity are
   entangled in the same angular subspace (consistent with the adversary result) — an honest negative.
3. **Bigger-if-true:** a `lift_over_raw ≥ 0.05` on any of care/fairness/legitimacy (register-bound
   under both prior attempts) would be the more important result.

## Bar & scope
Same discipline as every channel: a variant "passes" only if `structure_auroc` beats **max(k0_raw,
BoW)** by ≥ 0.10 with `fuzz > 1.0`. **Scope caveat (registered):** this is a FROZEN probe — a lower bar
than the fine-tuned lam=0 pass (purity 0.719 / loyalty 0.661). A frozen WIN is strong evidence and
motivates the fine-tuned+residual build (conditional follow-up); a frozen MISS is inconclusive about
that combination, not proof the geometry is useless. No claim of a shipped channel comes from this
probe alone. Result: `experiments/presence_polar_result.json`.

---

## RESULT (run 2026-07-13) — prediction 2 confirmed: removal does NOT help. 0/5.

| foundation | k0 raw | best variant | best AUROC | lift over raw | \|PC₁·domain\| | gate |
|---|---:|---|---:|---:|---:|:--|
| purity     | 0.535 | mean         | 0.561 | +0.026 | 0.968 | ❌ |
| loyalty    | 0.557 | mean         | 0.572 | +0.015 | 0.993 | ❌ |
| care       | 0.523 | k0_raw       | 0.523 | −0.000 | 0.988 | ❌ |
| fairness   | 0.515 | domain+pc1_2 | 0.529 | +0.014 | 0.997 | ❌ |
| legitimacy | 0.483 | pc1_3        | 0.507 | +0.025 | 0.996 | ❌ |

**No foundation clears `lift_over_raw ≥ 0.05`; none clears the 0.10 gate.** The pre-registered **rival/null
(prediction 2) holds:** removal doesn't help ⇒ register and identity are entangled in the same angular
subspace. The mechanism is decisive: **`|PC₁ · domain_dir| = 0.97–1.00`** for all five — the top principal
component essentially *is* the register axis, so shared-mode removal is precisely targeted, yet AUROC
barely moves and often drops. Removing the register direction removes the identity along with it. This
agrees with attempt-2b (adversary strips identity) and A2 (family collapses onto the shared factor):
three independent methods, one conclusion. Prediction 3 (bigger-if-true rescue of care/fairness/
legitimacy) did **not** occur. Per the registered scope caveat, this frozen probe does not disprove a
fine-tuned+residual variant — but the geometry (identity ⊂ shared mode, not ⊥ to it) argues against it.
