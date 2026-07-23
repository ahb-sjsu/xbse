# The bifactor readout, published — R3 of XBSE_REVIEW_1

**Provenance.** Pre-registration `experiments/prereg_bifactor_readout.md`
(registered 2026-07-12, before training G and before fitting any β).
Artifacts: `experiments/bifactor_A1_result.json` (the general-valence
channel), `experiments/bifactor_A2_result.json` (residualized transfer,
P1/P3 verdicts), `experiments/bifactor_overlap_result.json`. Published
2026-07-23 on the reviewer's request (§3 of the review): these numbers are
the empirical input F2 needs, and they are interesting regardless of
outcome. They are also consequential: **P1 failed**, and the prereg's own
demotion rule binds.

## A1 — the general-valence channel G is real and strong

Family-pooled signed corpora, standard cross-dataset gate, 3 seeds:

| metric | value |
|---|---|
| cross-dataset AUROC | **0.856 ± 0.008** |
| margin over max null | +0.337 |
| untrained null / BoW null | 0.480 / 0.519 |
| gate | **PASS** |
| n rows | 51,319 |

G passes the same pre-registered, baseline-relative bar as every named
axis — the general good/bad factor is not an artifact.

## A2 — residualized transfer: P1 FAILED (4 of 11 diagonal-dominant)

After residualizing each specific axis against G, the prereg required ≥ 8
of 11 axes to remain diagonal-dominant on the cross-corpus transfer
matrix. **Four did.** Per-axis: how much of each named axis is G
(`gpred` = G's prediction of the axis on independent corpora), and whether
the residual still best predicts its own axis:

| axis | gpred (G-share) | diag-dominant after residual | reading |
|---|---:|:---:|---|
| purity | **0.9997** | ✗ | is G |
| legitimacy | **0.9986** | ✗ | is G |
| loyalty | **0.9900** | ✗ | is G |
| care | **0.9873** | ✗ | is G |
| fairness | **0.9836** | ✗ | is G |
| epistemic | 0.9396 | ✗ | mostly G |
| physical_harm | 0.8197 | ✓ | mixed; residual survives |
| identity_attack | 0.6658 | ✗ | mixed |
| privacy | 0.6421 | ✓ | **specific** |
| environmental | 0.5538 | ✓ | **specific** |
| autonomy | 0.4945 | ✓ | **specific** |

Transfer summary: raw mean diagonal 0.954 vs off-diagonal 0.726;
after residualization 0.779 vs 0.527 (off-diagonal drop −0.199).
β instability is high on most axes (only `epistemic` β-stable) — the
G-loadings themselves are noisy, which is part of the finding.

## What this binds (the prereg's own words)

> "A FAIL demotes the affected channel into G — the readout displays G
> plus only the specifics that survive; it never shows a hollow axis."

Consequences, adopted:

1. **The 8/9 scorecard measures validated *transfer*, not validated
   *specificity*.** Care, fairness, legitimacy (and loyalty/purity in the
   extended set) are ≥ 0.98 predictable from general valence on
   independent corpora: their cross-dataset AUROCs are real signal that is
   substantially *g-moral*, not their named dimension. The scorecard now
   carries this label (README, scorecard section).
2. **The genuinely specific axes are privacy, autonomy, environmental**
   (and physical_harm's residual survives). This is a finding: moral text,
   as these corpora sample it, is one strong valence factor plus a small
   number of separable specifics — consistent with the rank-collapse
   measurement (effective rank ≈ 5.7 of 9) that motivated the prereg.
3. **The standing specificity gate** (`xbse.specificity`, registered
   margin 0.05) operationalizes the demotion rule going forward: a feeder
   that cannot beat its siblings on its own held-out pairs is displayed as
   G, never as a hollow axis.

*One measurement in nine coats is a finding, not a failure — labeled.*

## Addendum (2026-07-23): gate-level confirmation — the 12×12 matrix

The standing specificity gate has now been run on the production
checkpoints (`experiments/specificity_matrix.json`): all 11 named axes
plus the validated G channel as a rival, scored on every axis's held-out
pairs with the same gate metric as validation. Result, registered margin
0.05: **own-axis** — privacy (+.277), environmental (+.310),
identity_attack (+.246), autonomy (+.171), physical_harm (+.082);
**DEMOTE-to-G** — purity (−.124, G beats it on its own pairs), epistemic
(−.065), legitimacy (−.049), care (−.038), loyalty (−.024), fairness
(−.018). The G row is bimodal — 0.85–0.94 on the demoted axes' own
pairs, chance (0.43–0.51) on the specific ones — confirming the A2
residualization through independent math. One divergence between the two
methods: identity_attack (gate: decisively specific; A2 residual: mixed).
The gate is the registered criterion; the divergence is recorded, not
adjudicated away. Sibling interchangeability inside the G family is
total: care's feeder outscores epistemic's own feeder on epistemic's
pairs (0.875 vs 0.811) and nearly matches loyalty's on loyalty's (0.935
vs 0.911).
