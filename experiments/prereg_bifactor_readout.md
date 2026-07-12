# Pre-registration — Phase A: the bifactor MoralVector readout (G encoder + residualized specifics)

**Registered 2026-07-12, before training G and before fitting any β.** Executes MoralVector roadmap
Phase A. Converts the rank-collapse finding (effective rank 5.68 of 9; one dominant general factor
+ ~4–5 specifics) from a *reported data property* into *architecture*: ship a trained general-valence
channel `G` and residualize every specific axis against it, so the readout can never display a hollow
axis that is really just G in disguise.

This prereg fixes, before any result is seen: the hypotheses, the corpora, the β-fit protocol, and the
two falsification gates (P1, P3). It corrects the earlier roadmap overstatement that this document was
"already drafted" (it was not) and that a family-pool encoder was "already trained" (it was not — today's
`pc0` is a PCA projection, see §1).

## 1. What exists today, honestly (the thing A1 replaces)
`pc0` is **not** an encoder. It is `PCA(1, random_state=seed).fit(X[tr]).transform(X[te])[:,0]` over the
standardized feeder-score matrix X (gtc `spectrum/analyzer.py:196`, band-2 eigen-spectrum + band-3
leakage check). It has **no validation Report, no gate, no cross-corpus AUROC** — it is a linear summary
of the already-trained feeders, so it cannot be *consumed* as a channel and cannot anchor honest
residualization (residualizing against a projection of the same scores is partly circular). A1 replaces
it with a BSE-encoder general-valence channel that passes the *same* pre-registered gate as every axis.

## 2. Hypotheses (rival, registered before the run)
- **H_bifactor (primary):** moral text has a strong general good/bad factor **plus** separable specific
  axes. Predictions: (A1) a general-valence feeder trained on family-pooled signed corpora PASSES the
  standard cross-dataset gate; (A2) after residualizing each specific score s_k against G, the specifics
  remain **diagonal-dominant** on the cross-corpus transfer matrix (each residual best predicts its own
  axis) — i.e. real structure survives removal of G.
- **H_unifactor (rival, TDM-leaning):** morality is essentially one harm-templated factor. Predictions:
  residualization **collapses** the specifics (off-diagonal ≥ diagonal, or residual AUROCs fall to
  chance), and/or G explains independent-corpus axis variance at G > 0.55 (P3 fail). This is the same
  H_TDM tested at the gate in B1, now tested at the *readout-structure* level.

Either outcome is a finding. **A FAIL demotes the affected channel into G** — the readout displays G plus
only the specifics that survive; it never shows a hollow axis. (This is the D1 adjudication's structural
leg; D1 adds the eMFD-harm correlation and effective-rank legs later.)

## 3. Corpora
- **G (family-pooled):** all signed Social-Chem-101 RoTs pooled across foundation categories
  (care-harm, fairness-cheating, authority-subversion, loyalty-betrayal, sanctity-degradation, +
  uncategorized), signed by `action-moral-judgment` (+ good / − bad); **independent second corpus** =
  ETHICS-commonsense overall (label 1=wrong→'-', 0=ok→'+'). Both are the *general* valence direction,
  not a keyword-filtered foundation. This deliberately shares Social-Chem with the specific feeders —
  which is correct here: G is *meant* to be the shared factor. Residual independence (not corpus
  disjointness) is what A2 tests.
- **Residualization / transfer matrix (A2):** the existing per-axis held-out cross-corpus pairs used to
  gate each of the 11 learned axes. β_k fit on a **calibration split** (disjoint from every axis's eval
  fold; SHA-256 enforced, same discipline as the contraction leakage control), committed with results.

## 4. Gate (frozen before the run)
- **A1 gate (standard, identical to the 11 passing axes):** G's cross-dataset held-out AUROC beats BOTH
  the untrained-encoder null AND the TF-IDF BoW null by **≥ 0.10**, fuzz-ratio > 1.0, nulls frozen on the
  untrained BGE-M3 before training. Trained at **3 seeds**; report mean ± σ (retrain stability is part of
  the A-gate, unlike the single-seed axis gate).
- **P1 (residual diagonal-dominance):** on the cross-corpus transfer matrix of residuals r_k = s_k − β_k·G,
  each axis's residual predicts its **own** held-out label better than any other axis's (diagonal >
  every off-diagonal in its row) for a pre-registered majority (≥ 8 of 11 axes). β_k **stability**:
  |β_k(half-1) − β_k(half-2)| / |β_k| ≤ 0.25 as the manipulation check (unstable β ⇒ that axis's residual
  is not reported).
- **P3 (G is general, not a smuggled specific):** on independent-corpus dimensions, G's own AUROC for any
  *single* specific axis's label ≤ **0.55** beyond what a general-valence factor should explain — i.e. G
  must not secretly BE one foundation. Operationalized: max over axes of |AUROC(G→axis) − AUROC(G→pooled)|
  ≤ 0.10, and no single axis dominates G's loading.

FAIL on A1 ⇒ no G channel ships (keep the PCA proxy, disclosed as a proxy). FAIL on P1 for an axis ⇒ that
axis is demoted into G. FAIL on P3 ⇒ G is mislabeled and renamed to the foundation it actually tracks.

## 5. Parameters (frozen)
Encoder BGE-M3, mean-pool, `max_len=128`; `train_adversarial` epochs 6, batch 24, lr 2e-5,
max_steps 1200, `max_lambda=0.0`; **3 seeds** (0,1,2) via corpus-order perturbation. Builder
`build_general_valence_joint()` in `joint_builders.py` (family-pool + ETHICS second corpus, signed).
Script `scripts/build_bifactor_readout.py`: materialize G → gate at 3 seeds → fit β on calibration split
→ residual transfer matrix → P1/P3. `holdout_frac = 0.12`; calibration split disjoint & hash-enforced.
Result → `experiments/bifactor_readout_report.json`, transfer matrix → `bifactor_transfer.json`.

## 6. A4 (cheap convergence leg, no gate)
Cosine-align the trained G direction with Schramowski et al. 2022 "moral direction" (their template-PCA
vs our general factor); report cosine. Positions G in the computational literature; one figure. Not a
gate — a citation.

## RESULT
_(to be appended after the run — mirrors the B1 prereg's RESULT table + honest-caveats block)_
