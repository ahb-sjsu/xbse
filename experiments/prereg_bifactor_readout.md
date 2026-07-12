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

## RESULT (2026-07-12) — A1 PASSES: G is a real, transferable, retrain-stable axis.

The general-valence channel clears its own pre-registered gate at all 3 seeds. Nulls (frozen on the
untrained BGE-M3 before training): untrained 0.480, TF-IDF BoW 0.519 → **bar to beat = 0.619**.

| seed | trained AUROC | margin vs max-null (0.519) | fuzz | gate |
|---:|---:|---:|---:|:--:|
| 0 | 0.868 | +0.348 | 114.9 | ✅ |
| 1 | 0.850 | +0.330 | 125.7 | ✅ |
| 2 | 0.852 | +0.332 | 130.2 | ✅ |
| **mean** | **0.856 ± 0.008** | **+0.337** | min 114.9 | **✅ PASS** |

Corpus: 51,319 rows (Social-Chem pooled 32,606 / ETHICS 18,713), signs balanced (25,690 − / 25,629 +).
The BoW null (0.519) barely clears chance on cross-corpus general good/bad, so G learns transferable
structure **well beyond keywords** (+0.34). Retrain-stable (σ = 0.008). Data:
`experiments/bifactor_A1_result.json`; checkpoints `xbse_ckpt/general_valence_joint_s{0,1,2}.pt`.

**A1 gate satisfied ⇒ G ships as channel 0, replacing the ungated `pc0` PCA proxy.** This is the
gated encoder the readout-dishonesty deficit (#1: ~10 channels displayed, ~6 factors) required — a
trained channel-0 that residualization (A2) can honestly subtract, rather than a linear summary of the
same feeder scores.

**Honest caveats (carry with the numbers):**
1. **A1 is only the first leg.** Passing the standard gate proves G is a real axis; it does **not** yet
   prove the *bifactor* claim. That is A2/P1/P3 — do the specifics survive residualization against G
   (P1 diagonal-dominance + β-stability), and is G general rather than a smuggled foundation (P3)?
   Those run next; H_bifactor vs H_unifactor is decided there, not here.
2. **Shared corpus with the specifics — by design.** G pools Social-Chem, which the specific feeders
   also draw on; that is correct for a *shared* general factor but means A2's β-fit must use a
   calibration split disjoint from every axis's eval fold (hash-enforced) or the residual independence
   is confounded.
3. **A4 (Schramowski cosine) not yet run** — the cheap convergence leg is still to do.

## NEXT — A2 (registered here, before running)
Fit β_k (s_k = β_k·G + r_k) per axis on the disjoint calibration split; build the residual cross-corpus
transfer matrix; evaluate **P1** (residual diagonal-dominance ≥ 8/11 axes; β-stability |Δβ|/|β| ≤ 0.25
across split halves) and **P3** (G general, not a single foundation). FAIL on an axis ⇒ that channel is
demoted into G.

## RESULT (2026-07-12) — A2: the bifactor structure is REAL but PARTIAL. P1 and P3 both FAIL the strict gate — and the failure is the finding.

Per-item residualization of all 11 learned axes against G (seed-0), β_k fit on each axis's disjoint
calibration half, residual transfer AUROC on the test half. Data: `experiments/bifactor_A2_result.json`.

**Aggregate bifactor signature — present.** Removing G collapses cross-axis transfer toward chance while
the diagonal is retained:

| transfer matrix | mean diagonal | mean off-diagonal |
|---|---:|---:|
| raw scores | 0.954 | 0.726 |
| **after − β·G** | **0.779** | **0.527** |

Off-diagonal drops **−0.199 (→ 0.53, near chance)**: most cross-axis prediction was flowing **through
the shared general factor**. That is the bifactor claim, at the aggregate. **But the per-axis gates fail,
and they fail informatively — the 11 axes split into two clean populations:**

**Population A — collinear with G (the Social-Chem moral family): NOT separable.** care, fairness,
legitimacy, epistemic, loyalty, purity. β on G ≈ **0.98** (their per-item scores are ~identical to G),
G predicts them at **0.94–1.00**, and their residual diagonal collapses (0.59–0.77) below off-diagonals
→ **fail diagonal-dominance**. At the readout level these are facets of one factor, not independent axes.

**Population B — separable from G (independent-corpus axes): real specifics.** privacy, physharm,
identity_attack, autonomy (environmental borderline). β on G ≈ **0–0.36**, G predicts them only
**0.49–0.67** (autonomy at chance), residual diagonal stays **high (0.76–0.95) and dominates** its row
→ **pass diagonal-dominance**. These carry genuine axis-specific structure beyond G.

**Pre-registered verdict (honestly, as the gate was written):**
- **P1 FAILS:** 4/11 strictly diagonal-dominant (privacy, physharm, autonomy, identity_attack) vs the
  ≥8 threshold. *β-stability note:* the |Δβ|/|β| failures for autonomy (β≈0.02) and environmental
  (β≈−0.01) are **numerical artifacts of a near-zero loading** — a relative metric explodes when β≈0,
  which here *confirms* independence rather than instability. The genuine-instability cases are the
  family axes with large β.
- **P3 FAILS:** G-prediction spread across axes is **0.51** (min autonomy 0.49, max purity 1.00). G is
  **not uniformly general** — it is essentially the **shared valence of the Social-Chem moral-foundations
  family**, near-blind to independent-corpus axes.

**What FAIL means here (per the prereg's own rule "FAIL demotes channels into G"):**
1. **Relabel G honestly.** It is not a universal general moral factor; it is the *Social-Chem-family
   shared valence*. The channel name/scope must say so.
2. **Demote Population A into G** in the readout — displaying care/fairness/legitimacy/epistemic/loyalty/
   purity as six independent axes is the exact "hollow axis" dishonesty the bifactor readout exists to
   prevent. Effective independent structure ≈ **G + ~4 specifics ≈ 5**, consistent with the effective-rank
   5.68/9 finding — now with a *mechanism*, not just an eigenvalue.
3. **Keep Population B** (privacy, physharm, identity_attack, autonomy) as genuine residualized specifics.

**Critical confound (bounds the claim — do NOT over-read Population A):** G's training corpus (pooled
Social-Chem) **literally contains** Population A's training RoTs, so G→family ≈ 1.0 and β ≈ 0.98 are
**partly corpus overlap, not proven latent structure**. The Population B *separations are confound-free*
(different corpora, low G-pred), but "the family collapses into one true general moral factor" vs "the
family shares one corpus's valence direction" is **not settled here** — that is exactly **D1**
(cross-provenance residualization). So: Population B is a firm result now; Population A's collapse is
*demonstrated on shared-corpus data* and *flagged for D1*.

**Bearing on B1:** loyalty (β 0.97) and purity (β 0.96) passed their B1 gate as real transferable valence
axes, but A2 shows they are **not independent of G on this test** — precisely the B1 caveat #2. Honest
refinement, not a retraction: they are real moral-valence signals that are (on shared-corpus data)
largely the general factor. D1 decides whether that survives cross-provenance.

### A2-overlap diagnostic (2026-07-12) — the collapse is REAL, not corpus memorization. Do NOT delete the family axes.

The A2 confound resolved cheaply, *before* any readout demotion: partition each family axis's held-eval
items by whether G's training set contains that exact text (SHA-256), then recompute G→axis on the
**never-seen** half. Data: `experiments/bifactor_overlap_result.json`.

| axis | % held items in G-train | G→axis (seen) | **G→axis (UNSEEN)** | β (unseen) | residual self-AUROC (unseen) |
|---|---:|---:|---:|---:|---:|
| care | 26.6% | 0.995 | **0.990** | 0.987 | 0.606 |
| fairness | 28.0% | 0.995 | **0.993** | 0.984 | 0.669 |
| legitimacy | 31.1% | 0.999 | **0.997** | 0.984 | 0.584 |
| epistemic | 25.1% | 0.958 | **0.923** | 0.792 | 0.740 |
| loyalty | 28.9% | 1.000 | **0.992** | 0.957 | 0.756 |
| purity | 26.6% | 0.991 | **0.994** | 0.967 | 0.688 |
| **physharm (control)** | 14.0% | 0.976 | **0.805** | 0.369 | 0.750 |

**Verdict: the family collapse is a real, generalizing shared factor — not instance memorization.**
G predicts each family axis on text it *never trained on* essentially as well as on text it did (drop
≤ 0.035; β on unseen still 0.96–0.99). The **physharm control validates the test's sensitivity**: with
lower overlap it drops 0.976 → 0.805 and β 0.54 → 0.37, i.e. the method *does* expose overlap-inflation
when it exists — and physharm stays genuinely separable (low β, residual 0.75). So the ~27–31% literal
overlap is *not* what drives the family's collinearity with G.

**But the family axes are NOT hollow.** Every family residual (after removing G) is **above chance on
unseen items** (0.58–0.76). So the right representation is **not deletion and not six-independent-axes** —
it is the **bifactor form: G on channel 0 + a small residual r_k per axis**. That keeps *every* salient
dimension in the vector (no coverage lost — the end goal) while being honest that in this register the
family is ~0.96–0.99 the general factor. This *corrects the A2 "demote into G" language*: demote =
re-parameterize as G + r_k, **not** drop.

**Precise boundary (what is and isn't settled):** this rules out instance-memorization, but the unseen
items are still the *same register/provenance* (Social-Chem RoTs + ETHICS scenarios). The shared factor
is proven real **within the prescriptive-moral register**; whether care/loyalty/purity re-separate under
a *different* register or provenance is **D1/B5**, not answered here. What genuinely *adds* independent
coverage beyond G is the low-β set — privacy, physharm, identity_attack, autonomy — plus G's blind spots
(autonomy at chance) and the still-untested MAC/MFT dimensions (property, reciprocity → B4).

**A4 (Schramowski cosine) still not run.** Deferred with A-phase writeup.
