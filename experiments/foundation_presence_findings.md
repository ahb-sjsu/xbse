# Foundation-presence (identity) channels — attempt 1 (linear probe): DECLINED cross-register

**2026-07-12.** Follow-on to the D1/B5 finding that foundation *identity* is a real axis, distinct from
the valence axis G captures (linear probe separated foundations at 0.75 mean *within* MFRC). Goal: turn
that into gated channels — the "which foundation is engaged" signal the per-foundation valence feeders
discard. This records attempt 1 and why it fails the project's own gate.

## Method (attempt 1 — the cheap path)
Cross-register-gated **linear presence head** on frozen pretrained BGE-M3: train foundation-k vs
other-foundations on **Social-Chem** RoTs (`rot-moral-foundations` labels), test cross-register on
**MFRC** Reddit comments. Gate: cross-register AUROC beats the TF-IDF BoW cross-corpus null by ≥ 0.10
and clears 0.60 — the same cross-dataset transfer discipline every channel obeys.
Script: `scripts/build_foundation_presence.py`; data `experiments/foundation_presence_result.json`.

## Result — all 5 DECLINED
| foundation | cross-register (SC→MFRC) | BoW cross null | margin | within-MFRC CV (D1) |
|---|---:|---:|---:|---:|
| care | 0.562 | 0.592 | −0.030 | 0.793 |
| fairness | 0.510 | 0.517 | −0.007 | 0.635 |
| legitimacy | 0.577 | 0.558 | +0.019 | 0.619 |
| loyalty | 0.529 | **0.652** | **−0.122** | 0.792 |
| purity | 0.539 | 0.572 | −0.033 | 0.863 |

Foundation identity is clearly separable **within** MFRC (0.62–0.86, confirming D1) but **does not
transfer** Social-Chem → MFRC (0.51–0.58, at or below the lexical null). None clears the gate.

## Diagnosis — this is a method failure, not (yet) proof identity is register-bound
1. **The linear head overfits register.** For loyalty the **BoW null (0.65) *beats* the embedding head
   (0.53)** cross-register — raw keywords transfer better than the frozen-embedding probe. A probe that
   loses to bag-of-words on transfer is reading corpus-specific structure, not the construct.
2. **The single-corpus frozen probe has no transfer mechanism.** The valence feeders that clear a
   cross-dataset gate are *fine-tuned on two independent corpora jointly*; the attempt-1 head is a frozen
   linear probe trained on one corpus (Social-Chem) and evaluated on the other. It has neither the
   two-corpus contrastive signal nor a domain-adversarial term, so it cannot be expected to transfer —
   and doesn't. **[CORRECTION 2026-07-13:** an earlier version of this line asserted "the valence feeders
   only transfer *because of* domain-adversarial training." That over-attributed the effect: the B1
   loyalty/purity **valence** feeders passed the same cross-dataset gate with the adversary **off**
   (`max_lambda=0.0`, pre-registered), so the two-corpus joint-contrastive fine-tune is *sufficient* for
   transfer in those cases; the domain-adversarial term is an additional, not the sole, mechanism. The
   original valence feeders built via `run_joint.py` did use `--lambda 1.0`, so both configurations exist
   in the record — which is exactly why attempt 2 must isolate the adversary's contribution rather than
   assume it. See the attempt-2 section below.]**

By the project's own admission criterion (cross-dataset transfer beating both nulls), attempt-1 presence
channels are **DECLINED** — same discipline that retracted `threat` and declined `sexual_content`. The
saved `.npz` heads are NOT gated channels and are not shipped.

## What this does and doesn't establish
- **Does:** foundation identity, as captured by a plain linear head, is **register-bound** — it does not
  cross-transfer RoT→Reddit. The D1 "0.75 separable" number is *within-register*; it is not a
  transferable channel.
- **Does not:** prove identity is *inherently* non-transferable. The fair test is a **domain-adversarial
  contrastive presence feeder** (the same `train_adversarial` path the valence feeders use, with presence
  labels instead of valence) — it forces a register-invariant representation and might recover a
  transferable identity core. **Not yet run** (the expensive path: contrastive training per foundation).
  Genuinely uncertain: if identity is expressed *only* via register-specific features, domain-adversarial
  training will strip it and the feeder will fail — itself an honest finding.

## Bearing on the MoralVector (the end goal)
Until a presence feeder clears the cross-dataset gate, the honest vector is **G (valence — transfers) +
the valence residuals**. Foundation-*identity* channels remain **unproven cross-register**; adding them
requires the domain-adversarial build (attempt 2) or a concession that identity is register-specific and
must be scored per-register. Do not ship attempt-1 heads as if they were validated dimensions.

---

# Attempt 2 (2026-07-13): two-corpus joint-contrastive presence feeders

`build_foundation_presence_joint(k)` = Social-Chem-k-vs-other × MFRC-k-vs-other, fine-tuned through
`train_adversarial` with **presence** labels, then gated identically to every channel. Falsification
order (per the plan): loyalty + purity first — the two most lexically distinct foundations.
Result artifact: `experiments/foundation_presence_attempt2_result.json`. Script:
`scripts/build_presence_feeders.py` (now `--lam`/`--tag` parametrized).

## Attempt 2a — adversary OFF (`max_lambda=0.0`): joint-contrastive baseline

**Important honesty note.** The first attempt-2 sweep ran with **`max_lambda=0.0`**, which leaves the
DANN gradient-reversal **inert** — every training step logged `lam0.00` and the final `domain_acc=1.0`
(the encoder still names the corpus perfectly). So attempt-2a is **not** the domain-adversarial test; it
is the **two-corpus joint-contrastive** feeder with the adversary disabled. It still differs from
attempt 1 in two ways that matter (fine-tuned not frozen; trained jointly on *both* corpora, not one),
which is why some foundations now transfer where the linear probe couldn't.

| foundation | cross-dataset AUROC | max null | margin | fuzz | gate | ckpt hash |
|---|---:|---:|---:|---:|:--|---|
| care       | 0.574 | 0.523 | 0.051 | 15.4 | ❌ | efda193dc543b616 |
| fairness   | 0.569 | 0.517 | 0.051 | 16.9 | ❌ | d9dbaaa3f50cec02 |
| legitimacy | 0.585 | 0.500 | 0.085 | 15.6 | ❌ | cc0758cc4f78334d |
| **loyalty**| **0.661** | 0.557 | **0.104** | 28.4 | ✅ | 377f1bc8977fbf35 |
| **purity** | **0.719** | 0.535 | **0.185** | 21.3 | ✅ | 1997cb0b3ac9d12a |

**2/5 pass**, and exactly the predicted two — loyalty and purity, the most lexically distinct binding
foundations. **What 2a establishes:** two-corpus joint-contrastive fine-tuning (no adversary) recovers a
cross-register-transferable presence signal for **purity and loyalty**, but **not** care/fairness/
legitimacy (margins ≤ 0.085, below the 0.10 bar). **What 2a does NOT establish:** anything about the
domain-adversarial mechanism — it was off. Do not describe these as "domain-adversarial" channels.

## Attempt 2b — adversary ON (`max_lambda=1.0`): the genuine test — DONE 2026-07-13

Re-run with the DANN schedule actually engaged (`--lam 1.0`; logs confirm `lam` ramped 0→~1 and
`domain_acc` was driven from 1.0 toward chance ~0.12 — the reversal is **active**), same data/holdout/
nulls so it is a like-for-like comparison against 2a. Result: `foundation_presence_adv_result.json`.

| foundation | λ=0 (2a) | λ=1 (2b) | λ=1 margin | λ=1 fuzz | effect | λ=1 gate |
|---|---:|---:|---:|---:|---|:--|
| care       | 0.574 ❌ | 0.585 | 0.062 | 17.5 | neutral | ❌ |
| fairness   | 0.569 ❌ | 0.499 | −0.018 | **0.007** | **stripped to chance** (fuzz collapsed) | ❌ |
| legitimacy | 0.585 ❌ | 0.544 | 0.043 | 19.1 | hurt | ❌ |
| **loyalty**| **0.661 ✅** | 0.641 | 0.084 | 18.1 | **hurt — lost its pass** (margin 0.104→0.084) | ❌ |
| purity     | 0.719 ✅ | 0.699 | 0.164 | 29.6 | dinged but survives | ✅ |

**Outcome (iii): the adversary STRIPS identity.** λ=1 → **1/5 pass** vs λ=0's 2/5. Every foundation was
neutral-to-degraded; fairness collapsed to a coin flip; loyalty's validated pass was destroyed. The
adversary provably worked (domain_acc → chance) — it removed register-diagnostic features, and for
foundation presence that took the identity signal with it. **Conclusion: domain-adversarial training is
STRICTLY WORSE for presence channels than the λ=0 joint-contrastive config.** Identity and register are
entangled in the same representational subspace for care/fairness/legitimacy; only purity has enough
independent signal to survive de-registering.

### Cross-check — PolarQuant shared-mode removal (`presence_polar_result.json`, prereg `prereg_presence_polar.md`)
A geometric, non-adversarial alternative: remove the dominant shared **angular** modes (register/domain
axis + top PCs) from frozen BGE-M3, renormalize, re-gate. **Result: 0/5 lift ≥ 0.05; nothing passes.**
Best case purity +0.026 (still far below the 0.10 bar). The diagnostic is decisive: `|PC₁ · domain_dir|`
= **0.97–1.00** for every foundation — the top principal component *is* the register axis, so removal is
well-targeted, yet AUROC barely moves. **The identity signal is not in a subspace orthogonal to the
shared mode; it is entangled within it.** Removing the register direction removes the identity with it.
This is a third independent method agreeing with 2b and A2: for care/fairness/legitimacy, foundation
identity and register occupy the same angular subspace. (Caveat, as pre-registered: this is the frozen
lower bar; it does not disprove a fine-tuned+residual build, but the geometry argues strongly against
spending GPU time on one.)

## Bearing on the MoralVector (final for this line of work)
The defensible presence additions are **purity** (passes at λ=0 *and* λ=1 — robust) and **loyalty**
(passes at λ=0 only), built via the **λ=0 two-corpus joint-contrastive** config. **Do not use
domain-adversarial training for presence channels.** Care/fairness/legitimacy identity is **not
separately transferable** by any of the three methods tried (joint-contrastive, adversarial, geometric)
and stays scored per-register. The honest core remains **G (valence) + validated valence axes**;
purity/loyalty presence are a measured, caveated extension. Contrast with the **valence** channels,
which are register-*invariant* — see `lambda_comparison_result.json`: G, loyalty-valence, and
purity-valence all pass at λ=0 **and** λ=1 essentially unchanged. Valence transfers regardless; identity
does not. That dissociation is the headline.
