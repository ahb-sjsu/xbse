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

## Attempt 2b — adversary ON (`max_lambda=1.0`): the genuine test *(running 2026-07-13)*

Re-run with the DANN schedule actually engaged (`--lam 1.0`; logs show `lam` ramping 0→1, `domain_acc`
driven toward chance), same data/holdout/nulls so it is a like-for-like comparison against 2a. Purpose:
isolate the adversary's contribution. Three outcomes, each a finding: (i) more than 2/5 pass ⇒
register-invariance is doing real work beyond joint-contrastive; (ii) still 2/5 ⇒ the joint-contrastive
signal already carries the transfer and the adversary is redundant for presence; (iii) fewer pass ⇒ the
adversary *strips* identity along with register, i.e. foundation identity for those axes is expressed
via register-specific features (an honest negative — identity is partly register-bound). Checkpoints use
the `_adv` suffix so 2a is preserved. Result will land in `foundation_presence_adv_result.json`.

## Bearing on the MoralVector (updated)
Pending 2b, the defensible additions to the vector are **purity and loyalty presence channels** on the
strength of 2a's two-corpus transfer — but flagged **joint-contrastive, adversary-unverified** until 2b
says whether register-invariance holds or breaks them. Care/fairness/legitimacy **identity** remains
**not transferable** by either attempt so far and stays scored per-register (or not at all). The
honest core is still **G (valence) + valence residuals**; presence channels are a measured, caveated
extension, not a settled dimension.
