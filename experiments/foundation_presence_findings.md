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
2. **The valence feeders only transfer because of domain-adversarial training.** `train_adversarial`
   carries a `DomainHead` that *removes corpus-identifying features*, forcing register-invariance; that
   is why the 11 valence axes clear a cross-dataset gate at all. The linear probe here has **no such
   mechanism**, so it cannot be expected to transfer — and doesn't.

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

## Next (attempt 2, if pursued)
`build_foundation_presence_joint(k)` = Social-Chem-k-vs-other × MFRC-k-vs-other through `train_adversarial`
(domain-adversarial, presence labels) → gate identically. Falsification-order: prove on loyalty + purity
(most lexically distinct) before all five. Materialize MFRC to a local jsonl first.
