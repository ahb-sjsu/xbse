# Pre-registration: does dropping the residual shared corpus collapse the last off-diagonal?

**Registered before running.** Follow-up to `prereg_independent_corpora.md`, committed to git before
training. Pre-registers the dose-response prediction that §3.7 raised.

## Background

In the §3.7 4×4, the two re-corpused encoders (care_v2, fairness_v2) decoupled from the shared-corpus
family, but their **mutual** cross-transfer stayed the largest off-diagonal in the matrix:
`care_v2↔fair_v2` = 0.68 (gap +0.16, the weakest verdict). The reason is that care_v2 and fairness_v2
were only *partially* re-corpused — we replaced ETHICS with an independent corpus but **kept
Social-Chem** as the first corpus of both. So they still share Social-Chem, and the residual 0.68
tracks that residual overlap. The rest of the matrix supports a dose-response: share both corpora ≈
0.74; share Social-Chem only ≈ 0.49–0.68; the variation within "share Social-Chem" suggests it is the
quantity of shared *pairs*, not corpus identity, that drives cross-transfer.

## Intervention

Build `care_v3` and/or `fairness_v3` that **drop Social-Chem entirely**, pairing two fully-independent
corpora (for fairness: Measuring Hate Speech ↔ a second independent fairness/discrimination corpus;
for care: Moral-Stories ↔ a second independent care/empathy corpus). Re-run the `care↔fairness`
cross-transfer with no shared corpus between the two.

## Prediction (falsifiable)

- **Dose-response hypothesis:** with Social-Chem removed from both, `care_v3↔fair_v3` cross-transfer
  **drops from 0.68 toward chance (≤0.55)** — i.e. the last large off-diagonal falls once the residual
  shared corpus is gone. This would confirm cross-transfer tracks *degree of corpus/pair overlap*.
- **Falsifier:** if `care_v3↔fair_v3` stays ≈0.68 with **no** shared corpus, then the residual
  coupling is not corpus-overlap but genuine concept similarity (care and fairness *are* related moral
  dimensions), and the dose-response reading of §3.7 is wrong.

## Gate (manipulation check, inherited from the parent prereg)

care_v3 and fairness_v3 must each pass their own cross-dataset gate (beat max(null) by ≥0.10) on the
new all-independent pairing before the prediction is scored; a gate failure is "inconclusive — corpus
quality," not a falsification.

## Status

Registered 2026-07-10, before training care_v3/fairness_v3. Needs a second independent corpus for each
of care and fairness (sourcing); fairness has candidates (SBIC once loadable, or civil_comments
identity-attack slice); care is harder (same scarcity as legitimacy). No v3 encoder trained at
registration time.
