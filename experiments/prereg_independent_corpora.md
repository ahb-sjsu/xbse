# Pre-registration: do independent corpora separate the collapsed moral family?

**Registered before running.** Committed to git prior to training the encoders described below. The
point of this file is to fix the *prediction and the falsifier in advance*, so the result cannot be
reinterpreted after the fact — the discipline reviewer-2 asked for, and the same discipline the
project has applied to every prior control.

## Background

Under BGE-M3, four dimensions — `virtue_care`, `fairness_equity`, `legitimacy_trust`,
`epistemic_quality` — collapse into a single shared "commonsense-moral valence" axis (9×9 transfer
matrix + family-pool baseline; paper §3.2–3.3). All four currently draw from the **same** two
corpora (Social-Chem-101 + ETHICS), so the collapse is confounded with corpus-sharing. The open
question (paper §7 Q1): is the collapse a property of the **data** (removable by giving each
dimension an independent corpus) or of the **moral space** (irremovable)?

## Intervention

Replace the shared `Social-Chem ↔ ETHICS` pairing for each family dimension with a
`Social-Chem ↔ independent-corpus` pairing, where the independent corpus is a **different-provenance,
signed-valence** dataset targeting that specific concept. Cross-corpus positives then require the
encoder to align a concept across two genuinely different registers, not two commonsense-morality
corpora.

### Committed corpus choices (leading candidates; final ID per dimension fixed before *its* run)

| dimension | independent corpus | sign mapping | provenance vs Social-Chem |
|---|---|---|---|
| virtue_care | empathy / prosocial corpus (EmpatheticDialogues or Moral-Stories care-norm subset) | prosocial/supportive `+`, harmful/neglect `−` | different domain (dialogue/narrative), not ROTs |
| fairness_equity | discrimination / social-bias corpus (Social Bias Frames / hate-speech) | offensive-to-group/biased `−`, neutral `+` | different domain (social-media bias annotation) |
| legitimacy_trust | authority/legitimacy corpus (TBD — sourcing) | legitimate/authorized `+`, abuse/subversion `−` | to be fixed before run |
| epistemic_quality | misinformation/honesty corpus (LIAR / FEVER) | true/honest `+`, false/deceptive `−` | different domain (fact-checking), not ROTs |

`care` and `fairness` are **named now** (reviewer's explicit request). `legitimacy` and `epistemic`
corpora are being sourced by a research pass; each will be fixed in this file, committed, *then* run.
Moral-Stories carries a provenance caveat (its `norm` field may be seeded from Social-Chem); if used,
only its independent narrative *text* (situation + moral/immoral action) is relied on, and this is
recorded as a partial-independence corpus, not a clean one.

## Prediction (the falsifiable claim)

Re-run the **4×4 family cross-transfer** (care, fairness, legitimacy, epistemic) with the independent
corpora, comparing each dimension's off-diagonal cross-transfer to its within-transfer.

- **If the concepts are genuinely distinct** (data hypothesis): diagonal-dominance **emerges** —
  each dimension's mean off-diagonal drops **≥ 0.10 below** its within-transfer. Care and fairness in
  particular decouple from legitimacy/epistemic.
- **Falsifier** (moral-space hypothesis): if the family **still cross-contaminates** under
  independent corpora (off-diagonal within 0.10 of within-transfer, as it is now), then the ~5-axis
  collapse is a property of the moral space, not the data — and paper Implication 1 ("give every
  dimension independent corpora") is **wrong** and must be retracted.

## Controls committed in advance

1. **Register control.** The same test also addresses the symmetric confound (reviewer point 1):
   because the independent corpora are in *different registers*, a surviving diagonal-dominance is
   evidence of concept-specificity, not shared register.
2. **Same nulls.** Untrained + BoW nulls recomputed per new corpus; the pre-registered
   baseline-relative bar (margin ≥ 0.10 vs `max(null)`) is unchanged.
3. **Encoder held fixed** (BGE-M3, mean-pool, InfoNCE) so the comparison to the existing 9×9 is
   apples-to-apples; a gte-Qwen2 replication is secondary.

## Status

Registered 2026-07-10. Corpora for care/fairness named; legitimacy/epistemic pending sourcing.
No independent-corpus encoder has been trained at registration time.
