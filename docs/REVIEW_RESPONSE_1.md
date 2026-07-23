# Response to external review #1 — dispositions

**Provenance.** Responds to `XBSE_REVIEW_1.md` (2026-07-23), item by item,
per house convention. Review committed verbatim alongside this response.
Status marks: ✅ closed · 🔵 machinery shipped, measurement ⚪ · ⚪ registered,
not yet run.

## Dispositions

### R1 (F2, specificity gate + 9×9 matrix) — 🔵 machinery shipped; matrix run ⚪

**Accepted in full — and the review's suspicion was already confirmed by
data the reviewer asked for in §3.** The bifactor A2 run (now published,
`docs/BIFACTOR_READOUT.md`) shows P1 FAILED: only 4 of 11 axes remain
diagonal-dominant after residualizing against the general-valence channel;
care/fairness/legitimacy/loyalty/purity are ≥ 0.98 predictable from G on
independent corpora. Shipped: `xbse.specificity` — the full
`discrimination_matrix` (feeders × dimensions AUROC) and `SpecificityBar`
with the **registered `specificity_margin = 0.05`** (half the validation
margin; competitor is a trained sibling, not a null; may be tightened,
never loosened — enforced in code and by test). Failing feeders are
DEMOTED to G per the bifactor prereg's rule, never displayed as hollow
axes. The full 9×9 matrix on the production feeders requires the trained
checkpoints (Atlas) — column ⚪ until run; the A2 residual transfer matrix
stands as the interim evidence and is published. **Closes when** the
matrix lands in the scorecard; the gate code and its law are in force now.

### R2 (F1, calibration into Report + DEME weighting) — 🔵 shipped xbse-side; erisml consumption ⚪

**Accepted.** Shipped: `xbse.calibration` — Platt and isotonic maps fit on
held-out pairs only, `calibration_ece` + `raw_ece` + `reliability_curve`,
and the registered floor convention `reliability_weight = max(0, 2·AUROC − 1)`
(a 0.62 feeder enters at weight 0.24, a 0.85 feeder at 0.71; a consumer
may use the full calibrated posterior, never less). `calibration_fields()`
returns the exact Report block. **Remaining ⚪:** wiring the block through
the production validation runs (checkpoints on Atlas) and `erisml`'s
consumption of `reliability_weight` in `MoralVector` per-dimension
uncertainty — tracked as the cross-repo item; an unweighted consumer is
hereby a documented, deliberate exception until then.

### R3 (bifactor numbers) — ✅ published

`docs/BIFACTOR_READOUT.md`: A1 (G passes the standard gate at 0.856),
A2 per-axis g-shares and diagonal-dominance, transfer summary, β
stability, and the prereg's binding demotion consequences. The reviewer
was right that these are interesting regardless of outcome; the outcome
is that the review's F2 concern is not hypothetical — it is measured.

### R4 (third-corpus OOD column) — ⚪ registered

Accepted; the scorecard now carries the ⚪ third-corpus column commitment
for the top-3 dimensions (README), sourced per
`experiments/data_sourcing_plan.md`. Bar-tightening, no ceremony.

### R5 (LaBSE instance, P1–P3) — ⚪ accepted as designed

The controlled-study design (same base model/pooling/dimension, objective
as the only varying factor) is accepted as the right dissolution of the
encoder-comparability confound. Registered predictions P1–P3 stand as
written in the review (on record 2026-07-23). Requires Atlas GPU time and
the STRATA Phase-1 instruments; scheduled behind the specificity matrix
run.

### R6 (rights_respect phrasing) — ✅ fixed

README now reads: "hypothesized corpus-choice failure; discriminating
experiment: US civil-rights corpus (CourtListener), status ⚪." The
method-failure branch reopens if the run fails, and the README will say so.

### R7 (Trusted-Publishing backport to turboquant-pro) — ⚪ queued

Accepted as trivial; tracked for the next tqp release cycle so the
attestation lands with an actual artifact.

## The one-line summary the review earned

The scorecard's 8/9 measures **validated transfer**; the bifactor readout
shows much of that transfer is **general valence wearing nine coats**; the
specificity gate now exists so the coats must henceforth prove they are
worn by different people. Labeled, gated, published.
