# xbse — external review #1

**Provenance.** External review, 2026-07-23. Scope: `README.md` as pasted at
v0.1.0-era, the `xbse` PyPI page, and cross-references from
`moral-spectrum-analyzer` (the downstream consumer). Reviewer operates under
this project family's standing rules: measured targets, never promises;
guarantee-language banned; a claim's status is ⚪ until its evidence artifact
exists. Written to be committed verbatim; dispositions belong in a
`REVIEW_RESPONSE_1.md`, per house convention.

**Summary verdict.** The distinctive contribution is correctly identified by
the README itself: not the encoders, the discipline. Four findings below, two
of them HIGH — neither invalidates the 8/9 scorecard; both bound what the
scorecard is *evidence of*. One data request. One proposed experiment with
three registered predictions.

---

## 1. Context corrections (recorded for the file, not disputes)

**1.1 The x-axis.** This reviewer initially read `x` in xbse as "any
*encoder*" (a provenance wrapper). The repo's actual generalization is
stronger: `x` = any **invariance axis**. LaBSE quotients out language and
keeps meaning; LeBSE quotients citation surface and keeps legal holding; the
moral feeders quotient wording/framing/topic/corpus and keep one dimension's
valence. Same encoder, objective, and gate; only the `PairSource` varies. The
one-table framing is the contribution. The encoder-provenance sidecar
(`xbse-corpus/1` sketch; see `STRATA_RFC.md` §2.2 provenance conventions)
remains complementary and is assumed by §4 below.

**1.2 Doctrine lineage.** The cross-dataset correction is the third
instantiation of one epistemology across this project family:

| domain | easy number (artifact) | honest number (transfer) |
|---|---|---|
| KV keys (`KV_KEYS_FINDING.md`) | reconstruction cosine 0.995 | perplexity (10⁴ vs 15) |
| retrieval (`HUBNESS_PRIMER.md`) | aggregate recall | anti-hub / p05 tail |
| moral feeders (this repo) | within-dataset AUROC 0.75–0.955 | cross-dataset AUROC (~0.50 collapse) |

Stated once, for reuse: **the easy number measures the artifact; the honest
number measures the transfer.** `JointPairSource` is additionally the family's
most elegant *structural* enforcement to date: same-sign positives drawn
across corpora make surface features useless **to the loss itself** — the
optimizer cannot cheat because cheating does not reduce the objective.
Uncertain ⇒ miss, implemented in gradient space.

---

## 2. Findings

### F1 · Calibration gap — HIGH

**At issue.** AUROC is a *ranking* statistic; the DEME engine consumes
*scores*. The binary VALIDATED gate launders unequal reliabilities into equal
downstream authority: `physical_harm` (0.622 — weak per-comparison) enters
decisions with the same standing as `privacy_protection` (0.853).
`FeederValidationRecord` *archives* the margins; nothing shown *weights* by
them.

**Fix.**
- Per-feeder score calibration on the held-out pairs (isotonic or Platt);
  reliability diagram + `calibration_ece` in every `Report`.
- DEME decision confidence MUST be a function of the feeder's validated
  margin, not only of its pass bit — propagate `structure_auroc` and
  `lexical_margin` into the MoralVector's per-dimension uncertainty, not just
  the audit trail.
- Closes when: calibration fields ship in `Report` and `erisml` consumes
  them; an unweighted consumer is a documented, deliberate exception.

### F2 · Specificity is untested — HIGH (sharpest)

**At issue.** Cross-dataset transfer + the BoW control prove the signal is
**non-lexical** and **non-corpus-bound**. They do **not** prove the axis
learned is the dimension named. Two privacy corpora can share a distributed
confound invisible to bag-of-words: general negative valence, severity,
"badness." Nothing in the scorecard shows the privacy feeder stays quiet on
fairness violations. The repo itself gestures at the answer:
`experiments/rank_test.py`, "bifactor result" — a bifactor structure means a
general moral factor (g) plus specifics, i.e. some fraction of every feeder's
AUROC is g-moral, not its named dimension.

**Fix.**
- Promote cross-dimension discrimination from `experiments/` to a **standing
  gate metric**: feeder *d* must beat every *other* validated feeder on *d*'s
  held-out pairs by a registered `specificity_margin`. Publish the full 9×9
  discrimination matrix in the scorecard.
- Report each feeder's g-loading vs specific-loading from the bifactor fit.
  If a dimension is mostly g, say so — one measurement in nine coats is a
  finding, not a failure, but it must be *labeled*.
- Closes when: the matrix and margins are in the README table and the gate
  refuses a feeder that cannot out-score its neighbors on its own dimension.

### F3 · Held-out is in-distribution-by-corpus — MEDIUM

**At issue.** Evaluation is cross-dataset *by pairing*, but both corpora were
seen in training. This is honest and already stronger than the field norm; it
is not yet out-of-distribution. `data_sourcing_plan.md` already points at
third corpora.

**Fix.** A third, never-touched corpus per dimension as the next bar
tightening — "a bar may be tightened, never loosened" licenses this without
ceremony. Closes when: at least the top-3 dimensions carry a third-corpus
AUROC column, ⚪-marked until run.

### F4 · Phrasing: the `rights_respect` causal claim — LOW (discipline)

**At issue.** "A *corpus-choice* failure, not a method failure" is a
hypothesis wearing a conclusion's clothes until the CourtListener corpus is
run and passes.

**Fix.** Approved phrasing until then: "hypothesized corpus-choice failure;
discriminating experiment: US civil-rights corpus (CourtListener), status ⚪."
Closes when: the run exists, whichever way it lands. If it *fails*, the
method-failure branch reopens and the README says so.

---

## 3. Data request

Publish the `rank_test.py` bifactor numbers (loadings per dimension, fit
statistics, corpus/version provenance). They are the empirical input F2
needs, they already exist, and they are interesting regardless of outcome.

---

## 4. Proposed experiment — the LaBSE instance as a controlled study

**Design.** A language-invariance `PairSource` (translation pairs as
same-sign positives) on the **same base model, pooling, and dimension** as
the moral feeders (BGE-M3, mean-pool, L2-norm). This isolates the training
objective as the only varying factor — dissolving the encoder-comparability
confound by construction rather than by sidecar bookkeeping. Compare, on
identical texts with identical `tqp anatomy --by language` settings
(STRATA Phase-1 instruments; provenance sidecar mandatory):

1. base BGE-M3;
2. xbse language-invariant instance;
3. (P3 only) one validated moral feeder.

**Registered predictions** (on record 2026-07-23; the measurement may
embarrass the reviewer, which is how this works):

- **P1.** Under base BGE-M3, per-language count-skew `S_k` varies widely
  across the 37-language corpus; cross-lingual transit hubs concentrate in a
  semantically central / translationese region (a de facto Area 0).
- **P2.** The deliberately language-invariant instance *flattens* per-language
  skew and *concentrates* transit further — trained invariance produces a
  stronger interlingua backbone than BGE-M3's emergent multilinguality.
- **P3.** Valence fine-tuning creates **centrality hubs at the moral
  extremes** — severity-as-centrality, visible as `corr(count, centrality)`
  rising on feeder embeddings vs base embeddings of the same corpus.

**Deliverables.** Strata-report JSONs (publishable even where the corpora are
not — fingerprints, skews, τ, recall-by-decile leak no text); a public
companion notebook on the Gutenberg set; a workshop-paper skeleton. Noted
plainly: falsifiable predictions published next to their measurements are
this project family's working recruitment mechanism.

---

## 5. Recommendations ledger

| # | item | priority | closes when |
|---|---|---|---|
| R1 | specificity gate + 9×9 matrix (F2) | HIGH | matrix in scorecard, gate enforces |
| R2 | calibration into `Report` + DEME weighting (F1) | HIGH | fields shipped & consumed |
| R3 | bifactor numbers published (§3) | HIGH (cheap) | loadings in repo |
| R4 | third-corpus OOD column (F3) | MED | ⚪ columns exist, top-3 run |
| R5 | LaBSE instance + P1–P3 runs (§4) | MED | reports committed |
| R6 | rights_respect phrasing ⚪ (F4) | LOW | CourtListener run lands |
| R7 | Trusted-Publishing backport to `turboquant-pro` (already house practice here) | LOW (trivial) | tqp releases carry attestations |

## 6. Phrasing addendum (extends the README's language discipline)

**Approved:** "8 of 9 dimensions validated against pre-registered,
baseline-relative bars; signal shown non-lexical by an adversarial BoW
control; specificity across dimensions: see matrix." **Banned until their
artifacts exist:** "measures morality" · "understands values" · any
per-dimension claim that has not cleared the specificity gate · the
unqualified `rights_respect` causal story.

---

*Standing cross-repo ledger, for continuity: kv_block freeze decision ·
save-side economics (P1-M4) · second human. R7 above retires the
Trusted-Publishing thread on completion.*
