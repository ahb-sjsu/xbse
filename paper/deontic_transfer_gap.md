# The Deontic Gap: The 3×3 Structure of Moral Reasoning Predicts the Cross-Dataset Transferability of Moral-Dimension Encoders

**Andrew H. Bond**
Department of Computer Engineering, San José State University
`agi.hpc@gmail.com`

*Working draft — 2026-07-10. Circulated for critique; results are preliminary and several
interpretations are explicitly flagged as hypotheses. Comments welcome.*

---

## Abstract

We train nine sentence-embedding encoders, one per dimension of a 3×3 "MoralVector" taxonomy
(scope × mode: {Individual, Relational, Collective} × {What-Matters, Who-Decides, What-We-Know}),
and validate each on a held-out *second corpus* of the same dimension — the honest test of whether
an encoder has learned a transferable moral concept rather than one dataset's vocabulary. Eight of
nine dimensions clear a pre-registered, baseline-relative bar (cross-dataset AUROC beating both an
untrained-encoder null and a bag-of-words null by ≥0.10); one — `rights_respect` — fails across four
distinct corpus configurations. The central, unexpected finding is structural: **cross-dataset
transferability is organized by the *mode* (column) of the 3×3, not by scope (row).** The
epistemic column ("What-We-Know": privacy, honesty, societal/environmental) transfers best
(mean AUROC 0.83); the values column ("What-Matters": autonomy, care, fairness) is intermediate
(0.78); and the **deontic column ("Who-Decides": rights, physical-harm, legitimacy) transfers
worst (0.60)** and contains all three lowest-scoring dimensions. We argue this reflects a real
property of the moral domain rather than an artifact: deontic concepts encode *entitlements granted
by a governing framework*, which are jurisdiction-relative, whereas epistemic and (most) value
concepts track more universal states of the world. `rights_respect` is the extreme case — a right
is what a *legitimate* order grants, so rights are legitimacy-indexed — and even a stratified test
restricted to the most universal right (bodily integrity: European-Convention inhuman-treatment vs.
US excessive-force) fails to transfer, suggesting rights case-facts live in incommensurable legal
contexts. We report the method (cross-corpus contrastive training, an adversarial bag-of-words
control, dual-judge label-noise ceilings, and pre-registered bars), the negative results in full,
and the threats to validity, and we invite critique of the column-transferability hypothesis.

---

## 1. Introduction

AI systems that assist moral decisions increasingly rest on a *perception layer*: a component that
reads a situation and scores morally relevant features ("this harms someone," "this is unfair,"
"this exposes private data"). If that layer is unreliable, everything above it is unreliable. This
paper is about a specific, sobering discovery in building such a layer: **some moral dimensions do
not generalize across datasets, and which ones fail is predicted by their position in a simple 3×3
taxonomy of moral reasoning.**

Our taxonomy (the "MoralVector," the `k`-axis of the DEME moral tensor) is a 3×3 matrix:

|            | What Matters (values) | Who Decides (deontic) | What We Know (epistemic) |
|------------|-----------------------|-----------------------|--------------------------|
| Individual | autonomy_respect      | rights_respect        | privacy_protection       |
| Relational | virtue_care           | physical_harm         | epistemic_quality        |
| Collective | fairness_equity       | legitimacy_trust      | societal_environmental   |

Rows are the *scope* of a concern (whose good is at stake); columns are the *mode* (is the concern
about values, about who is entitled to decide, or about what is known). The taxonomy is convergent
with Moral Foundations Theory and Curry's Morality-as-Cooperation but is organized to be spanned by
a small set of encoders.

**Contributions.** (1) A cross-dataset validation regime for moral-dimension encoders, with an
adversarial bag-of-words control and pre-registered, label-noise-derived bars. (2) The empirical
finding that **transferability is column-structured** — the deontic column is systematically hard.
(3) A theory: deontic concepts are *framework-relative* (legitimacy-indexed), and a case study of
`rights_respect` failing across four corpus configurations, including a stratified universal-core
test. (4) A catalogue of coupling issues in the 3×3 that the transfer results expose.

We present this as a *negative-result-forward* paper. The strongest thing we can say is not "we
built nine good encoders" but "we built eight, one honestly failed, and the failure is legible."

## 2. Method

### 2.1 Encoders and the cross-dataset test

Each dimension's encoder is a BGE-M3 dual-encoder (mean-pooled, L2-normalized) fine-tuned with an
InfoNCE contrastive objective. The unit of supervision is a *signed valence*: each text is labeled
`+` (the dimension is upheld) or `−` (violated). The **honest metric is cross-dataset**: we train on
≥2 independent corpora per dimension and evaluate held-out AUROC on structural pairs whose anchor is
drawn from one corpus and whose same/different-sign comparisons are drawn from the *other* corpus.

This matters because single-corpus fine-tuning is deceptive. In earlier work every feeder scored
0.75–0.955 *within* its training corpus yet collapsed to ~0.47–0.55 (chance) on a second corpus of
the same dimension: the encoders had learned corpus surface, not moral structure.

### 2.2 Cross-corpus positives (`JointPairSource`)

The fix is to draw every training positive from a *different* corpus than its anchor: to pull an
A-text next to a same-sign B-text, the encoder cannot exploit A's surface — only structure the two
corpora share. Held-out AUROC is then cross-corpus by construction. An optional gradient-reversal
domain-adversary (DANN) was included but, empirically, rarely de-confounded (the domain classifier
usually stayed at ~100% accuracy); the cross-corpus positives, not the adversary, do the work.

### 2.3 Labels

Where corpora carried signed labels we used them (Social-Chem-101 moral-judgment sign; the ETHICS
benchmark; ECHR article-violation labels; toxicity/manipulation binaries). Where they did not
(privacy, environmental, and the CourtListener rights facts), we generated signs with a **dual-judge
pipeline** (two independent LLMs, `qwen3` and `glm-5`, temperature 0), averaging their signed
valence. This is a genuine threat to validity (§6) and we treat it as such.

### 2.4 Adversarial baselines and pre-registered bars

Two controls run in every gate, on the *same* held-out pairs:

- **Untrained null:** the base encoder's cross-dataset AUROC (typically ~0.5).
- **Bag-of-words null:** AUROC of TF-IDF cosine predicting same-structure. If the encoder barely
  beats bag-of-words, its "moral" signal is vocabulary.

A dimension is **VALIDATED** iff its cross-dataset AUROC beats *both* nulls by a pre-registered
margin (0.10). We chose this **baseline-relative** policy over an absolute label-noise ceiling after
discovering the latter is ill-posed here: a within-corpus noise ceiling (max AUROC achievable
against noisy labels) is not comparable to a *cross*-corpus AUROC, which additionally pays for
genre/jurisdiction gap. (We also caught and discarded a naïve noise estimator: Social-Chem's
`rot-agree` conflates rule-*applicability* agreement with *sign* correctness and overstated label
noise ~24×; a direct dual-judge sign-agreement estimate gave, e.g., 0.971 agreement on care →
label-flip ε≈0.015 → ceiling 0.878, against which our care model, 0.811, sits sensibly below.)
Bars are committed to version control *before* the validation run; the discipline is that a bar may
be tightened but never loosened after seeing a result, so git history is the pre-registration
record. Bar provenance is bound into a hash-chained audit artifact.

## 3. Results

### 3.1 The scorecard

Cross-dataset held-out AUROC, with both nulls and the baseline-relative verdict (margin 0.10):

| dimension | 2nd-corpus pairing | baseline | **cross** | BoW | margin | verdict |
|---|---|---:|---:|---:|---:|---|
| privacy_protection | privacy-RoTs + AITA scenarios | 0.551 | **0.853** | 0.542 | +0.31 | PASS |
| epistemic_quality | Social-Chem-honesty + ETHICS | 0.483 | **0.817** | 0.529 | +0.29 | PASS |
| societal_environmental | ClimateBERT + dual-judged env-claims | 0.426 | **0.817** | 0.483 | +0.33 | PASS |
| virtue_care | Social-Chem-care + ETHICS | 0.469 | **0.811** | 0.527 | +0.28 | PASS |
| fairness_equity | Social-Chem-fairness + ETHICS | 0.474 | **0.789** | 0.510 | +0.28 | PASS |
| autonomy_respect | ec-darkpattern + MentalManip | 0.515 | **0.747** | 0.529 | +0.22 | PASS |
| legitimacy_trust | Social-Chem-authority + ETHICS | 0.523 | **0.708** | 0.533 | +0.18 | PASS |
| physical_harm | BeaverTails + ETHICS-harm | 0.499 | **0.622** | 0.462 | +0.16 | PASS |
| rights_respect | ECHR + ETHICS-justice | 0.517 | **0.475** | 0.487 | −0.01 | **FAIL** |

The bag-of-words null is near chance (0.46–0.54) on every dimension, so the eight passing encoders
capture provably **non-lexical** structure — the cross-corpus construction that trains them is also
what starves bag-of-words (an A-anchor and its same-sign B-positive share little vocabulary). This
is our strongest defense against the "it just memorized valence-words" objection.

### 3.2 The column effect (the headline)

Grouping the cross-dataset AUROCs by the 3×3:

| mode (column) | dimensions | mean cross-dataset AUROC |
|---|---|---:|
| What-We-Know (epistemic) | privacy, epistemic, societal/env | **0.829** |
| What-Matters (values) | autonomy, care, fairness | **0.782** |
| **Who-Decides (deontic)** | rights, physical_harm, legitimacy | **0.602** |

| scope (row) | dimensions | mean cross-dataset AUROC |
|---|---|---:|
| Individual | autonomy, rights, privacy | 0.692 |
| Relational | care, physical_harm, epistemic | 0.750 |
| Collective | fairness, legitimacy, environmental | 0.771 |

**Transferability is organized by column (mode), not row (scope).** The column means are spread
0.60–0.83 and the three lowest-scoring dimensions in the entire framework — rights (0.475),
physical_harm (0.622), legitimacy (0.708) — are exactly the deontic column. Row means are compressed
(0.69–0.77) with no monotone structure. This is the paper's central claim, and we flag it as a
hypothesis: with n=9 (three per column) it is a striking pattern, not an established law (§6).

### 3.3 Case study: `rights_respect` fails across four configurations

Rights is the deontic column's extreme case. We attempted four cross-dataset configurations:

1. **ECHR ↔ ETHICS-justice** (European case-facts ↔ everyday justice scenarios). AUROC 0.475;
   training loss flat at chance throughout — cross-genre, no shared structure.
2. **ECHR ↔ CourtListener US civil-rights** (same genre, cross-jurisdiction). We improved the
   CourtListener extraction to target the *facts* section (cutting neutral dual-judge labels from
   87%→68%). AUROC 0.475; loss flat.
3. **ECHR ↔ CourtListener, class-balanced.** Rights litigation is overwhelmingly about
   *violations*, so the "respected" class was starved (+1,346 / −10,070). We mined the positive
   class specifically — civil-rights opinions whose *holdings* found no violation / granted
   qualified immunity — and dual-judge-confirmed 408 rights-respected cases (66% of candidates).
   Loss still flat.
4. **Stratified universal-core** (ECHR Articles 2–3, life/inhuman-treatment ↔ US excessive-force).
   Motivated by the observation that ECHR's aggregate "any-article-violated" label lumps
   incommensurable rights (its `−` class is 4,704 fair-trial + 1,421 property + 1,349
   inhuman-treatment + …). Restricting to a single, maximally universal right-type should make the
   cross-corpus positives coherent. Loss still flat (baseline 0.521).

Four configurations, one identical flat-loss signature. The method rehabilitated eight dimensions
and the bag-of-words control proves those are real; rights is not a method failure.

## 4. Interpretation

### 4.1 The deontic column is framework-relative

Values (care, fairness, autonomy) and epistemic states (privacy-as-exposure, honesty,
environmental facts) track features that are *largely universal*: cruelty, deception, and
data-exposure are recognizable across cultures and corpora. Deontic concepts are different: **a
right is an entitlement conferred by a governing order; legitimacy is the validity of that order;
welfare-as-duty is what the order owes.** These are *indexed to a framework*, so their surface
realizations differ not just in wording but in the institutional context that gives them meaning.
The prediction — deontic dimensions transfer worst — is what we observe.

### 4.2 Rights are legitimacy-indexed (a coupling, not an independence)

`rights_respect` and `legitimacy_trust` occupy the same deontic column (Individual vs. Collective
scope). They are coupled by construction: *specific* rights vary because they are granted by
*specific* legitimate orders. This predicts that rights should be the hardest dimension to transfer
across jurisdictions — and that even its most universal stratum (bodily integrity, which every
legitimate order nominally protects) may fail if the *case facts* are embedded in
jurisdiction-specific legal contexts (European detention/deportation vs. US police encounters). The
stratified test (config 4) is consistent with this stronger claim, though the small US-force sample
(369) leaves scarcity as a partial confound.

### 4.3 The privacy lesson: score the universal valence, not the legal category

Privacy is *legally* a right (ECHR Article 8 is literally "right to private life"), so one might
expect it to inherit rights' framework-relativity. Yet `privacy_protection` transfers best of all
(0.853). The reason is instructive: our privacy corpora encode privacy-as-**exposure-harm** ("was
personal information disclosed?" — universal), not privacy-as-legal-**right** (GDPR vs. US sectoral
rules — framework-relative). **The transferable signal is the universal valence beneath the legal
category.** This suggests the remedy for deontic dimensions is to target the underlying universal
good/harm rather than the jurisdiction-specific entitlement — and it is exactly what the failed
rights corpora do *not* do (court facts are irreducibly legal).

### 4.4 Other couplings the transfer results expose

- **care ↔ physical_harm** are near-inverse poles of one welfare axis (Relational row): protect vs.
  damage bodily/emotional welfare. They may be one signed dimension double-counted as two; an
  independence check (do their encoders' scores anti-correlate on shared scenarios?) is warranted.
- **fairness ↔ rights** collide on non-discrimination (ECHR Art. 14; US Equal Protection): a
  discrimination case is both, and our civil-rights corpus overlaps both — a potential
  double-counting that also muddies the rights labels.
- **autonomy ↔ legitimacy via consent** (Individual/values ↔ Collective/deontic): consent is
  autonomy exercised; legitimacy is "consent of the governed." A genuine cross-cell dependency.
- **epistemic_quality is a meta-dimension** for its column: honesty gates whether privacy and
  environmental claims can be trusted at all.
- **societal_environmental is a compound** (societal-harm + environmental-impact), internally
  heterogeneous and a candidate to split.

## 5. Implications for architecture

1. **Treat deontic dimensions as framework-conditioned**, not universal. Either stratify to a
   universal core, condition the encoder on the legitimacy framework, or score the underlying
   universal valence (the privacy strategy) rather than the legal category.
2. **Do not assume the 3×3 dimensions are independent.** At minimum, test care↔harm independence and
   resolve which dimension owns non-discrimination.
3. **Report transferability, not within-corpus fit.** Within-corpus AUROC is not evidence of a real
   dimension; the bag-of-words and untrained nulls should be standing metrics.

## 6. Threats to validity

- **Small n for the column claim.** Three dimensions per column. The 0.60/0.78/0.83 ordering is
  suggestive but could shift with different corpora; it is a hypothesis to be tested with more
  dimensions and more corpus pairings, not an established result.
- **Dual-LLM-judge labels.** Privacy, environmental, and the CourtListener rights signs are LLM
  labels. Two LLMs may share systematic bias, inflating apparent agreement and thus the label-noise
  ceiling; human-annotation validation is needed.
- **Confounds in the rights failure.** Config 4's US-force slice was small (369); scarcity cannot be
  fully ruled out. A within-US stratified pairing (excessive-force ↔ wrongful-detention, same
  jurisdiction) would isolate jurisdiction-gap from right-type mixing and is the cleanest next test.
- **Corpus construction.** The Social-Chem/ETHICS foundations share the "commonsense-morality
  vignette" genre; some transferability there may be genre similarity, not concept universality. We
  partially mitigate with the bag-of-words control (which is near chance), but this is not airtight.
- **Encoder and objective are fixed** (BGE-M3 + InfoNCE). We have not shown the column effect is
  invariant to encoder choice.
- **Keyword stratification** (config 4's force-keyword filter; the foundation keyword filters) is
  coarse and could introduce selection effects.

## 7. Open questions (for reviewers)

1. Is the column effect real, or an artifact of which corpora happen to exist for each dimension?
   What corpus design would falsify it?
2. Is `rights_respect` legitimately *unlearnable* cross-jurisdiction, or would a within-jurisdiction
   feeder (validated only on US law) be the honest unit — and if so, does "one rights encoder"
   even make sense, or must rights be jurisdiction-indexed?
3. Should the deontic column be modeled relationally (rights conditioned on legitimacy) rather than
   as independent axes?
4. Is care↔harm one signed dimension? Does collapsing them improve or harm downstream calibration?
5. Does the "universal valence beneath the legal category" strategy generalize — can we build a
   rights encoder on *conduct* (was a person brutalized / detained / silenced) rather than on *legal
   holdings*, and would that transfer where case-law does not?

## 8. Conclusion

Building a moral-perception layer forced a reckoning: not all moral dimensions are equally
learnable, and the pattern of which ones fail is not random. Transferability tracks the *mode* of
moral concern — epistemic > values > deontic — and the deontic column's difficulty appears to be a
real consequence of its framework-relativity, with `rights_respect` as the limiting case, coupled to
legitimacy and resistant across four corpus configurations. We offer the eight validated encoders,
the one honest failure, and the column hypothesis as an invitation: the most useful outcome would be
for someone to falsify the column claim, or to show that rights *can* be made to transfer by scoring
conduct rather than law. Either would advance a perception layer that a safety-critical ethics
engine can actually stand on.

## Acknowledgements & reproducibility

Encoders, the cross-dataset training harness, the bag-of-words control, and the pre-registered bars
are in `github.com/ahb-sjsu/xbse`; the per-dimension corpus map is in
`experiments/data_sourcing_plan.md`. The experimental pipeline (corpus assembly, dual-judge
labeling, training runs) was executed with substantial AI-assisted automation; all numerical results
are reproducible from the committed code and the documented corpora. This is a working draft
circulated for critique; the author welcomes correction, especially of the column-transferability
hypothesis and the dual-judge labeling.

## Selected references

- Forbes et al. *Social Chemistry 101.* EMNLP 2020.
- Hendrycks et al. *Aligning AI With Shared Human Values (ETHICS).* ICLR 2021.
- Chalkidis et al. *LexGLUE.* ACL 2022. (ECHR `ecthr_a`.)
- Sap et al. *Social Bias Frames (SBIC).* ACL 2020.
- Ji et al. *BeaverTails.* NeurIPS 2023.
- Graham, Haidt et al. *Moral Foundations Theory.*
- Curry et al. *Morality-as-Cooperation.*
- Ganin & Lempitsky. *Domain-Adversarial Training (DANN).* ICML 2015.
- Xiao et al. *BGE-M3.* 2024.
