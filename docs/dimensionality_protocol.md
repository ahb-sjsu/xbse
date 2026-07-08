# The Dimensionality of Moral-Representation Space — A Falsification Protocol

**Reframed question.** "How many moral dimensions are there?" is ill-posed as
stated. Dimension is task-relative: optimal k depends on the objective you
optimize. So we do not seek *a number*. We measure a **profile**:

1. a **floor** (from the invariance quotient — task-independent),
2. a **ceiling** (from the bond-schema rank — task-independent),
3. **optimal k under several independent objectives**, and
4. whether that optimal-k is **stable across objectives** (evidence it's a property
   of the phenomenon) or **swings** (evidence it's task-relative / not
   well-defined — itself a major finding).

Every step has a **matched non-moral control**. Any dimension result that the
non-moral corpus also produces is a property of text encoders, not of morality.

**Pre-registration:** freeze & hash this file before running. Fix estimators,
objectives, thresholds, and the control corpus now. No post-hoc edits;
deviations → AMENDMENTS.md.

---

## 0. What each outcome would mean (decided in advance)
- **Stable optimal-k across objectives, aligned directions, beats control** →
  evidence k is a property of moral-representation space. Report k with its CI.
  Rewrite Ch. 5 with the measured number, not the posited nine.
- **Optimal-k swings with objective** → moral representation is **task-relative**;
  "the dimension of moral space" is not well-defined. Major result; retires the
  fixed-9 axiom. Report the swing.
- **Same profile as non-moral control** → the structure is generic to text
  encoders, not moral-specific. Report the null. Cheapest, most important negative.

All three are publishable. None is a failure. This is wired so the program's
central assumption (fixed moral dimensionality) *can lose*.

## 1. Bounds (task-independent — compute first)

### 1.1 Floor — the invariance quotient rank
The BIP requires meaning-preserving transformations to identify situations. The
moral space must have enough dimensions to separate all situations that are NOT
so related. Estimate:
- Build the equivalence classes: cluster corpus situations by
  paraphrase/translation invariance (same MoBSE-positive group).
- The floor is the rank needed to keep distinct classes separable — estimate via
  the number of eigen-directions of the *between-class* scatter with
  eigenvalue above the *within-class* noise floor (an LDA-style bound).
- **Floor_k = # between-class directions exceeding within-class noise.**

### 1.2 Ceiling — the bond-schema generative rank
The bond schema (agent × patient × bond-type × Hohfeldian-state) has a bounded,
countable generative grammar. The representable structural distinctions cap the
usable dimension.
- **Ceiling_k = rank of the one-hot/embedded bond-feature design matrix** over the
  corpus (the number of independently-varying structural features actually
  attested). [AUTHOR: enumerate the schema's feature cardinalities.]

The true optimal k must lie in [Floor_k, Ceiling_k]. If any objective's optimal k
falls outside these bounds, the bound computation or the encoder is wrong —
investigate before trusting the k.

## 2. Objectives (each yields its own optimal k)
Train/評価 the SAME unsupervised MoBSE representation (judgment-relatedness only,
NO dimensional supervision). Then find optimal dimension under each objective
independently, by sweeping k (via PCA truncation / bottleneck width) and locating
the knee/optimum:

- **O1 — MDL / rate-distortion.** k minimizing (description length of data given
  k dims) + (model cost). Task-free-ish; the compression optimum.
- **O2 — judgment prediction.** k maximizing held-out accuracy predicting human
  moral judgment (Social-Chem / Scruples labels) from the k-dim projection.
- **O3 — cross-lingual transfer.** k maximizing held-out transfer (train one
  language, test others) — the invariance objective.
- **O4 — cross-corpus alignment.** k at which principal directions of
  independently-built corpora (Social-Chem, Moral Machine, Scruples, ETHICS, MFQ)
  best align (max mean canonical correlation across corpus pairs).
- **O5 — intrinsic dimension (estimator, not objective).** ball-growth / MLE /
  effective-rank, reported with inter-estimator spread. A geometry read-out, not
  a task optimum; included as a reference point.

## 3. Stability analysis (the actual result)
- Collect {k_O1 … k_O5}. Report the set, not a mean.
- **Stability statistic:** coefficient of variation of optimal-k across objectives.
  Pre-registered: CoV < 0.25 → "stable / phenomenon-level"; CoV ≥ 0.5 →
  "task-relative"; in between → "weakly stable, report both."
- **Direction alignment:** for the objectives that yield a subspace, compute
  pairwise principal-angle alignment. Stable *count* with *misaligned directions*
  is still task-relative (same number, different axes).

## 4. The non-moral control (load-bearing — run the whole thing twice)
Repeat §1–§3 on a matched non-moral corpus through the identical pipeline and
encoder. Candidates: product reviews with rated attributes, or arxiv with subject
tags (same-encoder, richly-but-non-morally annotated).
- If the non-moral corpus shows the **same** floor/ceiling/stability profile, the
  result is generic to text representation → moral-dimensionality claims get **no**
  support. Report as the primary null.
- The moral-specific result is always **(moral profile − control profile)**, never
  the moral number alone.

## 5. Decision table
| moral CoV | directions align | vs control | conclusion |
|---|---|---|---|
| < 0.25 | yes | differs | k is phenomenon-level; report k ∈ [floor, ceiling] with CI |
| < 0.25 | no | differs | stable count, task-relative axes; report both |
| ≥ 0.5 | — | differs | moral representation is task-relative; "the dimension" ill-defined |
| any | — | matches control | structure is generic to encoders; moral-specific null |

## 6. What this cannot establish
Even the strongest outcome (stable, aligned, beats control) establishes the
effective dimension of **human moral judgment as encoded from these corpora** —
a claim about moral cognition/annotation, not about morality-in-itself (no
mind-independent moral manifold is accessible). Keep the claim at that size.
Whitney/stratification is orthogonal: dimension may be a **local field**, not a
scalar — if §5 gives high spread, test local dimension per region (the
stratification protocol) before concluding a global k exists at all.

## 7. Deliverable
Report: Floor_k, Ceiling_k, the {k_Oi} set, the stability CoV, the direction-
alignment matrix, and every number minus its non-moral control. Plot k-vs-objective
as the headline figure. Publish whichever row of §5 obtains — including the nulls.
