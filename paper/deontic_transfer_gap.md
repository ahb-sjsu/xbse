# When Do Moral-Dimension Encoders Learn Distinct Concepts? Corpus Independence as the Load-Bearing Variable in Cross-Dataset Moral Encoding

**Andrew H. Bond**
Department of Computer Engineering, San José State University · `agi.hpc@gmail.com`

*Working draft — revised 2026-07-10 after external review and three additional control experiments
(a 9×9 off-diagonal transfer matrix, a generic-valence baseline, and an encoder-invariance
replication on a 4×-larger decoder). Claims are stated as hypotheses where the evidence is
preliminary; §Threats is expanded accordingly. Comments welcome.*

---

## Abstract

We set out to build a *perception layer* for an AI-ethics engine: nine sentence-embedding encoders,
one per dimension of a 3×3 "MoralVector" taxonomy (scope × mode). We validate each **across
datasets** — training on ≥2 independent corpora per dimension and testing on held-out cross-corpus
pairs — because within-corpus performance is deceptive (earlier single-corpus encoders scored
0.75–0.955 within their training corpus yet collapsed to ~0.5 on a second corpus). Under a
pre-registered, baseline-relative gate (beat an untrained null *and* a bag-of-words null by ≥0.10),
**eight of nine dimensions pass and one — `rights_respect` — fails across four corpus
configurations.** Two additional controls then reshape the interpretation. A **9×9 transfer matrix**
(each encoder evaluated on every dimension) shows that **dimension-specificity tracks corpus
independence, not the taxonomy**: the four dimensions with independent corpora (privacy,
environmental, autonomy, physical_harm) are diagonal-dominant and specific, while the four sharing a
single corpus family (care, fairness, legitimacy, epistemic — all from Social-Chem + ETHICS)
cross-contaminate, each firing on the others. A **valence-pool baseline** makes the collapse
decisive: an encoder trained on the pooled valence of the four shared-corpus dimensions *matches*
all four dedicated encoders (and, with ~4× data, exceeds two — we rely only on "matches") while
failing on the four independent ones (gaps +0.17 to +0.42) — those four are not four concepts but
**one**. We are symmetric about the confound: the four *independent*-corpus dimensions are distinct
**at the corpus level**, but their concept-level distinctness is itself confounded with register
shift and awaits a register-controlled test, so we do not overclaim it either. Both controls converge with an
independent rank test in implying the nine named dimensions occupy only **~5 effective axes** (a
general commonsense-valence factor + privacy, environmental, autonomy, physical_harm), plus rights
(which does not train). A cross-encoder replication then rules out the obvious objection that this is
a capacity artifact: a **4×-larger decoder embedder (gte-Qwen2-1.5B)** reproduces the same
care↔legitimacy collapse (cross-transfer ≥ within) and the same privacy separation, so the ~5-axis
structure is a property of the **data and labels, not the encoder**. We therefore argue the
load-bearing variable is **corpus independence**:
cross-dataset transfer *and* concept-distinctness both require it. The originally-striking "deontic column transfers worst" pattern is **confounded**
with corpus-sharing and is offered only as a hypothesis. The one robust concept-level result is
`rights_respect`: it fails even in a stratified, class-balanced, same-jurisdiction-adjacent design,
consistent with rights being *framework-relative* (a right is what a legitimate order grants). A
within-corpus decomposition sharpens this: rights labels are coherent *within* each corpus
(frozen-probe AUROC 0.71–0.77) but do not transfer — the largest within→cross drop of any dimension —
so the failure is in transfer, not learnability (small-corpus caveat aside). The contribution is methodological: a validation regime — cross-dataset gate
+ transfer matrix + valence baseline — that distinguishes encoders that learned a *concept* from
those that learned a *corpus*.

## 1. Introduction

AI systems that assist moral decisions need to *read* a situation and score morally relevant
features. This paper reports what happened when we tried to build that layer honestly — and the
central lesson is a caution: **an encoder can pass a cross-dataset test and still not have learned
the concept you named it after, if its two corpora share too much.** The tools that reveal this are
a cross-dataset validation gate, an off-diagonal transfer matrix, and a generic-valence baseline.

Our taxonomy (the "MoralVector," the `k`-axis of the DEME moral tensor) is a 3×3 matrix — rows =
scope {Individual, Relational, Collective}, columns = mode {What-Matters (values), Who-Decides
(deontic), What-We-Know (epistemic)}:

|            | What Matters      | Who Decides      | What We Know         |
|------------|-------------------|------------------|----------------------|
| Individual | autonomy_respect  | rights_respect   | privacy_protection   |
| Relational | virtue_care       | physical_harm    | epistemic_quality    |
| Collective | fairness_equity   | legitimacy_trust | societal_environmental |

**Contributions.** (1) A cross-dataset validation regime with a bag-of-words control and
pre-registered baseline-relative bars. (2) The finding that **dimension-specificity requires
independent corpora**: four of our dimensions have them and are demonstrably distinct; four share a
corpus and partially collapse into a shared moral-valence signal — shown by a 9×9 transfer matrix
and a generic-valence baseline. (3) A robust negative result: `rights_respect` fails across four
configurations, which we *hypothesize* reflects framework-relativity. (4) An honest retraction: the
"deontic column transfers worst" pattern from the first draft is confounded with corpus-sharing.

## 2. Method

**Encoders.** Each dimension is a BGE-M3 dual-encoder (mean-pool, L2-norm) fine-tuned with InfoNCE
on a *signed valence* label (`+` upheld, `−` violated).

**Cross-dataset validation.** We train on ≥2 corpora per dimension and draw every training positive
from a *different* corpus than its anchor (`JointPairSource`); held-out AUROC is thus cross-corpus
by construction. An optional gradient-reversal domain adversary was included but rarely
de-confounded (domain accuracy stayed ~1.0), so it is not load-bearing.

**Labels.** Native where available (Social-Chem judgment sign; ETHICS; ECHR article-violation;
toxicity/manipulation binaries); a dual-LLM-judge pipeline (`qwen3` + `glm-5`, temp 0, averaged)
where not (privacy, environmental, CourtListener rights). LLM labels are a threat to validity (§6).

**Controls.**
- *Untrained null* and *bag-of-words null* (TF-IDF cosine AUROC on the same held-out pairs). A
  dimension is VALIDATED iff it beats **both** by a pre-registered margin (0.10). We chose this
  baseline-relative policy over a label-noise ceiling because a within-corpus ceiling is not
  comparable to a cross-corpus AUROC (bars, nulls, and margin are committed to git before the run;
  they may be tightened, never loosened).
- *9×9 transfer matrix*: evaluate each trained encoder on **every** dimension's held-out pairs.
  Diagonal-dominance ⇒ concept-specificity; off-diagonal mass ⇒ shared structure.
- *Generic-valence baseline*: one encoder trained on `+/−` valence pooled across all dimensions,
  then evaluated per dimension. If it matches a per-dimension encoder, that dimension is "just
  valence."

## 3. Results

### 3.1 The cross-dataset scorecard (8/9 pass)

Cross-dataset held-out AUROC with both nulls and the baseline-relative verdict:

**Margin is defined consistently as `cross − max(untrained-null, BoW-null)`** — the gap to the
*stricter* of the two nulls — so the bar is the same test in every row (some earlier drafts silently
switched which null they measured against; this column is recomputed against the binding null).

| dimension | corpora | baseline | cross | BoW | margin | verdict |
|---|---|---:|---:|---:|---:|---|
| privacy_protection | privacy-RoTs + AITA | 0.55 | 0.853 | 0.54 | +0.30 | PASS |
| epistemic_quality | Social-Chem + ETHICS | 0.48 | 0.817 | 0.53 | +0.29 | PASS |
| societal_environmental | ClimateBERT + dual-judged claims | 0.43 | 0.817 | 0.48 | +0.34 | PASS |
| virtue_care | Social-Chem + ETHICS | 0.47 | 0.811 | 0.53 | +0.28 | PASS |
| fairness_equity | Social-Chem + ETHICS | 0.47 | 0.789 | 0.51 | +0.28 | PASS |
| autonomy_respect | ec-darkpattern + MentalManip | 0.52 | 0.747 | 0.53 | +0.22 | PASS |
| legitimacy_trust | Social-Chem + ETHICS | 0.52 | 0.708 | 0.53 | +0.18 | PASS |
| physical_harm | BeaverTails + ETHICS-harm | 0.50 | 0.622 | 0.46 | +0.12 | PASS |
| rights_respect | ECHR + ETHICS-justice | 0.52 | 0.475 | 0.49 | −0.05 | FAIL |

Bag-of-words is near chance throughout, so the passing encoders are not explained by our lexical
baseline. (Recomputed margins shift physical_harm +0.16→+0.12 and rights −0.01→−0.05 — both now
measured against their binding untrained null rather than BoW — but no verdict flips.) This *motivated* the first draft's story — but it does not, by itself, show the encoders
learned *distinct* concepts. That needs the transfer matrix.

### 3.2 The 9×9 transfer matrix — the pivotal result

Rows = trained-on, columns = evaluated-on (diagonal = self):

```
train↓ / eval→   priv   epis   env   care   fair   auto  legit  harm  rights
privacy          0.858  0.49  0.41  0.52  0.52  0.49  0.52  0.49  0.50
epistemic        0.52   0.843 0.44  0.75  0.68  0.54  0.66  0.48  0.49
environmental    0.56   0.41  0.817 0.45  0.44  0.53  0.48  0.49  0.44
care             0.55   0.871 0.46  0.825 0.77  0.51  0.74  0.39  0.50
fairness         0.51   0.853 0.51  0.80  0.788 0.51  0.69  0.40  0.48
autonomy         0.52   0.48  0.43  0.49  0.50  0.726 0.53  0.50  0.48
legitimacy       0.52   0.822 0.46  0.75  0.70  0.48  0.671 0.43  0.46
physical_harm    0.54   0.44  0.43  0.51  0.50  0.56  0.50  0.626 0.48
rights           0.66   0.50  0.53  0.49  0.50  0.48  0.50  0.51  0.468
```

Two groups fall out cleanly, split by **corpus independence, not by the taxonomy**:

- **Specific** (diagonal-dominant, off-diagonals ~0.4–0.55): **privacy, environmental, autonomy,
  physical_harm** — each with its *own* corpora. Their encoders fire only on their own dimension.
- **Collapsed**: **care, fairness, legitimacy, epistemic** — all from Social-Chem + ETHICS. The
  diagonal is not even the maximum: `care→epistemic 0.871 > care→care 0.825`; `fairness→epistemic
  0.853 > fairness→fairness 0.788`; `legitimacy→epistemic 0.822 > legitimacy→legitimacy 0.671`.
  These four fire on each other, with epistemic as the attractor — they share a corpus-specific
  "commonsense-moral valence," not four distinct concepts.

**The epistemic attractor is a clue, not a curiosity.** Everything in the family transfers to
epistemic *better than to itself* (`care→epistemic 0.871 > care→care 0.825`). The most economical
explanation is that epistemic_quality's held-out pairs are simply the **easiest** — cleanest labels
or strongest valence separation — which would *also* inflate its standalone PASS in §3.1. That is, the
transfer matrix's eval difficulty varies by dimension. A per-dimension pair-difficulty / label-
agreement diagnostic (planned, §6) would settle whether "epistemic is the attractor" is a fact about
the concept or about our eval set; we flag it rather than gloss it.

Notably, the couplings we *predicted from theory* (care↔harm; fairness↔rights) do **not** appear —
`care→harm` is 0.39 (below chance), because harm's independent corpus separates it. The actual
coupling is the shared-corpus family.

### 3.3 Valence-pool baselines — the family collapses into one concept

Two pooled encoders, each trained on `+/−` valence pooled across a set of dimensions, then evaluated
per dimension:

- **All-dimension pool** (positives paired across *all* dimensions): near chance (0.46–0.54) on
  every dimension. This shows no *universal* good-vs-bad axis captures anything — but the pairing is
  incoherent (privacy-'+' with harm-'+'), so it is weak evidence on its own.
- **Family pool** (care + fairness + legitimacy + epistemic only — the four sharing Social-Chem +
  ETHICS; positives are coherent *within* this family):

  | dimension | dedicated encoder | family pool | gap |
  |---|---:|---:|---:|
  | virtue_care | 0.825 | 0.834 | **−0.009** |
  | fairness_equity | 0.788 | 0.799 | **−0.011** |
  | legitimacy_trust | 0.671 | 0.795 | **−0.124** |
  | epistemic_quality | 0.843 | 0.914 | **−0.071** |
  | privacy_protection | 0.858 | 0.488 | +0.370 |
  | societal_environmental | 0.817 | 0.394 | +0.423 |
  | autonomy_respect | 0.726 | 0.533 | +0.193 |
  | physical_harm | 0.626 | 0.459 | +0.167 |

  **A single encoder trained on the pooled family valence matches all four dedicated family
  encoders** — and, with ~4× the training data, *exceeds* two of them (legitimacy 0.795 vs 0.671,
  epistemic 0.914 vs 0.843). The core inference needs only *matches* (which holds); we flag the
  excess as partly a sample-size effect and do **not** lean on "beats." Meanwhile the pool fails
  completely on the four independent-corpus dimensions. The collapse conclusion is robust:
  care/fairness/legitimacy/epistemic are **not four concepts but one** — a shared commonsense-moral
  valence.

  We are deliberately more cautious about the **mirror-image** claim — that the four
  independent-corpus dimensions are *concept*-distinct. Their low off-diagonals (§3.2) are confounded
  with **register/domain shift**: a privacy encoder scoring ~0.5 on care pairs is consistent with
  "privacy is a different concept" *and* with "the privacy encoder never saw Social-Chem's register."
  The same confound-logic we apply to the collapse must be pointed the other way. We therefore claim
  only that these four are **distinct at the corpus level**; establishing concept-level distinctness
  requires a register-controlled test (hold the concept fixed, move the register — §6, §7), which we
  pre-register rather than assert.

**Convergent evidence for a lower-rank moral space.** Under both the transfer matrix (§3.2) and the
family pool, the nine named dimensions reduce to **~five empirically-distinct axes**: {a shared
commonsense-valence factor collapsing care/fairness/legitimacy/epistemic, privacy, environmental,
autonomy, physical_harm}, plus rights (which does not train). This agrees with an independent
empirical rank test on scored MoralVectors (a bifactor structure: one dominant general
"moral-loading" factor + ~5 specifics; effective rank ≈ 3–6). Two unrelated methods — rank analysis
on the scored vectors, and encoder transfer/pooling — converge on a general valence factor plus a
handful of specifics.

### 3.4 `rights_respect` fails across four configurations

(1) ECHR ↔ ETHICS-justice: 0.475, flat loss. (2) ECHR ↔ CourtListener US civil-rights (facts-
targeted extraction): 0.475, flat. (3) ECHR ↔ CourtListener, class-balanced by mining 408 dual-
judge-confirmed no-violation/qualified-immunity holdings: flat. (4) Stratified universal-core (ECHR
Art 2–3 ↔ US excessive-force), with and without the adversary: flat loss across 5+ epochs, final
AUROC **0.506** (one adversarial run's *training* loss briefly collapsed to 0.54 but did not improve
held-out AUROC — an unstable outlier). Four configurations, one persistent failure.

**Is rights *unlearnable* or merely *untransferable*? (decomposition, per reviewer — now resolved,
§3.5.)** The 0.475 is cross-corpus by construction and cannot alone distinguish absent within-corpus
signal (**label incoherence**, unlearnable) from present-but-non-transferring signal
(**framework-relativity**, untransferable). A frozen within-corpus linear probe (§3.5) settles it:
rights labels are **strongly decodable within each corpus** — ECHR 0.77, CourtListener 0.71, well
above chance — so they are *coherent, not incoherent*. The flat cross-corpus training loss is
therefore a **transfer** failure, not a label-fitting failure, which **supports** — not weakens — the
framework-relativity reading of §4.

### 3.5 Within-corpus decomposition — rights is *untransferable*, not *unlearnable*

A frozen-encoder linear probe (BGE-M3 embeddings, **no fine-tuning**, 5-fold stratified
logistic-regression AUROC) measures whether each dimension's ± label is decodable *within a single
corpus*. This separates the two failure modes a cross-corpus number conflates — absent within-corpus
signal (label incoherence) vs present-but-non-transferring signal (framework-relativity):

| dimension | within-corpus probe AUROC (corpus A / B) | cross-dataset (§3.1) |
|---|---|---:|
| privacy_protection | 0.91 / 0.87 | 0.853 |
| societal_environmental | 0.99 / 0.95 | 0.817 |
| virtue_care | 0.997 / 0.76 | 0.811 |
| fairness_equity | 0.995 / 0.75 | 0.789 |
| legitimacy_trust | 0.995 / 0.69 | 0.708 |
| epistemic_quality | 0.99 / 0.81 | 0.817 |
| physical_harm | 0.95 / 0.74 | 0.622 |
| autonomy_respect | 0.99 / 0.72 | 0.747 |
| **rights_respect** | **0.77 / 0.71** | **0.475** |

Two things follow. **(1) Within-corpus success is universal and cheap.** Every dimension is linearly
decodable within its corpora (0.69–0.997) — including rights — by a probe that never fine-tuned. This
is the quantitative form of the paper's opening caution: within-corpus AUROC is deceptive because it
is a property *every* dimension has (and the Social-Chem side is trivially separable at ~0.99 for all
four family dimensions, which is exactly why they collapse into a shared valence). **(2) Rights is the
decisive case.** Its within-corpus signal is strong (0.77 ECHR, 0.71 CourtListener) yet its
cross-corpus AUROC craters to 0.475 — the **largest within→cross drop of any dimension**. Because a
frozen probe recovers the rights label *inside* each corpus, the labels are **coherent, not
incoherent**; the failure is specifically in *transfer across corpora/jurisdictions*. This is the
framework-relativity signature, with evidence: it upgrades §4's rights claim from "a hypothesis the
data has not yet earned" to "supported by decomposition."

*Caveat.* The within column is a frozen linear probe; the cross column is the fine-tuned InfoNCE
transfer — different estimators, so the exact numeric gap is not a clean subtraction. The *direction*
is unambiguous, however: strong within-corpus decodability rules out label incoherence for every
dimension, rights included, so the rights cross-failure can only be a transfer failure.

### 3.6 Encoder invariance — the collapse survives a 1.5B decoder

The natural objection to §3.2–3.3 is *capacity*: perhaps BGE-M3 (560M, encoder-only) is simply too
weak to tell care from legitimacy, and a larger model would separate them. We test this with
**gte-Qwen2-1.5B**, a decoder-based embedder ~4× the parameters from a different architecture family,
full-fine-tuned with the identical `JointPairSource` + InfoNCE recipe on three dimensions: the two
shared-corpus family members most at issue (**care, legitimacy**) and one independent-corpus control
(**privacy**). A 3×3 cross-transfer (rows = trained encoder, columns = held-out eval; diagonal =
within):

```
enc↓ / eval→     care    legit  privacy
care            0.793   0.761   0.513
legit           0.807   0.734   0.525
privacy         0.484   0.537   0.758
```

The structure is unchanged from BGE-M3. **The care↔legitimacy collapse persists**: the symmetric
cross-AUROC (0.784) *matches* the within-AUROC (min 0.734) — the legitimacy encoder scores **0.807
on care**, statistically indistinguishable from the care encoder's own 0.793 (a 0.014 gap on ~1k
pairs is within bootstrap noise — we do *not* claim it is "higher than itself") and clearly above
legitimacy on itself (0.734). The two encoders are interchangeable on care; there is no care-specific
structure a bigger model recovered. **Privacy stays distinct**: privacy↔care gap +0.26, privacy↔legitimacy gap
+0.20; privacy's encoder is at chance (0.48–0.54) on the family dimensions and both family encoders
are at chance (0.51–0.53) on privacy. A 4×-larger, architecturally-different encoder reproduces
*both* the collapse and the separation. The ~5-effective-axis structure is therefore a property of
the **data and labels**, not of BGE-M3's capacity — closing the most obvious reviewer objection.

*Caveat.* gte-Qwen2 was loaded as a native **causal** `Qwen2Model` (the vendor's bidirectional
inference code hard-requires `flash_attn`, which we bypassed), so this is "gte-Qwen2 weights with
last-token causal pooling," not the exact vendor embedder; and we replicated three dimensions, not
the full nine. Both are consistent with the narrow claim tested here — encoder capacity/architecture
does not dissolve the shared-corpus collapse — but a full-nine, bidirectional replication remains
future work.

## 4. Interpretation

**The primary finding is methodological: corpus independence is necessary for a dimension-specific
encoder.** Cross-dataset AUROC alone (§3.1) passed all four shared-corpus dimensions; only the
transfer matrix revealed they had learned a shared valence. Any framework that scores multiple moral
dimensions from overlapping corpora risks measuring one thing under many names.

**The rights failure is our most robust concept-level result**, now supported by the within-corpus
decomposition (§3.5): rights labels are coherent within-corpus (frozen-probe AUROC 0.71–0.77) yet do
not transfer, so the cross-corpus failure is a *transfer* failure, not label noise. We read this as
framework-relativity: a right is an entitlement conferred by a *legitimate* order, so specific rights
are jurisdiction-relative, and case-facts are embedded in incommensurable legal contexts even for
the same underlying right. `rights_respect` and `legitimacy_trust` are coupled in the taxonomy
(same deontic column), consistent with this. But we cannot exclude that rights needs longer context,
article-specific labels, within-jurisdiction pairing, or better labels (§6).

**An alternative hypothesis worth testing** (due to the reviewer): the real axis may not be
"deontic vs not" but **conduct/harm-level labels transfer better than institutionally-mediated
entitlement labels.** Privacy supports this — it transfers well when framed as exposure-*harm* even
though privacy is legally a *right*. This is possibly stronger and more defensible than the column
claim, and we flag it as the successor hypothesis.

**Retraction from the first draft.** We had reported that transfer is organized by the taxonomy's
*column* (epistemic > values > deontic). The 9×9 matrix shows this is **confounded with
corpus-sharing**: three of the four "hard" (low) dimensions are either the shared-corpus family or a
persistent failure. We withdraw the column claim as anything more than a hypothesis to be tested
after the shared-corpus confound is removed.

## 5. Implications
1. **Give every dimension independent corpora** before claiming it is a distinct concept. Our
   care/fairness/legitimacy/epistemic dimensions must be re-run with independent second corpora.
2. **Report the transfer matrix and a valence baseline**, not just cross-dataset AUROC.
3. **Treat rights (and likely the deontic dimensions) as candidate framework-relative concepts** —
   possibly requiring jurisdiction conditioning or conduct-level (not entitlement-level) labels.

## 6. Threats to validity
- **The column claim is retracted to a hypothesis** (small n; confounded with corpus-sharing).
- **Shared corpora** for four dimensions is the central confound the transfer matrix exposes; until
  fixed, those four are not established as distinct.
- **Register confound on the *specificity* side (symmetric to the collapse confound).** The four
  independent-corpus dimensions' low off-diagonals are confounded with domain/register shift: an
  encoder scoring ~chance off-diagonal is consistent with concept-specificity *and* with never having
  seen the other corpus's register. We therefore claim only corpus-level distinctness. The fix is a
  register-controlled test — hold the concept fixed and move the register (e.g. privacy-labeled text
  rewritten in Social-Chem style), or give two *different* dimensions corpora from the *same* family
  and check whether diagonal-dominance survives (§7).
- **Confidence intervals**: text-level bootstrap CIs (an afternoon on 410–1200 pairs) are being
  computed before wider circulation; several sub-0.02 gaps (e.g. the §3.6 0.807-vs-0.793) are
  reported as *matches*, not as ordered differences, pending those CIs.
- **Dual-LLM-judge labels** (privacy, environmental, rights) may share bias; human audit needed
  (planned: ~100–200 items/slice, report human↔LLM agreement, not only LLM↔LLM).
- **Rights confounds**: the stratified US-force slice was small (369); scarcity is not fully ruled
  out. Within-jurisdiction pairing and article-specific ECHR labels are the cleanest next tests.
- **Pseudo-replication**: AUROC is over structural pairs; text-level bootstrap CIs and per-dimension
  (n_pos, n_neg, unique-anchor, pair) counts are needed and not yet reported. Held-set sizes ranged
  410–1200 pairs per dimension.
- **Single encoder/objective** (BGE-M3 + InfoNCE): partially addressed (§3.6). A 4×-larger decoder
  (gte-Qwen2-1.5B) reproduces the care↔legitimacy collapse and the privacy separation on a 3-dim
  slice, so the *collapse* is not a BGE-M3 capacity artifact; a full-nine, bidirectional-mode
  replication is still outstanding.
- **Keyword-based** foundation and stratum filters are coarse.

## 7. Open questions and a pre-registered prediction
1. Do the four shared-corpus dimensions become distinct when given *independent* corpora — or is
   the ~5-dimensional collapse a property of the moral space rather than the data? (The family-pool
   control already confirms they collapse *under shared corpora*; this asks whether independence
   separates them.)
2. Does the concept-distinctness of the four independent-corpus dimensions survive a
   **register-controlled test** (concept fixed, register moved)? Diagonal-dominance that survives
   register control is concept-specificity; diagonal-dominance that vanishes was domain shift.
3. Is rights *unlearnable* or merely *untransferable*? *(Answered, §3.5: untransferable — within-corpus
   frozen-probe AUROC 0.71–0.77, cross 0.475, so the labels are coherent and the failure is transfer,
   not learnability.)* Still open: does a conduct-level rights label ("was someone
   brutalized/detained/silenced") transfer where case-law does not?
4. Is the (confounded) column pattern real once corpus-sharing is removed?
5. Does a stronger encoder separate the shared-corpus family that BGE-M3 collapses? *(Answered, §3.6:
   no — gte-Qwen2-1.5B reproduces the care↔legitimacy collapse. Open for the full nine dimensions in
   bidirectional mode.)*

**Pre-registered prediction (committed to git before the run).** We now give **virtue_care** and
**fairness_equity** independent second corpora — an empathy/prosocial corpus for care and a
discrimination/social-bias corpus for fairness — replacing the shared Social-Chem↔ETHICS pairing with
a Social-Chem↔independent-corpus pairing, and re-run the 4×4 family cross-transfer. **Prediction:** if
care and fairness are genuinely distinct concepts, their off-diagonal cross-transfer to each other
and to legitimacy/epistemic will *drop* below their (register-controlled) within-transfer by ≥0.10 —
diagonal-dominance emerges. **Falsifier:** if the family still cross-contaminates under independent
corpora, the ~5-axis collapse is a property of the moral space, not the data, and Implication 1 is
wrong. This prediction and the corpus choices are committed before training; see
`experiments/prereg_independent_corpora.md`.

## 8. Conclusion

Cross-dataset validation rescued eight of nine encoders from the illusion of within-corpus success —
but a transfer matrix then showed that four of the eight had learned a *shared corpus*, not a
*distinct concept*. The durable lesson is a methodological one: **to show an encoder learned the
moral dimension you named it after, it is not enough to beat a lexical baseline across two corpora;
the two corpora must be independent, and you must show the encoder does not fire on its neighbors.**
`rights_respect` remains the one dimension that resists every configuration, which we offer — with
appropriate caution — as possible evidence that some moral concepts are framework-relative and may
not admit a single, jurisdiction-independent encoder at all.

## Reproducibility & acknowledgements

Code, the cross-dataset harness (`train_adv.py`, `validate.py`, `JointPairSource`), the bag-of-words
control (`baselines.py`), pre-registered bars (`bar.py`, `joint_builders.py`), and the N×N
transfer-matrix script (`scripts/cross_transfer.py` — parameterizable over any dimension set, so it
regenerates both the 9×9 BGE-M3 matrix and the gte-Qwen2 3×3 of §3.6), and the within-corpus
frozen-probe decomposition of §3.5 (`scripts/within_corpus.py`) are in
`github.com/ahb-sjsu/xbse`; per-dimension corpora in `experiments/data_sourcing_plan.md`; the
pre-registered independent-corpora experiment in `experiments/prereg_independent_corpora.md`.
*Reproducibility caveat (being closed):* the family-pool baseline of §3.3 was run against the
harness with an ad-hoc pooled source; a dedicated committed `scripts/family_pool.py` is pending, and
the §3.2/§3.3 numbers should be regarded as harness-reproducible but not yet one-command-reproducible
until it lands. The experimental pipeline was executed with substantial
AI-assisted automation; results are reproducible from the committed code and documented corpora.
Circulated for critique — the author especially welcomes attempts to falsify the corpus-independence
claim and the rights framework-relativity hypothesis, and thanks the reviewer whose insistence on a
dimension-specificity test produced this paper's central result.
