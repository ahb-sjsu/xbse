# When Do Moral-Dimension Encoders Learn Distinct Concepts? Corpus Independence as the Load-Bearing Variable in Cross-Dataset Moral Encoding

**Andrew H. Bond**
Department of Computer Engineering, San José State University · `agi.hpc@gmail.com`

*Working draft — revised 2026-07-10 across five rounds of external review and six added controls: a
9×9 transfer matrix, a generic-valence baseline, a 4×-larger-decoder encoder-invariance replication, a
three-probe within-corpus decomposition, a pre-registered corpus-independence intervention, and an
Article-8 contamination check. Claims are stated at their licensed strength; two are visibly retracted.
Comments welcome.*

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
**one**. Symmetrically, the four *independent*-corpus dimensions are distinct at the **corpus level**;
their concept-level distinctness is itself confounded with register shift and awaits a
register-controlled test, so we do not overclaim it either. Both controls converge with an
independent rank test in implying the nine named dimensions occupy only **~5 effective axes** (a
general commonsense-valence factor + privacy, environmental, autonomy, physical_harm), plus rights
(which does not train). A cross-encoder replication then rules out the obvious objection that this is
a capacity artifact: a **4×-larger decoder embedder (gte-Qwen2-1.5B)** reproduces the same
care↔legitimacy collapse (cross-transfer ≥ within) and the same privacy separation, so the ~5-axis
structure is a property of the **data and labels, not the encoder**. A **pre-registered experiment**
then shows the collapse is **removable**: giving care and fairness *independent* corpora (corpora and
prediction committed to git before the run, with a manipulation-check gate) makes them **decouple**
from the family — off-diagonals fall toward chance in a dose-response tracking corpus overlap — while
dimensions kept on the shared corpus stay collapsed. With the kept-shared control, this establishes
*causally* that corpus-sharing drives the collapse (corpus-level); whether independence yields
*concept*-distinct encoders is a separate claim, still gated on a register-controlled test. We therefore
argue the load-bearing variable is **corpus independence**:
cross-dataset transfer *and* concept-distinctness both require it. The originally-striking "deontic column transfers worst" pattern is **confounded**
with corpus-sharing and is offered only as a hypothesis. The one robust concept-level result is
`rights_respect`: it fails even in a stratified, class-balanced, same-jurisdiction-adjacent design,
consistent with rights being *framework-relative* (a right is what a legitimate order grants). A
within-corpus decomposition sharpens this: rights labels are *decodable* within each corpus
(0.74–0.78, though largely lexically) but fine-tuning cannot lift their cross-transfer above chance —
the only dimension of nine where it cannot — so the failure is in transfer, not learnability. The contribution is methodological: a validation regime — cross-dataset gate
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
pre-registered baseline-relative bars. (2) The finding that **corpus-level specificity requires
independent corpora**: four of our dimensions have them and are distinct at the corpus level; four
share a corpus and partially collapse into a shared moral-valence signal — shown by a 9×9 transfer
matrix and a generic-valence baseline. (3) A **causal** demonstration (§3.7) that the collapse is
*caused* by corpus-sharing and *removed* by a pre-registered data intervention, with a kept-shared
internal control. (4) A three-probe **within-corpus decomposition** (§3.5) separating "unlearnable"
from "untransferable" and showing the within signal is largely lexical. (5) A robust negative result:
`rights_respect` is the only dimension fine-tuning cannot lift above chance cross-corpus, which we
*hypothesize* reflects framework-relativity. (6) Two honest retractions — the "deontic column" pattern
(confounded with corpus-sharing) and the "coherent, not incoherent" rights claim (the within signal is
lexical).

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

**Bootstrap 95% CIs and retrain variance (computed on the deployed checkpoints).** Re-evaluating each
deployed checkpoint on its held-out pairs, with a text-level bootstrap over unique anchors (so every
point estimate sits inside its own interval): privacy 0.853 [0.815, 0.891], environmental 0.816
[0.781, 0.850], care 0.813 [0.790, 0.837], epistemic 0.811 [0.782, 0.838], fairness 0.789 [0.760,
0.815], legitimacy 0.707 [0.680, 0.734], autonomy 0.699 [0.665, 0.731], physical_harm 0.629 [0.595,
0.662] — all clear their bars — and **rights 0.502 [0.468, 0.533], straddling chance.** These re-eval
point estimates differ from the training-time scorecard above by ≤0.05 (autonomy 0.747→0.699, rights
0.475→0.502); together with the Art-8 re-measurement (0.66→0.60) that puts **retrain/re-eval variance
at σ ≈ 0.03–0.05, reported here as its own quantity.** The bootstrap CIs (~±0.03) are eval uncertainty
and do *not* include this retrain variance; the two compose. Held-out sizes: 476–1200 pairs, 238–600
unique anchors, balanced ±.

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

(Matrix cells are training-time values from a separate eval run; per-cell differences from the §3.1
scorecard — e.g. privacy 0.858 vs 0.853, epistemic 0.843 vs 0.817, care 0.825 vs 0.811 — are within
the retrain/re-eval σ ≈ 0.03–0.05 quantified in §3.1, not distinct measurements.)

Two groups fall out cleanly, split by **corpus independence, not by the taxonomy**:

- **Specific** (diagonal-dominant, off-diagonals ~0.4–0.55): **privacy, environmental, autonomy,
  physical_harm** — each with its *own* corpora. Their encoders fire only on their own dimension.
- **Collapsed**: **care, fairness, legitimacy, epistemic** — all from Social-Chem + ETHICS. The
  diagonal is not even the maximum: `care→epistemic 0.871 > care→care 0.825`; `fairness→epistemic
  0.853 > fairness→fairness 0.788`; `legitimacy→epistemic 0.822 > legitimacy→legitimacy 0.671`.
  These four fire on each other, with epistemic as the attractor — they share a corpus-specific
  "commonsense-moral valence," not four distinct concepts.

**The epistemic attractor is a clue, not a curiosity.** Everything in the family transfers to
epistemic *better than to itself* (`care→epistemic 0.871 > care→care 0.825`). The obvious explanation —
that epistemic's held-out pairs are simply the **easiest** — is **not** supported by the data: a frozen
(untrained) probe scores epistemic's pairs at 0.483, mid-pack among the nine (§3.5 measure), so raw
pair-ease does not single it out. The attractor is therefore specific to encoders trained on the
*shared family valence*: epistemic's held-out pairs are the most cleanly separated *by that shared
axis*, not by any encoder. This still carries the consequence the raw-ease explanation would have:
epistemic's standalone 0.817 PASS (§3.1) is partly **the shared valence measured on exactly the pairs
that valence separates best**, not epistemic-specific structure — corroborated from the lexical side
by its mid-pack BoW null (0.53, §3.1).

Notably, the couplings we *predicted from theory* (care↔harm; fairness↔rights) do **not** appear —
`care→harm` is 0.39 (below chance), because harm's independent corpus separates it. The actual
coupling is the shared-corpus family.

**The one large off-diagonal outside the family block — `rights→privacy` (0.66) — is corpus
contamination, and we can catch it with our own method.** ECHR **Article 8 is the
right to private life**, so the rights training corpus *contains privacy cases*: the failed rights
encoder partially learned privacy. We test it directly — drop all Art-8 cases from the rights ECHR
corpus, retrain, re-evaluate: `rights→privacy` falls from **0.60** (our re-measurement of the 0.66
cell) **to 0.51 — chance** — while the (already near-random) rights encoder is otherwise unchanged
(`rights→rights` stays ~0.50). The one thing the failed rights encoder learned that transferred was
privacy, and it came from Art-8 contamination. This is the paper's contamination thesis catching a
live case *inside our own data* — a self-application of the transfer matrix as a diagnostic.

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
  valence — *when they share a corpus* (but §3.7: giving care and fairness independent corpora
  **decouples them at the corpus level**, so this is corpus-sharing, removable, not a fixed fact about
  the four concepts).

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

**Is rights *unlearnable* or merely *untransferable*? (decomposition, per reviewer — §3.5.)** The
0.475 is cross-corpus by construction and cannot alone distinguish absent within-corpus signal from
present-but-non-transferring signal. The within-corpus probes (§3.5) show rights labels *are*
decodable within each corpus (0.74–0.78) — though **largely lexically** (a bag of words does as well
or better), so this establishes only that the labels are not random, not a coherent rights concept.
The decisive fact is that InfoNCE fine-tuning rescues cross-transfer for all eight other dimensions
but **not** rights (frozen-cross 0.45 → fine-tuned 0.475): rights is the only dimension fine-tuning
cannot lift above chance. The failure is therefore **transfer**, not label-fitting — untransferable,
not unlearnable — consistent with the framework-relativity reading of §4.

### 3.5 Within-corpus decomposition — rights is *untransferable*, and the within signal is lexical

To decompose the rights failure — absent within-corpus signal (label incoherence, "unlearnable") vs
present-but-non-transferring signal (framework-relativity, "untransferable") — we run three
frozen-encoder probes with **no fine-tuning**, so both sides of the within→cross comparison use the
same estimator and we can separate concept signal from surface artifact:
**emb-within** (frozen BGE-M3 embeddings, 5-fold logistic AUROC within each corpus), **bow-within**
(TF-IDF logistic within each corpus — is the within signal merely lexical?), and **emb-cross** (probe
trained on corpus A, tested on B — does the raw space transfer?).

| dimension | emb-within | bow-within | emb-cross (frozen) | fine-tuned cross (§3.1) |
|---|---:|---:|---:|---:|
| privacy_protection | 0.89 | 0.85 | 0.63 | 0.853 |
| societal_environmental | 0.97 | 0.95 | 0.14 | 0.817 |
| virtue_care | 0.88 | 0.88 | 0.33 | 0.811 |
| fairness_equity | 0.87 | 0.83 | 0.25 | 0.789 |
| legitimacy_trust | 0.84 | 0.83 | 0.39 | 0.708 |
| epistemic_quality | 0.90 | 0.89 | 0.22 | 0.817 |
| physical_harm | 0.84 | 0.82 | 0.57 | 0.622 |
| autonomy_respect | 0.85 | 0.86 | 0.56 | 0.747 |
| **rights_respect** | **0.74** | **0.78** | **0.45** | **0.475** |

Three findings, the second of which **corrects an overclaim in the previous draft**.

**(1) Within-corpus success is universal.** Every dimension is decodable within its corpora (0.74–0.97
mean) — including rights — by a probe that never fine-tuned. Within-corpus AUROC is deceptive: it is a
property *every* dimension has.

**(2) That within signal is largely LEXICAL, so "decodable" is not "a coherent concept."** BoW-within
tracks emb-within almost exactly everywhere, and for **rights, BoW-within (0.78) is *higher* than the
embedding probe (0.74)** — a TF-IDF bag of words recovers the within-corpus label as well as the
neural encoder. The within signal is thus mostly topic / phrasing / outcome-section surface, not a
transferable concept. **We accordingly retract the "coherent, not incoherent" phrasing of the earlier
draft:** rights labels are *decodable* within corpus, but that establishes only that they are not
random — not that a coherent rights *concept* is present.

**(3) The rights-specific fact is that fine-tuning cannot rescue its transfer.** The frozen cross-probe
sits at chance or below for *most* dimensions (environmental 0.14, epistemic 0.22, fairness 0.25, care
0.33, legitimacy 0.39, rights 0.45): the raw BGE-M3 space does not align disparate corpora — which is
exactly what InfoNCE fine-tuning is *for*. (Environmental's **0.14 is not "no transfer" but *inverted*
transfer** — well below chance — so some surface feature is *anticorrelated* with the label between
ClimateBERT and the dual-judged claims, e.g. climate vocabulary marking positives in one corpus and
greenwashing-negatives in the other; that fine-tuning lifts an actively-*misaligned* 0.14 to 0.817 is
the sharpest illustration in the paper of why raw-space lexical alignment is untrustworthy.) And fine-tuning **does** rescue it — for all eight other
dimensions cross-transfer rises to 0.62–0.85. **Rights is the only dimension whose *fine-tuned*
cross-transfer stays at chance (0.475).** The honest decomposition is therefore: rights is
**untransferable, not unlearnable** — its within-corpus signal exists but is corpus/jurisdiction-specific
surface that fine-tuning cannot align across jurisdictions. That is consistent with
framework-relativity (each legal order's rights language is its own surface). We lean on **"the only
dimension fine-tuning cannot lift above chance,"** *not* on "largest drop" — the frozen-probe drop is
a continuum (environmental drops most), so the cliff is in the *fine-tuned* column, at rights alone.

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
on care**, which we treat as a **match pending CIs** with the care encoder's own 0.793 (a 0.014 gap
on ~1k pairs — we do *not* claim it is "higher than itself") and clearly above legitimacy on itself
(0.734). The two encoders are interchangeable on care; there is no care-specific
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

### 3.7 Independent corpora separate the collapsed family — the pre-registered test, confirmed

The central open question (does the collapse come from the *data* or the *moral space*?) was
pre-registered as a falsifiable prediction (`experiments/prereg_independent_corpora.md`, committed to
git **before** training). We give **care** and **fairness** independent second corpora — Moral-Stories
care-action narratives for care, Measuring Hate Speech for fairness — replacing the shared ETHICS
corpus while keeping Social-Chem, and re-run the 4×4 family cross-transfer. **Legitimacy and epistemic
keep the original Social-Chem↔ETHICS pairing as an internal control.**

**Manipulation check (committed gate).** Both re-corpused encoders first pass their
*own* cross-dataset gate on the new pairing — care_v2 `structure_auroc` 0.839 (margin +0.26 over its
nulls), fairness_v2 0.912 (margin +0.34) — so a null result would be interpretable, not a
corpus-quality artifact. Both pass; the transfer prediction is scorable.

4×4 cross-transfer (BGE-M3, rows = encoder, cols = eval; diagonal = within):

```
enc \ eval     care_v2   fair_v2    legit     epis
care_v2        0.839    0.701    0.432    0.423
fair_v2        0.658    0.912    0.438    0.493
legit          0.610    0.563    0.707    0.789
epis           0.553    0.563    0.693    0.811
```

| pair | shared corpora | symmetric cross | min within | gap | verdict |
|---|---|---:|---:|---:|---|
| **legit ↔ epis** *(control)* | Social-Chem **+** ETHICS | 0.741 | 0.707 | **−0.03** | **COLLAPSED** |
| care_v2 ↔ fair_v2 | Social-Chem only | 0.680 | 0.839 | +0.16 | decoupled (weakest) |
| fair_v2 ↔ epis | Social-Chem only | 0.528 | 0.811 | +0.28 | decoupled |
| care_v2 ↔ legit | Social-Chem only | 0.521 | 0.707 | +0.19 | decoupled |
| fair_v2 ↔ legit | Social-Chem only | 0.500 | 0.707 | +0.21 | decoupled |
| care_v2 ↔ epis | Social-Chem only | 0.488 | 0.811 | +0.32 | decoupled (strongest) |

**The prediction is confirmed — as *corpus-level decoupling*, and it comes as a dose-response.**
Re-corpusing works: **fairness_v2** (the clean flagship, fully-independent Measuring Hate Speech) and
care_v2 both fall away from the shared-corpus family, while the internal control (legit↔epis, kept on
*both* shared corpora) stays collapsed (−0.03) — the same run, only the corpus changed. But the effect
is a **gradient that tracks corpus overlap**, not "decoupling from everything":

- share **both** corpora (Social-Chem + ETHICS): `legit↔epis` = 0.74 (collapsed).
- `care_v2↔fair_v2` = **0.68** — the largest off-diagonal in the new matrix — precisely because these
  two *still share Social-Chem* (only ETHICS was replaced); it is also the smallest gap (+0.16).
- other pairs sharing Social-Chem only: 0.49–0.53.

It is not monotone in corpus *count* (`care_v2↔legit` also share Social-Chem yet sit at 0.52 vs
`care_v2↔fair_v2`'s 0.68), so what matters is the quantity of shared *pairs/overlap*, not corpus
identity alone. A gradient in cross-transfer that tracks degree of overlap is a stronger, more honest
headline than "decoupled," and it pre-registers the obvious follow-up: drop Social-Chem from one of the
two re-corpused feeders and predict the 0.68 falls.

**What this does and does not establish.** *Causally* — by intervention **plus** the kept-shared
control — it establishes that **the collapse is caused by corpus-sharing and is removable by a data
intervention.** It does **not** establish that independence produced *concept*-distinct encoders:
re-corpusing changed two things at once — the training signal *and the register of the eval pairs* — so
`care→epistemic` falling 0.87→0.42 is equally consistent with "care_v2 learned the care *concept*" and
with "care_v2 learned the Moral-Stories/MHS *register* and can no longer read Social-Chem↔ETHICS pairs"
— the identical register-shift confound we preserve for privacy (§3.3, §6). We therefore read the
matrix as **corpus-level decoupling**, not concept-level distinctness; the latter is gated on the
register-controlled test (open question 2), which this experiment does not discharge. Half the claim is
now causal (shared corpus → collapse, by intervention+control); the other half (independence → distinct
*concepts*) is not yet.

*Caveats (pre-committed).* Care's corpus (Moral-Stories) is *partial*-independence — its norms are a
Social-Chem subset, so only its narrative text is independent — which is why **fairness/MHS is the clean
flagship**; both arms agree. Legitimacy and epistemic stay collapsed only for want of clean independent
signed corpora (legitimacy especially — none redistributable exists, itself a finding).

## 4. Interpretation

**The primary finding is methodological, and now has a causal leg: corpus-sharing *causes* the
collapse, and a data intervention removes it — at the corpus level.** Cross-dataset AUROC alone (§3.1)
passed all four shared-corpus dimensions; only the transfer matrix revealed they had learned a shared
valence; and the pre-registered re-corpusing (§3.7) showed the collapse *reverses* when the shared
corpus is replaced, with the kept-shared control (legit↔epis) ruling out alternative causes. This
establishes corpus independence as **necessary and sufficient for corpus-level specificity**. It does
*not* yet establish concept-level distinctness — re-corpusing also moved the eval register, the same
confound we hold against the privacy off-diagonals (§3.3) — so concept-level sufficiency remains gated
on the register-controlled test (open question 2). Any framework that scores multiple moral dimensions
from overlapping corpora risks measuring one thing under many names — and can remove that specific
artifact by giving each dimension its own corpus.

**The rights failure is our most robust concept-level result**, now supported by the within-corpus
decomposition (§3.5): rights is the only dimension whose cross-transfer fine-tuning cannot lift above
chance, so its failure is a *transfer* failure, not label-fitting (its within-corpus signal is
decodable but largely lexical). We read this as framework-relativity: a right is an entitlement conferred by a *legitimate* order, so specific rights
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
1. **Give every dimension independent corpora** before claiming it is a distinct concept — and doing
   so *works*: re-corpusing care and fairness (§3.7) decoupled them from the family at the corpus level
   (pre-registered, confirmed). Legitimacy and epistemic still need clean independent signed corpora (legitimacy has
   none yet — a gap, not a refutation).
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
- **Confidence intervals** (computed, §3.1): text-level bootstrap over unique anchors; all eight
  passing dimensions' CIs clear their bars, and rights straddles chance [0.470, 0.533]. Sub-0.02 gaps
  (e.g. the §3.6 0.807-vs-0.793) are reported as *matches*, consistent with these CI widths (~±0.03).
- **Retrain / re-eval variance** is now measured three ways — Art-8 `rights→privacy` 0.66→0.60,
  autonomy 0.747→0.699, rights 0.475→0.502 — giving **σ ≈ 0.03–0.05** (§3.1). Applied to §3.7, the two
  *marginal* verdicts — the weakest decoupling (`care_v2↔fair_v2` gap +0.16) and the kept-shared
  control (−0.03) — sit within ~1–2 σ of their thresholds; the large cells (+0.28 to +0.32, and the
  0.87→0.42 before/after) are safe at many σ. A 3-seed replication of the marginal cell is **in
  progress** (`scripts/run_v2_seeds.py`; seed 0 reproduces +0.16) and its mean ± sd will land in §3.7;
  meanwhile "confirmed" rests on the large cells plus the intervention-with-control logic, not on +0.16.
- **Dual-LLM-judge labels** (privacy, environmental, rights) may share bias; human audit needed
  (planned: ~100–200 items/slice, report human↔LLM agreement, not only LLM↔LLM).
- **Rights confounds**: the stratified US-force slice was small (369); scarcity is not fully ruled
  out. Within-jurisdiction pairing and article-specific ECHR labels are the cleanest next tests.
- **Pseudo-replication**: AUROC is over structural pairs, so CIs resample **unique anchors**, not
  pairs (§3.1); per-dimension counts (pairs, unique anchors, ±) are now reported there. Held-out sizes
  range 476–1200 pairs / 238–600 anchors per dimension.
- **Single encoder/objective** (BGE-M3 + InfoNCE): partially addressed (§3.6). A 4×-larger decoder
  (gte-Qwen2-1.5B) reproduces the care↔legitimacy collapse and the privacy separation on a 3-dim
  slice, so the *collapse* is not a BGE-M3 capacity artifact; a full-nine, bidirectional-mode
  replication is still outstanding.
- **Keyword-based** foundation and stratum filters are coarse.

## 7. Open questions and a pre-registered prediction
1. Do the four shared-corpus dimensions become distinct when given *independent* corpora — or is
   the ~5-dimensional collapse a property of the moral space rather than the data? *(Answered, §3.7:
   they decouple (corpus-level) — care and fairness given independent corpora fall away from the family
   while the shared-corpus control stays collapsed. The collapse is data, not moral space.)* Remaining: repeat
   for legitimacy and epistemic once clean independent signed corpora are found.
2. Does the concept-distinctness of the four independent-corpus dimensions survive a
   **register-controlled test** (concept fixed, register moved)? Diagonal-dominance that survives
   register control is concept-specificity; diagonal-dominance that vanishes was domain shift.
3. Is rights *unlearnable* or merely *untransferable*? *(Answered, §3.5: untransferable — rights is the
   only dimension fine-tuning cannot lift above chance cross-corpus; its within signal is decodable
   but largely lexical, so the failure is transfer, not learnability.)* Still open: does a
   conduct-level rights label ("was someone brutalized/detained/silenced") transfer where case-law
   does not?
4. Is the (confounded) column pattern real once corpus-sharing is removed?
5. Does a stronger encoder separate the shared-corpus family that BGE-M3 collapses? *(Answered, §3.6:
   no — gte-Qwen2-1.5B reproduces the care↔legitimacy collapse. Open for the full nine dimensions in
   bidirectional mode.)*

**Pre-registered prediction — CONFIRMED (§3.7).** We gave **virtue_care** and **fairness_equity**
independent second corpora (Moral-Stories care actions; Measuring Hate Speech), replacing the shared
Social-Chem↔ETHICS pairing, and re-ran the 4×4 with legitimacy/epistemic kept shared as a control.
The prediction — off-diagonals drop ≥0.10 below within, diagonal-dominance emerges — **held** (gaps
+0.16 to +0.32), with a manipulation-check gate ensuring both re-corpused encoders trained (care_v2
0.839, fairness_v2 0.912). The falsifier (family still cross-contaminates) **did not occur**; the
shared-corpus control (legit↔epis) stayed collapsed (−0.03). Prediction, corpora, and gate were all
committed before training; see `experiments/prereg_independent_corpora.md` and
`experiments/independent_corpora_results.txt`.

## 8. Conclusion

Cross-dataset validation rescued eight of nine encoders from the illusion of within-corpus success —
but a transfer matrix then showed that four of the eight had learned a *shared corpus*, not a
*distinct concept*, and a **pre-registered intervention (§3.7) then made that causal**: replacing the
shared corpus for care and fairness decoupled them from the family while a kept-shared control stayed
collapsed — the same run, only the corpus changed. The durable lesson is methodological: **to show an
encoder learned the moral dimension you named it after, it is not enough to beat a lexical baseline
across two corpora; the two corpora must be independent, and you must show the encoder does not fire on
its neighbors** — and even then you have shown corpus-level, not concept-level, distinctness until a
register-controlled test separates the two. `rights_respect` remains the one dimension fine-tuning
cannot lift above chance, which we offer — with appropriate caution — as possible evidence that some
moral concepts are framework-relative and may not admit a single, jurisdiction-independent encoder at
all.

## Reproducibility & acknowledgements

All in `github.com/ahb-sjsu/xbse`: the cross-dataset harness (`train_adv.py`, `validate.py`,
`JointPairSource`), the bag-of-words control (`baselines.py`), pre-registered bars (`bar.py`,
`joint_builders.py`), and one script per result — the N×N transfer matrix (`scripts/cross_transfer.py`,
which regenerates the 9×9 BGE-M3 matrix §3.2 and the gte-Qwen2 3×3 §3.6), the within-corpus three-probe
decomposition (`scripts/within_corpus.py`, §3.5), the bootstrap CIs + pair-difficulty + counts
(`scripts/pair_stats.py`, §3.1), the family-pool baseline (`scripts/family_pool.py`, §3.3), the
pre-registered intervention (`scripts/run_v2_experiment.py`, §3.7), its 3-seed replication
(`scripts/run_v2_seeds.py`), and the Article-8 contamination check (`build_rights_no_art8`, §3.2).
Pre-registrations (committed **before** their runs, timestamps precede the training commits):
`experiments/prereg_independent_corpora.md` and `experiments/prereg_socialchem_drop.md`; per-run
outputs in `experiments/*_results.txt`; corpora in `experiments/data_sourcing_plan.md`. The
experimental pipeline was executed with substantial AI-assisted automation; results are reproducible
from the committed code and documented corpora.
Circulated for critique — the author especially welcomes attempts to falsify the corpus-independence
claim and the rights framework-relativity hypothesis, and thanks the reviewer whose insistence on a
dimension-specificity test produced this paper's central result.
