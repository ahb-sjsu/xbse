# The Moral Spectrum Analyzer: A Pre-Registered Instrument for Measuring, Validating, and Discovering Moral Dimensions in Language

**Andrew H. Bond**
San José State University · ORCID 0009-0003-2599-6158 · andrew.bond@sjsu.edu

*Draft — comprehensive first pass, 2026-07-13 (numbers verified against source artifacts; see Appendix B). Companion to "When Do Moral-Dimension Encoders Learn Distinct Concepts?" (the encoder-level corpus-independence result) and the DEME keystone architecture paper.*

---

## Abstract

Systems that moderate content or evaluate agents increasingly reason about *moral* properties of text — harm, fairness, autonomy, honesty — yet the scores they emit are typically **asserted rather than measured**: a classifier is trained on one corpus, reports a high in-domain AUROC, and is deployed as if it had captured a construct. We show that this practice is **unsound as measurement**, and we offer an instrument that replaces assertion with measurement.

We define the **MoralVector**, a per-item representation with one validated channel per moral dimension — nine canonical dimensions on a frozen 3×3 ontology plus a small number of validated *extension channels* — each cell carrying a signed valence, a confidence, and machine-checkable validation provenance. We then define the **Moral Spectrum Analyzer**, the measurement apparatus that produces and audits the vector: (i) a **pre-registered cross-dataset gate** in which a dimension is "validated" only if a held-out encoder beats *both* an untrained-encoder null and a bag-of-words null by a fixed margin, with all nulls frozen before training; (ii) **joint cross-corpus training** that removes the single-corpus surface shortcut which we show inflates in-domain numbers by up to 0.44 AUROC (autonomy) — but as little as 0.01 (privacy), a heterogeneity that is itself informative; (iii) an **invariance layer** with a measured decision-stability criterion; (iv) a **discovery loop** that surfaces missing dimensions and an **admission criterion with teeth** — over the program's history it has validated one discovered dimension, retracted another, and declined a third as non-moral policy.

Our central empirical result is a **dissociation between moral valence and moral identity**. Using a domain-adversarial ablation and an independent geometric ablation, we find that a dimension's *valence* (is it upheld or violated?) is register-invariant — it transfers across corpora with or without adversarial de-confounding — whereas a dimension's *identity* (which foundation is engaged?) is, for several foundations, register-bound: it occupies the *same* angular subspace as corpus/register and cannot be separated from it by adversarial training or by principal-component removal. Three complementary probes converge on this conclusion — two share a training pipeline and differ only in a domain adversary, the third is a fully independent geometric ablation on frozen embeddings. The finding has a direct engineering consequence — presence/identity channels must be built with a specific (non-adversarial) configuration — and a methodological one: **cross-dataset transfer, not in-domain accuracy, is the minimal evidence that a moral-dimension probe measures a construct at all.**

---

## 1. Introduction

A content-moderation pipeline that "detects toxicity," an agent evaluator that "scores honesty," a decision-support tool that "weighs autonomy against welfare" — each rests on a measurement claim: that some learned function maps text to a morally meaningful quantity. In practice these claims are rarely tested as measurement claims. The field-standard evidence is an in-domain held-out score, and an in-domain score answers a different question ("can the model separate *this corpus's* labels?") than the one that matters ("does the model track *the construct* across the distributions it will meet in deployment?").

This paper argues, and demonstrates, that the gap between those two questions is large enough to invalidate naive practice, and it supplies an instrument built around the gap.

**Contributions.**

1. **The MoralVector construct (§2).** A typed, versioned representation: nine canonical dimensions on a frozen "Nine Dimensions of Ethical Assessment" 3×3 matrix, plus validated extension channels, each cell a signed valence with confidence and validation provenance. A single source of truth defines the channel ontology and every consumer references it.

2. **The Moral Spectrum Analyzer (§3–4).** A measurement apparatus, not a model: a pre-registered cross-dataset gate with frozen nulls, joint cross-corpus training, an invariance layer with a measured decision-stability criterion, a learned contraction that turns validated dimensions into calibrated moderation decisions, and a discovery loop with a falsifiable admission criterion.

3. **A body of results that the instrument makes reportable (§5),** including several honest negatives that naive practice would never surface: the collapse of in-domain scores under cross-dataset test; a one-dominant-factor structure in the vector; strong canonical-layer invariance but weak raw-score invariance; and the valence/identity dissociation that is our headline.

4. **A reusable methodology (§6–8).** We frame the cross-dataset gate as a *measurement-invariance test* in the psychometric tradition, with a falsification bar and documented failures, and we argue that this is the difference between engineering a moral classifier and measuring a moral construct.

**Scope of the claim.** The instrument measures **corpus-encoded, predominantly Anglophone moral judgments** — the moral distinctions that the training and evaluation corpora (Social-Chemistry, ETHICS, Reddit, Twitter) encode — not morality-in-itself. Every validated channel is validated *for the registers and cultures its feeders were trained and tested on*, and the honest reading of a passing gate is "this probe tracks *this construct as these corpora operationalize it*, across the distributions we tested," not "this probe measures the moral property." The name "Moral Spectrum Analyzer" denotes the apparatus, not a claim to have instrumented morality; §7 states where the instrument should abstain.

We deliberately foreground what the instrument *rejects*. A measurement device earns trust by what it refuses to certify, and much of what follows is a record of refusals.

---

## 2. The MoralVector

### 2.1 The channel ontology

The MoralVector's canonical basis is the **DEME-9**, nine moral dimensions arranged as a 3×3 matrix of `{Individual, Relational, Collective}` scope × `{What Matters (values), Who Decides (deontic), What We Know (epistemic)}` mode:

| k | dimension | scope / mode |
|---|---|---|
| 0 | `physical_harm` | Relational / Who-Decides (consequences & welfare) |
| 1 | `rights_respect` | Individual / Who-Decides |
| 2 | `fairness_equity` | Collective / What-Matters |
| 3 | `autonomy_respect` | Individual / What-Matters |
| 4 | `privacy_protection` | Individual / What-We-Know |
| 5 | `societal_environmental` | Collective / What-We-Know |
| 6 | `virtue_care` | Relational / What-Matters |
| 7 | `legitimacy_trust` | Collective / Who-Decides |
| 8 | `epistemic_quality` | Relational / What-We-Know |

This basis is **frozen**: the nine names and their order are the tensor's positionally-indexed `k` axis and are duplicated, by design and under a drift-guard test, across the packages that consume them. Freezing the basis is what lets a rank-1..6 moral tensor be indexed positionally without silent misalignment.

**The frozen basis is an engineering interface, not a claim about dimensionality.** It fixes positional indexing so consumers can address channels without misalignment; it does *not* assert that morality-in-text has nine independent factors. The *measured* structure of the vector is the bifactor of §5.3 — one dominant general factor plus roughly four to five specifics (effective rank ≈ 5.7) — and the interface and the measured structure are deliberately decoupled. We posit nine channels for engineering, measure the true structure, and report the gap; the name "Nine Dimensions" labels the interface, not a discovered factor count.

Each dimension names a distinct moral concern the vector scores independently. The construct each channel is trained to measure — the thing a positive item *upholds* and a negative item *violates* — is:

| dimension | the moral concern it scores |
|---|---|
| `physical_harm` | bodily safety and material welfare: is a person shielded from, or exposed to, injury, violence, or physical damage |
| `rights_respect` | individual entitlements and liberties: consent, due process, freedom from coercion — honored vs infringed |
| `fairness_equity` | proportional treatment of comparable parties: equal treatment, distributive justice, absence of bias |
| `autonomy_respect` | a person's capacity for self-determined choice: informed agency vs manipulation, deception-into-action, paternalistic override |
| `privacy_protection` | personal information and private space: safeguarded vs surveilled, disclosed, or misused |
| `societal_environmental` | shared collective and ecological goods: public institutions, common resources, the environment — sustained vs degraded |
| `virtue_care` | compassion and responsiveness in relationships: attentiveness to others' needs vs cruelty or neglect |
| `legitimacy_trust` | warranted, accountable exercise of authority: legitimate governance and earned trust vs abuse of power |
| `epistemic_quality` | truthfulness and honest communication: evidence, accuracy, good-faith reasoning vs deception, misinformation, distortion |

A cell's **sign** records whether that concern is upheld (`+`) or violated (`−`); its **magnitude** records whether the concern is engaged at all (§2.2).

New moral foundations therefore do **not** widen the canonical basis. Following an established metadata-channel precedent, validated additions ride as **extension channels** outside the frozen nine. At the time of writing the extension channels are `identity_attack` (a discovered dimension, §5.6) and the two Moral Foundations Theory "binding" foundations `purity` and `loyalty` (validated in §5.6–5.7):

| extension channel | the moral concern it scores |
|---|---|
| `identity_attack` | dehumanization or demeaning of a person *for who they are* — a protected-class or group identity — as distinct from generic toxicity, which targets what someone *says* or *does* |
| `purity` | the sanctity/degradation intuition: bodily and spiritual purity, disgust, contamination, desecration of what is held sacred |
| `loyalty` | the in-group loyalty/betrayal intuition: solidarity and fidelity to one's group vs treachery and abandonment |

The full ordered vocabulary is the nine canonical dimensions followed by the extension channels, defined once in a single source-of-truth module and imported everywhere else.

![The MoralVector at a glance. The nine canonical dimensions sit on the frozen 3×3 `scope × mode` ontology; the three validated extension channels ride outside the frozen nine (metadata, not the `k`-axis). Validated channels are shaded by their *measured* general-factor loading β (§5.3): blue channels (β ≥ 0.8) share the single general moral factor G, vermillion channels (β < 0.8) occupy a separable subspace. The two dimensions without a validated second corpus — `rights_respect` and `societal_environmental` — are grayed as **pending**, not presented as confident channels. The two-color split is the paper's headline: valence transfers across register while identity, for the blue family, collapses onto G.](figs/fig0_moralvector.pdf){width=95%}

### 2.2 The cell

Each cell of the vector is a **signed valence** in `[-1, +1]` — `-1` violated, `0` not-engaged, `+1` upheld — accompanied by `confidence`, `uncertainty`, `direction`, `source_spans`, and an `explanation`. Two projections of the cell matter independently and recur throughout the paper: its **sign** (valence: upheld vs violated) and its **magnitude away from zero** (presence: is the dimension engaged at all?). A central empirical result (§5.7) is that these two projections have *different measurement properties* — valence transfers across register; presence, for several dimensions, does not.

Every cell also carries **validation provenance**: the identifier and checkpoint hash of the feeder that produced it, and the pre-registered bar that feeder cleared. A downstream consumer can refuse to construct a scorer from a checkpoint whose hash does not match a passing validation record. Measurement provenance is thus a first-class, machine-checkable property of the vector, not documentation.

---

## 3. The Moral Spectrum Analyzer

The Analyzer is the apparatus that produces, validates, and audits the vector. It has five parts.

**(1) Feeders.** Each dimension is measured by a domain-specific sentence-embedding encoder [@reimers2019] (a `*-BSE`) trained to be *invariant to surface* and *sensitive to that dimension's structure*. A feeder is not shipped because it trains; it is shipped only if it clears the gate.

**(2) The pre-registered cross-dataset gate (§4.1).** The non-negotiable validation mechanism. A dimension is `VALIDATED` iff a cross-dataset held-out AUROC beats *both* an untrained-encoder null *and* a TF-IDF bag-of-words null by ≥ 0.10, with a fuzz ratio > 1, where all nulls are measured on the untrained model and *frozen into the bar before training*.

**(3) Joint cross-corpus training (§4.2).** The load-bearing mechanism that lets a feeder clear the gate: train each dimension on ≥ 2 independent corpora simultaneously, drawing same-sign positives *across* corpora, so the encoder cannot separate pairs by any single corpus's surface.

**(4) The invariance layer (§4.3).** A canonicalization stage plus a *measured* decision-stability criterion (θ_d), so that the vector a decision consumes is a property of the input's meaning, not its surface form or language.

**(5) The discovery loop and admission criterion (§4.4).** A residual-analysis procedure that surfaces candidate missing dimensions, and a four-way admission criterion (validate / decline-as-policy / decline-as-duplicate / retract) that decides — falsifiably — whether a candidate joins the vector.

The instrument's governing discipline is **pre-registration with frozen nulls and documented failure.** Bars are derived from corpus properties (label-noise ceiling, baseline lift) and registered before training; they are never loosened after a run; and a failed gate is recorded, not hidden.

---

## 4. Methods

### 4.1 The gate: validation as a bar, not a number

Every dimension clears the *same gate mechanism* against a *per-dimension, pre-registered bar* that carries its own derivation. The shared, non-negotiable elements are:

- **Metrics.** (a) *Structure AUROC*: cross-dataset held-out AUROC separating same- vs different-structure pairs. (b) *Fuzz ratio*: how much more the embedding moves for a structural change than for a surface paraphrase; must exceed 1 (an earlier moral encoder had fuzz ≈ 0.2 — surface perturbations moved it *more* than moral ones; that must fail). (c) *Surface invariance*: a reported diagnostic.
- **Nulls, frozen before training.** The untrained-encoder AUROC and the TF-IDF bag-of-words AUROC are measured on the *untrained* model on the *same held-out pairs* and frozen into the bar. Beating BoW is the null that matters to the moral-lexicon literature: it certifies the win is not a keyword artifact.
- **The bar.** `VALIDATED` iff structure AUROC ≥ max(untrained, BoW) + 0.10 and fuzz > 1. Policy: baseline-relative, margin 0.10.
- **Hard stop.** If the gate fails, no downstream tool is built and no claim is made.

The cross-dataset construction is what makes the AUROC honest: every held-out structural pair places the anchor in one corpus and both comparisons in the *other*, so the structure AUROC *is* the cross-generalization number.

### 4.2 Joint cross-corpus training

A single-corpus feeder can achieve a high in-domain AUROC by separating that corpus's pairs on surface. To remove the shortcut, a dimension is assembled from a list of `(corpus, rows)` domains, and a training positive is same-valence but drawn from a *different* corpus than its anchor. To pull an anchor from corpus A next to a same-sign item from corpus B, the encoder cannot use A's surface — only the shared moral structure. This mechanism, not the domain-adversarial term [@ganin2016], is the load-bearing fix (§5.2, and the companion encoder paper).

### 4.3 The invariance layer

Invariance is a property of the **canonicalization layer**, not of raw feeder scores. We measure two things. (a) A **canonical-invariance index**: the cross-lingual / cross-paraphrase agreement of the canonicalized representation. (b) A **decision-stability criterion θ_d**: the residual drift in the *decision* after equivalence-class averaging — generate a class of meaning-preserving paraphrases of an item, average per-dimension scores over the class, then decide. θ_d is pre-registered at ≤ 0.5, and the deployment form generates the class at inference time (with a documented failure mode: a paraphrase generator that refuses harmful content yields a singleton class and no averaging — the invariance mechanism degrades on exactly the content that matters most, which we quantify).

### 4.4 The discovery loop

Beyond validating pre-specified dimensions, the Analyzer looks for dimensions the taxonomy is missing. A residual-analysis pass over labeled corpora surfaces a candidate; the candidate is then run through the *same* gate on two independent corpora. The admission criterion is four-way and falsifiable: **validate** (clears the gate on independent corpora → new channel), **decline-as-policy** (real signal but not a moral valence — route as a platform-configurable policy channel), **decline-as-duplicate** (residualizes away against an existing axis + G), or **retract** (fails after a fairer resample). §5.6 shows *three* of the four outcomes have actually occurred — validate, decline-as-policy, and retract; decline-as-duplicate is defined and available but has not yet been triggered — which is the evidence that the criterion has teeth rather than being a rubber stamp.

---

## 5. Results

### 5.1 In-domain accuracy is not evidence of a construct

Single-corpus feeders scored in-domain AUROC 0.75–0.955. On a held-out *second, independent* corpus of the same dimension, those numbers collapsed to ≈ 0.47–0.55 — chance — for **every** dimension. The most vivid case: `autonomy_respect` read **0.955 in-domain but 0.518 cross-dataset** before joint training — a pure surface artifact — and joint cross-corpus training recovered it to **0.747**. This is the paper's first and most consequential negative result: **an in-domain moral-dimension AUROC carries essentially no information about whether the probe measures the construct.** Cross-dataset test is the minimal bar.

After joint training, seven of nine canonical dimensions clear the gate with honest cross-dataset AUROCs:

| dimension | in-domain | **cross-dataset** | null |
|---|---:|---:|---:|
| `privacy_protection` | 0.865 | **0.853** | 0.551 |
| `epistemic_quality` | 0.90 | **0.817** | 0.483 |
| `virtue_care` | 0.85 | **0.811** | 0.469 |
| `fairness_equity` | 0.87 | **0.789** | 0.474 |
| `autonomy_respect` | 0.955 | **0.747** | 0.515 |
| `legitimacy_trust` | 0.80 | **0.708** | 0.523 |
| `physical_harm` | 0.88 | **0.622** | 0.499 |
| `rights_respect` | 0.75 | *pending* | — |
| `societal_environmental` | 0.77 | *pending* | — |

*Second-corpus pairings (each dimension is trained jointly on two independent corpora):* privacy = privacy-RoTs × AITA; epistemic = Social-Chem × ETHICS (honesty); care = Social-Chem care-harm × ETHICS; fairness = Social-Chem fairness × ETHICS; autonomy = dark-pattern × MentalManip; legitimacy = Social-Chem authority × ETHICS; physical-harm = BeaverTails × ETHICS (harm).

`physical_harm` is weakest (0.622) — the BeaverTails-vs-ETHICS genre gap is the widest — and two dimensions await a validated second corpus and are *not* claimed. `rights_respect` is additionally framework-relative and runs as a hand-specified rule channel rather than a learned feeder.

That `rights_respect` — arguably the most governance-critical concept — *cannot* be learned corpus-independently is a result, not a gap, and it names a design principle the architecture already follows: **learned channels for what transfers across corpora; specified rules for what does not; and the gate is the arbiter of which is which.** An instrument that can demote a concept from "learned" to "legislated" on measured evidence is doing its job, not failing.

*Reconciliation with the companion encoder paper.* The companion reports **eight** of nine passing; this paper reports **seven**. The two are the same result at different audit thresholds: both find `rights_respect` unlearnable corpus-independently, and the single swing dimension is `societal_environmental`, which cleared under dual-LLM-judge labels (the companion's count) but is held **pending** here until it clears on a human-audited second corpus. No dimension is counted as passing in one paper and failing in the other; the difference is exactly whether `societal_environmental` is admitted on machine-judged labels.

![In-domain AUROC overstates cross-dataset transfer. Grey bars: in-domain (inflated) AUROC; blue bars: honest cross-dataset AUROC after joint cross-corpus training. `autonomy_respect` shows the widest gap (0.95 → 0.75); before joint training its cross-dataset score was 0.52 — chance. The dashed line marks chance (0.5).](figs/fig1_crossdataset.pdf){width=88%}

### 5.2 Corpus independence is the load-bearing variable

The companion encoder paper isolates *why* joint training works: it is the **cross-corpus same-sign positives**, not the gradient-reversal domain adversary, that de-confounds. The adversary only helps when the two corpora are already surface-similar. This result — corpus independence as the load-bearing variable — recurs as the mechanism behind §5.7's dissociation.

### 5.3 Structure of the vector: one dominant factor plus specifics

A bifactor analysis of the vector's axes gives an **effective rank of ≈ 5.7 of 9** (`prereg_bifactor_readout.md`): one dominant general factor plus roughly four to five specifics. We trained and gated a **general-valence factor G** as an explicit channel (cross-dataset AUROC **0.856 ± 0.008**, 3 seeds; margin 0.34 over the frozen nulls, fuzz > 100), replacing an ungated PCA proxy, and residualized the specifics against it (per-item `r_k = s_k − β_k·G`, with β fit on a disjoint calibration half). The axes split into two regimes by their loading β on G:

- **Collapses onto G (β 0.96–0.99):** care 0.99, fairness 0.98, legitimacy 0.98, and — §5.7 — loyalty 0.98, purity 0.96 (epistemic is intermediate, 0.83). These are the shared-corpus (Social-Chem × ETHICS) family.
- **Separable from G (β ≤ 0.36):** autonomy 0.02, societal-environmental −0.01, privacy 0.25, physical-harm 0.35, identity_attack 0.36 — the independent-corpus axes.

An overlap diagnostic shows the collapse is a *real generalizing shared factor*, not corpus memorization: G predicts the family on text it never trained on almost as well as on seen text (care 0.99 seen vs 0.99 unseen; family gpred 0.98–1.00), while a sensitivity control (physical-harm, gpred 0.82) drops enough to prove the test discriminates. Crucially, the family residuals after G is removed stay *above chance* (**0.59–0.77**) — the family axes are not hollow. We therefore represent the vector in **bifactor form**: G on channel 0 plus a small residual per axis, keeping every dimension — no coverage lost, and no hollow axis displayed.

The same residualization, applied to the full cross-axis transfer matrix rather than to single axes, is the sharpest test of the *core-agreement* claim: after G is removed, does *same-dimension* agreement still exceed *cross-dimension* agreement on the residuals? It does, but modestly. Mean same-dimension (diagonal) transfer falls from **0.95 to 0.78** once G is residualized out, and cross-dimension (off-diagonal) transfer from **0.73 to 0.53** — leaving a **0.25** same-vs-cross gap that is real but far smaller than the raw matrix suggests, and only **4 of 11** axes retain clean diagonal dominance (per-axis β's are not individually stable, `bifactor_A2_result.json` `P1`). The reading is deliberately deflationary: the residualized core is genuine but thin — most of what looks like instrument agreement *is* the shared general factor, with a minority of axes carrying dimension-specific structure that survives its removal. This is the same effective-rank-≈ 5.7 finding stated as a commensurability result: not nine independent cross-instrument agreements, but one strong shared factor plus a few specifics.

![Two regimes in the vector. Loading β of each axis on the general factor G. A shared-corpus family (blue, β ≥ 0.8) collapses onto G; independent-corpus axes (vermillion, β < 0.8) stay separable. The split tracks corpus provenance, not the axis's nominal content. (Unvalidated feeders, e.g. `societal_environmental`, are included here only to report the factor estimate, not as validated channels — see §5.1.)](figs/fig3_bifactor.pdf){width=88%}

The honesty consequence is stated as a schema warning: a readout that displays ten channels while ~six independent factors are present risks over-claiming resolution, and the schema records both a discriminative margin *and* an invariance fuzz per channel so a consumer can require either or both.

### 5.4 Invariance: strong at the canonical layer, weak in raw scores

Raw feeder scores are only weakly invariant to paraphrase (ratio ≈ 0.60) — the trained axes *amplify* embedding drift. Canonical-layer invariance is strong: a same-language index **0.75 [0.64, 0.84]**, and **cross-lingual at scale 0.721 (BGE-M3) / 0.804 (LaBSE [@feng2022])** across Spanish, Arabic, Chinese, Hindi, and Swahili (four scripts), with **harmful ≈ benign** (BGE-M3 0.704 on harmful items vs 0.739 on benign — invariance is not a benign-only artifact); LaBSE wins cross-lingual, BGE-M3 is chosen as the same-language canonicalizer. Equivalence-class averaging as the decision-stability mechanism reaches **θ_d 0.42** as selection evidence (leave-scenario-out proxy) and, at scale on 60 held-out items with LLM-generated paraphrase classes, halves raw drift **0.407 → 0.219** — meeting the pre-registered ≤ 0.5 bar — with a red-team back-translation (NLLB [@nllb2022], six pivots) confirming θ_d **0.301 on harmful content** (toxicity ≥ 0.7). We report two honest caveats baked into the record: a **24.0% generator-refusal rate** (gpt-oss refuses 19/79 toxic items → singleton class → no averaging; the "attack-or-starve" hole, quantified — the NLLB red-team paraphraser has 0% refusal and closes it), and that the at-scale paraphrases are natural rather than adversarial, leaving adversarial robustness as a separate containment item.

### 5.5 From measurement to moderation: the learned contraction

A learned logistic contraction over the validated feeders turns the vector into calibrated moderation decisions on covered categories (leakage-controlled out-of-fold **AUROC 0.863 / F1 0.762**, N = 1523, 5-fold), a **+0.084 lift** over the 8-feeder baseline (0.779, computed on the identical disjoint sample), moving the pipeline from escalate-everything to **moderating 51.0%** (≈ 80% remove-precision, 95% allow-precision; escalate below bar). Leakage was measured, not assumed: the dominant feature's overlap with the fit corpus was quantified (**77/1600 = 4.8%**), the 77 rows excluded by content hash, and the contraction refit on the 1523 disjoint rows; the numbers barely moved (0.872 → 0.863), showing the lift is real signal, not the feeder scoring its own training data. An honest false positive (harsh-criticism → remove) is kept in the record rather than tuned away against the demo scenarios.

### 5.6 The discovery loop has teeth: validated, retracted, and declined

The admission criterion has produced three of its four possible outcomes, which is the evidence it is not a rubber stamp (the fourth, decline-as-duplicate, is defined and available but has not yet been triggered by a candidate):

- **Validated — `identity_attack`.** Surfaced as a labeled blind spot, built on two independent corpora (Jigsaw civil-comments identity-attack × Berkeley Measuring-Hate-Speech), and gated at cross-dataset **structure AUROC 0.8035, +0.25 over its untrained null (0.552) and clear of the BoW null (0.508)** — an anchor-bootstrap CI of 0.809 [0.783, 0.833]. A bag-of-words model with the same corpus choices cannot manufacture that margin, which is what makes a *discovered* dimension admissible. Wired as the tenth channel and the largest-weight feature in the contraction.
- **Retracted — `threat`.** Passed an initial screen but **failed after a balanced resample**; removed from the vector.
- **Declined-as-policy — `sexual_content`.** A real signal, but the analysis showed its moral weight is carried by already-covered harassment axes and consensual explicitness is not a moral violation; routed as a platform-configurable policy channel *outside* the moral tensor.
- **Validated — `purity` and `loyalty`.** The two MFT "binding" foundations absent from the DEME-9. Using Social-Chem × ETHICS valence corpora, both cleared the same gate: **loyalty AUROC 0.911, purity 0.811**. This motivates §5.7.

### 5.7 The headline: valence transfers, identity does not

`purity` and `loyalty` admit two distinct feeders — a **valence** feeder (upheld vs violated) and a **presence** feeder (which foundation is engaged, valence-agnostic). We asked, for both kinds and across five foundations, whether *identity/presence* transfers across register, and whether domain-adversarial de-confounding or geometric shared-mode removal recovers it where plain training fails. Three complementary probes converge (Probes 1–2 are the same training pipeline with the domain adversary toggled off/on; Probe 3 is a fully independent geometric ablation on frozen embeddings).

**Probe 1 — presence transfer via joint contrastive training (no adversary).** A two-corpus joint-contrastive presence feeder (Social-Chem RoTs × MFRC Reddit) recovers cross-register transfer for exactly the two most lexically distinct binding foundations — **purity 0.719, loyalty 0.661 (2/5 pass)** — while care, fairness, legitimacy presence fail (margins ≤ 0.085). That the two recovered foundations are also the two most lexically distinct raises a **rival hypothesis we do not yet exclude**: presence transfer may track *lexical separability* rather than a principled valence/identity divide. The direct adjudication — correlating per-foundation presence-transfer against a lexicon-separability measure across the five foundations — is a **pending test we flag rather than assert past**; Probe 3 argues against the rival (the identity signal is entangled *within* the register axis, not sitting in a separable lexical subspace), but the correlation should be measured and reported either way.

**Probe 2 — domain-adversarial ablation (λ = 0 vs λ = 1).** Turning the domain adversary *on* degrades presence for **all five** foundations in the same direction. We separate the robust signal from the fragile and rest the claim only on the robust. **Robust:** (i) the degradation direction is **consistent across all five** channels — a sign test on five same-direction outcomes gives p ≈ 0.03, independent of any single channel's magnitude; and (ii) fairness presence suffers a **fuzz collapse from 16.9 to 0.007**, a two-order-of-magnitude loss of structural sensitivity that dwarfs retrain noise. **Fragile (and not leaned on):** the per-channel AUROC deltas are small (≈ 0.01–0.02) and *within the companion paper's measured retrain σ ≈ 0.03–0.05*, so we do not treat any single pass-flip as a finding (loyalty presence 0.661 → 0.641 is sub-σ), and the 2/5 → 1/5 pass-count is reported for completeness only. The adversary provably works (domain accuracy driven from 1.0 toward chance), so the consistent, direction-preserving degradation plus the fairness fuzz collapse is the diagnostic: **stripping register-diagnostic features strips the identity signal with it.** By contrast, the **valence** channels are unmoved by the same adversary — G 0.856 → 0.850, loyalty-valence 0.911 → 0.892, purity-valence 0.811 → 0.801, every change within σ of unchanged — which is exactly the point: valence does not depend on register.

| channel | kind | λ = 0 | λ = 1 | effect |
|---|---|---:|---:|---|
| G (general valence) | valence | 0.856 | 0.850 | unchanged ✓ |
| loyalty | valence | 0.911 | 0.892 | unchanged ✓ |
| purity | valence | 0.811 | 0.801 | unchanged ✓ |
| care | presence | 0.574 | 0.585 | neutral ✗ |
| fairness | presence | 0.569 | 0.499 | stripped to chance ✗ |
| legitimacy | presence | 0.585 | 0.544 | hurt ✗ |
| loyalty | presence | 0.661 | 0.641 | down, sub-σ ✗ |
| purity | presence | 0.719 | 0.699 | survives ✓ |

*Per-channel λ=0→λ=1 deltas are ≈ 0.01–0.02, within the companion's retrain σ ≈ 0.03–0.05; the load-bearing evidence is the all-five direction consistency (sign-test p ≈ 0.03) and the fairness fuzz collapse (16.9 → 0.007), not any single channel's pass-flip.*

![The dissociation. Cross-dataset AUROC with the domain adversary off (λ=0) vs on (λ=1). Valence channels (blue) are unmoved by the adversary and stay well above their gates; presence channels (vermillion) all degrade in the same direction. The robust evidence is the consistent all-five degradation direction and fairness's fuzz collapse (16.9 → 0.007), not the individual per-channel deltas (≈ 0.01–0.02, within retrain σ). Filled circles pass the validation gate, open circles fail.](figs/fig2_dissociation.pdf){width=92%}

**Probe 3 — geometric shared-mode removal.** As an adversary-free alternative, we removed the dominant shared *angular* modes (the corpus/register axis and top principal components) from frozen embeddings and renormalized to the sphere. **0/5 foundations gain ≥ 0.05**; the best case is purity +0.026, far below the gate. The diagnostic is decisive: **|PC₁ · register-axis| = 0.97–1.00** for every foundation — the top principal component *is* the register axis — yet removing it barely moves AUROC. The identity signal is not in a subspace *orthogonal* to the shared mode; it is entangled *within* it, so removing register removes identity too.

**Synthesis.** Three complementary probes — joint-contrastive training and its domain-adversarial ablation (the same pipeline with the adversary toggled), plus a fully independent geometric PC-removal probe on frozen embeddings — agree that for care, fairness, and legitimacy, **foundation identity and register/corpus occupy the same angular subspace and are not separable**, while **valence is register-invariant regardless**. The two projections of a MoralVector cell therefore have qualitatively different measurement status. The engineering consequence: presence/identity channels must be built with the non-adversarial joint-contrastive configuration, and only `purity` (robust) and `loyalty` (λ = 0 only) currently qualify. The scientific consequence: **"is this foundation engaged?" is a harder measurement than "is it upheld or violated?", and for some foundations it may be register-bound in principle.**

---

## 6. Integration: how a decision consumes the vector

The Analyzer's output is consumed by a downstream moral tensor and decision layer (the DEME architecture, treated in the keystone paper; summarized here only as the vector's consumer). The nine canonical dimensions populate the frozen `k` axis of a rank-1..6 tensor; extension channels ride in tensor metadata rather than widening the axis; a decision layer collapses the vector into allow / remove / escalate via the learned contraction, with a categorical hard-constraint (veto) path kept separate from the graded channels; and a hash-chained audit proof binds the vector — **including the extension channels** — into a verifiable record. The single-source-of-truth ontology (§2.1) is what keeps the construct identical across the perception, decision, and audit stages.

Two properties matter for measurement integrity. First, **provenance flows end to end**: the audit proof records which validated feeder produced each channel and the equivalence-class members behind an invariance-averaged score. Second, the schema is **anti-Goodhart by construction** (§8): the vector is a measurement, not a reward, and any optimization *against* it re-incurs the adversarial gate.

---

## 7. Threats to validity

- **Second-corpus availability.** The gate needs two genuinely independent corpora per dimension. Two canonical dimensions lack one and are not claimed; some MFT sub-constructs (fairness split into equality vs proportionality; property/reciprocity) face the same bottleneck.
- **Register coverage.** Feeders are trained largely on prescriptive/social-norm registers; a per-axis register-transfer evaluation (chatbot dialogue, narrative, commonsense norm banks) is partial. §5.7 shows register is not incidental — for identity it is load-bearing.
- **Label provenance overlaps.** Some discovered-dimension labels overlap existing attribute sets (e.g. identity_attack ⊂ a Perspective attribute); the novelty is the *instrument and its cross-dataset validation*, not the label.
- **Invariance generator in the trust boundary.** The inference-time paraphrase generator can be attacked or starved; its refusal rate is a measured hole, not a solved problem.
- **Two canonical dimensions unclaimed.** `rights_respect` and `societal_environmental` lack a validated second corpus and are reported as pending, not measured; `rights_respect` additionally runs as a hand-specified rule channel.

---

## 8. Related work and framing

**Measurement invariance and construct validity.** The paper's core argument — a high in-domain AUROC is not evidence of a construct — is the classical construct-validity distinction [@cronbach1955] applied to learned textual probes. We position the cross-dataset gate explicitly in the psychometric measurement-invariance tradition [@vandenberg2000; @putnick2016], including the MFQ-2 cross-cultural invariance program [@atari2023]: the gate is a measurement-invariance test for learned textual representations, with a falsification bar and documented failures. This is the difference between reinventing psychometrics informally and adopting its discipline. The bifactor structure we report (§5.3) is read in the sense of [@reise2012].

**Shortcut learning and dataset bias.** The in-domain/cross-dataset collapse we quantify (§5.1) is the moral-domain instance of shortcut learning [@geirhos2020] and dataset bias [@torralba2011]: a probe exploits a single corpus's surface regularities and fails to transfer. Our frozen untrained-encoder and bag-of-words nulls play, for moral-dimension probes, the role that control tasks [@hewitt2019] play for syntactic probes — a floor the trained probe must clear to earn a construct interpretation.

**Moral foundations & moral direction.** The canonical basis and the binding-foundation extensions draw on Moral Foundations Theory [@graham2009; @graham2011; @atari2023] and the moral/care corpora — the Moral Foundations Twitter Corpus [@hoover2020], the Moral Foundations Reddit Corpus [@trager2022], and the extended Moral Foundations Dictionary [@hopp2021]. The general factor G connects to the "moral direction" found in language models [@schramowski2022] and to the Theory of Dyadic Morality's single-harm-template prediction [@schein2018] — which our data adjudicate as *distinct-but-correlated* dimensions on a shared factor rather than one factor.

**Norms & ethics datasets.** Training and evaluation draw on Social-Chemistry-101 [@forbes2020], ETHICS [@hendrycks2021], Delphi / Commonsense Norm Bank [@jiang2021] and its critique [@talat2022], Moral Stories [@emelin2021], and MoralExceptQA [@jin2022].

**Content moderation.** The contraction and escalation layer situates the vector as an auditable moderation substrate; the anti-Goodhart clause (the vector is a measurement, not a reward) is the reward-hacking analysis [@manheim2018] promoted to a schema-level warning.

---

## 9. Discussion

The instrument's value is less any single AUROC than the **discipline it enforces**: frozen nulls, cross-dataset bars, pre-registration, and documented refusal. Every headline result in §5 is one an in-domain-accuracy workflow would have missed or misreported — the 0.955 → 0.518 collapse, the one-factor structure behind a ten-channel display, the register-boundedness of identity. A moral-perception layer that cannot produce results like these on its own construct is not measuring; it is asserting.

The measured structure also sets the **honest limit of what the graded vector can decide**. With five-plus channels riding a single general factor at β ≥ 0.96 and an effective rank of ≈ 5.7, the graded portion of the vector is one calibrated general moral factor plus roughly four specific corrections — sufficient for moderation triage (the §5.5 contraction numbers are honest for allow / escalate / remove with human escalation) and *insufficient* for autonomous high-stakes vetoes. The architecture already respects this: **the graded channels advise; only specified rules veto; and the measured rank of the vector is the reason.** Reporting the rank is therefore not a limitation buried in an appendix — it is what licenses the decision architecture.

Two forward directions follow directly. **Per-axis measurement dossiers** — calibration, human-agreement panels, register-transfer tables, adversarial robustness — turn each channel into a documented instrument rather than a number. And the **discovery program** scales the residual-mining loop beyond modern moderation corpora, with the standing guard that a passing gate on two corpora, not a suggestive residual, is what admits a dimension.

---

## 10. Conclusion

We defined a moral-perception representation and, more importantly, the measurement apparatus that earns the right to emit it. The apparatus's central lesson is small and sharp: **cross-dataset transfer is the minimal evidence that a moral-dimension probe measures anything**, in-domain accuracy can mislead by up to 0.44 AUROC (autonomy) though by as little as 0.01 for others (privacy), and even given transfer, a dimension's *identity* can be register-bound where its *valence* is not. An instrument that reports these facts about its own construct — including the negatives — is the contribution.

---

## References {.unnumbered}

::: {#refs}
:::

---

## Appendix A. Reproducibility

Feeders, the joint cross-corpus trainer, the gate, and the invariance and contraction fitters are in the `xbse` framework; the discovery-loop records, invariance findings, and contraction artifacts are in the Moral Spectrum Analyzer repository (`gtc-prototype`); the tensor/decision/audit consumer and the single-source-of-truth ontology are in `erisml-compiler`. Checkpoints are referenced by sha-256 hash; a downstream scorer refuses a checkpoint whose hash does not match a passing validation report.

## Appendix B. Provenance of the numbers

Every quantitative claim in §5 has been reconciled against its source artifact (verification pass, 2026-07-13, America/Los_Angeles — all dates in this repository are stamped in Pacific time, not UTC). All figures below are drawn directly from committed result files; no number in the body is now a reconstruction. Source files are named by basename; feeder/discovery artifacts live under `xbse/experiments/`, invariance and contraction artifacts under `gtc-prototype/data/` and `gtc-prototype/docs/`.

- **§5.1 feeder table** — `moralvector_reference.md` (§2 table).
- **§5.3 bifactor** — `bifactor_A1_result.json` (G auroc 0.8563 ± 0.0081, min-fuzz 114.9, margin 0.337); `bifactor_A2_result.json` (per-axis β and residuals; `transfer_summary` residualized core — diagonal 0.78 vs off-diagonal 0.53, 4/11 axes diagonal-dominant); `bifactor_overlap_result.json` and the A2 `P3.gpred` block (seen/unseen G-prediction); effective rank 5.68/9 from `prereg_bifactor_readout.md`. *Correction applied this pass:* the β split is care / fairness / legitimacy / loyalty / purity ≈ 0.96–0.99 and epistemic 0.83, versus privacy 0.25, physical-harm 0.35, autonomy 0.02, environmental −0.01, identity-attack 0.36 — not a clean 0.98 / 0 dichotomy.
- **§5.4 invariance** — `theta_atscale_result.json` (θ_d raw 0.407 → mechanism 0.219, refusal 0.240); `theta_redteam_result.json` (0.482 → 0.301, 0% NLLB refusal); `identity_ci_and_rank.json`; cross-lingual 0.721 / 0.804 and same-language index 0.75 [0.64, 0.84] from `INVARIANCE_FINDINGS.md`.
- **§5.5 contraction** — `toxicity_contraction.json` (oof-auroc 0.8626, oof-f1 0.7624, moderate-rate 0.510, remove-precision 0.804, baseline 0.779, 77/1600 excluded).
- **§5.6 identity-attack** — `IDENTITY_ATTACK.md` (structure AUROC 0.8035, nulls 0.552 / 0.508) and `identity_ci_and_rank.json` (0.809 [0.783, 0.833]).
- **§5.7 dissociation** — this session's directly-run experiments: `lambda_comparison_result.json`, `foundation_presence_adv_result.json`, `foundation_presence_attempt2_result.json`, `presence_polar_result.json`, `b1_results.json`.

*(References are numbered `[n]` and formatted in IEEE style from `references.bib`, verified 2026-07-13 against DBLP, ACL Anthology, JMLR, arXiv, and Crossref. One residual: whether the Moral Foundations Reddit Corpus [@trager2022] later appeared at a formal venue is unconfirmed, so it is cited as an arXiv preprint.)*
