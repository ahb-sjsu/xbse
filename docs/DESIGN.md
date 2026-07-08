# xbse — Design Document

**Status:** draft v0.1 · **Scope:** the shared `*-BSE` framework, its modules, its
instance roster, and the disciplines that keep instances honest.

---

## 1. What xbse is

`xbse` is a framework for building **B**ond/**S**tructure **S**entence **E**ncoders:
domain-specific embeddings that are **invariant to surface variation** and
**separated by structural difference**.

The one-line definition (from the code):

> A `*-BSE` = shared `BSEEncoder` + shared contrastive `objective` + a per-domain
> `PairSource`, gated by one shared validation harness.

The **only** thing that differs between instances is the `PairSource` — the
definition of what should be invariant (positives: same structure, different
surface) and what should be separated (negatives: different structure).
Everything else is shared and validated once.

### Design commitments (non-negotiable)
1. **Shared core, isolated variation.** Encoder + objective are identical across
   all instances; domain knowledge lives only in a `PairSource`.
2. **Validation is a hard gate, not a convention.** An instance is not "built"
   until it passes the shared harness. No downstream tool may be constructed on
   an unvalidated instance.
3. **Anti-circularity is enforced in code.** Any boundary/label used for a
   downstream test must be provably disjoint from training pairs, or the test is
   rigged. This is checked, not trusted.
4. **Falsification order.** build instance → validate (hard stop) → only then earn
   downstream tools/probes. The dependency graph enforces this: tools import a
   *validated* core; they cannot be imported otherwise.

---

## 2. Module architecture

### 2.1 Core (`xbse/`) — domain-independent; builds and validates an embedding
| module | responsibility | status |
|---|---|---|
| `encoder.py` | `BSEEncoder`: base transformer → pooling → projection → L2-norm. Only base-model id + projection dim are config. | **exists** |
| `pairs.py` | `PairSource`, `Triplet`: the per-domain interface. Declares invariant (positives) and separated (negatives). Exposes train/held-out split. | **exists** |
| `objective.py` | Shared contrastive loss: InfoNCE + in-batch negatives + optional hard negatives; symmetric anchor↔positive. | **exists** |
| `splits.py` | Constructs train/held-out splits **and asserts disjointness** between training pairs and any reserved downstream-test boundary. Throws on violation. | **missing — build next** |
| `validate.py` | The hard gate. Standard battery every instance must pass: structure-vs-surface AUROC (held-out), non-domain control, surface-leak probe. Emits a signed report. Cannot mark "validated" without running. | **missing — build next** |
| `metrics.py` | Shared geometric measurements: intrinsic-dimension estimators (ball-growth / MLE / effective-rank, with spread), geodesic-preservation score, angular fidelity. Used by both `validate` and `probes`. | **missing** |
| `report.py` | Structured, hashable output for every validation/probe run: config, checkpoint hash, pre-registered thresholds, pass/fail. | **missing** |
| `augment.py` | Registry of surface-invariance transforms (case, whitespace, paraphrase, back-translation, language swap). Each `PairSource` declares which are meaning-preserving *for its domain*. | **missing** |

### 2.2 Probes (`xbse/probes/`) — falsifiable experiments *about* an embedding
Kept deliberately separate from core: these make **claims that can fail**, they are
not "operations."
| probe | claim it tests | verdict semantics |
|---|---|---|
| `dimension.py` | local intrinsic dimension varies & is spatially structured (strata-of-differing-dimension signature) | pass = varying + autocorrelated vs matched non-domain control |
| `discontinuity.py` | evaluation jumps at normative/structural boundaries (H1) | pass = localized jumps beating a smooth reference, surviving paraphrase-averaging |
| `first_law.py` | δ(entropy) ∝ δ(geometry) region-by-region | graded: null / running-coupling / universal (all beat shuffled+isotropic nulls) |

> Probes never live in core. A probe result is evidence, not an API guarantee.
> "The probe ran" ≠ "the claim is true." Each probe reports against a
> pre-registered threshold and a matched control.

### 2.3 Domain packages (`xbse-<domain>/`) — depend on a validated core
Domain-specific `PairSource` implementations + any domain concepts (e.g. moral
axes, Hohfeldian states, policy categories). **Nothing domain-specific enters
core.** Test for core-membership: *would a whale-song or code instance need it?*
If no → domain package.

### 2.4 Tools (`xbse-<domain>-tools/`) — gated behind validation, downstream
Reasoning APIs that manipulate embeddings (e.g. a moral tensor algebra). These
**presuppose a validated instance** and each op is itself a **testable hypothesis**
(does the transform preserve the invariant it claims to? does an interpolated path
stay in-manifold?). They live in a separate package that imports a validated core,
so the falsification order is enforced by the dependency graph.

> **Naming discipline for tools:** a tool must be named for what it *computes*, not
> for a claim it hasn't earned. E.g. the harm operator computes a
> **representation-invariant harm ledger** (invariance — testable), NOT a "Noether
> conserved quantity" (conservation — not established). A function name that
> overclaims is worse than a paper that does, because it *runs*.

---

## 3. Instance roster & the admission filter

### 3.1 The admission filter (the important part)
A domain admits a legitimate `*-BSE` **if and only if** all three have concrete
answers:
1. **Invariant structure** — a definable structure that *should* be preserved.
2. **Surface** — a definable class of transformations it should be invariant to.
3. **Independent labels** — same-structure/different-surface pairs labelable by a
   source **independent of the embedding itself**.

If "structure" is what you are trying to *discover* rather than *supervise*, or if
the only available label is the thing you want to predict (a confound), the domain
**does not** admit a `*-BSE` — building one would embed a hypothesis as if it were
data. This filter is the framework's scaling safeguard; apply it before spending
compute.

### 3.2 Roster
| instance | domain | invariant structure | surface | independent label source | verdict |
|---|---|---|---|---|---|
| **LaBSE** | multilingual (base) | meaning | language | parallel corpora | ✅ base |
| **LeBSE** | US case law | citation relatedness | phrasing | CourtListener citation graph | ✅ **validated** |
| **MoBSE** | moral judgment | moral judgment/attribute | paraphrase, language | Social-Chem-101 / Scruples labels | ⏳ in progress |
| **SciBSE** | scientific claims | the finding | wording, notation | citation graph / S2ORC | ✅ strong — data on hand (arxiv) |
| **CodeBSE** | programs | I/O behavior | syntax, language | test-equivalence / transpile pairs | ✅ strong — external check (behavior) |
| **ModBSE** | content moderation | policy violation | euphemism, obfuscation, language | policy labels (human) | ✅ strong — high practical value; ties to I-EIP |
| **ReaBSE** | reasoning (scoped) | proof strategy / ARC transformation-rule | surface presentation / grid instances | correctness oracle (Lean / ARC solve) | ✅ **only if scoped** to a correctness oracle |
| **PolBSE** | political argument | argument *form* (claim/warrant/evidence) | topic, rhetoric | argument-mining structure labels | ⚠️ medium — legitimate **only** as argument-form; stance is a confound to rule out |
| **MedBSE** | clinical | diagnosis/condition | vernacular vs clinical notes, language | UMLS / coded diagnoses | ⚠️ medium — high stakes, brutal validation bar |
| **HistBSE** | diachronic text | meaning | historical period | aligned period corpora | ⚠️ medium — nice generalization of cross-lingual→cross-time |
| **MusBSE** | music | harmonic/rhythmic structure | timbre, performance | score/composition labels | ⚠️ medium — pipeline exists (FMA+MERT) |
| **CogBSE** | "cognition" | — | — | **none independent** | ❌ cut — no external structure label; circularity |
| **AesBSE** | "beauty" | — | — | **none stable** | ❌ cut — surface *is* the thing; genre confound, cross-modal sign flips |

### 3.3 Build priority (by signal strength × external-check availability)
1. **SciBSE** — clean signal, data already on Atlas, immediate utility.
2. **CodeBSE** — behavior is externally testable; ties to structural-fuzzing.
3. **ModBSE** — highest practical value; hard checks (catches paraphrased
   violations surface filters miss); connects to I-EIP.
4. **ReaBSE (ARC-scoped)** — ties to the ARC repo; correctness oracle.
5. **MoBSE** — finish + validate; proceed to probes only if it clears the bar.

Rationale: SciBSE and CodeBSE validate the framework on domains with **hard
external checks**, which is where the work is strongest. If the abstraction
generalizes cleanly there, `*-BSE` is a real framework and not just a tidy
factoring of three encoders that happened to get built.

---

## 4. The build/validate lifecycle (per instance)
```
1. Implement a PairSource (domain package).
2. splits.py builds train/held-out; asserts boundary disjointness (throws if rigged).
3. Train: BSEEncoder + objective on the PairSource.  → instance checkpoint
4. validate.py runs the shared battery:
     - structure-vs-surface AUROC on held-out (must clear pre-registered bar)
     - non-domain control (same encoder, off-domain text) — beats it?
     - surface-leak probe (does z leak language/period/surface?)
   → signed report. PASS or FAIL. HARD STOP on FAIL.
5. Only on PASS: instance is "validated". Downstream probes/tools may be built.
6. Probes (optional): each pre-registered, each with a matched control, each able
   to return a null. Results are evidence, not API guarantees.
```

## 5. What keeps this honest (summary of the disciplines)
- **Small core.** Domain knowledge never enters `xbse/`.
- **Validation as a gate, in code.** `validate.py` must run; FAIL is a stop.
- **Anti-circularity, in code.** `splits.py` throws if a test boundary leaks into
  training.
- **Falsification order, via dependencies.** Tools import a validated core; they
  cannot be built first.
- **Honest naming.** Tools named for what they compute, not for unearned claims.
- **Admission filter.** No instance without an independent structure-label source.
- **Probes are experiments.** Kept out of core; each can fail; a passing run is
  not a proven claim.

---

## 6. Open questions / deliberately unresolved
- The strong "Whitney stratified" claim for moral space remains **posited**, not
  established: probes can show the *empirical shadow* (varying dimension, sharp
  boundaries) but not Whitney conditions A/B, which need a formal construction.
  Keep instance/tool claims at the size the evidence carries.
- Cross-instance transfer (does a validated SciBSE improve MoBSE via shared
  structure?) is an experiment, not an assumption.
- The tools layer (moral tensor algebra, agent) is **out of scope for v0.1** and
  gated behind MoBSE validation.
