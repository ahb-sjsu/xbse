# Coherence Auditing of Content-Moderation Harm Accounting — A Campaign

**What this is.** A multi-experiment campaign testing whether harm-accounting in
content moderation is *coherent* — invariant under meaning-preserving re-description,
non-annihilating, and path-independent. It audits **decision practice** (human
labels and/or classifier outputs), NOT harm-in-itself. Every claim is of the form
"system X violates property P at rate R," which is checkable and defensible without
resolving what is "truly" harmful.

**What this is NOT.** Not a measurement of real harm. Not a claim that harm is a
conserved quantity (it has no measurement procedure — that question is ill-posed;
see §0). Not human-subjects research, *provided* it audits existing labels/outputs
and collects no new human judgments (see §7 — the IRB line).

**Why it matters (AI-safety / policy relevance).** Incoherence is an *exploitable*
defect: invariance failure = adversarial evasion; path-dependence = order/anchoring
attacks; non-annihilation failure = systems that abandon the departed. You can
demonstrate incoherence without settling the ground truth — which is exactly what
makes it a rigorous, fundable audit.

---

## 0. Scope discipline (state in every writeup)
- We measure properties of the **accounting**, not the harm. "Not invariant" is a
  coherence defect regardless of the correct label.
- "Harm is conserved" is not tested and not testable (no measurement procedure,
  no boundary, no dynamics). We test invariance, additivity-direction, and
  path-independence — three checkable shadows.
- Human labels are contested and biased; for a *coherence* audit that is signal,
  not noise. Never treat any dataset's labels as ground truth.

## 1. Systems under audit (the "algorithm f")
Run every experiment against ≥2 classes of moderator so results generalize:
- **A. Human labels** — existing multi-annotator datasets (annotator disagreement
  on equivalent items = invariance signal). NO new human judgments collected.
- **B. Classifiers** — (i) an open toxicity model (e.g. Detoxify/Perspective-style
  open weights), (ii) an LLM-as-moderator (prompt a model to apply a fixed policy).
- **C. (optional) A commercial API** if terms permit black-box output logging.

## 2. Datasets (verify current license/access before use)
| dataset | why | which experiment |
|---|---|---|
| Kumar et al. 2021 (multi-annotator + demographics, ~5/item) | annotator disagreement + group structure | E1 invariance, E4 group-consistency |
| Measuring Hate Speech (Berkeley, IRT continuous score) | graded target for matched pairs | E1, E2 |
| HateCheck / MHC (functional matched-case suite) | pre-built surface-perturbation pairs | E1 invariance (primary) |
| Social Bias Frames (target/implication structure) | agent-target structure | E3 non-annihilation |
| ToxiGen (implicit toxic + benign controls) | benign-vs-violation discrimination | E0 discrimination baseline |
| HateXplain (rationale spans + targets, 3/item) | same-rationale consistency | E2 path-independence |

**Content-warning + duty-of-care:** these contain slurs, harassment, abuse. Prefer
working over labels/embeddings; minimize raw-text exposure; warn anyone involved;
no prolonged review sessions. (See §7.)

## 3. Pre-registration
Freeze & hash this file before running each experiment. Fix, per experiment: the
system(s), the dataset, the matched-pair construction, the perturbation set, the
control, and the decision thresholds below. Deviations → AMENDMENTS.md. Register
which model checkpoints / API snapshot dates are used (systems drift).

---

## E0 — Discrimination baseline (does the system separate benign from violation?)
*Precondition. If a system can't tell benign from violating content, coherence of
its "harm accounting" is moot.*
- ToxiGen benign-mention vs implicit-toxic, matched on surface topic.
- **Metric:** AUROC separating benign from violating. **Pass:** AUROC ≥ 0.75.
- Fail → report; the system is not a competent harm-accounter and E1–E4 are
  descriptive only.

## E1 — Invariance under meaning-preserving re-description (the headline)
*Same content, surface-perturbed → does the verdict move?*
- **Pairs:** (a) HateCheck matched functional cases; (b) constructed pairs via
  declared-invariant rewrites: casing, whitespace, punctuation (uncontroversial);
  synonym-swap, paraphrase, back-translation, benign leet/obfuscation (semantic —
  validated separately, see below).
- **Rewrite validation gate:** before a rewrite counts as invariant, confirm it is
  meaning-preserving *independently* (e.g. human check on a sample, or held-out
  agreement). A buggy rewrite that flips meaning would masquerade as system
  incoherence. [AUTHOR: validate the semantic rewrites on a held-out sample first.]
- **Metric:** flip rate = P(label changes | meaning-preserving rewrite). For graded
  scores: mean |Δscore| under rewrite.
- **Control (load-bearing):** a **known-benign non-perturbation** — re-embed the
  SAME text (identity rewrite) k times; any Δ here is measurement noise, the floor
  the real flip rate must exceed.
- **Pre-registered:** flip rate significantly above the identity-noise floor, on
  ≥2 systems, with the semantic-rewrite subset validated. Report per rewrite type
  (surface-only vs semantic) — they will differ and that difference is a result.
- **Connection:** this generalizes your 13.7% euphemism-flip from LLM verdicts to
  moderation systems on real data.

## E2 — Path-independence / holonomy (order and anchoring effects)
*Reach the same final item by different routes → same verdict?*
- **Human-label version:** in multi-annotator data with item *presentation order*
  metadata (if available), test whether an item's label depends on what preceded
  it (anchoring). If order metadata absent, skip the human version — do not fabricate.
- **Classifier/LLM version:** present the target item embedded in different benign
  contexts / conversation histories; measure verdict variance. HateXplain rationale:
  does the SAME rationale span yield the same label across items?
- **Metric:** holonomy = verdict variance of a fixed item across routes, vs the
  variance across genuinely different items (the scale).
- **Pre-registered:** route-variance significantly below item-variance would mean
  path-independent (good); route-variance approaching item-variance means the
  system is order-dominated (a real defect). Report the ratio.

## E3 — Non-annihilation direction (does removing the target reduce attributed harm?)
*The "you cannot eliminate harm by eliminating the harmed party" intuition,
operationalized as a directional coherence constraint.*
- **Matched scenarios (Social Bias Frames target structure):** (a) harm to a
  present/active target; (b) the same harm where the target has left / is absent /
  cannot respond. Matched on everything except target-presence.
- **Metric:** sign of Δ(attributed harm) from (a)→(b). Coherent: harm in (b) ≥ (a)
  (removing the victim does not reduce the wrong). Defect: (b) < (a) systematically.
- **Control:** length/complexity matched (so you're not measuring "less text =
  less harm") — a benign target-presence manipulation with no harm content, which
  should produce Δ ≈ 0.
- **Pre-registered:** fraction of matched pairs with (b) < (a), against the benign
  control's Δ distribution. A systematic negative Δ is the headline safety finding:
  the system can be gamed by removing the complainant.

## E4 — Group-consistency (LBI-style, ties to your fairness work)
*Treat-like-cases-alike, on the protected axis, for moderation.*
- Reuse the **LBI matched-pair k-NN machinery** from your COMPAS paper, but the
  "algorithm f" is the moderation label and the "protected attribute" is the
  identity group *mentioned or targeted* (from Kumar demographics / SBF targets).
- **Metric:** LBI(moderation, identity-group) — does content matched on legitimate
  features (severity, explicitness) get labeled differently by identity mentioned?
- **Control:** permutation null on the group label (exactly as in LBI/COMPAS).
- This directly connects the coherence audit to the accepted LBI paper — same
  instrument, new domain.

## 5. Cross-experiment synthesis
Report a **coherence profile** per system: (E0 competence, E1 invariance flip rate,
E2 holonomy ratio, E3 non-annihilation Δ, E4 group-LBI), each with its control and
CI. The deliverable is the *profile across systems*, not a single verdict. A system
can be competent (E0 pass) yet incoherent (E1–E4 fail) — that's the interesting and
likely case, and it's the exploitable-attack-surface result.

## 6. What each outcome means (pre-committed)
- **All coherent** → moderation harm-accounting is robust on this axis; strong,
  slightly surprising, report it.
- **Incoherent on E1 (likely)** → quantified evasion-robustness gap; directly useful
  to moderation practice and to adversarial-ML. Headline.
- **Incoherent on E3** → gameable by target-removal; a genuine, novel safety finding.
- **Incoherent on E4** → moderation shows LBI-style group bias; extends your
  fairness paper to a second domain.
- **Systems disagree** → coherence is system-specific, not intrinsic; also a result.
Publish the nulls as readily as the positives.

## 7. Ethics / IRB / duty-of-care (non-negotiable)
- **Auditing existing labels/outputs = likely IRB-exempt.** Collecting NEW human
  judgments = human-subjects research, IRB required BEFORE running. Do not cross
  this line without university IRB sign-off.
- **Duty of care** for anyone (esp. OPT students) touching raw content: content
  warnings, minimized exposure, no prolonged review, work over labels/embeddings
  where possible. Moderation-adjacent work has documented psychological cost.
- **Data licenses:** verify each dataset's current license and platform terms
  (Twitter-derived sets may require rehydration / be partly unavailable).
- **No new harmful content generated** beyond meaning-preserving rewrites of
  existing dataset items; the rewrite set is surface/benign, validated (E1 gate).

## 8. Deliverable
One paper: *"Coherence auditing of content-moderation harm accounting: invariance,
path-dependence, non-annihilation, and group consistency."* Per-system coherence
profile, every metric minus its control, all nulls reported. Natural venues: FAccT,
AIES, or a moderation/CSCW venue. Companion to the LBI paper (same matched-pair
philosophy, new domain). Reference implementation released; datasets cited, not
redistributed.
