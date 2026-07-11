# PREREG — Shadow Pilot (Phase 4, Gate to Queue-Ordering Authority)

**Status:** PRE-REGISTERED — to be finalized (bracketed values filled with [PILOT PARTNER]) and
committed **before first live traffic is scored**. **Rule:** tighten, never loosen, after commit.
**Intended location:** `experiments/prereg_shadow_pilot.md`, alongside the other pre-registrations, so
its commit timestamp sits with them.

**What this is.** A read-only deployment beside [PILOT PARTNER]'s incumbent moderation process: the
engine scores all in-scope traffic; nothing it outputs affects any user or any moderation action. The
gate decides one thing: does the engine earn **queue-ordering authority** (its scores may order the
human review queue). No other authority is obtainable from this pilot.

**What this is NOT.** Not an auto-action trial. Not a replacement-of-humans evaluation.
Severe-category escalation runs unconditionally throughout and is not part of the bet.

## Scoring function (pre-registered — which dimensions order the queue)

The queue-ordering score is a **frozen, pre-committed function of only the engine dimensions that
cleared the §3.1 cross-dataset gate** in the validation paper. This is load-bearing: the paper shows
(a) `rights_respect` fails its gate and (b) `care/fairness/legitimacy/epistemic` collapse into one
shared valence, so a naive nine-dimension score would ride an unvalidated axis or double-count a single
factor.

- **Included axes** (validated, corpus-independent): `privacy_protection`, `societal_environmental`,
  `autonomy_respect`, `physical_harm`, plus **one** `commonsense_valence` feature standing in for the
  collapsed family (NOT four separate features). Severe-category escalation is orthogonal and
  unconditional.
- **Excluded:** `rights_respect` (fails §3.1; straddles chance [0.468, 0.533]) contributes **no**
  queue-ordering weight. If the partner's traffic is rights-centric, that is a scope limit of v1, not a
  thing to paper over.
- The exact weights/aggregation and the mapping from partner categories to axes are committed in an
  appendix to this file before burn-in and are frozen thereafter (tighten-only applies to thresholds,
  not to the scoring function, which is fixed).

## Setup

- **Window:** [8] weeks of live traffic after a [1]-week burn-in. Burn-in data are used only for
  prevalence measurement, calibration checks, **and drift-σ estimation (see C4)**, never for gate
  scoring.
- **Ground truth:** the incumbent process's final adjudications (post-appeal where applicable), joined
  to engine scores logged at ingest. Adjudicators never see engine scores (blinding — the pilot's
  validity depends on it).
- **Imperfect-reference audit (pre-registered).** Because incumbent adjudications embed the incumbent's
  own biases, a random **[k_audit]%** of in-scope items is **independently double-adjudicated** by a
  second blinded rater (or panel). This subset (a) estimates incumbent label noise / inter-rater κ, and
  (b) lets "engine disagrees with incumbent" be separated from "engine is wrong": engine errors are
  scored against the double-adjudicated consensus on the audit subset, not against single-rater
  incumbent labels alone. Fairness (C3) is reported on both the incumbent labels **and** the audit
  consensus; a C3 pass requires it on the consensus.
- **Prevalence measurement:** per-channel base rates measured during burn-in; committed operating
  points from `prereg_calibration_operating_points.md` re-checked against them.
- **Power / N_min (pre-registered before burn-in).** For each of the confirmatory criteria (C1–C3) we
  compute the per-channel adjudication volume needed to detect the committed effect at power ≥ [0.8],
  α = [0.05]: C1 to detect a precision@k ratio of [1.15] vs baseline, C2 to bound ECE at [0.05], C3 to
  detect an FPR ratio of [1.25]. `[N_min]` per channel is the max of these. A channel-week below
  `[N_min]` contributes to monitoring but not to a confirmatory verdict (see INCONCLUSIVE).

## Confirmatory vs monitoring criteria; multiplicity

- **Confirmatory (the bet):** **C1** is the single primary endpoint (the product claim). C2–C3 are
  **co-primary safety gates** (a failure vetoes PASS regardless of C1).
- **Monitoring (standing SLOs, not primary):** C4, C5 — alarms and remediation, evaluated with the
  persistence rules below rather than as one-shot significance tests.
- **Multiplicity:** confirmatory tests are C1 (primary) + C2, C3 (co-primary vetoes). Across the
  8-week × per-channel × identity-slice family, per-week decisions use a **Benjamini–Hochberg FDR
  control at q = [0.1]** within each criterion, and every "≥ X of 8 weeks" / "two consecutive weeks"
  rule below is stated as a pre-registered decision rule (not post-hoc). Identity-slice tests (C3) are
  corrected across slices within channel-week.

## Pre-registered criteria (all must hold over the full window)

- **C1 — Triage value (PRIMARY):** ordering the review queue by engine score achieves precision@k whose
  **lower 95% bootstrap CI bound ≥ [1.15]× the baseline point estimate** (not point-estimate ≥ 1.15×),
  at k = the partner's actual daily review capacity, on ≥ [6] of [8] weeks with ≥ `[N_min]`
  adjudications that week. **Baseline is committed before burn-in** to the incumbent ranking (or
  chronological if the partner confirms no incumbent ranking exists — decided now, not conditionally
  later). If C1 fails, v1 has no reason to exist.
- **C2 — Calibration holds live (CO-PRIMARY):** per-channel ECE on live traffic ≤ [0.05] absolute,
  re-measured weekly with a bootstrap CI; no two consecutive weeks with the CI lower bound above the
  bound. Rare severe categories with < `[N_ece]` events/week are reported but excluded from the ECE
  gate (underpowered), and flagged.
- **C3 — Fairness holds live (CO-PRIMARY):** identity-slice benign FPR ratio ≤ [1.25]× overall, per
  channel, measured on the adjudicated sample **and confirmed on the double-adjudicated audit
  consensus**; any violated week (FDR-corrected across slices) triggers slice review; two violated
  weeks on the same slice = FAIL regardless of C1.
- **C4 — Drift within bounds (MONITORING):** score-distribution shift (PSI per channel) ≤ [0.2] and
  embedding drift within alert thresholds **set from the drift-σ measured during burn-in** (not the
  research retrain/re-eval σ, which is a different quantity — burn-in drift-σ is measured live before
  the window opens); alarms may fire, but an unexplained alarm that persists ≥ [1] week = FAIL.
- **C5 — Adversarial recurrence & missed-harm (MONITORING):** false negatives are **not** observable
  from the reviewed queue, so a random **[k_fn]%** sample of low-score / un-actioned traffic is
  independently adjudicated each week (pre-registered N `[N_fn]` for power to estimate the missed-harm
  rate to ±[e]). Of confirmed missed-harm items, < [1]% exhibit a known evasion pattern the adversarial
  gate (A1–A5 signatures) certified against; a certified pattern recurring above bound = FAIL and
  re-opens that adversarial gate.

## Verdicts

- **PASS (C1 primary **and** C2, C3 co-primary vetoes hold; C4–C5 not in FAIL):** engine earns
  queue-ordering authority at the committed operating points, with C2–C5 continuing as standing SLOs.
  Authority is revocable: any standing SLO violated for [2] consecutive weeks reverts to shadow mode
  automatically (`require_pass` semantics extended to runtime).
- **Post-authority honest measurement (pre-registered).** Once the engine orders the queue, precision@k
  under engine ordering is **self-fulfilling** (high-score items reviewed first → confirmed). So a
  pre-committed **[k_holdout]% of traffic is held out to a blinded stream** ordered by the incumbent /
  chronological baseline throughout the authority period; C1–C5 SLOs are measured on that held-out
  stream (or via interleaving), never on the engine-ordered majority. Losing the held-out stream =
  monitoring is blind = automatic revert to shadow.
- **FAIL:** engine stays in shadow mode; failure analysis is written up and published to the repo like
  any other negative result. C1 failure = product-thesis failure (report and reassess v1); C2/C3
  failure = safety veto; C4/C5 failure = remediate the specific property and re-run a [4]-week window.
- **INCONCLUSIVE:** adjudication volume below `[N_min]` per channel, blinding broken, or the
  double-adjudication audit reveals incumbent inter-rater κ below `[κ_min]` (reference standard too
  noisy to certify against) → window extended or re-run; no verdict from an underpowered, unblinded, or
  ungrounded window.

## Ethics & governance notes

Users are subject to the incumbent process only; the engine adds no user-facing effect during shadow.
Data handling per partner DPA; engine logs store scores + item hashes, not raw content, wherever the
partner's tooling permits re-join. The appeals path and model cards (Phase 3) must be live before PASS
converts to authority.

## Outputs

`experiments/shadow_pilot_results.md` — weekly C1–C5 tables (with CIs), the power/N_min appendix, the
frozen scoring-function appendix, the double-adjudication κ, the held-out-stream measurements, the
final verdict, and the authority decision record. Bracketed thresholds and the scoring function are
finalized with [PILOT PARTNER] and committed before burn-in; after that, thresholds tighten-only and
the scoring function is frozen.
