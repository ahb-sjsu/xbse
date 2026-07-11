# Shadow-pilot results — SIMULATED dry-run (NOT a deployment result)

Methods rehearsal of `prereg_shadow_pilot.md` using NRP LLMs as simulated adjudicators.
`scripts/shadow_sim.py`, Atlas, 2026-07-10. **This is a pipeline rehearsal, not evidence about real
deployment.** Two caveats hold throughout: (1) LLM-simulated adjudicators are not human subjects;
(2) the engine and the LLM judge may share a harm notion, which could inflate or distort C1.

## Setup
- Traffic: 360 Measuring-Hate-Speech items (hate_speech_score>0.5 oracle; race/gender/religion slices),
  chunked into 8 "weeks".
- Engine: xbse commonsense-valence violation score (from care_joint; the validated-family axis).
- Incumbent adjudicator (blinded): gemma-small. Double-adjudication audit: kimi (different family).
  (First run used qwen3/glm-5 — reasoning models that returned content=null under max_tokens=4;
  fixed by raising max_tokens and using non-reasoning-fast models.)

## Results
- **kappa(incumbent, audit) = 0.302** — below the 0.40 floor.
- C1 (precision@k, engine vs chronological): 0/8 weeks pass -> FAIL. Weekly N~45 makes precision@k
  noisy, and the commonsense-valence axis is not hate-speech-specific.
- C3 (identity-slice benign FPR ratio, op-threshold q0.80; overall FPR 0.20): gender-targeted slice
  FPR 0.33, ratio 1.61 (>1.25) -> FAIL. race 0.76, religion 0.52.
- C5 (missed-harm among engine low-score): 0.20.

## Verdict: INCONCLUSIVE
kappa 0.30 < 0.40, so the pre-registered `INCONCLUSIVE-if-kappa<kappa_min` rule fires: the reference
standard (agreement between the two simulated adjudicators) is too noisy to certify against, and NO
pass/fail verdict is issued. This is the intended behavior — the double-adjudication audit I added to
the prereg is load-bearing: without it, a FAIL would have been reported on an unreliable reference.

## What the dry-run demonstrated
1. The C1/C3/C5 measurement machinery computes end-to-end.
2. The INCONCLUSIVE guardrail correctly refuses a verdict on a noisy reference standard.
3. The feared C1-inflation from a shared engine/judge harm notion did NOT occur here (C1 failed),
   because the engine's valence axis does not align with hate-speech and weekly samples are small.
Real deployment would use human adjudicators, per-partner scoring function, and the power/N_min +
held-out-blinded-stream machinery the prereg specifies.
