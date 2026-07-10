# Pre-registration: do independent corpora separate the collapsed moral family?

**Registered before running.** Committed to git prior to training the encoders described below. The
point of this file is to fix the *prediction and the falsifier in advance*, so the result cannot be
reinterpreted after the fact — the discipline reviewer-2 asked for, and the same discipline the
project has applied to every prior control.

## Background

Under BGE-M3, four dimensions — `virtue_care`, `fairness_equity`, `legitimacy_trust`,
`epistemic_quality` — collapse into a single shared "commonsense-moral valence" axis (9×9 transfer
matrix + family-pool baseline; paper §3.2–3.3). All four currently draw from the **same** two
corpora (Social-Chem-101 + ETHICS), so the collapse is confounded with corpus-sharing. The open
question (paper §7 Q1): is the collapse a property of the **data** (removable by giving each
dimension an independent corpus) or of the **moral space** (irremovable)?

## Intervention

Replace the shared `Social-Chem ↔ ETHICS` pairing for each family dimension with a
`Social-Chem ↔ independent-corpus` pairing, where the independent corpus is a **different-provenance,
signed-valence** dataset targeting that specific concept. Cross-corpus positives then require the
encoder to align a concept across two genuinely different registers, not two commonsense-morality
corpora.

### Committed corpus choices (leading candidates; final ID per dimension fixed before *its* run)

| dimension | independent corpus | sign mapping | provenance vs Social-Chem |
|---|---|---|---|
| virtue_care | empathy / prosocial corpus (EmpatheticDialogues or Moral-Stories care-norm subset) | prosocial/supportive `+`, harmful/neglect `−` | different domain (dialogue/narrative), not ROTs |
| fairness_equity | **Measuring Hate Speech** (`ucberkeley-dlab/measuring-hate-speech`) | mean `hate_speech_score` > 0.5 → `−`, < −1.0 → `+` | **clean** — YouTube/Reddit/Twitter comments, not ROTs (27,768 rows) |
| legitimacy_trust | authority/legitimacy — MFTC Authority/Subversion (needs tweet rehydration) | legitimate/authorized `+`, subversion `−` | **deferred** (no clean redistributable signed corpus; hardest dim) |
| epistemic_quality | FEVER (`fever/fever`) — available if extended | `SUPPORTS` → `+`, `REFUTES` → `−` | clean (Wikipedia); secondary run |

**FINAL (locked before training, 2026-07-10):** fairness = Measuring Hate Speech (`mhs_fairness.jsonl`,
27,768 rows, clean independent); care = Moral-Stories care-norm subset (`care_signed.jsonl`, 2,426
rows: `moral_action`→+, `immoral_action`→−). **Care carries a documented partial-independence caveat**
— Moral-Stories norms are a filtered subset of Social-Chem, so only its independent narrative *text*
(the action sentences) is relied on; it is a partial-independence corpus, not a clean one. Fairness is
the **clean flagship**; care is the caveated secondary. Legitimacy/epistemic are deferred (legitimacy
has no clean redistributable signed corpus — itself a finding). The pairing replaces ETHICS with the
independent corpus while keeping Social-Chem as the first corpus, so the change is isolated.

## Prediction (the falsifiable claim)

Re-run the **4×4 family cross-transfer** over {care_v2, fairness_v2, legitimacy (orig), epistemic
(orig)} — reusing the existing BGE-M3 legitimacy/epistemic checkpoints and training only the two
re-corpused encoders — comparing each dimension's off-diagonal cross-transfer to its within-transfer.

**Manipulation check FIRST (gate on the prediction; committed per reviewer-3 point 4).** A null
result is only interpretable if the new corpora actually produced *working* encoders. So **before the
transfer prediction is scored**, care_v2 and fairness_v2 must each **pass their own cross-dataset gate
on the new Social-Chem↔independent pairing** — i.e. cross-dataset held-out `structure_auroc` beats
`max(untrained-null, BoW-null)` by ≥ 0.10 (the same pre-registered bar as every other feeder). If a
re-corpused encoder **fails its own gate**, that arm is scored **"inconclusive — corpus quality,"
NOT a falsification** of the moral-space hypothesis. Only encoders that pass the gate contribute to
the transfer test below.

- **If the concepts are genuinely distinct** (data hypothesis): for a gate-passing encoder,
  diagonal-dominance **emerges** — its mean off-diagonal drops **≥ 0.10 below** its within-transfer.
  Care and fairness in particular decouple from legitimacy/epistemic.
- **Falsifier** (moral-space hypothesis): if a **gate-passing** re-corpused encoder **still
  cross-contaminates** (off-diagonal within 0.10 of within-transfer, as it is now), then the ~5-axis
  collapse is a property of the moral space, not the data — and paper Implication 1 ("give every
  dimension independent corpora") is **wrong** and must be retracted.
- **Inconclusive:** a re-corpused encoder that fails its own gate says nothing about moral space —
  only that this corpus was too weak; report as such, do not count it as evidence either way.

## Controls committed in advance

1. **Register control.** The same test also addresses the symmetric confound (reviewer point 1):
   because the independent corpora are in *different registers*, a surviving diagonal-dominance is
   evidence of concept-specificity, not shared register.
2. **Same nulls.** Untrained + BoW nulls recomputed per new corpus; the pre-registered
   baseline-relative bar (margin ≥ 0.10 vs `max(null)`) is unchanged.
3. **Encoder held fixed** (BGE-M3, mean-pool, InfoNCE) so the comparison to the existing 9×9 is
   apples-to-apples; a gte-Qwen2 replication is secondary.
4. **Register-test rewrite-fidelity check** (reviewer-3 point 5, for the *separate* register-controlled
   test, not this run): if that test uses LLM rewriting to hold concept fixed while moving register, a
   frozen probe trained on the *original* corpus must still decode the labels on the *rewritten* text
   at near its original within-corpus AUROC **before** the rewritten set is used — otherwise a
   specificity drop is ambiguous between "was domain shift" and "rewrite erased the concept." Run both
   directions (privacy text → Social-Chem register and family text → privacy register).

## Status

Registered 2026-07-10 **before training either re-corpused encoder**. Corpora LOCKED: fairness =
Measuring Hate Speech (clean flagship, 27,768 rows), care = Moral-Stories care-subset (caveated
secondary, 2,426 rows). Manipulation-check gate committed. Legitimacy/epistemic deferred. No
independent-corpus encoder trained at registration time.
