# Pre-registration — B1: loyalty & purity feeders (MoralVector roadmap Phase B1)

**Registered 2026-07-12, before training.** Executes MoralVector roadmap B1: build `loyalty` and
`purity` moral-axis feeders — the two MFT "binding" foundations the DEME taxonomy misses — through the
*same* pre-registered admission → gate used for `identity_attack`.

## Hypotheses (rival, registered before the run)
- **H_MFT (primary):** loyalty and purity are **real, transferable moral axes**. Prediction: each feeder's
  cross-dataset held-out AUROC beats BOTH nulls by ≥ 0.10 (PASS the gate).
- **H_TDM (rival):** morality is one harm-templated factor (Theory of Dyadic Morality); the "binding"
  foundations are not separable transferable axes in text. Prediction: one or both **fail** the
  cross-dataset gate, or pass only via lexical overlap (fail the BoW null margin).

Either outcome is a finding. A feeder that fails is **declined**, exactly as `threat` was retracted and
`sexual_content` declined — the admission criterion has teeth.

## Corpora — and an honest substitution
The roadmap specified **MFTC × MFRC** (two MFT-labeled social-media corpora). On check (2026-07-12):
- **MFTC** is not readily available on HF (Twitter text is not redistributable without hydration; the
  Twitter API path is closed).
- **MFRC** (`USC-MOLA-Lab/MFRC`, split `train_dedup`, 53,827 rows) **is** available, but its `annotation`
  field is foundation **presence** (Care / Loyalty / Purity / …), **not valence** — it does not encode
  loyalty-*upheld* vs -*betrayed*, which a signed moral-valence axis requires.

So the two independent **valence-labeled** corpora are:
- **loyalty** = Social-Chem-101 `loyalty-betrayal` RoTs (41,290; signed by `action-moral-judgment`)
  × ETHICS-commonsense (loyalty-keyword scenarios; label 1=wrong→'-', 0=ok→'+').
- **purity** = Social-Chem-101 `sanctity-degradation` RoTs (15,477; signed) × ETHICS-commonsense
  (purity-keyword scenarios).
Both are genuinely independent datasets **and different registers** (prescriptive RoTs vs everyday
scenarios) — a stronger cross-register test than two social-media corpora. **Disclosed caveat:** this is
*not* the roadmap's MFTC×MFRC; it reuses Social-Chem (a corpus shared with the existing DEME family
feeders), so if loyalty/purity later **collapse into the family/G**, corpus-sharing is a candidate
explanation to be settled by the D1 residualization test (deferred — G not yet built).
- **MFRC presence cross-check (bonus, NOT the gate):** the trained valence feeder should also separate
  MFRC `Loyalty`/`Purity`-annotated comments from `Non-Moral` at AUROC > 0.60 — an independent-register,
  independent-provenance generalization leg. Reported, but a pass/fail here does not gate admission.

## Gate (identical policy to the 8 passing feeders + identity_attack, prereg 2026-07-10)
VALIDATED iff cross-dataset held-out AUROC beats **BOTH** the untrained-encoder null **AND** the TF-IDF
bag-of-words null by **≥ 0.10** on the same held-out pairs, with **fuzz-ratio > 1.0**. The nulls are
measured on the **untrained** BGE-M3 before training and frozen into the Bar (properties of
untrained-model + corpus, not of the trained feeder). `holdout_frac = 0.12`.

**Lexical-null note.** The roadmap's B-gate wants **eMFD** (the field-standard moral-foundations
dictionary) to join TF-IDF as a *second* lexical null — beating the dictionary method is the null that
matters to the MFT literature. eMFD scoring is not yet wired here, so this run uses the **TF-IDF BoW
null only** (disclosed); adding the eMFD null is a follow-up before any paper claim.

## Parameters (frozen)
Encoder BGE-M3, mean-pool, `max_len=128`; `train_adversarial` epochs 6, batch 24, lr 2e-5,
max_steps 1200, `max_lambda=0.0`; seed via corpus order (deterministic). Scripts:
`scripts/build_mft_foundations.py` (materialize + gate), builders `build_foundation_joint("loyalty"|"purity")`
in `joint_builders.py`. Result → `loyalty_joint_report.json`, `purity_joint_report.json`.

## RESULT (2026-07-12) — H_MFT confirmed: both PASS. Loyalty and purity are real transferable axes.

| axis | trained AUROC | untrained null | BoW null | margin vs max-null | fuzz | gate | MFRC presence x-check |
|---|---:|---:|---:|---:|---:|:--:|---:|
| **loyalty** | 0.911 | 0.411 | 0.505 | **+0.406** | 130.2 | ✅ PASS | 0.608 |
| **purity** | 0.811 | 0.404 | **0.656** | **+0.156** | 39.0 | ✅ PASS | 0.635 |

**H_MFT is supported over H_TDM at the gate:** both feeders learn transferable cross-corpus valence
that beats their nulls by the pre-registered ≥ 0.10 margin — they are **not** absorbed into an
undifferentiated harm signal at this test. Purity is the stronger result *because* its bar was hardest:
disgust language is highly lexical (BoW null 0.656), yet the feeder beats the disgust lexicon by +0.156,
so it is learning structure **beyond** the keywords. Both clear the independent-provenance MFRC presence
cross-check (Reddit comments, > 0.60). Corpus tallies: loyalty 23,739 rows (Social-Chem 23,137 / ETHICS
602); purity 9,637 (9,300 / 337). Data: `experiments/b1_results.json`.

**Honest caveats (carry these with the numbers):**
1. **Thin second corpus.** The independent ETHICS leg is small (602 loyalty / 337 purity rows), so the
   cross-corpus held-out is small and the very high fuzz ratios (130, 39) are **unstable** — treat the
   exact AUROC as provisional pending a larger independent second corpus (MFTC if hydratable, or a
   third foundation corpus).
2. **Passing the gate ≠ independent of G.** Both share Social-Chem with the DEME family, so whether
   loyalty/purity are separable from the general factor / the collapsed family is the **D1
   residualization test**, which needs the Phase-A `G` encoder (not yet built). The MFRC cross-check
   speaks to cross-provenance generalization, not to residual independence.
3. **Lexical null = TF-IDF only.** The roadmap's B-gate also wants the **eMFD** dictionary as a second
   lexical null; it is not wired yet, so beating eMFD (the field-standard moral-foundations lexicon)
   remains to do before a paper claim — especially for purity, given its lexical baseline.
4. **Not MFTC×MFRC** — Social-Chem×ETHICS substitution (see Corpora, above).
