# xBSE data-sourcing plan — ≥2 independent corpora per DEME-9 dimension

**Purpose.** Single-dataset feeders learned corpus *surface*, not the moral dimension (within-AUROC
0.75–0.955 but cross-dataset ~0.47–0.55). The fix is joint cross-dataset training (same-sign
positives paired *across* corpora). That requires ≥2 genuinely independent, genre-distinct corpora
per dimension with a signed valence label ('+' = dimension upheld, '−' = violated). This document
maps every DEME-9 dimension to its corpora, synthesizing three research sweeps (general datasets +
two legal-evidence sweeps). Verified 2026-07-09.

**Cross-dataset results so far** (joint held-out AUROC vs untrained baseline): privacy 0.551→**0.853**,
care 0.469→**0.811**. fairness/legitimacy/epistemic + physharm/autonomy training. See
`moralvector_reference.md` for the honest scorecard.

## Per-dimension corpora

| dim | corpus A (have) | corpus B | corpus C (legal-evidence / API) | label→sign |
|---|---|---|---|---|
| physical_harm | BeaverTails (QA, phys-harm cats) | ETHICS harm-keyword scenarios | **CourtListener criminal** (assault/homicide; USSG §2A2.2 injury tier, 18 USC §113/§1111); **openFDA** adverse-events/recalls API | harm inflicted / higher injury tier = '−' |
| rights_respect | ECHR / LexGLUE ecthr | **CourtListener civil-rights NOS 440/441/443/446/448, §1983** | UPR-Info/UHRI (OHCHR); HR Violations Reporting Dataset (832k paras) | liability found = '−'; QI granted / defense verdict = '+' |
| fairness_equity | Social-Chem fairness-cheating | ETHICS (fairness-kw) | **CourtListener Title VII NOS 442/445**; HateXplain; CrowS-Pairs; CLAUDETTE ToS (unfair clauses) | discrimination/unfair found = '−' |
| autonomy_respect | ec-darkpattern (UI) | MentalManip (dialogue) | **CourtListener fraud/undue-influence NOS 370/371/470**; SemEval-2020 PTC propaganda; Mathur dark-patterns crawl | manipulation/fraud = '−' |
| privacy_protection | dual-judge privacy RoTs | AITA-privacy scenarios | **CourtListener FCRA NOS 480 / TCPA 485 / ECPA**; OPP-115; CFPB complaints API | violation / liability = '−' |
| societal_**environmental** | *(was zero)* | **EcoVerse** (native env-impact +/−); **ClimateBERT** climate_sentiment (risk/opportunity) | **CourtListener NOS 893 + NEPA EIS (FONSI vs significant-impact)**; **EPA ECHO** API/bulk; ClimateStance; CLIMATE-FEVER; GDELT ENV_* | env harmed = '−'; protected/enforced = '+' |
| virtue_care | Social-Chem care-harm | ETHICS (care-kw) ✅ joint 0.811 | Jigsaw Toxic (threat col); HR-violation narratives | harm/cruelty = '−' |
| legitimacy_trust | Social-Chem authority-subversion | ETHICS (authority-kw) | **CourtListener APA NOS 899 + SSA §405(g) 861–865**; CLAUDETTE (illegitimate clauses) | agency action vacated / arbitrary = '−'; upheld = '+' |
| epistemic_quality | Social-Chem honesty RoTs | ETHICS (honesty-kw) | **LIAR** (PolitiFact graded); **Ott deceptive-opinion-spam**; Google Fact Check / ClaimReview API | deceptive/false = '−'; truthful = '+' |

**Priority to break the ETHICS-as-universal-partner risk** (care/fairness/legitimacy/epistemic all
currently pair Social-Chem↔ETHICS): add a *third*, genre-distinct corpus to each — LIAR/Ott for
epistemic, HateXplain/CrowS-Pairs/CLAUDETTE for fairness, CourtListener APA for legitimacy,
Jigsaw-threat for care.

## Legal-evidence infrastructure (CourtListener, 63GB on Atlas)

CourtListener is a **cross-dimensional evidence corpus** — each dimension has a case type where the
impact is entered as labeled evidence. Mechanics (verified against FLP source):
- **`nature_of_suit`** is a **free-text** field on `/dockets/` (NOT numeric-coded) — match with
  `startswith` / the Search API `?type=r&nature_of_suit=`; populated **only for federal PACER/RECAP**
  (blank on scraped state/appellate). For the local 63GB, use FLP **quarterly bulk CSV dumps**, not
  the API.
- **Two-tier labeling.** Tier 1: **FJC Integrated Database** (joined into CL dockets) `judgment`
  field (1=Plaintiff, 2=Defendant) — cheap coarse winner for *adjudicated* civil cases (civil-rights
  ~44k, SSA ~18k, employment ~15k/yr). Tier 2: **LLM opinion-text** for *holdings* ("was the right
  actually violated"), all appellate affirm/reverse, all APA dispositions, and all criminal
  guilt/injury-tier (RECAP does NOT store criminal charges → must extract from text).
- **Label traps to encode:** vacating a *protective* rule = '−' but vacating a *rollback* = '+'
  (same disposition, opposite sign); standing/procedural dismissals = `no_valence`; government role
  is not fixed (enforcing = '+', defending a rollback = '−'). Never hardcode "gov = +". Use hybrid:
  metadata prior → dual-judge on facts → keep concordant, human-review disagreements.

**Corrections banked:** 444 Welfare is RETIRED; FCRA = NOS **480** (not 890); 442 = employment
*discrimination* (wage/labor = 710/720/740); 440s exclude prisoners (550/555/560); 485 TCPA exists
only on JS-44 Rev.03/24. CERCLA/RCRA/ESA/OPA are not explicitly listed under 893 → 893/890 by
practice (audit our schema's NOS fill-rate before relying on it).

## Queryable APIs (free, verified)

| API | dimension | endpoint | label | license |
|---|---|---|---|---|
| **EPA ECHO** | environmental | `echodata.epa.gov/echo/case_rest_services.get_cases` (+bulk ICIS FE&C) | Case-Summary narrative + penalty → '+' enforcement | public domain |
| **openFDA** | physical_harm | `open.fda.gov/apis` (no key) | adverse-event/recall narratives → '−' | public domain |
| **CFPB** | privacy/fairness | `cfpb.github.io/api/ccdb/` | complaint + company-response outcome | public domain |
| **GDELT GKG** | environmental/rights | `data.gdeltproject.org/api/v2` | themed events + tone → sign the tone | open |
| **Google Fact Check / ClaimReview** | epistemic | `developers.google.com/fact-check/tools/api` | reviewRating (True…False) graded | schema.org open |

## Remaining true gaps / caveats

- **Environmental** is now *data-solvable* (was zero) — but every env corpus is genre-narrow
  (tweets / reports / enforcement / court); build the joint across ≥3 of them and expect modest
  numbers. Legal outcome-labeling is noisy (the rule-direction trap) → prefer NEPA-EIS FONSI labels
  and dual-judge on facts.
- **Cross-genre transfer** (legal ↔ vignette) is the hardest test; a low number there is honest, not
  a bug. Report per-corpus-pair AUROC, not just pooled.
- License watch: `climatebert/climate_sentiment` is CC-BY-NC (non-commercial); most legal/gov sources
  are public domain.
