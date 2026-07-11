# IRB Protocol Summary — Human Audit of Machine-Generated Moral Valence Labels (DRAFT)

**PI:** Andrew H. Bond, Computer Engineering, SJSU · [contact]
**Site investigator (SSU):** [name/contact] · **Sites:** SJSU, SSU (reliance agreement or
parallel determinations requested)
**PI–instructor relationship:** [PI does / does not teach the recruiting courses — state
explicitly]. If the PI is instructor of record, the blinding + equal-credit mitigations below
are the required COI controls and are drafted accordingly.
**Requested category (two tiers — see Procedures):**
- **Student tier** (non-sensitive items only): **Exempt — Category 2** (anonymous adult
  survey, minimal risk) *requested*, but we submit anticipating the board may prefer
  **Expedited (Category 7)** given moral-judgment content, and do not rely on the exempt
  determination.
- **Sensitive tier** (targeted hate speech, graphic/legal-case material): **not rated by
  students at all** — handled under a separate determination by the PI/RAs/paid adult
  annotators. The board determines final categories.

## Purpose (1 paragraph)

An ongoing ML research project trains text encoders to score moral dimensions of sentences
(privacy, fairness, harm, etc.). Some training labels were produced by AI "judge" models. This
protocol audits those labels: human raters judge a stratified sample of the same items
(Upholds / Neutral / Violates, plus an unpenalized Skip), and the research reports aggregate
human↔AI and human↔human agreement. A **pre-registered** agreement threshold
(`prereg_label_audit.md`) gates whether the AI-generated labels may be used downstream. Results
appear in publications and a public repository in aggregate/anonymized form only.

## Participants & recruitment

Volunteer students (≥18) from [courses] at SJSU and SSU; expected 40–60 participants of a
100–150 pool per semester. Recruitment via a Canvas assignment offering **two equal-credit
options of comparable effort** — (A) participate as a rater, or (B) a non-research written
alternative — where Option B is **required-course-credit-equivalent, not extra credit**.
**Coercion mitigations:** instructor of record is blind to option choice and participation
identity until final grades post (anonymous completion codes verified by a TA/automated
process); consent is a separate first screen; declining/withdrawing has no academic effect and
the consent says so.

## Procedures (two-tier)

**Student tier (this protocol's human subjects).** Web app, single session ≤ 50 items
(~25–40 min): consent + 18+ attestation → short practice set → **3-way moral-valence rating —
Upholds / Neutral / Violates — with a separate unpenalized Skip** (Neutral is a substantive
"neither" matching the labels' signed-valence dead-band; Skip = abstain/uncomfortable, so
disagreement is never confounded with abstention) → optional anonymous demographics (age band,
gender, primary language, region raised) → completion code. No compensation beyond course
credit (equal for Option B).

**Sensitive tier (not students).** Items containing identity-directed hate speech, graphic
violence, torture, or legal-case facts are **excluded from the student pool entirely** (not
merely the high-severity subset) and rated only by the PI/RAs or paid adult annotators under a
separate determination. The tiering boundary is a hash-pinned, pre-committed item list.

## Sampling & scientific design (stratified to test the labels actually in question)

The audited sample is stratified by **dimension × AI-judge agreement**, oversampling where the
audit has the most value:
- **Dual-judged dimensions** (privacy, societal_environmental, rights) — the AI-labeled dims
  reviewers questioned; natively-labeled dims (Social-Chem/ETHICS) need less auditing.
- **AI-judge *disagreement* items** — where the two judge models split (highest information for
  calibration).
- **The rights slice specifically** — if humans are *also* incoherent on rights labels, that
  distinguishes framework-relativity from label-noise (the load-bearing question for the paper's
  rights result). Rights items are drawn from the non-sensitive tier (case-fact summaries,
  targeted hate speech excluded).

**Power / coverage (pre-registered in `prereg_label_audit.md`, before rating).** The prereg
commits items-per-dimension × **raters-per-item ≥ [R]** so human↔human κ and human↔AI agreement
are each estimable to a target CI half-width [±e]; channel-weeks/dimensions below that coverage
are reported but excluded from the gate. The **gate threshold** (what human↔AI agreement, per
dim, licenses downstream use) is committed there too — tighten-only after commit.

## Content & risk management

Student-tier items come from public research datasets of everyday moral judgments **with all
identity-directed hate speech and severe material removed** (sensitive tier). Residual risk is
transient discomfort from morally-charged (non-targeted) text, managed by (i) a content advisory
in the consent, (ii) per-item Skip and stop-anytime, (iii) campus counseling contacts in the
consent, (iv) no screening required given minimal-risk student content, but the advisory names
the general nature of the material. No deception.

## Data & privacy

Ratings stored under random identifiers only; no names, student IDs, or emails in the ratings
database. The completion-code/credit pathway is technically and administratively separate from
ratings data and inaccessible to analysts. FERPA: no education records leave Canvas; the app
collects no roster data. Retention [X years] per policy; public release is aggregate or
anonymized per-rating records without identifiers, and no demographics cross-linkage below
cell-size [5].

## Limitations (disclosed)

**Volunteer self-selection:** students who choose rating (Option A) over the alternative
(Option B) may skew less-sensitive to morally-charged text, biasing agreement estimates upward;
reported as a limitation, and mitigated by removing targeted hate speech from the student tier.

## Consent

Written (electronic) consent screen (attached: consent_form.md) with 18+ attestation, content
advisory, voluntariness and grade-protection language, withdrawal terms (already-submitted
anonymous ratings are unlinkable and cannot be individually deleted — stated in consent), and
IRB contacts for both campuses.

## Attachments

consent_form.md · canvas_assignment.md (recruitment text incl. equal-credit alternative) ·
prereg_label_audit.md + amendment 1 (scientific protocol, stratification, power, gate threshold) ·
rating-app screen flow (3-way scale + Skip) · item-tiering policy (hash-pinned list, student vs
sensitive tier).
