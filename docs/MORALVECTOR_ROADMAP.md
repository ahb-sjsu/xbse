# MoralVector Improvement Roadmap — from validated prototype to a practical, general-purpose moral perception layer

**Status:** planning document (v1.1, 2026-07-12; corrected after a repo cross-check — Phase A's
"already drafted / already trained" claims were overstated and are fixed below, the effective-rank
number is synced to the 9-feeder value, and the invariance line reflects the at-scale results).
Sequencing note up front, per the cut line: nothing here preempts the GTC Charter's committed items.
Phases are ordered so the earliest ones double as Charter work where they overlap, and the rest queue
behind mid-October. **B1 DONE (2026-07-12): loyalty + purity feeders built and PASS the pre-registered
gate (AUROC 0.911 / 0.811) → the MFT "binding" foundations are real transferable axes, not one harm
factor.** See `experiments/prereg_mft_loyalty_purity.md` for the result + caveats.

**Definition.** *MoralVector* = the per-item output of the perception layer: one scored channel per
validated moral axis, each carrying value, confidence, validation provenance, and invariance
metadata; plus a general-valence channel, policy channels (explicitly non-moral), hard-constraint
channels, and the audit binding. The goal of this roadmap: make that vector *trustworthy enough to
consume* in any ethical-modeling scenario — moderation, agent evaluation, scenario judgment,
decision support, embedded interlocks, comparative/cross-cultural analysis — with every property
measured, not asserted.

---

## 0. Current state (what v1 is, honestly)

**[demonstrated]**
- 8 learned axes pass the pre-registered cross-dataset gate; `rights_respect` fails (framework-
  relative) and runs as a hand-specified rule channel; `identity_attack` discovered, gated
  (AUROC 0.80, +0.25 over null), wired as a 10th channel; **B1 (2026-07-12)** added `loyalty`
  (AUROC 0.911) and `purity` (0.811, beats the disgust-lexicon BoW null +0.156) through the same
  gate → **11/12 learned axes validated** (of 12 built: 11 learned pass, `rights` is the hand-rule
  channel). *Not yet wired into the DEME decision vector — that integration is a separate step.*
- Effective rank ≈ 5.19 of 8 on transfer data (8 validated feeders); **5.68 of 9** once
  `identity_attack` is added — one dominant general factor (pc0) + ~4–5 specifics; a *data property,
  causally removable* (deontic_transfer_gap §3.7), independently reconfirmed.
- Within-corpus signal is largely lexical; raw score invariance weak (ratio 0.60); canonical-layer
  invariance strong (index 0.75 same-language; **cross-lingual at scale 0.721 BGE-M3 / 0.804 LaBSE,
  incl. harmful**); equivalence-class averaging selected at θ_d 0.42 (LSO proxy) and **met at scale
  on LLM-generated classes, θ_d 0.219** (red-team back-translation confirms it holds on harmful
  content, 0.301).
- Learned contraction (leakage-controlled OOF AUROC 0.863) turns the vector into decisions on
  covered categories; admission criterion executed in all three directions (validate / decline /
  retract).

**Known deficits this roadmap targets**
1. Readout dishonesty risk: 10 channels displayed, ~6 independent factors present.
2. Coverage: no loyalty, authority-as-binding, purity, property, reciprocity-as-distinct axes —
   the dimensions two decades of moral psychology evidence say exist (MFT, MAC).
3. Per-axis measurement quality is uneven and partially uncharacterized (calibration, register
   coverage, human agreement, adversarial robustness).
4. Rights: no framework-conditioned deontic channel.
5. The construct's scientific standing is asserted by our papers alone; the prior-art convergences
   (and adjudicable disputes) are unexploited.
6. Practicality gaps: no stable schema, no calibrated probabilities, no per-axis model cards, no
   cost/latency envelope, no anti-Goodhart guidance.

---

## Phase A — Structural readout: the bifactor MoralVector (v1.5)

*The cheapest large improvement; converts the rank-collapse finding into architecture.*

- **A1. Promote the general factor. ✓ DONE 2026-07-12 — G trained + PASSES the gate.** Trained a
  general-valence BSE feeder (pooled signed Social-Chem all-categories × all signed ETHICS, 51,319 rows)
  at 3 seeds: **AUROC 0.856 ± 0.008, +0.337 over max-null (0.519), fuzz ≥ 115 → PASS.** Ships as channel 0,
  replacing the ungated `pc0` PCA projection. See `experiments/prereg_bifactor_readout.md` +
  `bifactor_A1_result.json`. Psychometric ancestry: bifactor model (Reise 2012); MFT
  individualizing/binding structure (Graham et al. 2009, 2011).
- **A2. Residualize the specifics. ✓ DONE 2026-07-12 — bifactor REAL but PARTIAL; P1/P3 fail strict,
  informatively.** Per-item r_k = s_k − β_k·G, β_k on a disjoint calibration half. Removing G collapses
  mean off-diagonal transfer 0.726 → 0.527 (near chance) — the shared-factor signature. But the 11 axes
  split two ways: **(A) the Social-Chem family** (care, fairness, legitimacy, epistemic, loyalty, purity)
  is **collinear with G** (β ≈ 0.98); **(B) independent-corpus axes** (privacy, physharm, identity_attack,
  autonomy) are **separable** (β ≈ 0, dominant residual diagonal). **Overlap diagnostic (2026-07-12)
  settles the confound cheaply:** G predicts the family on text it *never trained on* as well as on seen
  text (drop ≤ 0.035; β 0.96–0.99 on unseen) — the collapse is a **real generalizing shared factor, NOT
  corpus memorization** (physharm control drops 0.98→0.81, proving the test is sensitive). **But family
  residuals stay above chance (0.58–0.76)**, so the family axes are **not hollow**. ⇒ Represent as the
  **bifactor form — G on channel 0 + a small residual r_k per axis — keeping every dimension** (no coverage
  lost), not deletion and not six-independent-axes. Boundary: proven **within the prescriptive register**;
  cross-register/provenance separation is **D1/B5**. Independent coverage beyond G comes from the low-β set
  (privacy/physharm/identity_attack/autonomy) + G's blind spots + untested MAC dims (property/reciprocity,
  B4). See prereg A2 RESULT + `bifactor_A2_result.json` + `bifactor_overlap_result.json`.
- **A3. Two quality numbers per channel.** The margin↔fuzz dissociation is measured (legitimacy:
  weakest margin, high fuzz; environmental: reverse) — the vector schema carries **both** per axis:
  discriminative margin and invariance fuzz. A consumer can require either or both.
- **A4. Convergence replication (cheap, citable).** Align pc0 with Schramowski et al.'s "moral
  direction" (their template-PCA vs. our general factor; report cosine). One afternoon; one figure;
  positions G in the computational literature.
- **Gate:** residualized specifics diagonal-dominant on the transfer matrix (prereg P1); G ≤ 0.55 on
  independent-corpus dimensions (P3). FAIL demotes channels into G — the readout never displays a
  hollow axis.

## Phase B — Coverage completion from the prior art (v2 axes)

*The MFT corpora dissolve the second-corpus bottleneck for exactly the missing dimensions.*

- **B1. Loyalty and purity feeders. ✓ DONE 2026-07-12 — both PASS (AUROC 0.911 / 0.811).** Two
  independent labeled corpora exist per foundation: Moral Foundations Twitter Corpus (Hoover et al.
  2020, ~35k tweets) × Moral Foundations Reddit Corpus (Trager et al. 2022, ~16k comments). The **exact
  template exists**: `build_identity_attack_joint` (two independent corpus jsonl files → JointPairSource)
  plus the pre-reg bar machinery (`_BASELINE_NULL`/`_prereg_bar`) in `joint_builders.py`. B1 adds
  `build_loyalty_joint` = MFTC(loyalty) × MFRC(loyalty) and `build_purity_joint` = MFTC(purity) ×
  MFRC(purity), materialised by a `build_mft_foundations.py` script, then runs them through admission →
  gate identically to `identity_attack`. *(The pre-existing `build_foundation_joint` is
  Social-Chem×ETHICS category-filtered — a shared-corpus source, NOT the B1 path; the two MFT corpora
  are the genuinely-independent pair.)* Pre-register the MFT-derived prediction (these are real,
  transferable axes) *and* the rival TDM-derived prediction (they collapse into G) — either outcome is
  a finding (see D1).
- **B2. Authority: admission analysis before building.** `legitimacy_trust` already exists;
  MFT's authority-respect may be the same construct, a component, or distinct. Run the admission
  analysis first (does MFTC/MFRC authority signal residualize away against legitimacy + G?);
  build only if distinct. Document either way — a *decline-as-duplicate* extends the admission
  vocabulary (validate / decline-as-policy / decline-as-duplicate / retract).
- **B3. Fairness split test.** MFQ-2 (Atari et al. 2023) splits fairness into **equality** vs
  **proportionality**. Test whether `fairness_equity` transfers to both sub-constructs (corpus
  candidates: MFRC fairness split by annotation, equality/proportionality vignette sets); if it
  transfers to only one, the axis is mislabeled and should be split or renamed. **Corpus-availability
  caveat:** unlike B1, the *two-independent-corpus* bar is not obviously met here — MFRC-by-annotation
  is one source, and vignette sets are a different register; scout whether a genuine second
  equality/proportionality-labeled corpus exists before committing to a build (same caveat, more
  acute, applies to B4's property/reciprocity).
- **B4. Property/ownership scouting.** MAC (Curry et al. 2019) attests property as a cross-cultural
  moral domain; none of our axes covers it. Corpus scouting task: Social-Chem property RoTs,
  theft/ownership subsets of ETHICS-justice, legal petit-theft corpora; needs two independent
  sources before a build is attempted. Also reciprocity-as-exchange (MAC) distinct from fairness.
- **B5. Register expansion for existing axes.** Add third-register eval (not training) legs:
  Moral Integrity Corpus (Ziems et al. 2022, chatbot dialogue), Delphi's Commonsense Norm Bank
  (Jiang et al. 2021) for G, Moral Stories (Emelin et al. 2021) for narrative register. Per-axis
  register-transfer table becomes part of the dossier (C1).
- **Gate:** every new axis through the identical pre-registered bar; eMFD (Hopp et al. 2021)
  joins TF-IDF as a *second lexical null* for the MFT-derived axes — beating the field-standard
  dictionary method is the null that matters to that literature.

## Phase C — Per-axis measurement dossiers (the psychometric upgrade)

*One page per axis; the vector is only as practical as its worst-documented channel.*

- **C1. Dossier contents:** cross-corpus AUROC + bootstrap CI + retrain σ (3 seeds); margins vs all
  nulls; fuzz ratio; **calibration** (isotonic per register; ECE reported — scores become usable
  probabilities); register-transfer row (B5); human-agreement row (the IRB'd student panel:
  100–200 items/axis, κ + human↔score agreement); adversarial row (per-axis attack-detection AUROC
  from the containment work); known confounds and the axis's decline/retract history.
- **C2. Frame the gate as measurement invariance.** Position the cross-corpus criterion explicitly
  in the psychometric tradition (Vandenberg & Lance 2000; Putnick & Bornstein 2016; Atari et al.
  2023 ran formal invariance across 25 countries for MFQ-2): *the gate is a measurement-invariance
  test for learned textual representations, with a falsification bar and documented failures.*
  This sentence, in the tool paper's related work, is the difference between "rigorous newcomers"
  and "engineers reinventing psychometrics."
- **C3. Anti-Goodhart clause.** MoralVector is a **measurement, not a reward**. If any consumer
  optimizes against it, the lexical-signal findings predict gaming; the dossier states this, and
  any optimization use requires the adversarial gate re-run against the optimized system. (This is
  the reward-hacking analysis from the production plan, promoted to a schema-level warning.)

## Phase D — Scientific adjudications that harden the construct

*Each is a pre-registered study that both improves the vector and produces a publishable result.*

- **D1. TDM vs MFT rank adjudication. ◑ PARTIAL 2026-07-12 (cross-register leg done).** Theory of Dyadic
  Morality (Schein & Gray 2018; Gray et al. 2012) predicts one harm-templated factor; MFT predicts ~5–6.
  **Cross-register result (MFRC Reddit):** our *valence* feeders show no foundation selectivity across
  register (all ≈ 0.5), BUT a linear probe on pretrained BGE-M3 embeddings **separates the foundations at
  mean AUROC 0.748** (care 0.73 … purity 0.83). ⇒ **foundations are distinct-but-correlated dimensions;
  the A2 collapse is a property of the VALENCE readout, not the moral space** — MFT-leaning, anti-strong-TDM,
  with a new evidence type. The identity axis is orthogonal to G. **Register caveat (2026-07-12):** identity separability is
  *within-register* — a linear presence head trained on Social-Chem does **NOT** transfer to MFRC
  (0.51–0.58, ≤ BoW null; `foundation_presence_findings.md`), because a plain probe overfits register
  where the valence feeders transfer only via domain-adversarial training. Fix is *additive but needs the
  right method*: keep G + valence residuals; a transferable foundation-presence channel requires a
  **domain-adversarial contrastive presence feeder** (attempt 2, not yet run) — or identity is
  register-bound. See prereg D1/B5 + `d1_register_result.json` + `d1_probe_result.json` +
  `foundation_presence_result.json`. Still to do: (i) pc0↔eMFD-harm; (iii) extended-vector effective rank.
- **D2. Circumplex geometry test.** Schwartz's values (1992; 2012 refinement) organize on a validated
  circle — adjacency = compatibility, opposition = conflict. Test whether feeder correlation
  geometry reproduces opposition structure (candidate: autonomy vs legitimacy/authority). A
  "geometric ethics" program should check its geometry against the one validated geometric result
  in the values literature.
- **D3. Blind recovery + canon program.** Already pre-registered (`DISCOVERY_ROADMAP.md`); note the
  B1 axes become *additional planted targets* for blind recovery, and the canon experiment's
  loyalty/purity predictions get modern-label validation legs from B1 — the phases interlock.
- **D4. Exception structure (deontic redesign input).** MoralExceptQA (Jin, Levine et al. 2022 —
  when rule-breaking is permissible) is framework-flexibility as a dataset; probe whether the
  vector + G predicts human exception judgments. Feeds the rights/deontic v3 design (E3).

## Phase E — The practical surface (what "all ethical modeling scenarios" actually requires)

- **E1. Stable schema (v2).** JSON: per channel {value ∈ [−1,1], calibrated_p, confidence, margin,
  fuzz, validation:{auroc, CI, bar, registered, ckpt_hash}, register_coverage}; channel classes:
  `general` (G) / `moral_axis` (residualized) / `policy` (explicitly non-moral; sexual_content) /
  `hard_constraint` (rights rules) / `experimental`; plus framework_projections[], moral_residue,
  equivalence_class (hashes), audit_hash. Versioned; consumers pin schema versions.
- **E2. Scenario adapters.** (i) Moderation: contraction + escalation (shipped); (ii) conversational
  agents: MIC-register calibration + per-turn vectors; (iii) scenario/narrative judgment: Moral
  Stories + ETHICS adapters; (iv) decision support: DEME contraction with residue surfaced;
  (v) embedded: the EPU frame (hard-constraint channels only can VETO — per the 0.7 rule);
  (vi) comparative analysis: per-corpus emphasis spectrograms.
- **E3. Deontic v3.** Conduct-level rights labels (the paper's own successor hypothesis) +
  framework-conditioned scoring (jurisdiction/policy as input); long-term: framework projections
  *learned* from tradition corpora (canon program deliverable) instead of hand-authored.
- **E4. Ops envelope.** Throughput-per-dollar with equivalence-class averaging ON (the shipped
  invariance configuration), latency tiers (single-pass vs audit-grade), and the small-encoder vs
  LLM-moderation cost comparison (Charter's efficiency item — same measurement, reused).

## Sequencing & gates (one table)

| step | depends on | gate to proceed | earliest slot |
|---|---|---|---|
| A1–A4 bifactor | family-pool ckpt | prereg P1/P3 pass | overlaps Charter dev window |
| B1 loyalty/purity | MFTC+MFRC ingest | admission + standard gate | post-Charter-green or gated stretch |
| B2 authority admission | A2 residuals | decline-or-build decision doc | with B1 |
| C1 dossiers | A, B axes final | every shipped axis has one | rolling; blocks v2 tag |
| D1–D2 adjudications | A, B1 | prereg committed first | post-October |
| D3 recovery/canon | Charter green | pinned prereg (done) | December+ |
| E1 schema | A | consumer sign-off | with v2 tag |
| E3 deontic v3 | D4 input | separate design review | 2027 track |

## References (verify years/venues before citing — from memory of a stable literature)

Graham, Haidt & Nosek 2009, *JPSP* (moral foundations, liberals/conservatives) · Graham et al. 2011,
*JPSP* (MFQ) · Atari et al. 2023, *JPSP* (MFQ-2; cross-cultural measurement invariance) · Hoover et
al. 2020, *Soc. Psych. & Personality Science* (MFTC) · Trager et al. 2022, arXiv (MFRC) · Hopp et
al. 2021, *Behavior Research Methods* (eMFD) · Schein & Gray 2018, *Pers. & Soc. Psych. Review*
(TDM) · Gray, Young & Waytz 2012, *Psych. Inquiry* (dyadic morality) · Schwartz 1992 (*Adv. Exp.
Soc. Psych.*), 2012 refinement (values circumplex) · Curry, Mullins & Whitehouse 2019, *Current
Anthropology* (MAC, 60 societies) · Schramowski et al. 2022, *Nature Machine Intelligence* (moral
direction in LMs) · Jiang et al. 2021/2022, arXiv (Delphi; Commonsense Norm Bank) · Talat et al.
2022 (Delphi critique) · Ziems et al. 2022, *ACL* (Moral Integrity Corpus) · Hendrycks et al. 2021,
*ICLR* (ETHICS) · Emelin et al. 2021, *EMNLP* (Moral Stories) · Forbes et al. 2020, *EMNLP*
(Social-Chemistry-101) · Jin et al. 2022, *NeurIPS* (MoralExceptQA) · Vandenberg & Lance 2000,
*Organizational Research Methods* (measurement invariance review) · Putnick & Bornstein 2016,
*Developmental Review* (invariance conventions) · Reise 2012, *Multivariate Behavioral Research*
(bifactor) · plus internal: deontic_transfer_gap; REGATE; IDENTITY_ATTACK; SEXUAL_CONTENT_ADMISSION;
INVARIANCE_FINDINGS; DISCOVERY_ROADMAP; prereg_bifactor_readout.
