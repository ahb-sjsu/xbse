> **SUPERSEDED (2026-09-01).** The plan-of-record for MoBSE is now
> `erisml-compiler/experiments/MOBSE_PLAN.md` (post-review checkpoint,
> 2026-07-13), which supersedes this 2026-07-07 copy. Kept for history.
> See `erisml-lib/docs/CONCEPT_REGISTRY.md` §5.

# MoBSE — plan of record (post-review checkpoint, 2026-07-07)

Checkpointed because we're near context limit and the *sequencing discipline* is the part
that gets lost when the agent wraps. If you read one thing: **build in falsification order,
gate every step on the last, and never let a tool name assert something we retired.**

## The spine (keep this — it's the good part)
- **MoBSE** = a purpose-built moral sentence embedding (the morality analogue of LaBSE/LeBSE):
  situations near iff same *moral structure*, invariant to wording/language/framing/gauge.
- It's the instrument the whole geometric-ethics empirical program needs. Every prior probe
  failed for one reason: **BGE-M3 is a general encoder** — it encodes surface smoothly
  (age 17≈19, $999≈$1001), so it's blind to normative discontinuities, and its geometry is
  generic (arxiv looked just like legal in H3). MoBSE is the fix at the root.
- **The toolkit is simultaneously the reasoning API and the probe battery.** Every tool is a
  falsifiable bet about the geometry. That's what makes this science, not architecture.
  KEEP THE SENTENCE: *"most of these tools are testable hypotheses, not guaranteed
  operations — each op is a claim about the geometry that either holds or doesn't."*

## Prior findings this rests on (don't relitigate)
- **First-law: NO FIRST LAW** (m=1500 full-rank). Real H1_R²=0.151 beats nulls ~10× (weak,
  real, structure-dependent S–G co-variation) but sub-threshold and κ region-dependent
  (CoV 0.70). Not a law. Non-moral gutenberg control R²=0.059 → moral ~2.6× enriched (weak).
- **Jump/discontinuity test on BGE-M3: SMOOTH, but that's ENCODER-BLINDNESS, not falsification.**
  All 6 legal-threshold families smooth (after fixing the battery caption-vs-code artifact:
  year-and-a-day, wording constant). A general encoder can't see a stratum it never learned.
- **H3 (varying dimension): real but GENERIC.** legal autocorr 0.84 vs arxiv-BGE 0.607 vs
  isotropic null 0.007 — structured varying dimension is a property of BGE-M3 text embeddings,
  NOT moral-specific. Deflates the H3 "pass" as thesis evidence.
- Net: on a general encoder we cannot separate moral stratification from generic text-manifold
  structure. **MoBSE is the only instrument that could adjudicate the thesis.**

## TWO FIXES FROM REVIEW (apply before any code)

### Fix 1 — sequencing. The honest plan is brutally short. Hard stops.
1. **Build MoBSE v1** — contrastive fine-tune off BGE-M3 on Social-Chem-101 + Scruples +
   Moral-Machine(TRAIN split), positives = same-judgment + paraphrase (surface-invariance),
   negatives = opposite-judgment. Supersedes the Sefaria z_bond (and sidesteps the
   json_cache_full loader fight). Data cached: `/archive/ethics-corpora/{social-chem-101/
   social-chem-101/social-chem-101.v1.0.tsv, scruples/dilemmas, moral_machine_llm/data.zip}`.
2. **VALIDATE MoBSE like LeBSE was validated. HARD STOP.** Held-out structure-vs-surface
   AUROC + the fuzz test (surface changes move z LITTLE, moral changes move it A LOT; ratio
   > 1). v10.14 failed this — ratio inverted to 0.20×. **If MoBSE fails the gate: STOP.
   No tools, no claims. Iterate the encoder.** Do NOT build tool 2–9 before this passes.
3. **Only if it passes:** build `encode`, `distance`, `decompose` (the safe three).
4. **Then build ONE probe tool: `crosses_stratum`**, and run the pre-registered **H1
   discontinuity test on held-out boundaries** (Moral-Machine TEST split + untrained legal
   thresholds — disjoint from training, or the test is circular). **That result tells us
   whether the stratification thesis lives.**
   The nine-tool algebra + Erebus agent are the REWARD after the geometry earns them.

### Fix 2 — language. Kill the retired overclaim before it becomes a function name.
- **`harm_account` computes a REPRESENTATION-INVARIANT HARM LEDGER — invariance, testable,
  real. NOT a "Noether conserved quantity."** There is no conserved current; that framing
  dissolved on inspection and was renamed to a coherence constraint. A function name that
  lies is worse than a paper that lies, because it runs and gets a green checkmark.
- **`holonomy(loop)`**: compute the path-dependence/curl (does moral evaluation depend on the
  ORDER factors are considered?), report whether it's nonzero, STOP. Do not narrate "the moral
  field is non-conservative therefore [grand claim]."

## Falsification order for the eventual toolkit (each op = a pre-registered hypothesis)
- `gauge(z, swap)` → test E(d)=E(d′) on held-out Hohfeldian swaps. Holds? green. Fails? finding.
- `crosses_stratum(path)` → fires at real normative boundaries, quiet elsewhere? (= H1).
- `geodesic(a,b)` → does the interpolated path stay on-manifold?
- `analogy(a,b,c)` → does a moral offset transfer across situations?
- Promote hypothesis→operation ONLY on a green light. Each red light is a result.

## Circularity discipline (non-negotiable)
Train MoBSE for INVARIANCE (positives) + generic moral-judgment signal — NOT on the specific
boundaries you'll test. Test discontinuity on HELD-OUT boundaries. If a MoBSE trained only to
be surface-agnostic spontaneously shows jumps at boundaries it never saw → real, non-circular
evidence. If jumps only appear at trained boundaries → circular, no claim.

## Also-not-Whitney reminder
Even a clean positive buys "empirically stratified — discontinuity + varying dimension," NOT
Whitney A/B (the tangent/secant regularity conditions are a posited theoretical layer, not
empirically reachable from point clouds). The murder/absolutism argument forces *stratification*
(a −∞ wall smooth utility can't represent); Whitney-A is morally motivated; Whitney-B is an
honestly-labelable regularity/tractability postulate.
