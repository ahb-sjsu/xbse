# Per-dimension risk–coverage of MoralVector sub-BSEs — methods & results

**Status:** 2026-07-08. Instrument = `xbse` per-foundation MoBSE sub-BSEs (BGE-M3 dual-encoder,
contrastive). This report covers (1) how the sub-BSEs and the shared gate are built, (2) the
label-ceiling measurement `C_moral`, (3) the risk–coverage / selective-prediction methodology,
(4) a train/held split-determinism bug that contaminated earlier numbers and its fix, and
(5) the corrected @350-vs-@2500 comparison for sanctity and fairness.

> **Honesty note up front.** Several figures in earlier working notes (sanctity 0.951; sanctity
> "overfitting" 0.951→0.785; fairness 0.81/0.87/0.94 across runs) were **split noise**, not real
> effects — the PairSource split was non-deterministic across processes (see §4). The numbers in
> §5 use a deterministic split and supersede them. Where a pre-fix number is quoted it is marked
> *(pre-fix, split-noisy)* and used only qualitatively.

---

## 1. What a sub-BSE is, and the shared gate

Each `*-BSE` is the same architecture (base transformer → mean-pool → L2-norm) trained to be
**invariant to surface** and **sensitive to structure**; only the `PairSource` differs. A MoBSE
per-foundation sub-BSE keys structure on the **moral fingerprint** = (Moral Foundation ×
judgment-sign) from Social-Chem-101 human annotation. Because a per-foundation instance fixes the
foundation, its structure reduces to the **judgment sign** (+/–/0).

- **positives** (invariance): same fingerprint, *different situation/topic* → must stay near.
- **negatives** (sensitivity): same topic, *different fingerprint* → must separate.

The runs here use `clean=False`: the **full distribution including contested items** (no
`rot-agree≥3` filter), so contested cases are in both train and eval.

**Shared validation gate** (`validate.py`): pass iff `structure_auroc > 0.97` **and**
`fuzz_ratio > 1`. The 0.97 bar is inherited verbatim from LeBSE's held-out citation-retrieval
AUROC (0.971). `fuzz_ratio` = how much more the embedding moves for structure than for surface;
`surface_invariance` is reported as a diagnostic but does not gate.

## 2. Label ceiling `C_moral`

The gate scores AUROC against a **noisy human label**, so a Bayes ceiling `C` bounds the best
achievable AUROC. For a per-foundation sub-BSE the label is the judgment sign, whose per-item
reliability is given by Social-Chem's `action-agree` 0–4 consensus bin (→ `p`, midpoint map
0:.01 / 1:.15 / 2:.50 / 3:.825 / 4:.95). Ceiling = AUROC of the Bayes soft-match predictor
`P(relabelled signs match)` vs the observed same/different label, over the held-out pairs.

**Result (clean population, `rot-agree≥3`):** `C ≈ 0.997–0.999` for every foundation
(mean sign reliability `p̄ = 0.86`). Contestedness-restricted: `action-agree=4` → `C≈1.000`;
`action-agree=2` (~50% agree, genuinely contested) → `C≈0.910`; the clean pool is ~99%
`action-agree≥3`, so `C≈1.0`. **Implication:** on the population these encoders train/eval on,
the 0.97 bar sits *inside* the moral labels' own ceiling — the gate failures are **real encoder
gaps, not label noise**. (This ceiling is a property of the label population, robust to which 10%
is held out, so it is unaffected by the split bug in §4.)

## 3. Risk–coverage (selective prediction) methodology

For a trained encoder and its held-out structural pairs:
1. cosine similarity `s` per pair; decision threshold `τ*` = Youden-J point on the ROC.
2. per-pair **confidence** = `|s − τ*|`; sort pairs most-confident first.
3. at coverage `c`, keep the top-`c` fraction and report **selective AUROC** (ranking quality on
   the confident subset) and **selective accuracy** (`(s>τ*)==y`). Abstain on the rest.
4. `coverage@0.97/0.95/0.90` = the largest coverage (restricted to ≥0.5) whose selective AUROC
   clears that bar. 95% CIs on full AUROC by 1000-sample bootstrap.

Selective prediction only helps if **confidence tracks correctness** (a rising selective-AUROC
curve). A flat/declining curve means abstention buys nothing.

> **Tail artifact:** below ~0.6 coverage the confident subset becomes class-imbalanced (pos%
> drifts to 8–79%), so selective AUROC destabilizes. Read only the **100→70% range**.

## 4. The split-determinism bug (and fix) — commit `414e2d2`

`_split()` (and the shared `splits.split_by_key`, whose docstring *claimed* "deterministic /
resumable") assigned train/held via Python's built-in `hash()`, which is **salted per process**
(`PYTHONHASHSEED`). Consequences:
- every run drew a **different held-out set** → AUROC numbers not comparable across runs;
- on an unlucky salt, a degenerate held set (one rerun drew a 3-pair, all-positive eval).

**Fix:** `stable_frac(key)` = md5-based bucket in `pairs.py`, used in `splits.py` and all six
instances (mobse, rights, scibse, reabse, gabse, codebse). Same key → same bucket, every process,
every machine — so @350 and @2500 now see the **same** held set. (A separate runner bug — passing
the whole `heldout_eval()` dict instead of its `structural_pairs` — also caused the 3-pair
artifact and was fixed in the harness.)

## 5. Results — deterministic @350 vs @2500 (sanctity, fairness)

Full held-out AUROC (BGE-M3 dual-encoder, `clean=False`, max_len 192, lr 2e-5, batch 48):

| run | full AUROC [95% CI] | n pairs |
|---|---|---|
| sanctity@350  | 0.806 [0.687–0.907] | 84 |
| sanctity@2500 | 0.800 [0.700–0.905] | 84 |
| fairness@350  | 0.872 [0.814–0.922] | 208 |
| fairness@2500 | 0.866 [0.815–0.916] | 208 |

Selective AUROC by coverage, with fairness selective *accuracy* in parentheses
(meaningful 100→70% range):

| coverage | sanctity@350 | sanctity@2500 | fairness@350 | fairness@2500 (selAcc) |
|---|---|---|---|---|
| 100% | 0.806 | 0.800 | 0.872 | 0.866 (0.84) |
| 90%  | 0.798 | 0.808 | 0.876 | 0.874 (0.85) |
| 80%  | 0.772 | 0.796 | 0.880 | 0.877 (0.86) |
| 70%  | 0.726 | 0.761 | 0.884 | 0.886 (0.86) |
| 60%  | —     | 0.756 | 0.880 | 0.896 (0.86) |

**Findings:**
1. **Longer training helps neither dimension.** sanctity@350 (0.806) ≈ sanctity@2500 (0.800),
   and fairness@350 (0.872) ≈ fairness@2500 (0.866) — every pair of CIs fully overlaps. On the
   deterministic split, 2500 steps buys nothing over 350. The @2500 hypothesis is **falsified**
   for both. The pre-fix apparent gains ("sanctity 0.951→0.785 overfit", "fairness 0.867→0.935
   improve") were both **split noise** — opposite-signed artifacts of drawing different held sets.
2. **Sanctity was never the best dimension.** The 0.951 *(pre-fix, split-noisy)* was a lucky
   split; on the fixed split sanctity is ~0.80 (wide CI, n=84).
3. **Confidence is at best weakly informative.** In the reliable coverage range, sanctity's
   selective AUROC *declines* (0.806→0.726) and fairness's rises only marginally
   (0.866→0.896 at 60% coverage). **No run reaches `coverage@0.97/0.95/0.90`.** Selective
   *accuracy* is respectable (fairness ~0.86–0.90), but selective *ranking* does not lift these
   sub-BSEs to legal-grade at any usable coverage.

## 6. Exploratory 7-dimension sweep *(pre-fix — split-noisy, qualitative only)*

A pre-fix @350 sweep (randomized split) over all foundations is retained for one robust
*qualitative* signal only: **the broad MoBSE's confidence is flat/anti-informative, while the
per-foundation specialists' confidence is at least weakly informative.** That is the strongest
evidence that "MoBSE is too broad" holds at the level of *calibration*, not just accuracy. The
absolute AUROCs from that sweep (e.g. sanctity 0.951, rights 0.78→0.89, authority 0.67→0.85) are
**not** reliable — §5 shows split noise can move sanctity by ~0.15 — and are superseded wherever a
deterministic number exists.

## 7. Discussion — the practical-system question

The design constraint is that the deployed monitor must **not discard dimensions or contested
cases**. Risk–coverage delivers the honest version of that: every dimension keeps a per-coverage
operating point instead of a binary pass/fail, and abstains where confidence is low. But the
deterministic results are sober:
- selective prediction **does not reach legal-grade (0.97)** at usable coverage for sanctity or
  fairness; and the confidence signal is weaker than the pre-fix curves suggested.
- so the path to a shippable legal-grade instrument is **better encoders**, not more abstention —
  and *not* longer training of the current recipe (both sanctity and fairness: @2500 ≈ @350).

The most likely lever is **data**, not steps: the small foundations are data-limited (sanctity
held-out n=84), which caps both the encoder and the statistical resolution.

## 8. Data added & next steps

- **Moral Stories** (Emelin et al., 12,000 stories) pulled to
  `/archive/ethics-corpora/moral_stories/moral_stories_full.jsonl`. Schema: `norm, situation,
  intention, moral_action, moral_consequence, immoral_action, immoral_consequence, label`. Each
  story is a **surface-matched, judgment-flipped pair** (same situation; moral vs immoral action)
  — ideal hard negatives, which should raise `structure_auroc` legitimately (better negatives, not
  a lower bar).
- **Next:** (a) add a Moral-Stories PairSource and retrain fairness/care with the matched-pair
  negatives; (b) MFTC/MFRC as an independent-scheme held-out set to break residual circularity;
  (c) larger per-foundation corpora (Commonsense Norm Bank) to lift the small dims off their n≈100
  eval ceiling.

## 9. Artifacts & honesty ledger

- Checkpoints: `/home/claude/xbse_ckpt/{sanctity,fairness}_{350,2500}_det.pt` (deterministic
  split), plus the pre-fix `{name}.pt`. Results JSON: `risk_coverage_det.json`.
- Split fix committed `414e2d2` on `main`.
- **Corrections made in this report:** (i) sanctity 0.951 and the sanctity "overfitting" story
  were split noise; (ii) the strong rising risk-coverage curves were partly split noise; (iii) the
  earlier `n=3` result was a runner dict-passing bug, not the hash split. The hash split was a
  *real, separate* bug and is fixed; the deterministic numbers here are the ones to trust.
