"""estimate_noise_ceiling.py — derive a per-dimension validation bar from LABEL NOISE, not results.

A bar chosen after seeing model scores is calibrated to bless them. This script derives the
maximum achievable AUROC (the "noise ceiling") from properties of the CORPUS ONLY, then prints a
ready-to-paste `Bar(...)` at ceiling * factor. Commit the Bar before the training run; the git
history is the pre-registration record. A bar may be tightened before a run, never loosened after.

Ceiling logic: a perfect scorer knows the TRUE sign of every text, but the gate evaluates against
OBSERVED labels, which flip the true sign with some per-item probability eps. Even the perfect
scorer's AUROC against noisy labels is < 1. We estimate eps three ways:

  --social-chem TSV [--foundation care] : per-item eps from rot-agree (k of N_ANNOTATORS agree on
        the RoT); Laplace-smoothed p_correct = (k+1)/(n+2), eps_i = 1 - p_correct.
  --dual-judge JSONL --fields a,b       : two independent judge columns; per-corpus eps solved
        from observed sign-agreement a = eps^2 + (1-eps)^2  =>  eps = (1 - sqrt(2a-1)) / 2.
  --agreement A                          : agreement rate known from elsewhere; same solve.

Then simulate: true signs ~ observed sign base-rate; observed = true flipped w.p. eps_i;
perfect scorer score = true sign (+ tiny tie-break noise); AUROC(score, observed). Report mean
ceiling over trials and the suggested bar.

Usage examples:
  python scripts/estimate_noise_ceiling.py --social-chem /archive/.../social-chem-101.v1.0.tsv \
      --foundation care --name virtue_care
  python scripts/estimate_noise_ceiling.py --agreement 0.87 --name privacy_protection
  python scripts/estimate_noise_ceiling.py --dual-judge privacy_labeled.jsonl \
      --fields privacy_qwen3,privacy_glm5 --name privacy_protection
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import sys

import numpy as np

N_ANNOTATORS = 5  # Social-Chem-101 RoT workers per item (rot-agree is k of these)


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def eps_from_agreement(a: float) -> float:
    """Solve a = eps^2 + (1-eps)^2 for the smaller root (independent-flip judge model)."""
    if a < 0.5:
        raise SystemExit(f"agreement {a} < 0.5 — the two label sources disagree more than chance.")
    return (1.0 - math.sqrt(2.0 * a - 1.0)) / 2.0


def eps_from_rot_agree(tsv: str, foundation: str | None) -> tuple[dict[str, np.ndarray], float]:
    """Per-item eps from Social-Chem rot-agree, under TWO estimators:
      raw      : eps = 1 - k/n            (trusts unanimity fully)
      laplace  : eps = 1 - (k+1)/(n+2)    (never trusts unanimity fully)
    They disagree most exactly where it matters (high-agreement items). Downstream we take the
    STRICTER (higher) resulting ceiling: leniency must never come from an estimation knob."""
    raw, smooth, pos, n_rows = [], [], 0, 0
    with open(tsv, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                k = int(row.get("rot-agree") or 0)
                j = float(row.get("action-moral-judgment") or 0)
            except (TypeError, ValueError):
                continue
            if j == 0:
                continue
            if foundation and foundation not in (row.get("rot-moral-foundations") or "").lower():
                continue
            raw.append(1.0 - k / N_ANNOTATORS)
            smooth.append(1.0 - (k + 1) / (N_ANNOTATORS + 2))
            pos += j > 0
            n_rows += 1
    if not n_rows:
        raise SystemExit("no rows matched — check --foundation spelling / file path.")
    return {"raw": np.asarray(raw), "laplace": np.asarray(smooth)}, pos / n_rows


def eps_from_dual_judge(path: str, fields: list[str], db: float = 0.05) -> tuple[float, float]:
    """Per-corpus eps from the sign-agreement of two judge fields in a jsonl."""
    fa, fb = fields
    agree = tot = pos = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
                a, b = float(d[fa]), float(d[fb])
            except (ValueError, TypeError, KeyError):
                continue
            sa = 1 if a > db else (-1 if a < -db else 0)
            sb = 1 if b > db else (-1 if b < -db else 0)
            if sa == 0 or sb == 0:
                continue
            tot += 1
            agree += sa == sb
            pos += sa > 0
    if tot == 0:
        raise SystemExit(f"no doubly-signed rows in {path} for fields {fields}.")
    return eps_from_agreement(agree / tot), pos / tot


def simulate_ceiling(
    eps: np.ndarray, pos_rate: float, n: int = 4000, trials: int = 40, seed: int = 0
) -> tuple[float, float]:
    """Perfect-scorer AUROC against labels flipped per-item w.p. eps. Returns (mean, std)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(trials):
        e = rng.choice(eps, size=n, replace=True) if len(eps) > 1 else np.full(n, eps[0])
        true = (rng.random(n) < pos_rate).astype(int)
        flip = rng.random(n) < e
        observed = np.where(flip, 1 - true, true)
        score = true + rng.normal(0, 1e-6, n)  # perfect scorer, tie-break jitter
        a = _auroc(score, observed)
        if not math.isnan(a):
            out.append(a)
    return float(np.mean(out)), float(np.std(out))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--name", default="dimension", help="dimension name for the printed Bar")
    ap.add_argument("--social-chem", help="Social-Chem-101 TSV (uses rot-agree)")
    ap.add_argument("--foundation", help="MFT substring filter, e.g. care-harm")
    ap.add_argument("--dual-judge", help="jsonl with two judge score fields")
    ap.add_argument("--fields", help="comma-separated pair of judge fields, e.g. a,b")
    ap.add_argument(
        "--agreement", type=float, help="known sign-agreement rate of two label sources"
    )
    ap.add_argument("--factor", type=float, default=0.90, help="bar = ceiling * factor")
    ap.add_argument("--fuzz-min", type=float, default=1.0)
    a = ap.parse_args()

    sources = []
    if a.social_chem:
        eps_by_est, pr = eps_from_rot_agree(a.social_chem, (a.foundation or "").lower() or None)
        n = len(eps_by_est["raw"])
        for est, eps in eps_by_est.items():
            sources.append(
                (f"rot-agree[{est}](n={n}, foundation={a.foundation or 'all'})", eps, pr, "rot")
            )
    if a.dual_judge:
        if not a.fields or "," not in a.fields:
            raise SystemExit("--dual-judge needs --fields judgeA,judgeB")
        e, pr = eps_from_dual_judge(a.dual_judge, a.fields.split(",", 1))
        sources.append((f"dual-judge({a.dual_judge})", np.array([e]), pr, "judge"))
    if a.agreement is not None:
        e = eps_from_agreement(a.agreement)
        sources.append((f"agreement={a.agreement}", np.array([e]), 0.5, "judge"))
    if not sources:
        ap.print_help()
        sys.exit(1)

    print(f"== noise-ceiling estimate for '{a.name}' ==")
    by_group: dict[str, list[float]] = {}
    labels_used = []
    for label, eps, pr, group in sources:
        mean_eps = float(np.mean(eps))
        ceil, sd = simulate_ceiling(eps, pr)
        by_group.setdefault(group, []).append(ceil)
        labels_used.append(label)
        print(
            f"  {label}: mean_eps={mean_eps:.3f} pos_rate={pr:.2f} -> ceiling AUROC {ceil:.3f} ± {sd:.3f}"
        )

    # Within one noise source (e.g. the two rot-agree estimators) take the STRICTER (max) ceiling:
    # when estimators disagree, leniency must never come from the estimation knob. Across
    # independent label sources take the min: a scorer trained on both is capped by the weaker.
    per_source_ceilings = [max(v) for v in by_group.values()]
    ceiling = min(per_source_ceilings)
    bar = round(ceiling * a.factor, 3)
    today = datetime.date.today().isoformat()
    src_desc = " + ".join(labels_used)
    print(
        f"\n  ceiling = min over label sources of (stricter estimator) = {ceiling:.3f};"
        f"  suggested bar = ceiling * {a.factor} = {bar}"
    )
    print("\n-- paste into the instance builder and COMMIT BEFORE the training run --\n")
    print(f"""from xbse.bar import Bar

{a.name.upper()}_BAR = Bar(
    auroc_min={bar},
    fuzz_min={a.fuzz_min},
    source="noise-ceiling({src_desc}) * {a.factor}",
    derivation=(
        "Perfect-scorer AUROC against labels flipped with per-item eps estimated from "
        "annotator/judge agreement (label-noise ceiling {ceiling:.3f}), taken * {a.factor}; "
        "derived from corpus properties only, before training."
    ),
    registered="{today}",
)""")


if __name__ == "__main__":
    main()
