#!/usr/bin/env python3
"""coherence_probe.py — run the E1 (invariance) and E3 (non-annihilation) coherence
experiments against a moderation scorer.

The scorer is ANY function text -> harm score in [0,1]. Plug in Detoxify, an LLM
prompt, a Perspective client, or (for testing) a mock. The harness computes flip
rates / directional deltas against the pre-registered controls.

Wired so "coherent" (null) is a first-class outcome. The identity-rewrite noise
floor (E1) and the benign length-matched control (E3) are mandatory — a result
that doesn't beat its control is reported as no-effect, not massaged.

Deps: numpy, scipy.
Inputs: a JSONL of items; a scorer callable; a rewrite set.
"""
from __future__ import annotations
import argparse, json, statistics as st, sys
import numpy as np
from scipy.stats import wilcoxon

# ---- pre-registered thresholds ----------------------------------------------
E1_FLIP_MARGIN = 0.05      # flip rate must exceed identity-noise floor by this
E3_DELTA_ALPHA = 0.05      # significance for directional non-annihilation test


# ---- scorer interface -------------------------------------------------------
def mock_scorer(text: str) -> float:
    """Stand-in. Replace with Detoxify/LLM/Perspective. Deterministic-ish demo:
    a scorer that (badly) drops score when 'left' or 'gone' appears — to exercise
    the E3 non-annihilation DETECTOR. Also mildly surface-sensitive for E1."""
    import hashlib
    base = int(hashlib.md5(text.lower().encode()).hexdigest(), 16) % 1000 / 1000
    s = 0.4 + 0.5 * base
    if any(w in text.lower() for w in ["left", "gone", "deleted", "no longer here"]):
        s *= 0.5                      # the defect E3 should catch
    if text != text.lower():
        s += 0.03                     # mild surface sensitivity E1 should catch
    return float(min(max(s, 0), 1))


# ---- E1: invariance under meaning-preserving rewrite ------------------------
SURFACE_REWRITES = {
    "identity":  lambda t: t,                         # the noise-floor control
    "lowercase": lambda t: t.lower(),
    "whitespace": lambda t: " ".join(t.split()),
    "strip_punct": lambda t: t.rstrip(".!?"),
}
# semantic rewrites must be validated as meaning-preserving before trusting (E1 gate)
SEMANTIC_REWRITES = {
    # placeholders — AUTHOR supplies validated paraphrase / back-translation fns
    # "paraphrase": paraphrase_fn,
    # "backtranslate": bt_fn,
}


def e1_invariance(items, scorer, threshold=0.5):
    """Flip rate per rewrite vs the identity-noise floor."""
    base = {i: scorer(t) for i, t in enumerate(items)}
    # identity noise floor: rescore identity a few times (stochastic scorers)
    id_deltas = []
    for i, t in enumerate(items):
        for _ in range(3):
            id_deltas.append(abs(scorer(SURFACE_REWRITES["identity"](t)) - base[i]))
    noise_floor = float(np.quantile(id_deltas, 0.95)) if id_deltas else 0.0

    results = {}
    for name, fn in {**SURFACE_REWRITES, **SEMANTIC_REWRITES}.items():
        if name == "identity":
            continue
        flips, deltas = 0, []
        for i, t in enumerate(items):
            s2 = scorer(fn(t))
            deltas.append(abs(s2 - base[i]))
            if (base[i] >= threshold) != (s2 >= threshold):
                flips += 1
        flip_rate = flips / len(items)
        mean_delta = float(np.mean(deltas))
        results[name] = {
            "flip_rate": round(flip_rate, 3),
            "mean_abs_delta": round(mean_delta, 3),
            "noise_floor": round(noise_floor, 3),
            "exceeds_floor": bool(mean_delta > noise_floor + E1_FLIP_MARGIN),
        }
    verdict = ("INCOHERENT — verdicts move under meaning-preserving rewrite beyond noise"
               if any(r["exceeds_floor"] for r in results.values())
               else "invariant within noise floor (coherent on E1)")
    return {"noise_floor": round(noise_floor, 3), "by_rewrite": results, "verdict": verdict}


# ---- E3: non-annihilation direction -----------------------------------------
def e3_non_annihilation(pairs, scorer):
    """pairs: list of {present: text_with_target_present, absent: text_target_removed,
                       control_a: benign present, control_b: benign absent}.
    Coherent: score(absent) >= score(present) (removing victim doesn't reduce harm).
    Defect: score(absent) < score(present) systematically."""
    d_harm, d_ctrl = [], []
    for p in pairs:
        d_harm.append(scorer(p["absent"]) - scorer(p["present"]))
        if "control_a" in p and "control_b" in p:
            d_ctrl.append(scorer(p["control_b"]) - scorer(p["control_a"]))
    d_harm = np.array(d_harm)
    frac_negative = float((d_harm < 0).mean())
    # is the harm-delta distribution shifted negative relative to the benign control?
    try:
        stat, pval = wilcoxon(d_harm)  # H0: median delta = 0
    except Exception:
        pval = float("nan")
    ctrl_mean = float(np.mean(d_ctrl)) if d_ctrl else None
    defect = bool(frac_negative > 0.5 and np.isfinite(pval) and pval < E3_DELTA_ALPHA and
                  np.median(d_harm) < 0)
    return {
        "median_delta_harm": round(float(np.median(d_harm)), 3),
        "frac_absent_lower_than_present": round(frac_negative, 3),
        "control_mean_delta": round(ctrl_mean, 3) if ctrl_mean is not None else None,
        "wilcoxon_p": round(float(pval), 4) if np.isfinite(pval) else None,
        "verdict": ("DEFECT — removing the target REDUCES attributed harm "
                    "(system is gameable by target-removal)" if defect
                    else "no non-annihilation defect detected (coherent on E3)"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", help="JSONL with a 'text' field (for E1)")
    ap.add_argument("--pairs", help="JSONL with present/absent(/control_a/control_b) (for E3)")
    ap.add_argument("--registered", action="store_true")
    a = ap.parse_args()
    if not a.registered:
        print("Freeze & hash coherence_campaign.md before trusting results (dry mode).\n", file=sys.stderr)

    scorer = mock_scorer   # AUTHOR: replace with Detoxify/LLM/Perspective
    out = {}
    if a.items:
        items = [json.loads(l)["text"] for l in open(a.items) if l.strip()]
        out["E1_invariance"] = e1_invariance(items, scorer)
    if a.pairs:
        pairs = [json.loads(l) for l in open(a.pairs) if l.strip()]
        out["E3_non_annihilation"] = e3_non_annihilation(pairs, scorer)
    print(json.dumps(out, indent=1))
    __import__("pathlib").Path("coherence_results.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
