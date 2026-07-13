"""Phase A1: build + gate the general-valence channel G (bifactor MoralVector readout).

Replaces the PCA proxy pc0 (a linear summary of feeder scores, no gate) with a TRAINED general-valence
BSE encoder that passes the SAME pre-registered cross-dataset gate as every axis, measured at 3 seeds
(retrain stability is part of the A-gate). See experiments/prereg_bifactor_readout.md (prereg 2026-07-12).

Corpus: pooled signed Social-Chem (ALL foundation categories, no keyword filter) x all signed
ETHICS-commonsense — the general good/bad direction, not a foundation. Gate: cross-dataset held-out
AUROC beats BOTH the untrained-encoder null AND the TF-IDF BoW null by >= 0.10, fuzz > 1.0; nulls
measured on the untrained BGE-M3 before training and frozen. GPU 1 only.

A2 (residualize specifics against G -> P1/P3) is a SEPARATE stage, run only if A1 passes — falsification
order: no bifactor readout is worth building if a trained G cannot even clear its own gate.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # leave GPU 0
os.environ.setdefault("HF_HOME", "/archive/cache/huggingface")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
from collections import Counter  # noqa: E402

sys.path.insert(0, os.path.expanduser("~/xbse/src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from xbse.bar import Bar  # noqa: E402
from xbse.encoder import BSEEncoder  # noqa: E402
from xbse.instances.joint_builders import build_general_valence_joint  # noqa: E402
from xbse.train_adv import train_adversarial  # noqa: E402
from xbse.validate import gate  # noqa: E402

CKDIR = os.path.expanduser("~/xbse_ckpt")
SEEDS = [0, 1, 2]


def fresh_encoder():
    return BSEEncoder(base_model="BAAI/bge-m3", pooling="mean", max_len=128, device="cuda")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lam", type=float, default=0.0, help="domain-adversarial max_lambda (0.0=inert)"
    )
    ap.add_argument(
        "--tag", default="", help="output tag; suffixes ckpts/report/dir, empty=lam=0 baseline"
    )
    a = ap.parse_args()
    lam, suffix = a.lam, (f"_{a.tag}" if a.tag else "")
    out = os.path.expanduser(f"~/bifactor_A1{suffix}")
    os.makedirs(out, exist_ok=True)
    os.makedirs(CKDIR, exist_ok=True)
    print(f"[config] lam={lam} tag='{a.tag}' out={out}", flush=True)
    print("\n===== general_valence (G) — Phase A1 =====", flush=True)
    src = build_general_valence_joint(holdout_frac=0.12)
    rows = src._rows()
    print(
        f"[G] rows={len(rows)} by-domain={Counter(r[0] for r in rows)} "
        f"by-sign={Counter(r[2] for r in rows)}",
        flush=True,
    )
    ev = src.heldout_eval()

    # --- nulls: measured ONCE on the untrained model, before any training (frozen into the bar) ---
    enc0 = fresh_encoder()
    base = gate(enc0, ev)
    null, bow = float(base["structure_auroc"]), float(base["bow_auroc"])
    max_null = max(null, bow)
    print(
        f"[G] UNTRAINED null structure_auroc={null:.4f} bow={bow:.4f} -> max_null={max_null:.4f}",
        flush=True,
    )
    del enc0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    bar = Bar(
        auroc_min=round(min(max(max_null + 0.10, 0.5001), 0.999), 3),
        fuzz_min=1.0,
        policy="baseline_relative",
        margin=0.10,
        baseline_auroc=round(max_null, 3),
        source=f"baseline-relative(margin 0.10) over max(untrained {null:.3f}, BoW {bow:.3f})",
        derivation=(
            "VALIDATED iff cross-dataset held-out AUROC beats BOTH the untrained-encoder null "
            f"({null:.3f}) and the TF-IDF BoW null ({bow:.3f}) by >=0.10 on the same held-out "
            "pairs; nulls are properties of (untrained model + corpus)."
        ),
        registered="2026-07-12",
    )

    per_seed = []
    for s in SEEDS:
        print(f"\n----- seed {s} -----", flush=True)
        random.seed(s)
        torch.manual_seed(s)
        np.random.seed(s)
        enc = fresh_encoder()
        src.bar = bar
        ckpt = f"{CKDIR}/general_valence_joint{suffix}_s{s}.pt"
        rpt = f"{CKDIR}/general_valence_joint{suffix}_report_s{s}.json"
        train_adversarial(
            enc,
            src,
            epochs=6,
            batch_size=24,
            lr=2e-5,
            max_steps=1200,
            max_lambda=lam,
            checkpoint_path=ckpt,
            report_path=rpt,
        )
        fin = gate(enc, ev)
        tr, bown = float(fin["structure_auroc"]), float(fin["bow_auroc"])
        fuzz = float(fin.get("fuzz_ratio", float("nan")))
        margin = tr - max(null, bown)
        passed = bool(margin >= 0.10 and fuzz > 1.0)
        row = {
            "seed": s,
            "trained_auroc": round(tr, 4),
            "bow_null": round(bown, 4),
            "margin_vs_max_null": round(margin, 4),
            "fuzz_ratio": round(fuzz, 4),
            "gate_passed": passed,
            "checkpoint": f"xbse_ckpt/general_valence_joint{suffix}_s{s}.pt",
        }
        per_seed.append(row)
        print("SEED_RESULT " + json.dumps(row), flush=True)
        del enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    aurocs = [r["trained_auroc"] for r in per_seed]
    margins = [r["margin_vs_max_null"] for r in per_seed]
    fuzzes = [r["fuzz_ratio"] for r in per_seed]
    result = {
        "dim": f"general_valence_joint{suffix}",
        "max_lambda": lam,
        "domain_adversarial": lam > 0.0,
        "auroc_mean": round(float(np.mean(aurocs)), 4),
        "auroc_std": round(float(np.std(aurocs)), 4),
        "margin_mean": round(float(np.mean(margins)), 4),
        "min_fuzz": round(float(min(fuzzes)), 4),
        "untrained_null": round(null, 4),
        "bow_null": round(bow, 4),
        "max_null_frozen": round(max_null, 4),
        # strict A-gate: mean margin clears 0.10, every seed individually passes, all fuzz > 1.0
        "gate_passed": bool(np.mean(margins) >= 0.10 and all(r["gate_passed"] for r in per_seed)),
        "n_rows": len(rows),
        "by_domain": dict(Counter(r[0] for r in rows)),
        "n_seeds": len(SEEDS),
        "per_seed": per_seed,
    }
    json.dump(result, open(f"{out}/bifactor_A1_result.json", "w"), indent=2)
    print("\nRESULT " + json.dumps(result), flush=True)
    print(f"SAVED {out}/bifactor_A1_result.json", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
