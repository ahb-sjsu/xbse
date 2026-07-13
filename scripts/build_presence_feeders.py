"""Attempt 2: foundation-PRESENCE (identity) feeders through the joint cross-corpus path + cross-register gate.

Attempt 1 (linear probe on frozen BGE-M3) failed cross-register transfer (foundation_presence_findings.md)
because a plain head overfits register. This trains a joint contrastive feeder on TWO independent
foundation-labeled corpora (Social-Chem rot-moral-foundations x MFRC Reddit annotations) with PRESENCE
labels ('+' = foundation-k engaged, '-' = a different foundation), optionally with the domain-adversarial
DomainHead engaged (`--lam` > 0 forces register-invariance by stripping corpus-identifying features).

IMPORTANT (2026-07-13): `--lam` is the domain-adversarial coefficient ceiling (`train_adversarial`'s
`max_lambda`). At `--lam 0.0` the adversary is INERT — its gradient is never reversed into the encoder,
so this is a pure two-corpus joint-contrastive feeder (final domain_acc stays ~1.0). The first attempt-2
sweep (results in `~/presence_feeders/`) ran at lam=0.0 by mistake and must NOT be described as
domain-adversarial. Run `--lam 1.0` for the genuine adversarial test.

Gate (identical to every channel): cross-dataset held-out presence AUROC beats BOTH the untrained-encoder
null AND the TF-IDF BoW null by >= 0.10, fuzz > 1.0; nulls measured on the untrained BGE-M3 before
training and frozen. PASS => the foundation-identity axis has a register-INVARIANT core (transfers
RoT<->Reddit). FAIL => identity is register-bound (honest finding). GPU 1 only.
"""

import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ.setdefault("HF_HOME", "/archive/cache/huggingface")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import json  # noqa: E402
import sys  # noqa: E402
from collections import Counter  # noqa: E402

sys.path.insert(0, os.path.expanduser("~/xbse/src"))

from xbse.bar import Bar  # noqa: E402
from xbse.encoder import BSEEncoder  # noqa: E402
from xbse.instances.joint_builders import build_foundation_presence_joint  # noqa: E402
from xbse.train_adv import train_adversarial  # noqa: E402
from xbse.validate import gate  # noqa: E402

CKDIR = os.path.expanduser("~/xbse_ckpt")
FOUNDATIONS = ["care", "fairness", "legitimacy", "loyalty", "purity"]


def build_one(f, lam, suffix):
    print(f"\n===== {f} presence (lam={lam}) =====", flush=True)
    src = build_foundation_presence_joint(f, holdout_frac=0.12)
    rows = src._rows()
    print(
        f"[{f}] rows={len(rows)} by-domain={Counter(r[0] for r in rows)} "
        f"by-sign={Counter(r[2] for r in rows)}",
        flush=True,
    )
    if len(rows) < 200:
        return {"dim": f, "error": "insufficient rows", "n_rows": len(rows)}

    ev = src.heldout_eval()
    enc = BSEEncoder(base_model="BAAI/bge-m3", pooling="mean", max_len=128, device="cuda")

    base = gate(enc, ev)  # untrained null, before training
    null, bow0 = float(base["structure_auroc"]), float(base["bow_auroc"])
    max_null = max(null, bow0)
    print(
        f"[{f}] UNTRAINED null structure={null:.4f} bow={bow0:.4f} max_null={max_null:.4f}",
        flush=True,
    )

    src.bar = Bar(
        auroc_min=round(min(max(max_null + 0.10, 0.5001), 0.999), 3),
        fuzz_min=1.0,
        policy="baseline_relative",
        margin=0.10,
        baseline_auroc=round(max_null, 3),
        source=f"baseline-relative(margin 0.10) over max(untrained {null:.3f}, BoW {bow0:.3f})",
        derivation=(
            "VALIDATED iff cross-dataset held-out PRESENCE AUROC beats BOTH the untrained-encoder null "
            f"({null:.3f}) and the TF-IDF BoW null ({bow0:.3f}) by >=0.10 on the same held-out pairs."
        ),
        registered="2026-07-12",
    )

    ckpt = f"{CKDIR}/{f}_presence{suffix}_joint.pt"
    train_adversarial(
        enc,
        src,
        epochs=6,
        batch_size=24,
        lr=2e-5,
        max_steps=1200,
        max_lambda=lam,
        checkpoint_path=ckpt,
        report_path=f"{CKDIR}/{f}_presence{suffix}_joint_report.json",
    )

    fin = gate(enc, ev)
    tr, bown = float(fin["structure_auroc"]), float(fin["bow_auroc"])
    fuzz = float(fin.get("fuzz_ratio", float("nan")))
    margin = tr - max(null, bown)
    passed = bool(margin >= 0.10 and fuzz > 1.0)
    rec = {
        "dim": f"{f}_presence{suffix}_joint",
        "max_lambda": lam,
        "domain_adversarial": lam > 0.0,
        "trained_auroc": round(tr, 4),
        "untrained_null": round(null, 4),
        "bow_null": round(bown, 4),
        "margin_vs_max_null": round(margin, 4),
        "fuzz_ratio": round(fuzz, 4),
        "gate_passed": passed,
        "n_rows": len(rows),
        "by_domain": dict(Counter(r[0] for r in rows)),
        "checkpoint": f"xbse_ckpt/{f}_presence{suffix}_joint.pt",
    }
    print("RESULT " + json.dumps(rec), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lam",
        type=float,
        default=0.0,
        help="domain-adversarial coefficient ceiling (train_adversarial max_lambda); "
        "0.0 = adversary INERT (joint-contrastive only), 1.0 = genuine domain-adversarial",
    )
    ap.add_argument(
        "--tag",
        default="",
        help="output tag; checkpoints -> {f}_presence_{tag}_joint.pt, results -> ~/presence_feeders_{tag}/. "
        "Empty tag reproduces the original lam=0 baseline paths.",
    )
    a = ap.parse_args()
    suffix = f"_{a.tag}" if a.tag else ""
    out = os.path.expanduser(f"~/presence_feeders{suffix}")
    os.makedirs(out, exist_ok=True)
    os.makedirs(CKDIR, exist_ok=True)
    print(f"[config] lam={a.lam} tag='{a.tag}' out={out}", flush=True)

    results = [build_one(f, a.lam, suffix) for f in FOUNDATIONS]
    json.dump(results, open(f"{out}/presence_results.json", "w"), indent=2)
    print("\nSAVED " + f"{out}/presence_results.json", flush=True)
    for r in results:
        print(
            f"  {r['dim']}: passed={r.get('gate_passed')} auroc={r.get('trained_auroc')} "
            f"margin={r.get('margin_vs_max_null')}",
            flush=True,
        )
    n = sum(1 for r in results if r.get("gate_passed"))
    print(f"PASSED {n}/{len(FOUNDATIONS)}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
