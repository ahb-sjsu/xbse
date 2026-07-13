"""Build + gate the B1 loyalty & purity feeders (MoralVector roadmap Phase B1).

Same pre-registered admission gate as identity_attack (prereg_mft_loyalty_purity.md):
VALIDATED iff cross-dataset held-out AUROC beats BOTH the untrained-encoder null AND the TF-IDF
bag-of-words null by >= 0.10 on the same held-out pairs, fuzz > 1.0. Nulls measured on the untrained
BGE-M3 before training, frozen into the Bar. Two independent VALENCE corpora per axis:
Social-Chem (loyalty-betrayal / sanctity-degradation, signed) x ETHICS-commonsense (keyword scenarios).
Plus an independent-register MFRC PRESENCE cross-check (bonus, not the gate). GPU 1 only.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # leave GPU 0
os.environ.setdefault("HF_HOME", "/archive/cache/huggingface")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import argparse
import json
import sys
from collections import Counter

sys.path.insert(0, os.path.expanduser("~/xbse/src"))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from xbse.bar import Bar  # noqa: E402
from xbse.encoder import BSEEncoder  # noqa: E402
from xbse.instances.joint_builders import build_foundation_joint  # noqa: E402
from xbse.scorer import DimensionScorer  # noqa: E402
from xbse.train_adv import train_adversarial  # noqa: E402
from xbse.validate import gate  # noqa: E402

CKDIR = os.path.expanduser("~/xbse_ckpt")
AXES = ["loyalty", "purity"]
MFRC_POS = {"loyalty": {"Loyalty"}, "purity": {"Purity"}}


def mfrc_crosscheck(enc, src, report, axis):
    """Bonus: does the trained valence feeder separate MFRC foundation-present from Non-Moral?
    Presence, not valence, so we score |valence| (either pole = foundation engaged). AUROC > 0.60 = the
    English-scenario feeder generalises to an independent-register, independent-provenance corpus.
    """
    try:
        from datasets import load_dataset

        d = load_dataset("USC-MOLA-Lab/MFRC", split="train_dedup")
        pos, neg = [], []
        for r in d:
            t = (r.get("text") or "").strip()
            if len(t) < 12:
                continue
            ann = str(r.get("annotation") or "")
            if ann in MFRC_POS[axis] and len(pos) < 800:
                pos.append(t)
            elif ann == "Non-Moral" and len(neg) < 800:
                neg.append(t)
            if len(pos) >= 800 and len(neg) >= 800:
                break
        n = min(len(pos), len(neg))
        if n < 50:
            return {"note": f"insufficient MFRC rows (pos={len(pos)}, neg={len(neg)})"}
        texts = pos[:n] + neg[:n]
        y = [1] * n + [0] * n
        sc = DimensionScorer.from_pairsource(enc, src, report, report.checkpoint_hash)
        mag = [abs(float(v.value)) for v in sc.score_batch(texts)]
        return {"auroc_presence": round(float(roc_auc_score(y, mag)), 4), "n_per_class": n}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:160]}


def build_one(axis, lam, suffix):
    print(f"\n===== {axis} (lam={lam}) =====", flush=True)
    src = build_foundation_joint(axis, holdout_frac=0.12)
    rows = src._rows()
    print(
        f"[{axis}] rows={len(rows)} by-domain={Counter(r[0] for r in rows)} "
        f"by-sign={Counter(r[2] for r in rows)}",
        flush=True,
    )
    if len(rows) < 200:
        print(f"[{axis}] TOO FEW ROWS — aborting this axis", flush=True)
        return {"dim": axis, "error": "insufficient rows", "n_rows": len(rows)}

    ev = src.heldout_eval()
    enc = BSEEncoder(base_model="BAAI/bge-m3", pooling="mean", max_len=128, device="cuda")

    base = gate(enc, ev)  # untrained null — BEFORE training, frozen into the bar
    null, bow0 = float(base["structure_auroc"]), float(base["bow_auroc"])
    print(f"[{axis}] UNTRAINED null structure_auroc={null:.4f} bow={bow0:.4f}", flush=True)

    src.bar = Bar(
        auroc_min=round(min(max(null + 0.10, 0.5001), 0.999), 3),
        fuzz_min=1.0,
        policy="baseline_relative",
        margin=0.10,
        baseline_auroc=round(null, 3),
        source=f"baseline-relative(margin 0.10) over untrained null {null:.3f} + BoW null",
        derivation=(
            "VALIDATED iff cross-dataset held-out AUROC beats BOTH the untrained-encoder null "
            f"({null:.3f}) and the TF-IDF bag-of-words null by >=0.10 on the same held-out pairs."
        ),
        registered="2026-07-12",
    )

    ckpt = f"{CKDIR}/{axis}{suffix}_joint.pt"
    report_path = f"{CKDIR}/{axis}{suffix}_joint_report.json"
    train_adversarial(
        enc,
        src,
        epochs=6,
        batch_size=24,
        lr=2e-5,
        max_steps=1200,
        max_lambda=lam,
        checkpoint_path=ckpt,
        report_path=report_path,
    )

    fin = gate(enc, ev)  # trained
    tr, bown = float(fin["structure_auroc"]), float(fin["bow_auroc"])
    fuzz = float(fin.get("fuzz_ratio", float("nan")))
    margin = tr - max(null, bown)
    passed = bool(margin >= 0.10 and fuzz > 1.0)

    from xbse.report import Report

    report = Report(**json.load(open(report_path)))
    xcheck = mfrc_crosscheck(enc, src, report, axis)

    result = {
        "dim": f"{axis}{suffix}_joint",
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
        "checkpoint": ckpt,
        "mfrc_presence_crosscheck": xcheck,
    }
    print("RESULT " + json.dumps(result), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=0.0, help="domain-adversarial max_lambda (0.0=inert)")
    ap.add_argument("--tag", default="", help="output tag; suffixes ckpts/report/dir, empty=lam=0 baseline")
    a = ap.parse_args()
    suffix = f"_{a.tag}" if a.tag else ""
    out = os.path.expanduser(f"~/mft_b1{suffix}")
    os.makedirs(out, exist_ok=True)
    print(f"[config] lam={a.lam} tag='{a.tag}' out={out}", flush=True)
    results = [build_one(a2, a.lam, suffix) for a2 in AXES]
    json.dump(results, open(f"{out}/b1_results.json", "w"), indent=2)
    print("\nSAVED " + f"{out}/b1_results.json", flush=True)
    for r in results:
        print(
            f"  {r['dim']}: passed={r.get('gate_passed')} auroc={r.get('trained_auroc')} "
            f"margin={r.get('margin_vs_max_null')} mfrc={r.get('mfrc_presence_crosscheck')}",
            flush=True,
        )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
