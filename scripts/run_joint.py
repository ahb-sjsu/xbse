"""Run domain-adversarial joint (cross-dataset) training for one dimension.

    python run_joint.py <dim_joint> [--steps N] [--batch B] [--lambda L] [--holdout F]

Prints the honest CROSS-DATASET gate: structure_auroc (headline), severity_auroc (polar radius /
general factor), domain_acc (adversary — want near chance). Also runs a BASELINE cross-dataset
AUROC on the untrained encoder for reference, so the lift is visible.
"""

import argparse
import sys

sys.path.insert(0, "/home/claude/xbse/src")

from xbse.encoder import BSEEncoder
from xbse.instances.joint_builders import BUILDERS
from xbse.train_adv import train_adversarial
from xbse.validate import gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dim")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0)
    ap.add_argument("--holdout", type=float, default=0.12)
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args()

    src = BUILDERS[a.dim](holdout_frac=a.holdout)
    rows = src._rows()
    from collections import Counter

    print(
        f"[{a.dim}] rows={len(rows)} by-domain={Counter(r[0] for r in rows)} "
        f"by-sign={Counter(r[2] for r in rows)}",
        flush=True,
    )

    enc = BSEEncoder(base_model="BAAI/bge-m3", max_len=src.max_len, device="cuda")

    # baseline: untrained encoder's cross-dataset AUROC (the number the single-corpus model degraded to)
    base = gate(enc, src.heldout_eval())
    print(
        f"[{a.dim}] BASELINE cross-dataset structure_auroc={base['structure_auroc']:.4f} "
        f"fuzz={base['fuzz_ratio']:.3f}",
        flush=True,
    )

    ckpt = a.ckpt or f"/home/claude/xbse_ckpt/{a.dim}.pt"
    train_adversarial(
        enc,
        src,
        epochs=6,
        batch_size=a.batch,
        lr=a.lr,
        max_steps=a.steps,
        max_lambda=a.lam,
        checkpoint_path=ckpt,
        report_path=f"/home/claude/xbse_ckpt/{a.dim}_report.json",
    )


if __name__ == "__main__":
    main()
