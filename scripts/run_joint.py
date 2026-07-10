"""Run domain-adversarial joint (cross-dataset) training for one dimension.

    python run_joint.py <dim_joint> [--steps N] [--batch B] [--lambda L] [--holdout F]

Prints the honest CROSS-DATASET gate: structure_auroc (headline), severity_auroc (polar radius /
general factor), domain_acc (adversary — want near chance). Also runs a BASELINE cross-dataset
AUROC on the untrained encoder for reference, so the lift is visible.
"""

import argparse
import os
import sys

# Prefer the installed package; fall back to the in-repo src/ (or $XBSE_SRC) for dev/remote runs.
try:
    import xbse  # noqa: F401
except ImportError:
    sys.path.insert(
        0,
        os.environ.get(
            "XBSE_SRC",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"),
        ),
    )

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
    ap.add_argument("--base-model", dest="base_model", default="BAAI/bge-m3")
    ap.add_argument("--pooling", default="mean", choices=["mean", "last"])
    ap.add_argument("--trust-remote-code", dest="trust", action="store_true")
    ap.add_argument("--max-len", dest="max_len", type=int, default=None)
    a = ap.parse_args()

    src = BUILDERS[a.dim](holdout_frac=a.holdout)
    rows = src._rows()
    from collections import Counter

    print(
        f"[{a.dim}] rows={len(rows)} by-domain={Counter(r[0] for r in rows)} "
        f"by-sign={Counter(r[2] for r in rows)}",
        flush=True,
    )

    enc = BSEEncoder(
        base_model=a.base_model,
        max_len=a.max_len or src.max_len,
        device="cuda",
        pooling=a.pooling,
        trust_remote_code=a.trust,
    )

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
