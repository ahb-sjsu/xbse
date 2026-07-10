"""Pre-registered independent-corpora experiment (experiments/prereg_independent_corpora.md).

Trains the two re-corpused family encoders (care_v2 = Social-Chem-care + Moral-Stories;
fairness_v2 = Social-Chem-fairness + Measuring-Hate-Speech) on BGE-M3, and applies the
MANIPULATION-CHECK GATE (reviewer-3 point 4) before the decoupling prediction is scored:

  a re-corpused encoder must beat max(untrained-null, BoW-null) by >= 0.10 on its OWN cross-dataset
  held-out set. A gate failure => "inconclusive - corpus quality", not a falsification.

Saves checkpoints to /home/claude/xbse_ckpt/{dim}.pt; the 4x4 cross-transfer is run separately with
cross_transfer.py over {care_v2, fairness_v2, legitimacy_joint, epistemic_joint}.
"""

import json
import os
import sys

sys.path.insert(
    0,
    os.environ.get(
        "XBSE_SRC",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"),
    ),
)

from xbse.encoder import BSEEncoder  # noqa: E402
from xbse.instances.joint_builders import BUILDERS  # noqa: E402
from xbse.train_adv import train_adversarial  # noqa: E402
from xbse.validate import gate  # noqa: E402

MARGIN = 0.10
CKDIR = os.environ.get("XBSE_CKPT_DIR", "/home/claude/xbse_ckpt")
DIMS = ["care_v2_joint", "fairness_v2_joint"]

results = {}
for dim in DIMS:
    src = BUILDERS[dim](holdout_frac=0.12)
    rows = src._rows()
    from collections import Counter

    print(
        f"\n[{dim}] rows={len(rows)} by-domain={Counter(r[0] for r in rows)} "
        f"by-sign={Counter(r[2] for r in rows)}",
        flush=True,
    )
    ev = src.heldout_eval()
    enc = BSEEncoder(base_model="BAAI/bge-m3", pooling="mean", max_len=128, device="cuda")

    base = gate(enc, ev)  # untrained null
    print(
        f"[{dim}] UNTRAINED null structure_auroc={base['structure_auroc']:.4f} "
        f"bow={base['bow_auroc']:.4f}",
        flush=True,
    )

    ckpt = os.path.join(CKDIR, f"{dim}.pt")
    train_adversarial(
        enc, src, epochs=6, batch_size=24, lr=2e-5, max_steps=1200,
        max_lambda=0.0, checkpoint_path=ckpt,
    )
    fin = gate(enc, ev)  # trained
    null, bow, tr = base["structure_auroc"], fin["bow_auroc"], fin["structure_auroc"]
    margin = tr - max(null, bow)
    passed = margin >= MARGIN
    print(
        f"[{dim}] TRAINED structure_auroc={tr:.4f}  bow={bow:.4f}  null={null:.4f}  "
        f"margin(vs max-null)={margin:+.4f}  GATE={'PASS' if passed else 'FAIL'}",
        flush=True,
    )
    results[dim] = {
        "trained_auroc": round(tr, 4), "untrained_null": round(null, 4),
        "bow_null": round(bow, 4), "margin": round(margin, 4), "gate_passed": bool(passed),
        "checkpoint": ckpt,
    }

print("\n=== MANIPULATION-CHECK GATE (prereg pt4) ===", flush=True)
print(json.dumps(results, indent=2), flush=True)
print(
    "\nNext: only gate-passing encoders are scored in the 4x4 transfer. "
    "Run cross_transfer.py over {care_v2, fairness_v2, legitimacy_joint, epistemic_joint}.",
    flush=True,
)
