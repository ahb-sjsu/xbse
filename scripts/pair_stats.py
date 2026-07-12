"""Per-dimension pair statistics: bootstrap CIs, pair difficulty, and counts.

Addresses reviewer-2 points 5 (CIs before circulating) and 6 (pair difficulty explains the epistemic
attractor), plus the requested per-dimension (n_pos, n_neg, unique-anchor, pair) counts. For each
dimension it loads the trained joint encoder and reports, on the SAME held-out structural pairs:

  frozen-difficulty : untrained BGE-M3 structure_auroc  (how easy are the pairs before training?)
  trained-auroc     : trained encoder structure_auroc   (matches paper §3.1)
  95% CI            : text-level bootstrap over UNIQUE ANCHORS (respects pseudo-replication)
  counts            : n_pairs, n_unique_anchors, n_pos, n_neg

If epistemic's frozen-difficulty is the highest, its held-out pairs are simply the easiest — which
would explain its being the transfer-matrix attractor AND inflate its standalone PASS.
"""

import os
import sys

import numpy as np

sys.path.insert(
    0,
    os.environ.get(
        "XBSE_SRC",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"),
    ),
)

import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from xbse.encoder import BSEEncoder  # noqa: E402
from xbse.instances.joint_builders import BUILDERS  # noqa: E402

CKDIR = os.environ.get("XBSE_CKPT_DIR", "/home/claude/xbse_ckpt")
BOOT = int(os.environ.get("XBSE_BOOTSTRAP", "1000"))
DIMS = [
    "privacy_joint",
    "environmental_joint",
    "rights_joint",
    "care_joint",
    "fairness_joint",
    "legitimacy_joint",
    "epistemic_joint",
    "physharm_joint",
    "autonomy_joint",
]


def _sims(enc, sp):
    za = enc.encode([a for a, _b, _l in sp]).cpu().numpy()
    zb = enc.encode([b for _a, b, _l in sp]).cpu().numpy()
    return (za * zb).sum(-1)


def _boot_ci(sims, y, anchors, n=BOOT):
    uniq = list(set(anchors))
    by = {u: [] for u in uniq}
    for i, a in enumerate(anchors):
        by[a].append(i)
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = [i for k in pick for i in by[uniq[k]]]
        yy, ss = y[idx], sims[idx]
        if 0 < yy.sum() < len(yy):
            out.append(roc_auc_score(yy, ss))
    return (
        (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))
        if out
        else (float("nan"),) * 2
    )


rows = []
for dim in DIMS:
    ck = os.path.join(CKDIR, f"{dim}.pt")
    if not os.path.exists(ck):
        print(f"[{dim}] no checkpoint at {ck} — skip", flush=True)
        continue
    src = BUILDERS[dim](holdout_frac=0.12)
    sp = src.heldout_eval()["structural_pairs"]
    anchors = [a for a, _b, _l in sp]
    y = np.array([1 if l else 0 for _a, _b, l in sp])

    frozen = BSEEncoder(base_model="BAAI/bge-m3", pooling="mean", max_len=128, device="cuda")
    fro_auroc = roc_auc_score(y, _sims(frozen, sp))
    frozen.load_state_dict(torch.load(ck, map_location="cuda"))
    frozen.eval()
    s = _sims(frozen, sp)
    tr_auroc = roc_auc_score(y, s)
    lo, hi = _boot_ci(s, y, anchors)
    nuniq = len(set(anchors))
    print(
        f"[{dim}] frozen={fro_auroc:.3f} trained={tr_auroc:.3f} CI95=[{lo:.3f},{hi:.3f}] "
        f"pairs={len(sp)} anchors={nuniq} pos={int(y.sum())} neg={int((1 - y).sum())}",
        flush=True,
    )
    rows.append((dim, fro_auroc, tr_auroc, lo, hi, len(sp), nuniq, int(y.sum())))
    del frozen
    torch.cuda.empty_cache()

print("\n=== SUMMARY (sorted by frozen-difficulty; highest = easiest pairs) ===", flush=True)
print(
    f"{'dimension':<22}{'frozen':>8}{'trained':>9}{'CI95':>18}{'pairs':>7}{'anchors':>9}{'pos':>6}"
)
for dim, fr, tr, lo, hi, npr, na, npos in sorted(rows, key=lambda r: -r[1]):
    print(
        f"{dim:<22}{fr:>8.3f}{tr:>9.3f}   [{lo:.3f},{hi:.3f}]{npr:>7}{na:>9}{npos:>6}", flush=True
    )
