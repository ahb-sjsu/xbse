"""Bootstrap 95% CIs for every cell of an N x N cross-transfer matrix.

Firms the marginal verdicts of §3.7 (the +0.16 care_v2↔fair_v2 decoupling and the -0.03 legit↔epis
control) by putting text-level bootstrap CIs (resampling unique anchors) on each cross-transfer cell,
rather than relying on point estimates given the ~±0.06 retrain variance.

Usage (same arg form as cross_transfer.py):
  python matrix_ci.py care_v2:care_v2_joint:/ck/care_v2.pt fair_v2:fairness_v2_joint:/ck/fair_v2.pt ...
Env: XBSE_BASE_MODEL (BAAI/bge-m3), XBSE_POOLING (mean), XBSE_MAX_LEN (128), XBSE_BOOTSTRAP (1000).
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

BASE = os.environ.get("XBSE_BASE_MODEL", "BAAI/bge-m3")
POOLING = os.environ.get("XBSE_POOLING", "mean")
MAXLEN = int(os.environ.get("XBSE_MAX_LEN", "128"))
BOOT = int(os.environ.get("XBSE_BOOTSTRAP", "1000"))

PAIRS = [tuple(a.split(":", 2)) for a in sys.argv[1:]]
assert PAIRS and all(len(p) == 3 for p in PAIRS), "args: label:builder:ckpt"
LABELS = [p[0] for p in PAIRS]

evs = {}
for label, builder, _ck in PAIRS:
    evs[label] = BUILDERS[builder](holdout_frac=0.12).heldout_eval()["structural_pairs"]
    print(f"[eval] {label} built ({len(evs[label])} pairs)", flush=True)


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
        idx = [i for k in rng.integers(0, len(uniq), len(uniq)) for i in by[uniq[k]]]
        yy, ss = y[idx], sims[idx]
        if 0 < yy.sum() < len(yy):
            out.append(roc_auc_score(yy, ss))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


cells = {}
for enc_label, _builder, ck in PAIRS:
    enc = BSEEncoder(base_model=BASE, pooling=POOLING, max_len=MAXLEN, device="cuda")
    enc.load_state_dict(torch.load(ck, map_location="cuda"))
    enc.eval()
    for ev in LABELS:
        sp = evs[ev]
        anchors = [a for a, _b, _l in sp]
        y = np.array([1 if l else 0 for _a, _b, l in sp])
        s = _sims(enc, sp)
        au = roc_auc_score(y, s)
        lo, hi = _boot_ci(s, y, anchors)
        cells[(enc_label, ev)] = (au, lo, hi)
        tag = "within" if enc_label == ev else "cross"
        print(f"  {enc_label:>8} -> {ev:<8} {au:.3f} [{lo:.3f},{hi:.3f}] {tag}", flush=True)
    del enc
    torch.cuda.empty_cache()

print("\n=== marginal §3.7 verdicts with CIs ===", flush=True)
for a, b in [(LABELS[0], LABELS[1])] + ([(LABELS[2], LABELS[3])] if len(LABELS) >= 4 else []):
    cross = (cells[(a, b)][0] + cells[(b, a)][0]) / 2
    within = min(cells[(a, a)][0], cells[(b, b)][0])
    print(
        f"  {a}<->{b}: cross~{cross:.3f} (cells [{cells[(a, b)][1]:.3f},{cells[(a, b)][2]:.3f}] / "
        f"[{cells[(b, a)][1]:.3f},{cells[(b, a)][2]:.3f}]) min-within {within:.3f} gap {within - cross:+.3f}",
        flush=True,
    )
