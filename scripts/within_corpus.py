"""Within-corpus AUROC per dimension — decomposes 'unlearnable' vs 'untransferable'.

For each dimension, for each of its corpora SEPARATELY, measure whether the +/- valence label is
linearly decodable from a FROZEN base encoder via stratified k-fold logistic regression (roc_auc).
Frozen (no fine-tune) isolates the reviewer's decomposition:

  within-corpus AUROC ~ 0.5   -> no within-corpus signal        -> LABEL INCOHERENCE  (unlearnable)
  within-corpus AUROC high, cross (paper 3.1) low -> signal exists but does not cross  -> UNTRANSFERABLE
                                                                    (framework-relativity signature)

This is the cheap, checkpoint-free first cut: it asks whether the *labels* carry within-corpus signal
at all, which is exactly what separates the two failure modes for rights_respect.

Env: XBSE_BASE_MODEL (default BAAI/bge-m3), XBSE_POOLING (mean), XBSE_MAX_LEN (128),
     XBSE_MAX_PER_CORPUS (cap rows/corpus for speed, default 8000).
"""

import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(
    0,
    os.environ.get(
        "XBSE_SRC",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"),
    ),
)

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from xbse.encoder import BSEEncoder  # noqa: E402
from xbse.instances.joint_builders import BUILDERS  # noqa: E402

BASE = os.environ.get("XBSE_BASE_MODEL", "BAAI/bge-m3")
POOLING = os.environ.get("XBSE_POOLING", "mean")
MAXLEN = int(os.environ.get("XBSE_MAX_LEN", "128"))
CAP = int(os.environ.get("XBSE_MAX_PER_CORPUS", "8000"))

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
# paper §3.1 cross-dataset trained AUROC, for side-by-side context
CROSS = {
    "privacy_joint": 0.853, "environmental_joint": 0.817, "rights_joint": 0.475,
    "care_joint": 0.811, "fairness_joint": 0.789, "legitimacy_joint": 0.708,
    "epistemic_joint": 0.817, "physharm_joint": 0.622, "autonomy_joint": 0.747,
}

enc = BSEEncoder(base_model=BASE, pooling=POOLING, max_len=MAXLEN, device="cuda")
rng = np.random.default_rng(0)


def _cap(idx, labels):
    """Stratified down-sample of row indices to CAP total, preserving class balance."""
    if len(idx) <= CAP:
        return idx
    idx = np.array(idx)
    y = np.array(labels)
    pos, neg = idx[y == 1], idx[y == 0]
    k_pos = int(round(CAP * len(pos) / len(idx)))
    k_neg = CAP - k_pos
    keep = np.concatenate([
        rng.choice(pos, min(k_pos, len(pos)), replace=False),
        rng.choice(neg, min(k_neg, len(neg)), replace=False),
    ])
    return keep.tolist()


def within_auroc(texts, labels, folds=5):
    y = np.array(labels)
    if y.sum() < folds or (len(y) - y.sum()) < folds:
        return float("nan")
    X = enc.encode(texts).cpu().numpy()
    aucs = []
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=0).split(X, y):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.decision_function(X[te])))
    return float(np.mean(aucs))


print(f"[within] base={BASE} pooling={POOLING} cap={CAP}/corpus", flush=True)
summary = []
for dim in DIMS:
    try:
        src = BUILDERS[dim]()
        rows = src._rows()  # (domain_idx, text, sign)
        names = src.domain_names
    except Exception as e:  # noqa: BLE001
        print(f"\n=== {dim} === builder failed: {e}", flush=True)
        continue
    bydom = defaultdict(list)
    for di, text, sign in rows:
        bydom[di].append((text, 1 if sign == "+" else 0))
    print(f"\n=== {dim} (cross={CROSS.get(dim, float('nan')):.3f}) ===", flush=True)
    for di, drows in sorted(bydom.items()):
        allidx = list(range(len(drows)))
        labs = [drows[i][1] for i in allidx]
        keep = _cap(allidx, labs)
        texts = [drows[i][0] for i in keep]
        y = [drows[i][1] for i in keep]
        au = within_auroc(texts, y)
        nm = names[di] if di < len(names) else str(di)
        print(f"  within[{nm:<14}] n={len(y):5d} pos={sum(y):5d} AUROC={au:.4f}", flush=True)
        summary.append((dim, nm, len(y), au))

print("\n=== SUMMARY (within-corpus frozen-probe AUROC vs cross-dataset trained) ===", flush=True)
print(f"{'dimension':<22}{'corpus':<16}{'n':>7}{'within':>9}{'cross':>8}{'  gap(within-cross)'}")
for dim, nm, n, au in summary:
    cx = CROSS.get(dim, float("nan"))
    gap = au - cx if au == au else float("nan")
    print(f"{dim:<22}{nm:<16}{n:>7}{au:>9.4f}{cx:>8.3f}{gap:>12.3f}", flush=True)
