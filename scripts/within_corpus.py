"""Within-corpus decomposition — 'unlearnable' vs 'untransferable', with same-estimator controls.

For each dimension we compute three linear-probe AUROCs (NO fine-tuning), so both sides of the
within->cross comparison use the SAME estimator (reviewer-3 point 1b), and we can tell concept signal
from surface artifact (reviewer-3 point 1a):

  emb-within  : frozen-embedding logistic, k-fold WITHIN each corpus      (is the label decodable?)
  bow-within  : TF-IDF logistic, k-fold WITHIN each corpus                (is it just lexical?)
  emb-cross   : frozen-embedding logistic trained on corpus A, tested on B (does it transfer?)

Decomposition:
  emb-within high, emb-cross ~0.5  -> signal exists but does not transfer  -> framework-relativity
  emb-within ~ bow-within          -> the within signal is largely lexical/surface (weakens "concept")
  drop = mean(emb-within) - emb-cross   (SAME estimator; reviewer-3 point 2 defines the statistic)

Env: XBSE_BASE_MODEL (BAAI/bge-m3), XBSE_POOLING (mean), XBSE_MAX_LEN (128), XBSE_MAX_PER_CORPUS (8000).
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

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
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

enc = BSEEncoder(base_model=BASE, pooling=POOLING, max_len=MAXLEN, device="cuda")
rng = np.random.default_rng(0)


def _cap(pairs):
    if len(pairs) <= CAP:
        return pairs
    y = np.array([p[1] for p in pairs])
    idx = np.arange(len(pairs))
    pos, neg = idx[y == 1], idx[y == 0]
    k_pos = int(round(CAP * len(pos) / len(idx)))
    keep = np.concatenate(
        [
            rng.choice(pos, min(k_pos, len(pos)), replace=False),
            rng.choice(neg, min(CAP - k_pos, len(neg)), replace=False),
        ]
    )
    return [pairs[i] for i in keep]


def _kfold_auroc(X, y, folds=5):
    y = np.array(y)
    if y.sum() < folds or (len(y) - y.sum()) < folds:
        return float("nan")
    aucs = []
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=0).split(X, y):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.decision_function(X[te])))
    return float(np.mean(aucs))


def _cross_auroc(Xa, ya, Xb, yb):
    ya, yb = np.array(ya), np.array(yb)
    if ya.sum() == 0 or (len(ya) - ya.sum()) == 0 or yb.sum() == 0 or (len(yb) - yb.sum()) == 0:
        return float("nan")
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xa, ya)
    return roc_auc_score(yb, clf.decision_function(Xb))


summary = []
print(f"[within] base={BASE} pooling={POOLING} cap={CAP}/corpus", flush=True)
for dim in DIMS:
    try:
        src = BUILDERS[dim]()
        rows = src._rows()
        names = src.domain_names
    except Exception as e:  # noqa: BLE001
        print(f"\n=== {dim} === builder failed: {e}", flush=True)
        continue
    bydom = defaultdict(list)
    for di, text, sign in rows:
        bydom[di].append((text, 1 if sign == "+" else 0))

    emb, lab, embw, boww = {}, {}, {}, {}
    print(f"\n=== {dim} ===", flush=True)
    for di in sorted(bydom):
        pairs = _cap(bydom[di])
        texts = [t for t, _ in pairs]
        y = [l for _, l in pairs]
        X = enc.encode(texts).cpu().numpy()
        emb[di], lab[di] = X, y
        embw[di] = _kfold_auroc(X, y)
        Xt = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2).fit_transform(texts)
        boww[di] = _kfold_auroc(Xt.toarray() if Xt.shape[1] < 4000 else Xt, y)
        nm = names[di] if di < len(names) else str(di)
        print(
            f"  {nm:<16} n={len(y):5d} emb-within={embw[di]:.4f} bow-within={boww[di]:.4f}",
            flush=True,
        )

    # frozen cross-probe: train on each corpus, test on the others; mean over ordered pairs
    doms = sorted(bydom)
    crosses = []
    for a in doms:
        for b in doms:
            if a != b:
                crosses.append(_cross_auroc(emb[a], lab[a], emb[b], lab[b]))
    crosses = [x for x in crosses if x == x]
    emb_cross = float(np.mean(crosses)) if crosses else float("nan")
    mean_within = float(np.nanmean([embw[d] for d in doms]))
    mean_bow = float(np.nanmean([boww[d] for d in doms]))
    drop = mean_within - emb_cross
    print(
        f"  -> mean-emb-within={mean_within:.4f} emb-CROSS-probe={emb_cross:.4f} drop={drop:.4f} "
        f"(bow-within={mean_bow:.4f})",
        flush=True,
    )
    summary.append((dim, mean_within, mean_bow, emb_cross, drop))

print(
    "\n=== SUMMARY (same-estimator frozen linear probe) ranked by within->cross drop ===",
    flush=True,
)
print(
    f"{'dimension':<22}{'emb-within':>11}{'bow-within':>11}{'emb-cross':>11}{'drop':>8}{'  at-chance?'}"
)
for dim, mw, mb, ec, dp in sorted(summary, key=lambda r: -r[4]):
    chance = "  <-- CHANCE" if (ec == ec and ec < 0.55) else ""
    print(f"{dim:<22}{mw:>11.4f}{mb:>11.4f}{ec:>11.4f}{dp:>8.3f}{chance}", flush=True)
print(
    "\nReading: emb-within~bow-within => within signal is largely lexical; emb-cross~0.5 => no "
    "transfer at the representation level (framework-relativity if within is high).",
    flush=True,
)
