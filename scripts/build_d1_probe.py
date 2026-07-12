"""D1/B5 disambiguator: is foundation IDENTITY linearly present in MFRC embedding space at all?

The valence-feeder |value| test found no foundation selectivity in MFRC — but valence feeders aren't
foundation classifiers, and |value| is confounded by valence-extremity (all feeders peak on purity).
This test removes our feeders from the loop: embed MFRC comments with the PRETRAINED BGE-M3 (a general
semantic space, not BSE-fine-tuned) and train plain logistic probes to separate each foundation.

  * DISTINCTNESS ceiling: foundation-j vs OTHER family foundations, 5-fold CV AUROC.
    probe >> 0.5  => foundations ARE linearly distinct in MFRC (our valence feeders just miss it;
                     the dimensions are real -> build foundation-presence feeders, don't collapse).
    probe ~ 0.5   => foundations are NOT separable even in a general space => genuinely one factor.
  * PRESENCE ceiling: foundation-j vs Non-Moral, 5-fold CV AUROC (sanity: is anything moral separable?).
GPU 1 only; embedding + logistic CV only.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ.setdefault("HF_HOME", "/archive/cache/huggingface")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import json  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.expanduser("~/xbse/src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from xbse.encoder import BSEEncoder  # noqa: E402

OUT = os.path.expanduser("~/bifactor_A2")
FAM = {
    "care": {"Care"},
    "fairness": {"Proportionality", "Equality"},
    "legitimacy": {"Authority"},
    "loyalty": {"Loyalty"},
    "purity": {"Purity"},
}
FOUNDATIONS = list(FAM.keys())
CAP = 500

from datasets import load_dataset  # noqa: E402

d = load_dataset("USC-MOLA-Lab/MFRC", split="train_dedup")
found_texts = {f: [] for f in FOUNDATIONS}
nonmoral = []
label_of = {lbl: f for f, lbls in FAM.items() for lbl in lbls}
for r in d:
    t = (r.get("text") or "").strip()
    if len(t) < 12:
        continue
    ann = str(r.get("annotation") or "")
    if ann in label_of and len(found_texts[label_of[ann]]) < CAP:
        found_texts[label_of[ann]].append(t)
    elif ann == "Non-Moral" and len(nonmoral) < CAP:
        nonmoral.append(t)
for f in FOUNDATIONS:
    print(f"[mfrc] {f:12s} n={len(found_texts[f])}", flush=True)
print(f"[mfrc] non-moral n={len(nonmoral)}", flush=True)

# embed the unique pool once with PRETRAINED bge-m3 (no BSE checkpoint loaded)
pool = sorted({t for f in FOUNDATIONS for t in found_texts[f]} | set(nonmoral))
pidx = {t: i for i, t in enumerate(pool)}
enc = BSEEncoder(base_model="BAAI/bge-m3", pooling="mean", max_len=128, device="cuda")
enc.eval()
embs = []
B = 64
for i in range(0, len(pool), B):
    z = enc.encode(pool[i : i + B])
    z = z.detach().cpu().numpy() if hasattr(z, "detach") else np.asarray(z)
    embs.append(z.astype("float32"))
X = np.concatenate(embs, 0)
print(f"[emb] pool={len(pool)} dim={X.shape[1]}", flush=True)


def cv_auroc(pos_texts, neg_texts):
    pi = [pidx[t] for t in pos_texts]
    ni = [pidx[t] for t in neg_texts]
    Xd = np.concatenate([X[pi], X[ni]])
    y = np.array([1] * len(pi) + [0] * len(ni))
    Xs = StandardScaler().fit_transform(Xd)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    sc = cross_val_score(clf, Xs, y, cv=5, scoring="roc_auc")
    return float(sc.mean()), float(sc.std())


distinct = {}
presence = {}
for j in FOUNDATIONS:
    pos = set(found_texts[j])
    neg = set()
    for m in FOUNDATIONS:
        if m != j:
            neg |= set(found_texts[m])
    neg -= pos
    a, s = cv_auroc(sorted(pos), sorted(neg))
    distinct[j] = {"auroc": round(a, 4), "std": round(s, 4)}
    ap, sp = cv_auroc(found_texts[j], nonmoral)
    presence[j] = {"auroc": round(ap, 4), "std": round(sp, 4)}
    print(
        f"[probe] {j:12s} distinct(vs other foundations)={a:.4f}±{s:.3f}  "
        f"presence(vs non-moral)={ap:.4f}",
        flush=True,
    )

dmean = float(np.mean([distinct[j]["auroc"] for j in FOUNDATIONS]))
result = {
    "phase": "D1_B5_linear_probe_MFRC",
    "encoder": "pretrained BAAI/bge-m3 (not BSE-fine-tuned)",
    "distinctness_vs_other_foundations": distinct,
    "presence_vs_nonmoral": presence,
    "mean_distinctness_auroc": round(dmean, 4),
    "verdict_key": (
        "mean distinctness >> 0.5 => foundations ARE linearly distinct in MFRC (valence feeders miss it; "
        "build foundation-presence feeders). ~0.5 => genuinely one factor across registers."
    ),
}
os.makedirs(OUT, exist_ok=True)
json.dump(result, open(f"{OUT}/d1_probe_result.json", "w"), indent=2)
print("\nRESULT " + json.dumps(result["distinctness_vs_other_foundations"]), flush=True)
print(f"mean_distinctness={dmean:.4f}", flush=True)
print(f"SAVED {OUT}/d1_probe_result.json", flush=True)
print("DONE", flush=True)
