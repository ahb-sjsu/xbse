"""Build foundation-PRESENCE (identity) channels — the axis the valence feeders throw away.

D1/B5 showed foundation identity is real and separable (0.75 within MFRC), orthogonal to the valence
axis G captures. This builds it into actual channels, gated cross-dataset like every other channel:

  * TRAIN a linear presence head on Social-Chem foundation labels (prescriptive RoTs), on frozen
    PRETRAINED BGE-M3 embeddings — foundation-k RoTs (+) vs other-foundation RoTs (-).
  * TEST cross-register on MFRC (Reddit) — foundation-k comments (+) vs other-foundation comments (-).
    This is STRONGER than D1's within-MFRC CV: it requires the identity signal to TRANSFER from
    prescriptive RoTs to Reddit dialogue.
  * GATE: cross-corpus (SocialChem->MFRC) AUROC beats the TF-IDF BoW cross-corpus null by >= 0.10 and
    clears 0.60 — the presence signal must be semantic/transferable, not keyword overlap (purity's BoW
    will be high: disgust lexicon). Nulls measured before, reported with the number.

A presence channel = shared frozen BGE-M3 + a small linear head (coef + scaler), saved for deployment.
Additive to the vector: valence (G) x foundation-identity (these). GPU 1 only.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ.setdefault("HF_HOME", "/archive/cache/huggingface")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.expanduser("~/xbse/src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from xbse.encoder import BSEEncoder  # noqa: E402
from xbse.instances.joint_builders import SOCIAL_CHEM, _open  # noqa: E402

OUT = os.path.expanduser("~/foundation_presence")
os.makedirs(OUT, exist_ok=True)

# foundation -> (Social-Chem rot-moral-foundations category substring, MFRC annotation label set)
FOUND = {
    "care": ("care-harm", {"Care"}),
    "fairness": ("fairness-cheating", {"Proportionality", "Equality"}),
    "legitimacy": ("authority-subversion", {"Authority"}),
    "loyalty": ("loyalty-betrayal", {"Loyalty"}),
    "purity": ("sanctity-degradation", {"Purity"}),
}
FS = list(FOUND.keys())
CAP_SC = 1200  # Social-Chem RoTs per foundation
CAP_MF = 500  # MFRC comments per foundation
MARGIN = 0.10


# ---- 1. Social-Chem RoTs grouped by foundation (presence = category present, valence ignored) ----
def load_social_chem():
    by_f = {f: [] for f in FS}
    cats = {f: FOUND[f][0] for f in FS}
    with _open(SOCIAL_CHEM, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            try:
                if int(row.get("rot-agree") or 0) < 3:
                    continue
            except (TypeError, ValueError):
                continue
            rot = (row.get("rot") or "").strip()
            if len(rot) < 12:
                continue
            col = (row.get("rot-moral-foundations") or "").lower()
            for f in FS:
                if cats[f] in col and len(by_f[f]) < CAP_SC:
                    by_f[f].append(rot)
    return by_f


# ---- 2. MFRC comments grouped by foundation ----
def load_mfrc():
    from datasets import load_dataset

    d = load_dataset("USC-MOLA-Lab/MFRC", split="train_dedup")
    by_text = {}
    for r in d:
        t = (r.get("text") or "").strip()
        if len(t) < 12:
            continue
        by_text.setdefault(t, set()).add(str(r.get("annotation") or ""))
    by_f = {f: [] for f in FS}
    label_to_f = {lbl: f for f, (_c, lbls) in FOUND.items() for lbl in lbls}
    for t, anns in by_text.items():
        fs = {label_to_f[a] for a in anns if a in label_to_f}
        if len(fs) == 1:  # single-foundation comments -> clean identity labels
            f = next(iter(fs))
            if len(by_f[f]) < CAP_MF:
                by_f[f].append(t)
    return by_f


print("[load] Social-Chem by foundation", flush=True)
SC = load_social_chem()
for f in FS:
    print(f"  SC {f:12s} n={len(SC[f])}", flush=True)
print("[load] MFRC by foundation", flush=True)
MF = load_mfrc()
for f in FS:
    print(f"  MF {f:12s} n={len(MF[f])}", flush=True)

# ---- 3. embed everything once with frozen pretrained BGE-M3 ----
enc = BSEEncoder(base_model="BAAI/bge-m3", pooling="mean", max_len=128, device="cuda")
enc.eval()
sc_pool = sorted({t for f in FS for t in SC[f]})
mf_pool = sorted({t for f in FS for t in MF[f]})
sc_idx = {t: i for i, t in enumerate(sc_pool)}
mf_idx = {t: i for i, t in enumerate(mf_pool)}


def embed(texts):
    out = []
    for i in range(0, len(texts), 64):
        z = enc.encode(texts[i : i + 64])
        z = z.detach().cpu().numpy() if hasattr(z, "detach") else np.asarray(z)
        out.append(z.astype("float32"))
    return np.concatenate(out, 0)


Xsc = embed(sc_pool)
Xmf = embed(mf_pool)
print(f"[emb] SC={Xsc.shape} MF={Xmf.shape}", flush=True)


def balanced(pos_texts, neg_pool_by_f, k):
    neg = []
    others = [m for m in FS if m != k]
    per = max(1, len(pos_texts) // len(others))
    for m in others:
        neg.extend(neg_pool_by_f[m][:per])
    return pos_texts, neg


def auroc(y, s):
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


rows = []
for k in FS:
    # train on Social-Chem: k vs other-foundations
    pos_tr, neg_tr = balanced(SC[k], SC, k)
    Xtr = np.concatenate([Xsc[[sc_idx[t] for t in pos_tr]], Xsc[[sc_idx[t] for t in neg_tr]]])
    ytr = np.array([1] * len(pos_tr) + [0] * len(neg_tr))
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(scaler.transform(Xtr), ytr)

    # test cross-register on MFRC: k vs other-foundations
    pos_te, neg_te = balanced(MF[k], MF, k)
    Xte = np.concatenate([Xmf[[mf_idx[t] for t in pos_te]], Xmf[[mf_idx[t] for t in neg_te]]])
    yte = np.array([1] * len(pos_te) + [0] * len(neg_te))
    p = clf.decision_function(scaler.transform(Xte))
    cross_auroc = auroc(yte, p)

    # TF-IDF BoW cross-corpus null (lexical control): fit on SC train texts, test on MFRC
    tr_txt = list(pos_tr) + list(neg_tr)
    te_txt = list(pos_te) + list(neg_te)
    vec = TfidfVectorizer(min_df=2, max_features=40000, ngram_range=(1, 2)).fit(tr_txt)
    bow = LogisticRegression(max_iter=3000, C=1.0).fit(vec.transform(tr_txt), ytr)
    bow_auroc = auroc(yte, bow.decision_function(vec.transform(te_txt)))

    # within-MFRC CV reference (the D1 number)
    Xcv = np.concatenate([Xmf[[mf_idx[t] for t in pos_te]], Xmf[[mf_idx[t] for t in neg_te]]])
    cv = float(
        np.mean(
            cross_val_score(
                LogisticRegression(max_iter=2000),
                StandardScaler().fit_transform(Xcv),
                yte,
                cv=5,
                scoring="roc_auc",
            )
        )
    )

    margin = cross_auroc - bow_auroc
    passed = bool(margin >= MARGIN and cross_auroc >= 0.60)
    rec = {
        "foundation": k,
        "n_sc": len(SC[k]),
        "n_mf": len(MF[k]),
        "cross_register_auroc": round(cross_auroc, 4),
        "bow_cross_null": round(bow_auroc, 4),
        "margin_vs_bow": round(margin, 4),
        "within_mfrc_cv": round(cv, 4),
        "gate_passed": passed,
    }
    rows.append(rec)
    print("ROW " + json.dumps(rec), flush=True)

    # persist the channel (frozen BGE-M3 + this head)
    np.savez(
        f"{OUT}/presence_{k}.npz",
        coef=clf.coef_.astype("float32"),
        intercept=clf.intercept_.astype("float32"),
        scaler_mean=scaler.mean_.astype("float32"),
        scaler_scale=scaler.scale_.astype("float32"),
    )

n_pass = sum(r["gate_passed"] for r in rows)
result = {
    "phase": "foundation_presence_channels",
    "encoder": "frozen pretrained BAAI/bge-m3 + linear presence head",
    "gate": "cross-register (SocialChem->MFRC) AUROC beats TF-IDF BoW cross null by >=0.10 and >=0.60",
    "n_passed": n_pass,
    "n_total": len(FS),
    "channels": rows,
}
json.dump(result, open(f"{OUT}/foundation_presence_result.json", "w"), indent=2)
print("\nRESULT " + json.dumps({"n_passed": n_pass, "channels": rows}), flush=True)
print(f"SAVED {OUT}/foundation_presence_result.json", flush=True)
print("DONE", flush=True)
