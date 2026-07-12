"""D1/B5: do the Social-Chem-family axes SEPARATE in a different register/provenance (MFRC Reddit)?

A2+overlap proved the family (care, fairness, legitimacy, loyalty, purity) is collinear with G *within
the prescriptive Social-Chem/ETHICS register* — a real shared factor, not memorization. Open question:
is that collinearity REGISTER-SPECIFIC? If in a different register the family feeders selectively detect
their OWN foundation (where G, foundation-agnostic, cannot), the foundations are genuinely distinct
dimensions that merely appear collinear in prescriptive text.

Test bed: Moral Foundations Reddit Corpus (USC-MOLA-Lab/MFRC, train_dedup) — Reddit comments annotated
with a single MFT foundation. Genuinely different register + provenance from Social-Chem RoTs.

Method: score MFRC comments with each family feeder's |value| (foundation ENGAGEMENT magnitude, per B1)
and with G. Build:
  * SELECTIVITY matrix  Sel[k][j] = AUROC(|feeder_k|, foundation-j comments vs OTHER-family-foundations).
    Diagonal-dominance => feeder-k is selective for foundation-k => foundations are DISTINCT in MFRC.
  * PRESENCE matrix     Pres[k][j] = AUROC(|feeder_k|, foundation-j vs Non-Moral)  (supporting).
  * G row on both: G is the foundation-AGNOSTIC baseline — expected ~0.5 selectivity (can't tell
    foundations apart). If family feeders beat G on the diagonal, distinctness is register-real.
  * G-residualized selectivity: resid_k = |s_k| - beta*|g| (beta = OLS over the pool) — does removing
    the general magnitude SHARPEN foundation selectivity (the clean-register bifactor prediction)?
GPU 1 only; scoring only.
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
from sklearn.metrics import roc_auc_score  # noqa: E402

from xbse.encoder import BSEEncoder  # noqa: E402
from xbse.instances.joint_builders import BUILDERS  # noqa: E402
from xbse.report import Report, require_pass  # noqa: E402
from xbse.scorer import DimensionScorer  # noqa: E402

CK = os.path.expanduser("~/xbse_ckpt")
OUT = os.path.expanduser("~/bifactor_A2")
os.makedirs(OUT, exist_ok=True)

# family feeder -> MFRC annotation label(s) for its foundation
FAM = {
    "care": {"Care"},
    "fairness": {"Proportionality", "Equality"},
    "legitimacy": {"Authority"},
    "loyalty": {"Loyalty"},
    "purity": {"Purity"},
}
FOUNDATIONS = list(FAM.keys())
CAP = 500  # comments per foundation


def load_encoder(ckpt):
    enc = BSEEncoder(base_model="BAAI/bge-m3", pooling="mean", max_len=128, device="cuda")
    enc.load_state_dict(torch.load(ckpt, map_location="cuda"))
    enc.eval()
    return enc


def build_scorer(enc, src, report):
    require_pass(report, report.checkpoint_hash)
    train, _held = src._split()
    pos = [t for (_d, t, s) in train if s == "+"][:400]
    neg = [t for (_d, t, s) in train if s == "-"][:400]
    return DimensionScorer.fit(enc, pos, neg, name=getattr(src, "name", ""))


def auroc(y, s):
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def ols(s, g):
    gm = g - g.mean()
    d = float((gm * gm).sum())
    return float(((s - s.mean()) * gm).sum() / d) if d > 1e-9 else 0.0


# ---- 1. collect MFRC comments per foundation (single dominant annotation) + Non-Moral ----
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
    if ann in label_of:
        f = label_of[ann]
        if len(found_texts[f]) < CAP:
            found_texts[f].append(t)
    elif ann == "Non-Moral" and len(nonmoral) < CAP:
        nonmoral.append(t)
for f in FOUNDATIONS:
    print(f"[mfrc] {f:12s} n={len(found_texts[f])}", flush=True)
print(f"[mfrc] non-moral n={len(nonmoral)}", flush=True)

# unique text pool to score once per feeder
pool = sorted({t for f in FOUNDATIONS for t in found_texts[f]} | set(nonmoral))
pos_index = {t: i for i, t in enumerate(pool)}


def score_pool(scorer):
    vals = scorer.score_batch(pool)
    return np.array([abs(float(v.value)) for v in vals], dtype="float32")  # |value| = engagement


# ---- 2. score with G + each family feeder ----
print("\n[load] G", flush=True)
gE = load_encoder(f"{CK}/general_valence_joint_s0.pt")
gScorer = build_scorer(
    gE,
    BUILDERS["general_valence_joint"](holdout_frac=0.12),
    Report(**json.load(open(f"{CK}/general_valence_joint_report_s0.json"))),
)
MAG = {"G": score_pool(gScorer)}
del gE
torch.cuda.empty_cache()

for f in FOUNDATIONS:
    print(f"[load] {f}", flush=True)
    axis = f + "_joint"
    eK = load_encoder(f"{CK}/{axis}.pt")
    sc = build_scorer(
        eK, BUILDERS[axis](holdout_frac=0.12), Report(**json.load(open(f"{CK}/{axis}_report.json")))
    )
    MAG[f] = score_pool(sc)
    del eK
    torch.cuda.empty_cache()


def idx(texts):
    return np.array([pos_index[t] for t in texts])


# ---- 3. selectivity + presence matrices ----
def selectivity(mag, j):
    """AUROC(|feeder|, foundation-j vs OTHER family foundations), excluding overlap texts."""
    pos_set = set(found_texts[j])
    neg_set = set()
    for m in FOUNDATIONS:
        if m != j:
            neg_set |= set(found_texts[m])
    neg_set -= pos_set  # a text tagged both -> treat as pos, drop from neg
    pos = idx(sorted(pos_set))
    neg = idx(sorted(neg_set))
    y = np.array([1] * len(pos) + [0] * len(neg))
    s = np.concatenate([mag[pos], mag[neg]])
    return auroc(y, s)


def presence(mag, j):
    pos = idx(found_texts[j])
    neg = idx(nonmoral)
    y = np.array([1] * len(pos) + [0] * len(neg))
    s = np.concatenate([mag[pos], mag[neg]])
    return auroc(y, s)


rows_feeders = ["G"] + FOUNDATIONS
Sel = {k: {j: round(selectivity(MAG[k], j), 4) for j in FOUNDATIONS} for k in rows_feeders}
Pres = {k: {j: round(presence(MAG[k], j), 4) for j in FOUNDATIONS} for k in rows_feeders}

# G-residualized selectivity: resid_k = |s_k| - beta*|G| (beta OLS over pool); does removing the
# general magnitude SHARPEN foundation selectivity?
g = MAG["G"]
SelResid = {}
beta_on_G = {}
for k in FOUNDATIONS:
    b = ols(MAG[k], g)
    beta_on_G[k] = round(b, 4)
    resid = MAG[k] - b * g
    SelResid[k] = {j: round(selectivity(resid, j), 4) for j in FOUNDATIONS}


def diag_dominant(mat, keys):
    n = 0
    for k in keys:
        off = [mat[k][j] for j in FOUNDATIONS if j != k]
        if mat[k][k] > max(off):
            n += 1
    return n


sel_diag = float(np.mean([Sel[k][k] for k in FOUNDATIONS]))
sel_off = float(np.mean([Sel[k][j] for k in FOUNDATIONS for j in FOUNDATIONS if j != k]))
g_sel = float(np.mean([Sel["G"][j] for j in FOUNDATIONS]))
resid_sel_diag = float(np.mean([SelResid[k][k] for k in FOUNDATIONS]))

result = {
    "phase": "D1_B5_cross_register_MFRC",
    "register": "MFRC Reddit (foundation-annotated) vs Social-Chem prescriptive RoTs",
    "n_per_foundation": {f: len(found_texts[f]) for f in FOUNDATIONS},
    "n_nonmoral": len(nonmoral),
    "selectivity_matrix": Sel,
    "selectivity_residualized_vs_G": SelResid,
    "beta_on_G": beta_on_G,
    "presence_matrix": Pres,
    "summary": {
        "family_selectivity_diag_dominant_axes": diag_dominant(Sel, FOUNDATIONS),
        "residualized_selectivity_diag_dominant_axes": diag_dominant(SelResid, FOUNDATIONS),
        "n_foundations": len(FOUNDATIONS),
        "family_selectivity_mean_diag": round(sel_diag, 4),
        "family_selectivity_mean_offdiag": round(sel_off, 4),
        "G_mean_selectivity": round(g_sel, 4),
        "residualized_mean_diag": round(resid_sel_diag, 4),
        "interpretation_key": (
            "family diag >> G_mean_selectivity (~0.5) => foundations DISTINCT in MFRC (register-specific "
            "collapse). family diag ~ G ~ 0.5 => one factor across registers."
        ),
    },
}
json.dump(result, open(f"{OUT}/d1_register_result.json", "w"), indent=2)
print("\nRESULT " + json.dumps(result["summary"]), flush=True)
print(f"SAVED {OUT}/d1_register_result.json", flush=True)
print("DONE", flush=True)
