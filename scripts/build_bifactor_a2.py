"""Phase A2: residualize the 11 learned specifics against the trained G channel; test P1 & P3.

A1 proved G is a real axis. A2 tests the BIFACTOR claim (prereg_bifactor_readout.md):
  P1 — after r_k = s_k - beta_k*G, does each residual still predict its OWN axis better than any
       other (diagonal-dominant residual transfer matrix, >= 8/11 axes)? + beta-stability |db|/|b|<=0.25.
  P3 — is G GENERAL (predicts all axes similarly) rather than a smuggled single foundation?

Mechanism (per-item, using the signed DimensionScorer .value in [-1,1], the same readout the
contraction + B1 cross-check use):
  * each scorer's valence axis is fit on that dimension's TRAIN rows only (no eval leakage);
  * every axis j's HELD items are split cal/test (disjoint); beta_k is fit on axis-k's CAL items;
  * residual transfer AUROC is measured on TEST items.
Headline bifactor signature: residualizing against G should collapse OFF-diagonal transfer toward
0.5 (shared valence removed) while the DIAGONAL stays elevated (axis-specific residual survives).
GPU 1 only. Scoring only — no training.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # leave GPU 0
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

AXES = [
    "privacy_joint",
    "care_joint",
    "fairness_joint",
    "legitimacy_joint",
    "epistemic_joint",
    "physharm_joint",
    "autonomy_joint",
    "environmental_joint",
    "identity_attack_joint",
    "loyalty_joint",
    "purity_joint",
]
SHORT = {a: a.replace("_joint", "") for a in AXES}
CAP_PER_SIGN = 300  # held items per sign per axis (balanced); half cal, half test


def load_encoder(ckpt):
    enc = BSEEncoder(base_model="BAAI/bge-m3", pooling="mean", max_len=128, device="cuda")
    enc.load_state_dict(torch.load(ckpt, map_location="cuda"))
    enc.eval()
    return enc


def build_scorer(enc, src, report):
    """Valence axis fit on TRAIN rows only (avoid eval leakage); gate-checked."""
    require_pass(report, report.checkpoint_hash)
    train, _held = src._split()
    pos = [t for (_d, t, s) in train if s == "+"][:400]
    neg = [t for (_d, t, s) in train if s == "-"][:400]
    return DimensionScorer.fit(enc, pos, neg, name=getattr(src, "name", ""))


def report_for(name):
    return Report(**json.load(open(f"{CK}/{name}_report.json")))


# ---- 1. materialise each axis's held items, balanced, split cal/test (deterministic) ----
ITEMS = {}
srcs = {}
for a in AXES:
    src = BUILDERS[a](holdout_frac=0.12)
    srcs[a] = src
    _train, held = src._split()
    pos = [t for (_d, t, s) in held if s == "+"][:CAP_PER_SIGN]
    neg = [t for (_d, t, s) in held if s == "-"][:CAP_PER_SIGN]
    texts = pos + neg
    y = np.array([1] * len(pos) + [0] * len(neg))
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(texts))
    cal_mask = np.zeros(len(texts), bool)
    cal_mask[perm[: len(texts) // 2]] = True  # deterministic 50/50 cal/test
    ITEMS[a] = {"texts": texts, "y": y, "cal": cal_mask}
    print(
        f"[items] {SHORT[a]:14s} n={len(texts)} (+{len(pos)}/-{len(neg)}) "
        f"cal={cal_mask.sum()} test={(~cal_mask).sum()}",
        flush=True,
    )


def score_all_axes(scorer):
    """Return {axis_j: np.array of per-item .value over ITEMS[j].texts}."""
    out = {}
    for j in AXES:
        vals = [v.value for v in scorer.score_batch(ITEMS[j]["texts"])]
        out[j] = np.array(vals, dtype="float32")
    return out


# ---- 2. G scores over every axis's items ----
print("\n[load] G (seed 0)", flush=True)
gE = load_encoder(f"{CK}/general_valence_joint_s0.pt")
gReport = Report(**json.load(open(f"{CK}/general_valence_joint_report_s0.json")))
gScorer = build_scorer(gE, BUILDERS["general_valence_joint"](holdout_frac=0.12), gReport)
G = score_all_axes(gScorer)
del gE
torch.cuda.empty_cache()

# ---- 3. each axis-k scorer over every axis's items ----
S = {}
for k in AXES:
    print(f"[load] {SHORT[k]}", flush=True)
    eK = load_encoder(f"{CK}/{k}.pt")
    sc = build_scorer(eK, srcs[k], report_for(k))
    S[k] = score_all_axes(sc)
    del eK
    torch.cuda.empty_cache()


def auroc(y, score):
    try:
        return float(roc_auc_score(y, score))
    except Exception:
        return float("nan")


def ols_slope(s, g):
    gm = g - g.mean()
    denom = float((gm * gm).sum())
    return float(((s - s.mean()) * gm).sum() / denom) if denom > 1e-9 else 0.0


# ---- 4. beta_k on axis-k CAL items; residual + raw transfer matrices on TEST items ----
beta = {}
beta_stab = {}
for k in AXES:
    cal = ITEMS[k]["cal"]
    s_cal, g_cal = S[k][k][cal], G[k][cal]
    beta[k] = ols_slope(s_cal, g_cal)
    idx = np.where(cal)[0]
    h = len(idx) // 2
    b1 = ols_slope(S[k][k][idx[:h]], G[k][idx[:h]])
    b2 = ols_slope(S[k][k][idx[h:]], G[k][idx[h:]])
    beta_stab[k] = abs(b1 - b2) / (abs(beta[k]) + 1e-9)

RAW = {}  # RAW[k][j]  = AUROC(y_test_j, s_k on test_j)
RES = {}  # RES[k][j]  = AUROC(y_test_j, r_k on test_j),  r = s_k - beta_k*g
GPRED = {}  # G's own AUROC per axis (P3)
for j in AXES:
    test = ~ITEMS[j]["cal"]
    yj = ITEMS[j]["y"][test]
    GPRED[j] = auroc(yj, G[j][test])
for k in AXES:
    RAW[k], RES[k] = {}, {}
    for j in AXES:
        test = ~ITEMS[j]["cal"]
        yj = ITEMS[j]["y"][test]
        s = S[k][j][test]
        r = s - beta[k] * G[j][test]
        RAW[k][j] = auroc(yj, s)
        RES[k][j] = auroc(yj, r)

# ---- 5. P1 diagonal-dominance + beta-stability ----
p1_rows = {}
diag_dom = 0
for k in AXES:
    off = [RES[k][j] for j in AXES if j != k]
    dominates = bool(RES[k][k] > max(off))
    stable = bool(beta_stab[k] <= 0.25)
    p1_rows[k] = {
        "diag_residual": round(RES[k][k], 4),
        "max_offdiag_residual": round(max(off), 4),
        "diag_dominates": dominates,
        "beta": round(beta[k], 4),
        "beta_rel_instability": round(beta_stab[k], 4),
        "beta_stable": stable,
    }
    if dominates:
        diag_dom += 1

raw_diag = np.nanmean([RAW[k][k] for k in AXES])
raw_off = np.nanmean([RAW[k][j] for k in AXES for j in AXES if j != k])
res_diag = np.nanmean([RES[k][k] for k in AXES])
res_off = np.nanmean([RES[k][j] for k in AXES for j in AXES if j != k])

# ---- 6. P3 G-generality ----
gp = np.array([GPRED[j] for j in AXES])
p3 = {
    "gpred_per_axis": {SHORT[j]: round(GPRED[j], 4) for j in AXES},
    "gpred_mean": round(float(np.nanmean(gp)), 4),
    "gpred_min": round(float(np.nanmin(gp)), 4),
    "gpred_max": round(float(np.nanmax(gp)), 4),
    "gpred_spread": round(float(np.nanmax(gp) - np.nanmin(gp)), 4),
    # G is "general" if it predicts most axes above chance with a bounded spread (no single-axis spike
    # AND no axis left at chance). Report; the prereg reads spread + min together.
    "general_pass": bool(np.nanmin(gp) >= 0.60 and (np.nanmax(gp) - np.nanmin(gp)) <= 0.25),
}

P1_PASS = bool(diag_dom >= 8 and all(p1_rows[k]["beta_stable"] for k in AXES))

result = {
    "phase": "A2_bifactor_residualization",
    "n_axes": len(AXES),
    "P1": {
        "diag_dominant_axes": diag_dom,
        "threshold": 8,
        "all_beta_stable": bool(all(p1_rows[k]["beta_stable"] for k in AXES)),
        "pass": P1_PASS,
        "per_axis": {SHORT[k]: p1_rows[k] for k in AXES},
    },
    "P3": p3,
    "transfer_summary": {
        "raw_mean_diag": round(float(raw_diag), 4),
        "raw_mean_offdiag": round(float(raw_off), 4),
        "residual_mean_diag": round(float(res_diag), 4),
        "residual_mean_offdiag": round(float(res_off), 4),
        "offdiag_drop_after_residual": round(float(raw_off - res_off), 4),
        "diag_retained_after_residual": round(float(res_diag), 4),
    },
    "matrices": {
        "raw": {SHORT[k]: {SHORT[j]: round(RAW[k][j], 4) for j in AXES} for k in AXES},
        "residual": {SHORT[k]: {SHORT[j]: round(RES[k][j], 4) for j in AXES} for k in AXES},
    },
}
json.dump(result, open(f"{OUT}/bifactor_A2_result.json", "w"), indent=2)
print(
    "\nRESULT " + json.dumps({k: result[k] for k in ("P1", "P3", "transfer_summary")}), flush=True
)
print(f"SAVED {OUT}/bifactor_A2_result.json")
print("DONE", flush=True)
