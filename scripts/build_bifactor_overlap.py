"""Phase A2 follow-up: how much of Population A's collapse into G is CORPUS OVERLAP vs a real
shared factor?  (Answers the confound A2 flagged, cheaply, before any readout demotion.)

G was trained on pooled Social-Chem; the Population-A family feeders (care, fairness, legitimacy,
epistemic, loyalty, purity) draw on the SAME Social-Chem categories. So G->family ~ 1.0 and beta ~ 0.98
could be (a) a real shared prescriptive-valence factor, or (b) G having literally trained on the family
axis's eval texts. Decisive split: partition each family axis's held-eval items by whether G's TRAINING
set contains that exact text (SHA-256, normalized), then recompute G->axis AUROC and beta on
  * G-SEEN items (in-distribution / memorised) vs
  * G-UNSEEN items (true generalization).
If G-unseen ~ G-seen ~ high -> the collapse is a REAL shared factor. If G-unseen drops toward the
independent-corpus level -> it was substantially corpus overlap, and the family dimensions are more
distinct than the shared-corpus beta implied (do NOT demote them; fix the measurement instead).
physharm (independent corpus, ~0% overlap) is the built-in control. GPU 1 only; scoring only.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ.setdefault("HF_HOME", "/archive/cache/huggingface")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import hashlib  # noqa: E402
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
FAMILY = [
    "care_joint",
    "fairness_joint",
    "legitimacy_joint",
    "epistemic_joint",
    "loyalty_joint",
    "purity_joint",
]
CONTROL = ["physharm_joint"]  # independent corpus, expected ~0% overlap
CAP_PER_SIGN = 400


def norm(t):
    return " ".join((t or "").split()).lower()


def h(t):
    return hashlib.sha256(norm(t).encode("utf-8")).hexdigest()


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


# ---- G training-text hash set (both pooled Social-Chem AND ETHICS train rows) ----
gsrc = BUILDERS["general_valence_joint"](holdout_frac=0.12)
g_train, _g_held = gsrc._split()
G_TRAIN_HASHES = {h(t) for (_d, t, _s) in g_train}
print(f"[G] train rows={len(g_train)} unique-hashes={len(G_TRAIN_HASHES)}", flush=True)

gE = load_encoder(f"{CK}/general_valence_joint_s0.pt")
gReport = Report(**json.load(open(f"{CK}/general_valence_joint_report_s0.json")))
gScorer = build_scorer(gE, gsrc, gReport)

rows = []
for axis in FAMILY + CONTROL:
    src = BUILDERS[axis](holdout_frac=0.12)
    _tr, held = src._split()
    pos = [t for (_d, t, s) in held if s == "+"][:CAP_PER_SIGN]
    neg = [t for (_d, t, s) in held if s == "-"][:CAP_PER_SIGN]
    texts = pos + neg
    y = np.array([1] * len(pos) + [0] * len(neg))
    seen_mask = np.array([h(t) in G_TRAIN_HASHES for t in texts])
    frac_seen = float(seen_mask.mean())

    g_vals = np.array([v.value for v in gScorer.score_batch(texts)], dtype="float32")
    # axis's own scorer for the residual-diagonal-on-unseen leg
    aE = load_encoder(f"{CK}/{axis}.pt")
    aSc = build_scorer(aE, src, Report(**json.load(open(f"{CK}/{axis}_report.json"))))
    s_vals = np.array([v.value for v in aSc.score_batch(texts)], dtype="float32")
    del aE
    torch.cuda.empty_cache()

    def leg(mask):
        if mask.sum() < 20 or len(set(y[mask])) < 2:
            return {"n": int(mask.sum()), "g_auroc": None, "beta": None, "self_auroc": None}
        b = ols(s_vals[mask], g_vals[mask])
        resid = s_vals[mask] - b * g_vals[mask]
        return {
            "n": int(mask.sum()),
            "g_auroc": round(auroc(y[mask], g_vals[mask]), 4),  # G -> axis
            "self_auroc": round(auroc(y[mask], s_vals[mask]), 4),  # axis -> itself (raw)
            "beta": round(b, 4),
            "resid_self_auroc": round(auroc(y[mask], resid), 4),  # axis residual after removing G
        }

    rec = {
        "axis": axis.replace("_joint", ""),
        "n_total": len(texts),
        "frac_in_G_train": round(frac_seen, 4),
        "G_SEEN": leg(seen_mask),
        "G_UNSEEN": leg(~seen_mask),
    }
    rows.append(rec)
    print("ROW " + json.dumps(rec), flush=True)

result = {"phase": "A2_overlap_diagnostic", "cap_per_sign": CAP_PER_SIGN, "axes": rows}
os.makedirs(OUT, exist_ok=True)
json.dump(result, open(f"{OUT}/bifactor_overlap_result.json", "w"), indent=2)
print(f"\nSAVED {OUT}/bifactor_overlap_result.json", flush=True)
print("DONE", flush=True)
