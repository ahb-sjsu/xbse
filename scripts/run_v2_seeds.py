"""3-seed replication of the §3.7 marginal cell (care_v2 <-> fair_v2) — reviewer-5/6 top open item.

The +0.16 decoupling gap and the control rest on single training runs, within the ~±0.06 retrain
variance documented elsewhere. This retrains care_v2 and fairness_v2 under 3 seeds (seed 0 = the
canonical checkpoints), recomputes the care_v2<->fair_v2 within/cross each time, and reports mean ± sd
of the gap so "decoupled" carries a seed-level error bar.
"""

import os
import random
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
from xbse.train_adv import train_adversarial  # noqa: E402

CKDIR = os.environ.get("XBSE_CKPT_DIR", "/home/claude/xbse_ckpt")
SEEDS = [0, 1, 2]
DIMS = {"care_v2": "care_v2_joint", "fair_v2": "fairness_v2_joint"}

# build eval sets once (deterministic)
evs = {lab: BUILDERS[b](holdout_frac=0.12).heldout_eval()["structural_pairs"] for lab, b in DIMS.items()}


def _auroc(enc, sp):
    za = enc.encode([a for a, _b, _l in sp]).cpu().numpy()
    zb = enc.encode([b for _a, b, _l in sp]).cpu().numpy()
    y = [1 if l else 0 for _a, _b, l in sp]
    return roc_auc_score(y, (za * zb).sum(-1))


def _ckpt(lab, seed):
    if seed == 0:  # canonical checkpoints from run_v2_experiment
        return os.path.join(CKDIR, f"{DIMS[lab]}.pt")
    return os.path.join(CKDIR, f"{lab}_s{seed}.pt")


def _get_enc(lab, seed):
    ck = _ckpt(lab, seed)
    enc = BSEEncoder(base_model="BAAI/bge-m3", pooling="mean", max_len=128, device="cuda")
    if os.path.exists(ck):
        enc.load_state_dict(torch.load(ck, map_location="cuda"))
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        src = BUILDERS[DIMS[lab]](holdout_frac=0.12)
        train_adversarial(enc, src, epochs=6, batch_size=24, lr=2e-5, max_steps=1200,
                          max_lambda=0.0, checkpoint_path=ck)
    enc.eval()
    return enc


gaps = []
for seed in SEEDS:
    print(f"\n--- seed {seed} ---", flush=True)
    care = _get_enc("care_v2", seed)
    fair = _get_enc("fair_v2", seed)
    cc = _auroc(care, evs["care_v2"])
    cf = _auroc(care, evs["fair_v2"])
    fc = _auroc(fair, evs["care_v2"])
    ff = _auroc(fair, evs["fair_v2"])
    cross = (cf + fc) / 2
    within = min(cc, ff)
    gap = within - cross
    gaps.append(gap)
    print(f"  care->care {cc:.3f} fair->fair {ff:.3f} | care->fair {cf:.3f} fair->care {fc:.3f} | "
          f"cross {cross:.3f} min-within {within:.3f} GAP {gap:+.3f}", flush=True)
    del care, fair
    torch.cuda.empty_cache()

g = np.array(gaps)
print(f"\n=== care_v2<->fair_v2 gap over {len(SEEDS)} seeds: {g.mean():+.3f} ± {g.std(ddof=1):.3f} "
      f"(gaps {[round(x, 3) for x in gaps]}) ===", flush=True)
print("decoupling is safe if mean - sd stays > 0 (and comfortably > the +0.10 threshold).", flush=True)
