"""Family-pool baseline (paper §3.3) — the reproducibility script reviewer-2/3 asked to be committed.

Trains ONE encoder on the pooled +/- valence of the four shared-corpus family dimensions
(care, fairness, legitimacy, epistemic — all Social-Chem + ETHICS), then evaluates it on EACH
family dimension's held-out pairs and compares to that dimension's dedicated encoder. If the pool
matches the dedicated encoders, the four are one shared valence, not four concepts (§3.3).

Leakage control: pairs whose anchor text was in the pool's TRAIN split are dropped from each
dimension's held-out eval, so the pool is never scored on text it trained on.
"""

import os
import sys

sys.path.insert(
    0,
    os.environ.get(
        "XBSE_SRC",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"),
    ),
)

import torch  # noqa: E402

from xbse.encoder import BSEEncoder  # noqa: E402
from xbse.instances.joint import JointPairSource  # noqa: E402
from xbse.instances.joint_builders import (  # noqa: E402
    BUILDERS,
    FOUNDATIONS,
    _ethics_rows,
    _social_chem_rows,
)
from xbse.train_adv import train_adversarial  # noqa: E402
from xbse.validate import structure_vs_surface_auroc  # noqa: E402

CKDIR = os.environ.get("XBSE_CKPT_DIR", "/home/claude/xbse_ckpt")
FAMILY = ["care", "fairness", "legitimacy", "epistemic"]


def build_family_pool(holdout_frac: float = 0.12) -> JointPairSource:
    sc, eth = [], []
    for f in FAMILY:
        cat, kw = FOUNDATIONS[f]
        sc += _social_chem_rows(cat, kw)
        eth += _ethics_rows(kw)
    return JointPairSource(
        name="family_pool",
        domains=[("socialchem", sc), ("ethics", eth)],
        holdout_frac=holdout_frac,
        label_source="pooled care+fairness+legitimacy+epistemic valence",
    )


pool_src = build_family_pool()
pool_train_texts = {t for t, _d in pool_src.train_rows()}
print(
    f"[family_pool] pooled rows={len(pool_src._rows())} train-texts={len(pool_train_texts)}",
    flush=True,
)

pool_ck = os.path.join(CKDIR, "family_pool_repro.pt")
pool_enc = BSEEncoder(base_model="BAAI/bge-m3", pooling="mean", max_len=128, device="cuda")
if os.environ.get("XBSE_REUSE_POOL") and os.path.exists(pool_ck):
    pool_enc.load_state_dict(torch.load(pool_ck, map_location="cuda"))
    print("[family_pool] loaded existing pool checkpoint", flush=True)
else:
    train_adversarial(
        pool_enc,
        pool_src,
        epochs=6,
        batch_size=24,
        lr=2e-5,
        max_steps=1500,
        max_lambda=0.0,
        checkpoint_path=pool_ck,
    )
pool_enc.eval()

print("\n=== §3.3 family-pool vs dedicated (leakage-filtered held-out) ===", flush=True)
print(f"{'dimension':<14}{'dedicated':>10}{'family-pool':>13}{'gap':>8}{'  n_clean/n':>14}")
for f in FAMILY:
    fsrc = BUILDERS[f"{f}_joint"](holdout_frac=0.12)
    sp = fsrc.heldout_eval()["structural_pairs"]
    spc = [(a, b, l) for a, b, l in sp if a not in pool_train_texts]
    if len([1 for _a, _b, l in spc if l]) < 5 or len([1 for _a, _b, l in spc if not l]) < 5:
        print(f"{f:<14} too few clean pairs ({len(spc)}) — skip", flush=True)
        continue
    ded_ck = os.path.join(CKDIR, f"{f}_joint.pt")
    ded = BSEEncoder(base_model="BAAI/bge-m3", pooling="mean", max_len=128, device="cuda")
    ded.load_state_dict(torch.load(ded_ck, map_location="cuda"))
    ded.eval()
    d_au = structure_vs_surface_auroc(ded, spc)
    p_au = structure_vs_surface_auroc(pool_enc, spc)
    print(
        f"{f:<14}{d_au:>10.3f}{p_au:>13.3f}{p_au - d_au:>+8.3f}{f'  {len(spc)}/{len(sp)}':>14}",
        flush=True,
    )
    del ded
    torch.cuda.empty_cache()
print(
    "\nReading: family-pool >= dedicated (gap >= ~0) => the four are one shared valence (§3.3).",
    flush=True,
)
