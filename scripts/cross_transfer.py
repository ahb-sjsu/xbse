"""N x N cross-transfer between trained gte-Qwen2 feeders (dimensional-distinctness test).

For each trained encoder X and each held-out eval set Y, compute the cross-dataset structure_auroc.
The decisive comparison is INTERNAL and apples-to-apples: on the SAME eval set Y, compare
Y's own encoder (within = diagonal) against another dim's encoder (cross = off-diagonal).

  diagonal - off-diagonal gap large  => distinct dimensions (each encoder learned its own axis)
  off-diagonal ~ diagonal            => collapsed / shared valence axis (dims are not distinct)

privacy is the distinct-dimension control (separate corpus); care & legitimacy are the
ETHICS/Social-Chem family that collapsed into one shared valence under BGE-M3.
"""

import os
import sys

import torch

sys.path.insert(0, os.environ.get("XBSE_SRC", "/work/xbse/src"))

from xbse.encoder import BSEEncoder  # noqa: E402
from xbse.instances.joint_builders import BUILDERS  # noqa: E402
from xbse.validate import gate  # noqa: E402

BASE = os.environ.get("XBSE_BASE_MODEL", "Alibaba-NLP/gte-Qwen2-1.5B-instruct")
POOLING = os.environ.get("XBSE_POOLING", "last")  # "last" for decoder embedders, "mean" for BGE
MAX_LEN = int(os.environ.get("XBSE_MAX_LEN", "192"))

# (short label, builder name, checkpoint path). Override on the command line to run any N x N matrix
# (e.g. the full 9x9 over BGE-M3 checkpoints): pass one "label:builder_name:/path/ckpt.pt" per dim.
#   python cross_transfer.py care:care_joint:/ck/care.pt fair:fairness_joint:/ck/fair.pt ...
# With no args, defaults to the gte-Qwen2 care/legitimacy/privacy 3-dim slice (paper §3.5).
_DEFAULT = [
    ("care", "care_joint", "/work/gte_care.pt"),
    ("legit", "legitimacy_joint", "/work/gte_legit.pt"),
    ("privacy", "privacy_joint", "/work/gte_privacy.pt"),
]
if len(sys.argv) > 1:
    PAIRS = [tuple(a.split(":", 2)) for a in sys.argv[1:]]
    assert all(len(p) == 3 for p in PAIRS), "each arg must be label:builder_name:/path/ckpt.pt"
else:
    PAIRS = _DEFAULT
LABELS = [p[0] for p in PAIRS]

# Build each held-out eval set once (deterministic split inside the builder).
evs = {}
for short, name, _ck in PAIRS:
    src = BUILDERS[name](holdout_frac=0.12)
    evs[short] = src.heldout_eval()
    print(f"[eval] {short} ({name}) built", flush=True)


def load(path: str) -> BSEEncoder:
    enc = BSEEncoder(base_model=BASE, pooling=POOLING, max_len=MAX_LEN, device="cuda")
    sd = torch.load(path, map_location="cuda")
    enc.load_state_dict(sd)
    enc.eval()
    return enc


results = {}
for enc_short, _name, ckpt in PAIRS:
    print(f"[load] {enc_short} <- {ckpt}", flush=True)
    enc = load(ckpt)
    for ev_short in LABELS:
        m = gate(enc, evs[ev_short])
        results[(enc_short, ev_short)] = float(m["structure_auroc"])
        tag = "within" if enc_short == ev_short else "CROSS"
        print(
            f"  ENC={enc_short:<8} EVAL={ev_short:<8} structure_auroc={m['structure_auroc']:.4f} [{tag}]",
            flush=True,
        )
    del enc
    torch.cuda.empty_cache()

print(f"\n=== N x N CROSS-TRANSFER MATRIX ({BASE})  rows=encoder, cols=eval ===", flush=True)
hdr = "enc \\ eval  " + "".join(f"{c:>10}" for c in LABELS)
print(hdr)
for r in LABELS:
    row = f"{r:<11}" + "".join(f"{results[(r, cc)]:>10.4f}" for cc in LABELS)
    print(row)

diag = sum(results[(x, x)] for x in LABELS) / len(LABELS)
offs = [(r, cc, results[(r, cc)]) for r in LABELS for cc in LABELS if r != cc]
off_mean = sum(v for _r, _c, v in offs) / len(offs)
print(f"\nmean diagonal (within)     = {diag:.4f}")
print(f"mean off-diagonal (cross)  = {off_mean:.4f}")
print(f"overall separation gap     = {diag - off_mean:.4f}")

# Symmetric pairwise cross means vs the min of the two within-AUROCs (collapse if cross ~ within).
print("\nper-pair (symmetric cross mean vs. min within):")
seen = set()
for r in LABELS:
    for cc in LABELS:
        if r == cc or (cc, r) in seen:
            continue
        seen.add((r, cc))
        cross = (results[(r, cc)] + results[(cc, r)]) / 2
        within = min(results[(r, r)], results[(cc, cc)])
        verdict = "COLLAPSED (shared axis)" if (within - cross) < 0.10 else "distinct"
        print(
            f"  {r:<8}<->{cc:<8} cross={cross:.4f}  min-within={within:.4f}  gap={within - cross:+.4f}  {verdict}"
        )
print("interpretation: gap<0.10 => collapsed/shared axis; gap>=0.10 => distinct dims", flush=True)
