"""PolarQuant-style shared-mode-removal probe for foundation-PRESENCE cross-register transfer.

Question (from the MoralVector work): does removing the dominant shared ANGULAR mode(s) from FROZEN
BGE-M3 embeddings — the register/domain axis and/or top principal components — then renormalizing to
the unit sphere, improve cross-register presence transfer? This is the geometric, NON-adversarial
alternative to domain-adversarial training, which HURT transfer (attempt-2b: adversary degraded all
five foundations and cost loyalty its pass).

Design — pure geometry, NO fine-tuning, so the removal's contribution is isolated:
  * Removal basis fit on TRAIN rows ONLY (heldout excluded by the JointPairSource split) -> no leakage.
  * Basis uses only corpus/domain id + unsupervised PCs — NEVER the presence label -> no target leakage.
  * Wrap the frozen encoder: z' = normalize(z - (z.R^T).R); feed to the SAME gate() every channel uses.
  * Compare each removal variant's cross-domain structure_auroc against k0 (raw frozen) and the BoW null.

Falsification order: purity + loyalty first (the two that passed lam=0), but ALL five computed — a lift
on care/fairness/legitimacy (register-bound under both prior attempts) would be the bigger finding.

Scope caveat (stated up front): a FROZEN probe is a LOWER bar than the fine-tuned lam=0 pass
(purity 0.719 / loyalty 0.661). A frozen WIN is strong evidence the geometry helps and motivates the
fine-tuned+residual follow-up; a frozen MISS is inconclusive about that combination, not proof the
geometry is useless. GPU 1 only.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ.setdefault("HF_HOME", "/archive/cache/huggingface")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import json  # noqa: E402
import sys  # noqa: E402
from collections import Counter  # noqa: E402

sys.path.insert(0, os.path.expanduser("~/xbse/src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from xbse.encoder import BSEEncoder  # noqa: E402
from xbse.instances.joint_builders import build_foundation_presence_joint  # noqa: E402
from xbse.validate import gate  # noqa: E402

OUT = os.path.expanduser("~/presence_polar")
os.makedirs(OUT, exist_ok=True)
FOUNDATIONS = ["purity", "loyalty", "care", "fairness", "legitimacy"]
# the fine-tuned lam=0 structure_auroc these must be read against (from presence_feeders/, 2a)
LAM0_FT = {"purity": 0.7194, "loyalty": 0.6611, "care": 0.5742, "fairness": 0.5687, "legitimacy": 0.585}


class ResidualEncoder:
    """Frozen base encoder + fixed removal of orthonormal directions R, then angular renorm."""

    def __init__(self, base, R):
        self.base = base
        self.R = (
            None
            if R is None or len(R) == 0
            else torch.tensor(np.asarray(R, dtype="float32"), device=base.device)
        )

    def encode(self, texts, batch_size=64):
        z = self.base.encode(texts, batch_size=batch_size)  # (n,d) L2-normed
        if self.R is not None:
            z = z - (z @ self.R.t()) @ self.R  # remove components along R
            z = F.normalize(z, dim=-1)  # back to the unit sphere (angular)
        return z


def orthonormal(vecs):
    """Rows -> orthonormal rows (QR)."""
    M = np.asarray(vecs, dtype="float64")
    Q, _ = np.linalg.qr(M.T)  # columns orthonormal, span = row space of M
    return Q.T.astype("float32")


def main():
    base = BSEEncoder(base_model="BAAI/bge-m3", pooling="mean", max_len=128, device="cuda")
    all_results = []
    for f in FOUNDATIONS:
        print(f"\n===== {f} polar probe =====", flush=True)
        src = build_foundation_presence_joint(f, holdout_frac=0.12)
        train = src.train_rows()  # [(text, domain_idx)] — TRAIN split only
        texts = [t for t, _ in train]
        dom = np.array([d for _, d in train])
        Z = base.encode(texts).detach().cpu().numpy().astype("float64")  # L2-normed, TRAIN only
        print(f"[{f}] train n={len(texts)} by-domain={dict(Counter(dom.tolist()))}", flush=True)

        mu = Z.mean(0)
        mean_dir = mu / (np.linalg.norm(mu) + 1e-9)
        d0, d1 = Z[dom == 0].mean(0), Z[dom == 1].mean(0)
        dom_dir = d0 - d1
        dom_dir = dom_dir / (np.linalg.norm(dom_dir) + 1e-9)  # the register/corpus axis
        _, _, Vt = np.linalg.svd(Z - mu, full_matrices=False)
        pcs = Vt[:4]  # top-4 principal directions of centered TRAIN embeddings
        dom_align = [float(abs(np.dot(pcs[i], dom_dir))) for i in range(len(pcs))]
        print(f"[{f}] |PC_i . domain_dir| top4 = {[round(x, 3) for x in dom_align]}", flush=True)

        variants = {
            "k0_raw": [],
            "mean": [mean_dir],
            "domain": [dom_dir],
            "pc1": [pcs[0]],
            "pc1_2": [pcs[0], pcs[1]],
            "pc1_3": [pcs[0], pcs[1], pcs[2]],
            "domain+pc1_2": [dom_dir, pcs[0], pcs[1]],
        }
        ev = src.heldout_eval()
        rows, k0 = [], None
        for vname, vecs in variants.items():
            R = None if not vecs else orthonormal(vecs)
            g = gate(ResidualEncoder(base, R), ev)  # raw metrics only; our own margin below
            au, bow, fuzz = (
                float(g["structure_auroc"]),
                float(g["bow_auroc"]),
                float(g["fuzz_ratio"]),
            )
            if vname == "k0_raw":
                k0 = au
            rows.append(
                {
                    "variant": vname,
                    "n_removed": len(vecs),
                    "structure_auroc": round(au, 4),
                    "bow_auroc": round(bow, 4),
                    "fuzz_ratio": round(fuzz, 4),
                }
            )
            print(f"  {vname:14s} auroc={au:.4f} bow={bow:.4f} fuzz={fuzz:.3f}", flush=True)

        for r in rows:
            base_null = max(k0, r["bow_auroc"])
            r["margin_vs_raw_and_bow"] = round(r["structure_auroc"] - base_null, 4)
            r["gate_passed"] = bool(r["margin_vs_raw_and_bow"] >= 0.10 and r["fuzz_ratio"] > 1.0)
        best = max(rows, key=lambda r: r["structure_auroc"])
        rec = {
            "foundation": f,
            "k0_raw_auroc": round(k0, 4),
            "lam0_finetuned_auroc": LAM0_FT.get(f),
            "best_variant": best["variant"],
            "best_auroc": best["structure_auroc"],
            "best_margin_vs_raw_and_bow": best["margin_vs_raw_and_bow"],
            "best_gate_passed": best["gate_passed"],
            "lift_over_raw": round(best["structure_auroc"] - k0, 4),
            "pc_domain_alignment": [round(x, 3) for x in dom_align],
            "variants": rows,
        }
        all_results.append(rec)
        print("RESULT " + json.dumps(rec), flush=True)

    json.dump(all_results, open(f"{OUT}/presence_polar_results.json", "w"), indent=2)
    print("\nSAVED " + f"{OUT}/presence_polar_results.json", flush=True)
    n_lift = sum(1 for r in all_results if r["lift_over_raw"] >= 0.05)
    print(f"SUMMARY {n_lift}/{len(FOUNDATIONS)} foundations lift >=0.05 over raw-frozen", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
