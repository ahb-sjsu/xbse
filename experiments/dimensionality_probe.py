#!/usr/bin/env python3
"""dimensionality_probe.py — measure moral-space dimensionality as a PROFILE.

Computes, from an embedding matrix (+ optional labels / groups / bond-features):
  - Floor_k   (invariance-quotient rank, LDA-style between/within bound)
  - Ceiling_k (bond-schema design-matrix rank)
  - optimal k under several objectives (MDL, judgment-prediction, intrinsic-dim)
  - stability across objectives (CoV) -> phenomenon-level vs task-relative
  - the same, on a matched non-moral control, and the DIFFERENCE

Designed so "task-relative" and "same as control" are first-class outcomes.

Deps: numpy, scikit-learn.
Inputs (npz):
  Z            (N,D)  moral embeddings (unsupervised MoBSE)
  groups       (N,)   invariance-class id (paraphrase/translation group) [for Floor]
  labels       (N,)   moral-judgment label [for O2], optional
  bond_feats   (N,F)  one-hot/embedded bond features [for Ceiling], optional
  Z_control    (N,D)  matched non-moral embeddings [for the control], optional
"""
from __future__ import annotations
import argparse, json, sys
import numpy as np
from numpy.linalg import matrix_rank, eigvalsh

# pre-registered thresholds
STABLE_COV = 0.25
TASKREL_COV = 0.50
NOISE_QUANTILE = 0.95   # within-class eigenvalue noise floor


def floor_k(Z, groups):
    """Between-class directions exceeding within-class noise (LDA-style bound)."""
    if groups is None:
        return None
    Z = Z - Z.mean(0)
    classes = np.unique(groups)
    if len(classes) < 3:
        return None
    # within-class scatter eigen-spectrum (noise floor)
    within = []
    Sb = np.zeros((Z.shape[1], Z.shape[1]))
    gmean = Z.mean(0)
    for c in classes:
        Zc = Z[groups == c]
        if len(Zc) < 2:
            continue
        within.append(np.cov(Zc.T))
        d = (Zc.mean(0) - gmean)[:, None]
        Sb += len(Zc) * (d @ d.T)
    Wbar = np.mean(within, axis=0)
    noise = np.quantile(eigvalsh(Wbar), NOISE_QUANTILE)
    between_eigs = eigvalsh(Sb / len(classes))
    return int((between_eigs > noise).sum())


def ceiling_k(bond_feats):
    if bond_feats is None:
        return None
    return int(matrix_rank(bond_feats - bond_feats.mean(0), tol=1e-6))


def pca_spectrum(Z):
    Z = Z - Z.mean(0)
    C = np.cov(Z.T)
    ev = np.sort(eigvalsh(C))[::-1]
    return np.clip(ev, 0, None)


def k_mdl(Z):
    """MDL-ish knee: k minimizing reconstruction MSE + k*(model cost per dim)."""
    ev = pca_spectrum(Z)
    N, D = Z.shape
    total = ev.sum() + 1e-12
    recon_err = (total - np.cumsum(ev)) / total          # fraction variance lost
    model_cost = np.arange(1, len(ev) + 1) * (np.log(N) / N)  # per-dim penalty
    obj = recon_err + model_cost
    return int(np.argmin(obj) + 1)


def k_participation(Z):
    """Effective dimension = participation ratio of the eigenspectrum."""
    ev = pca_spectrum(Z)
    return float((ev.sum() ** 2) / (np.sum(ev ** 2) + 1e-12))


def k_judgment(Z, labels):
    """k maximizing held-out judgment-prediction accuracy from k-dim PCA."""
    if labels is None:
        return None
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    Kmax = min(40, Z.shape[1] - 1)
    best_k, best_acc = 1, -1
    for k in range(1, Kmax + 1, max(1, Kmax // 20)):
        P = PCA(k).fit_transform(Z)
        try:
            acc = cross_val_score(LogisticRegression(max_iter=500), P, labels, cv=3).mean()
        except Exception:
            continue
        if acc > best_acc:
            best_acc, best_k = acc, k
    return best_k


def profile(npz, prefix=""):
    Z = npz["Z"] if prefix == "" else npz.get("Z_control")
    if Z is None:
        return None
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    groups = npz.get("groups") if prefix == "" else None
    labels = npz.get("labels") if prefix == "" else None
    bond = npz.get("bond_feats") if prefix == "" else None

    ks = {}
    ks["mdl"] = k_mdl(Z)
    ks["participation"] = round(k_participation(Z), 1)
    kj = k_judgment(Z, labels)
    if kj is not None:
        ks["judgment"] = kj

    kvals = [v for v in ks.values() if isinstance(v, (int,)) or float(v) == int(float(v))]
    kvals = [float(v) for v in ks.values()]
    cov = float(np.std(kvals) / (np.mean(kvals) + 1e-9)) if len(kvals) >= 2 else float("nan")
    stability = ("stable/phenomenon-level" if cov < STABLE_COV else
                 "task-relative" if cov >= TASKREL_COV else "weakly-stable")
    return {"floor_k": floor_k(Z, groups), "ceiling_k": ceiling_k(bond),
            "optimal_k_by_objective": ks, "stability_CoV": round(cov, 3),
            "stability_verdict": stability}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="npz with Z, groups, labels, bond_feats, Z_control")
    ap.add_argument("--registered", action="store_true")
    a = ap.parse_args()
    if not a.registered:
        print("Freeze & hash the protocol before trusting results (dry mode).\n", file=sys.stderr)

    npz = np.load(a.data)
    moral = profile(npz, "")
    control = profile(npz, "control")

    out = {"moral": moral, "control": control}
    if moral and control:
        # every claim is moral MINUS control
        out["moral_specific"] = {
            "delta_CoV": round(moral["stability_CoV"] - control["stability_CoV"], 3),
            "same_profile_as_control": bool(
                abs(moral["stability_CoV"] - control["stability_CoV"]) < 0.1 and
                abs(np.mean(list(map(float, moral["optimal_k_by_objective"].values()))) -
                    np.mean(list(map(float, control["optimal_k_by_objective"].values())))) < 2),
        }
        if out["moral_specific"]["same_profile_as_control"]:
            out["VERDICT"] = ("NULL — moral dimensionality profile matches non-moral control. "
                              "Structure is generic to text encoders, not moral-specific.")
        elif moral["stability_verdict"] == "task-relative":
            out["VERDICT"] = ("TASK-RELATIVE — optimal k swings across objectives beyond control. "
                              "'The dimension of moral space' is not well-defined. Retire fixed-k.")
        else:
            out["VERDICT"] = (f"PHENOMENON-LEVEL — stable k in [{moral['floor_k']},{moral['ceiling_k']}], "
                              "distinct from control. Report k with CI; rewrite Ch.5 with measured value.")
    else:
        out["VERDICT"] = "control missing — cannot separate moral-specific from generic. Run with Z_control."
    print(json.dumps(out, indent=1, default=str))
    __import__("pathlib").Path("dimensionality_results.json").write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
