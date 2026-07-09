"""Empirical rank test — the intrinsic dimensionality of a moral-scenario × dimension matrix.

The question "what is the MINIMAL set of dimensions needed to model any moral scenario" is a
low-rank question about the dimension covariance Σ (cf. GDT prereg-sigma-v1): given an N×D matrix
whose columns are candidate moral axes scored over N scenarios, how many independent directions
does it actually contain? A redundant axis (a linear combination of others) does not add rank.

This module is DATA-agnostic: feed it any (N, D) score matrix + column names. It reports
  - the eigenspectrum of the correlation matrix,
  - participation ratio PR = (Σλ)² / Σλ²   — a continuous "effective rank",
  - variance-threshold ranks (#components for 90% / 95% variance),
  - Horn's PARALLEL ANALYSIS: #factors whose eigenvalue beats the 95th percentile of eigenvalues
    from column-permuted (independent) data — the standard, null-calibrated factor count,
  - redundant pairs (|corr| > 0.7),
  - a permutation null on PR (independent columns → PR ≈ D; observed PR ≪ D = real collapse).

Pre-registration discipline: freeze the candidate columns and thresholds BEFORE running, so
"the rank is k" is a test, not a fit. Nothing here is fit to a target.
"""
from __future__ import annotations

import numpy as np


def _corr_eigs(X: np.ndarray) -> np.ndarray:
    """Descending, non-negative eigenvalues of the column-correlation matrix of X."""
    R = np.corrcoef(X, rowvar=False)
    R = np.nan_to_num(R, nan=0.0)
    eig = np.linalg.eigvalsh(R)                 # ascending, symmetric
    return np.clip(eig[::-1], 0.0, None)


def _participation_ratio(eig: np.ndarray) -> float:
    s = eig.sum()
    return float(s * s / np.square(eig).sum()) if s > 0 else 0.0


def effective_rank(X: np.ndarray, dim_names: list[str], n_perm: int = 500, seed: int = 0,
                   var_thresholds: tuple[float, ...] = (0.90, 0.95),
                   redundancy_r: float = 0.70) -> dict:
    """Empirical dimensionality of an (N, D) scenario×dimension score matrix.

    X columns must correspond 1:1 to dim_names. Rows are scenarios. Missing values should be
    imputed by the caller (or dropped) — this expects a dense float array.
    """
    X = np.asarray(X, dtype=float)
    N, D = X.shape
    if D != len(dim_names):
        raise ValueError(f"X has {D} columns but {len(dim_names)} names")
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)

    eig = _corr_eigs(Xs)
    pr = _participation_ratio(eig)
    cumvar = np.cumsum(eig) / eig.sum()
    var_ranks = {f"{int(t*100)}%": int(np.searchsorted(cumvar, t) + 1) for t in var_thresholds}

    R = np.nan_to_num(np.corrcoef(Xs, rowvar=False), nan=0.0)
    redundant = sorted(
        ((dim_names[i], dim_names[j], float(R[i, j]))
         for i in range(D) for j in range(i + 1, D) if abs(R[i, j]) >= redundancy_r),
        key=lambda t: -abs(t[2]))

    # Horn's parallel analysis + PR null: permute each column independently (kills cross-column
    # structure, preserves marginals) -> null eigenspectrum for an "independent D-dim" world.
    rng = np.random.default_rng(seed)
    null_eigs = np.empty((n_perm, D))
    null_pr = np.empty(n_perm)
    for p in range(n_perm):
        Xp = np.column_stack([rng.permutation(Xs[:, c]) for c in range(D)])
        e = _corr_eigs(Xp)
        null_eigs[p] = e
        null_pr[p] = _participation_ratio(e)
    null_p95 = np.percentile(null_eigs, 95, axis=0)          # per-rank 95th pctile of null eigs
    horn_k = int(np.sum(eig > null_p95))                     # retained factors (real > null)

    return {
        "n_scenarios": int(N), "n_candidate_dims": int(D),
        "eigenspectrum": [float(x) for x in eig],
        "participation_ratio": pr,
        "participation_ratio_null_mean": float(null_pr.mean()),
        "participation_ratio_null_p2.5_97.5": [float(np.percentile(null_pr, 2.5)),
                                               float(np.percentile(null_pr, 97.5))],
        "variance_ranks": var_ranks,
        "horn_parallel_retained": horn_k,
        "redundant_pairs": redundant,
        "dim_names": list(dim_names),
    }


def print_report(rep: dict) -> None:
    print(f"N={rep['n_scenarios']} scenarios × D={rep['n_candidate_dims']} candidate dims")
    print("eigenspectrum:", " ".join(f"{x:.2f}" for x in rep["eigenspectrum"]))
    pr, lo_hi = rep["participation_ratio"], rep["participation_ratio_null_p2.5_97.5"]
    print(f"participation ratio (effective rank): {pr:.2f}  "
          f"[independent-null: {rep['participation_ratio_null_mean']:.2f} "
          f"({lo_hi[0]:.2f}-{lo_hi[1]:.2f})]")
    print(f"variance ranks: {rep['variance_ranks']}")
    print(f"Horn parallel-analysis retained factors: {rep['horn_parallel_retained']}  "
          f"(of {rep['n_candidate_dims']})")
    if rep["redundant_pairs"]:
        print("redundant pairs (|r|>=0.70):")
        for a, b, r in rep["redundant_pairs"]:
            print(f"   {a} ~ {b}: r={r:+.2f}")
    else:
        print("redundant pairs (|r|>=0.70): none")


if __name__ == "__main__":
    # self-test: 8 columns but only 3 latent factors -> effective rank should be ~3, Horn ~3
    rng = np.random.default_rng(0)
    F = rng.standard_normal((4000, 3))
    load = rng.standard_normal((3, 8))
    X = F @ load + 0.15 * rng.standard_normal((4000, 8))
    names = [f"d{i}" for i in range(8)]
    print_report(effective_rank(X, names, n_perm=200))
