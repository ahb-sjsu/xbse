"""Shared geometric measurements — used by validate.py and by probes/.

Ported from the legal_h3_v2 fix: local intrinsic dimension via local-PCA participation ratio
(effective rank — stable in high-D, unlike Levina-Bickel MLE) + spatial autocorrelation of the
dimension field (the discriminator between structured strata and estimator noise). Kept in core
because both the validation gate and the dimension probe need identical, trusted estimators.
"""
from __future__ import annotations
import numpy as np


def local_effrank(X: np.ndarray, anchors: np.ndarray, m: int = 200) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=m + 1).fit(X)
    _, idx = nn.kneighbors(X[anchors])
    d = np.empty(len(anchors))
    for a in range(len(anchors)):
        R = X[idx[a]]; R = R - R.mean(0)
        lam = np.linalg.svd(R, compute_uv=False) ** 2
        d[a] = (lam.sum() ** 2) / (np.sum(lam ** 2) + 1e-12)   # participation ratio = effective dim
    return d


def dimension_field(X: np.ndarray, m: int = 200, n: int = 1500, seed: int = 0) -> dict:
    """Local dimension distribution + spatial autocorrelation. High autocorr = structured strata."""
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)
    anch = rng.choice(len(X), min(n, len(X)), replace=False)
    d = local_effrank(X, anch, m)
    nn = NearestNeighbors(n_neighbors=11).fit(X[anch]); _, nb = nn.kneighbors(X[anch])
    d_nbr = np.array([d[nb[i, 1:]].mean() for i in range(len(anch))])
    return {"dim_median": float(np.median(d)),
            "dim_iqr": float(np.percentile(d, 75) - np.percentile(d, 25)),
            "dim_range": [float(d.min()), float(d.max())],
            "spatial_autocorr": float(np.corrcoef(d, d_nbr)[0, 1])}


def cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two views of the same items — for the DAG "do lenses add info?" test."""
    def gram(A):
        A = A - A.mean(0)
        return A @ A.T
    Kx, Ky = gram(X), gram(Y)
    hsic = (Kx * Ky).sum()
    return float(hsic / (np.sqrt((Kx * Kx).sum() * (Ky * Ky).sum()) + 1e-12))
