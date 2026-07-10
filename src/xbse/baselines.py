"""Adversarial baselines — the controls a hostile reviewer reaches for first.

A high cross-dataset AUROC is only evidence of *moral structure* if a dumb model that sees only
surface can't match it. The core claim ("cross-corpus positives force the encoder off surface") is
falsified if a bag-of-words model does just as well. So every gate should report, next to the
encoder's `structure_auroc`, a **lexical baseline** on the SAME held-out structural pairs:

    bow_auroc = AUROC( tfidf_cosine(a, b) , same_structure )

If `structure_auroc - bow_auroc` is small, the encoder is a lexical matcher wearing a transformer
and the headline is surface, not structure. The margin is a standing honesty metric, not a one-off.
"""

from __future__ import annotations

import numpy as np


def _tfidf_cosines(
    structural_pairs, ngram_max: int = 2, min_df: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Fit TF-IDF on all texts in the pairs; return (cosine(a,b) per pair, same-structure labels).

    sklearn TF-IDF rows are L2-normalized, so cosine = dot product. A word/char-ngram bag is the
    strongest *purely lexical* model available with no training beyond counting."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [t for a, b, _ in structural_pairs for t in (a, b)]
    vec = TfidfVectorizer(
        lowercase=True, ngram_range=(1, ngram_max), min_df=min_df, sublinear_tf=True
    ).fit(texts)
    a_mat = vec.transform([a for a, _b, _s in structural_pairs])
    b_mat = vec.transform([b for _a, b, _s in structural_pairs])
    sims = np.asarray(a_mat.multiply(b_mat).sum(axis=1)).ravel()  # L2-normed rows -> dot = cosine
    labels = np.array([1 if s else 0 for _a, _b, s in structural_pairs])
    return sims, labels


def bow_structure_auroc(structural_pairs) -> float:
    """AUROC of a TF-IDF lexical model predicting same-structure on the held-out pairs.

    This is the number the encoder must beat to claim it captures more than vocabulary."""
    from sklearn.metrics import roc_auc_score

    if len(structural_pairs) < 4:
        return float("nan")
    sims, labels = _tfidf_cosines(structural_pairs)
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, sims))


def lexical_margin(structure_auroc: float, bow_auroc: float) -> float:
    """How much the encoder beats bag-of-words. Small (or negative) ⇒ the win is lexical."""
    if np.isnan(bow_auroc):
        return float("nan")
    return float(structure_auroc - bow_auroc)
