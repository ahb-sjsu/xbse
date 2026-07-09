"""JointPairSource — train one dimension on >=2 corpora at once (cross-dataset rehabilitation).

Within-dataset AUROC was inflated: the encoder separated a corpus's pairs by its *surface*. Two
mechanisms here remove that shortcut:

1. **Cross-dataset positives.** A training positive is same-valence but from a DIFFERENT corpus
   (anchor from A, positive from B). To pull an A-text next to a same-sign B-text the encoder
   cannot use A's surface — only the shared moral structure. This is the core fix; it works even
   without the adversary.

2. **Domain labels** for the gradient-reversal head (see `xbse.adversarial`).

The held-out gate is CROSS-DOMAIN by construction: every structural pair puts the anchor in one
corpus and both comparisons in the *other*, so `structure_auroc` IS the honest generalization
number — the thing that read ~0.50 for the single-corpus encoders.

A dimension is assembled from a list of domain specs: `(domain_name, rows)` where each row is
`(text, sign)` with sign in {'+','-'} in ONE agreed valence convention for that dimension. Thin
per-dimension builders live in `joint_builders.py`; all the mechanism is here.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from ..admission import AdmissionCriteria
from ..pairs import PairSource, Triplet, _norm, stable_frac


class JointPairSource(PairSource):
    max_len = 192

    def __init__(
        self,
        name: str,
        domains: Sequence[tuple[str, Sequence[tuple[str, str]]]],
        holdout_frac: float = 0.1,
        invariant_structure: str = "",
        label_source: str = "",
    ):
        self.name = name
        self._domain_specs = list(domains)
        self.domain_names = [d[0] for d in self._domain_specs]
        self.holdout_frac = holdout_frac
        self.admission = AdmissionCriteria(
            invariant_structure=invariant_structure or f"{name} valence (shared across corpora)",
            surface_class="wording / which corpus the text is drawn from",
            independent_label_source=label_source or f"union of per-corpus {name} labels",
        )
        self._rows_cache = None

    # rows: (domain_idx, text, sign)
    def _rows(self):
        if self._rows_cache is None:
            seen, rows = set(), []
            for di, (_dname, drows) in enumerate(self._domain_specs):
                for text, sign in drows:
                    text = (text or "").strip().replace("\n", " ")
                    key = _norm(text)
                    if len(text) < 12 or sign not in ("+", "-") or key in seen:
                        continue
                    seen.add(key)
                    rows.append((di, text[:800], sign))
            self._rows_cache = rows
        return self._rows_cache

    def _split(self):
        train, held = [], []
        for r in self._rows():
            key = f"joint|{self.name}|{r[0]}|{r[1][:60]}"
            (held if stable_frac(key) < self.holdout_frac else train).append(r)
        return train, held

    @staticmethod
    def _index(rows):
        # by (domain, sign) and by sign
        by_ds: dict[tuple[int, str], list[int]] = {}
        by_sign: dict[str, list[int]] = {}
        for i, (d, _t, s) in enumerate(rows):
            by_ds.setdefault((d, s), []).append(i)
            by_sign.setdefault(s, []).append(i)
        return by_ds, by_sign

    @staticmethod
    def _pick_cross(by_ds, rows, anchor_domain, sign, rng, n_domains):
        """Same sign, DIFFERENT domain if any such row exists; else same-sign any domain."""
        others = [d for d in range(n_domains) if d != anchor_domain and by_ds.get((d, sign))]
        if others:
            d = others[int(rng.integers(len(others)))]
            pool = by_ds[(d, sign)]
            return pool[int(rng.integers(len(pool)))]
        pool = by_ds.get((anchor_domain, sign), [])
        return pool[int(rng.integers(len(pool)))] if pool else None

    def train_rows(self):
        """(text, domain_idx) for the adversary's minibatches."""
        train, _ = self._split()
        return [(t, d) for (d, t, _s) in train]

    @property
    def n_domains(self) -> int:
        return len(self._domain_specs)

    def train_triplets(self) -> Iterator[Triplet]:
        for t, _d in self.train_triplets_tagged():
            yield t

    def train_triplets_tagged(self) -> Iterator[tuple[Triplet, int]]:
        """Yield (triplet, anchor_domain) so the adversary can de-confound the SAME anchor
        embeddings the task uses — no extra encoder pass (memory) and a tighter coupling."""
        train, _ = self._split()
        by_ds, by_sign = self._index(train)
        rng = np.random.default_rng(0)
        opp = {"+": "-", "-": "+"}
        for _i, (d, text, s) in enumerate(train):
            jp = self._pick_cross(by_ds, train, d, s, rng, self.n_domains)  # cross-domain +
            jn = self._pick_cross(by_ds, train, d, opp[s], rng, self.n_domains)  # cross-domain -
            if jp is None or jn is None:
                continue
            yield Triplet(anchor=text, positive=train[jp][1], negative=train[jn][1]), d

    def heldout_eval(self) -> dict:
        _, held = self._split()
        by_ds, by_sign = self._index(held)
        rng = np.random.default_rng(1)
        opp = {"+": "-", "-": "+"}
        structural_pairs, surface_pairs = [], []
        for _i, (d, text, s) in enumerate(held):
            # force BOTH comparisons into a different corpus than the anchor -> cross-domain AUROC
            jp = self._pick_cross(by_ds, held, d, s, rng, self.n_domains)
            jn = self._pick_cross(by_ds, held, d, opp[s], rng, self.n_domains)
            if jp is None or jn is None:
                continue
            structural_pairs.append((text, held[jp][1], True))
            structural_pairs.append((text, held[jn][1], False))
            surface_pairs.append(
                (text, ("Reportedly, " + text[0].lower() + text[1:]) if text else text)
            )
            if len(surface_pairs) >= 600:
                break
        return {
            "structural_pairs": structural_pairs,
            "surface_pairs": surface_pairs,
            "ood_texts": [],
        }
