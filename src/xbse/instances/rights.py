"""RightsBSE — the rights_respect dimension of the MoralVector, as a gated sub-BSE.

un_declarations (74 chunks) was too small; ECHR is the same idea at real scale: European Court of
Human Rights judgments labeled by which Convention ARTICLE (right) was at stake — a clean,
independent structure label from the courts.

invariance axis  -> positives = two cases about the SAME article (right), different facts
sensitivity axis -> negatives = a case about a DIFFERENT article
independent label -> ECHR judgments (which Convention article was violated)

Data: pulled from HF in-pod (like MBPP / recipes).
"""
from __future__ import annotations
from typing import Iterator

import numpy as np

from ..pairs import PairSource, Triplet, stable_frac
from ..admission import AdmissionCriteria

RIGHTS_CONFIG = {"base_model": "BAAI/bge-m3", "holdout_frac": 0.1, "max_cases": 9000}
_DATASETS = [("coastalcph/lex_glue", "ecthr_a"), ("lex_glue", "ecthr_a"),
             ("lex_glue", "ecthr_b"), ("ecthr_cases", "alleged-violation-prediction")]


class RightsBSEPairSource(PairSource):
    name = "rights"
    admission = AdmissionCriteria(
        invariant_structure="the right at stake (ECHR Convention article)",
        surface_class="case wording, facts, jurisdiction, language",
        independent_label_source="ECHR court judgments — which Convention article was violated",
    )

    def __init__(self, max_cases: int = 9000, holdout_frac: float = 0.1):
        self.max_cases = max_cases
        self.holdout_frac = holdout_frac
        self._rows_cache = None

    def _rows(self):
        if self._rows_cache is None:
            from datasets import load_dataset
            ds = None
            for name, cfg in _DATASETS:
                try:
                    ds = load_dataset(name, cfg, split="train") if cfg else load_dataset(name, split="train")
                    break
                except Exception:
                    continue
            if ds is None:
                raise RuntimeError("no ECHR dataset loadable from HF")
            rows = []
            for i, d in enumerate(ds):
                if i >= self.max_cases:
                    break
                facts = d.get("text") or d.get("facts")           # lex_glue uses "text" (list of paragraphs)
                facts = " ".join(facts) if isinstance(facts, list) else str(facts or "")
                labels = d.get("labels") or d.get("allegedly_violated_articles") or d.get("violated_articles") or []
                if isinstance(labels, list) and labels and len(facts) > 120:
                    rows.append((str(i), facts.strip()[:1500], str(labels[0])))   # primary article
            self._rows_cache = rows
        return self._rows_cache

    def _split(self):
        train, held = [], []
        for r in self._rows():
            (held if stable_frac(self.name + "|" + r[0]) < self.holdout_frac else train).append(r)
        return train, held

    @staticmethod
    def _by_article(rows):
        idx = {}
        for i, (_id, _facts, art) in enumerate(rows):
            idx.setdefault(art, []).append(i)
        return idx

    @staticmethod
    def _sample(pool, rng, exclude=None, tries=12):
        for _ in range(tries):
            j = pool[int(rng.integers(len(pool)))]
            if j != exclude:
                return j
        return None

    def train_triplets(self) -> Iterator[Triplet]:
        train, _ = self._split()
        by_art = self._by_article(train)
        all_idx = list(range(len(train)))
        rng = np.random.default_rng(0)
        for i, (_id, facts, art) in enumerate(train):
            same = by_art[art]
            if len(same) < 2:
                continue
            jp = self._sample(same, rng, exclude=i)
            jn = int(rng.integers(len(train)))
            if train[jn][2] == art:                        # ensure different article
                jn = (jn + 1) % len(train)
            if jp is not None:
                yield Triplet(anchor=facts, positive=train[jp][1], negative=train[jn][1])

    def heldout_eval(self) -> dict:
        _, held = self._split()
        by_art = self._by_article(held)
        rng = np.random.default_rng(1)
        structural_pairs, surface_pairs = [], []
        for i, (_id, facts, art) in enumerate(held):
            same = by_art[art]
            if len(same) < 2:
                continue
            jp = self._sample(same, rng, exclude=i)
            jn = int(rng.integers(len(held)))
            if held[jn][2] == art:
                jn = (jn + 1) % len(held)
            if jp is None:
                continue
            structural_pairs.append((facts, held[jp][1], True))    # same article -> near
            structural_pairs.append((facts, held[jn][1], False))   # different article -> far
            surface_pairs.append((facts[:750], facts[:1500]))      # same case, different span -> surface
            if len(surface_pairs) >= 600:
                break
        return {"structural_pairs": structural_pairs, "surface_pairs": surface_pairs, "ood_texts": []}
