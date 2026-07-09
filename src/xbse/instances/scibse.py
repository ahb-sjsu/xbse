"""SciBSE — the scientific-claim *-BSE. FIRST framework-proving instance (design §3.3).

Chosen first because its label source is HARD and independent: a paper's title (compressed
finding) and abstract (full finding) describe the SAME finding in very different surface forms,
written by the authors — not by any embedding. Same-paper -> near, different-paper -> far is a
real retrieval-style check, so passing the gate here means the abstraction is real, not a tidy
factoring of encoders that happened to exist.

invariance axis  -> positives = (abstract, title) of the SAME paper   (surface: length/wording)
sensitivity axis -> negatives = a DIFFERENT paper's abstract          (different finding)

Data: arxiv_papers on Atlas (680K rows, title+abstract+categories). Split by arxiv_id.
"""
from __future__ import annotations
from typing import Iterator

from ..pairs import PairSource, Triplet, stable_frac
from ..admission import AdmissionCriteria

SCIBSE_CONFIG = {"base_model": "BAAI/bge-m3", "max_papers": 50000, "holdout_frac": 0.1}


class SciBSEPairSource(PairSource):
    name = "scibse"
    admission = AdmissionCriteria(
        invariant_structure="the scientific finding (paper identity)",
        surface_class="wording, length, notation (title vs abstract)",
        independent_label_source="arXiv paper identity + categories (authors' title & abstract "
                                 "independently describe one finding; citation graph available for v2)",
    )

    def __init__(self, max_papers: int = 50000, holdout_frac: float = 0.1):
        self.max_papers = max_papers
        self.holdout_frac = holdout_frac
        self._cache = None

    def _rows(self):
        if self._cache is None:
            import sys
            sys.path.insert(0, "/archive/ethics-corpora")
            import embed_v2 as e
            c = e.get_db_conn(); cur = c.cursor()
            cur.execute("select arxiv_id, title, abstract, categories from arxiv_papers "
                        "where title is not null and abstract is not null limit %s", (self.max_papers,))
            self._cache = [(r[0], r[1].strip(), r[2].strip(), r[3] or "") for r in cur.fetchall()]
            c.commit()
        return self._cache

    def _split(self):
        train, held = [], []
        for r in self._rows():
            b = stable_frac(self.name + "|" + str(r[0]))
            (held if b < self.holdout_frac else train).append(r)
        return train, held

    def train_triplets(self) -> Iterator[Triplet]:
        train, _ = self._split()
        n = len(train)
        for i, (_aid, title, abstract, _cat) in enumerate(train):
            neg_abstract = train[(i + n // 2) % n][2]     # a different paper's abstract
            yield Triplet(anchor=abstract, positive=title, negative=neg_abstract)

    def heldout_eval(self) -> dict:
        _, held = self._split()
        m = min(len(held), 1000)
        structural_pairs, surface_pairs = [], []
        for i in range(m - 1):
            _aid, title, abstract, _cat = held[i]
            structural_pairs.append((abstract, title, True))               # same paper -> near
            structural_pairs.append((abstract, held[i + 1][1], False))     # other paper title -> far
            surface_pairs.append((abstract, title))                        # same finding, different surface
        return {"structural_pairs": structural_pairs, "surface_pairs": surface_pairs, "ood_texts": []}
