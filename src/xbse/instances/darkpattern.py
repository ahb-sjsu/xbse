"""AutonomyDarkBSE — the autonomy_respect feeder, from ec-darkpattern (retry after MentalManip).

MentalManip's long movie-dialogue manipulation did NOT transfer (AUROC 0.51). ec-darkpattern is a
clean, short, ~balanced binary corpus of manipulative interface/e-commerce text (dark-pattern vs
not; Mathur 2019 positives + segmented negatives). A dark pattern manipulates a user's choice —
a direct violation of autonomy_respect. Short + clean is what worked for the other feeders.

invariance  -> positives = same valence (both manipulative or both respectful), different category
sensitivity -> negatives = opposite valence
label       -> ec-darkpattern binary dark-pattern annotation (independent of z)
"""
from __future__ import annotations
import csv
from typing import Iterator

import numpy as np

from ..pairs import PairSource, Triplet, stable_frac
from ..admission import AdmissionCriteria

DARKPATTERN_CONFIG = {
    "base_model": "BAAI/bge-m3",
    "tsv": "/archive/ethics-corpora/darkpattern/dataset.tsv",  # page_id, text, label, Pattern Category
}


class AutonomyDarkBSEPairSource(PairSource):
    name = "autonomy_dark"
    max_len = 96
    admission = AdmissionCriteria(
        invariant_structure="manipulation valence of interface text (dark pattern vs respectful)",
        surface_class="wording / product / pattern category",
        independent_label_source="ec-darkpattern binary dark-pattern annotation",
    )

    def __init__(self, tsv: str | None = None, holdout_frac: float = 0.1):
        self.tsv = tsv or DARKPATTERN_CONFIG["tsv"]
        self.holdout_frac = holdout_frac
        self._rows_cache = None

    def _rows(self):
        # (topic=pattern-category, text, manip)  manip: 1 dark pattern, 0 respectful
        if self._rows_cache is None:
            seen, rows = set(), []
            with open(self.tsv, newline="", encoding="utf-8", errors="replace") as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    text = (row.get("text") or "").strip().replace("\n", " ")
                    if len(text) < 10 or text in seen:
                        continue
                    try:
                        lab = int(row.get("label"))
                    except (TypeError, ValueError):
                        continue
                    seen.add(text)
                    topic = (row.get("Pattern Category") or "none").strip() or "none"
                    rows.append((topic, text[:400], lab))
            self._rows_cache = rows
        return self._rows_cache

    def _split(self):
        train, held = [], []
        for r in self._rows():
            (held if stable_frac("autonomy_dark|" + r[1][:60]) < self.holdout_frac else train).append(r)
        return train, held

    @staticmethod
    def _index(rows):
        by_lab = {0: [], 1: []}
        for i, (_t, _x, lab) in enumerate(rows):
            by_lab[lab].append(i)
        return by_lab

    @staticmethod
    def _sample_diff_topic(pool, rows, topic, rng, tries=12):
        for _ in range(tries):
            j = pool[int(rng.integers(len(pool)))]
            if rows[j][0] != topic:
                return j
        return pool[int(rng.integers(len(pool)))] if pool else None

    def train_triplets(self) -> Iterator[Triplet]:
        train, _ = self._split()
        by_lab = self._index(train)
        rng = np.random.default_rng(0)
        for i, (topic, text, lab) in enumerate(train):
            jp = self._sample_diff_topic(by_lab[lab], train, topic, rng)
            opp = by_lab[1 - lab]
            jn = opp[int(rng.integers(len(opp)))] if opp else None
            if jp is None or jn is None:
                continue
            yield Triplet(anchor=text, positive=train[jp][1], negative=train[jn][1])

    def heldout_eval(self) -> dict:
        _, held = self._split()
        by_lab = self._index(held)
        rng = np.random.default_rng(1)
        structural_pairs, surface_pairs = [], []
        for i, (topic, text, lab) in enumerate(held):
            jp = self._sample_diff_topic(by_lab[lab], held, topic, rng)
            opp = by_lab[1 - lab]
            jn = opp[int(rng.integers(len(opp)))] if opp else None
            if jp is None or jn is None:
                continue
            structural_pairs.append((text, held[jp][1], True))
            structural_pairs.append((text, held[jn][1], False))
            surface_pairs.append((text, ("Reportedly, " + text[0].lower() + text[1:]) if text else text))
            if len(surface_pairs) >= 600:
                break
        return {"structural_pairs": structural_pairs, "surface_pairs": surface_pairs, "ood_texts": []}
