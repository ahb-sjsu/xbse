"""AutonomyBSE — the autonomy_respect dimension (manipulation/consent proxy), from MentalManip.

No clean full-axis autonomy/consent dataset exists; MentalManip (ACL 2024) human-labels dialogues
as manipulative vs not (with an 11-technique / 5-vulnerability taxonomy). Manipulation is the core
violation of autonomy_respect (overriding another's agency), so this encoder learns the manipulation
valence as the feeder.

invariance  -> positives = same valence (both manipulative or both respectful), different bucket
sensitivity -> negatives = opposite valence
label       -> MentalManip human manipulative/non-manipulative annotation (independent of z)
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from ..admission import AdmissionCriteria
from ..pairs import PairSource, Triplet, stable_frac

MENTALMANIP_CONFIG = {
    "base_model": "BAAI/bge-m3",
    "hf": "audreyeleven/MentalManip",
    "config": "mentalmanip_con",
}


class AutonomyBSEPairSource(PairSource):
    name = "autonomy"
    max_len = 192
    admission = AdmissionCriteria(
        invariant_structure="manipulation valence of a dialogue (manipulative vs respectful of agency)",
        surface_class="wording / speakers / scenario of the dialogue",
        independent_label_source="MentalManip human manipulative/non-manipulative annotation",
    )

    def __init__(self, holdout_frac: float = 0.1):
        self.holdout_frac = holdout_frac
        self._rows_cache = None

    def _rows(self):
        # (topic=bucket, text, manip)  manip: 1 manipulative, 0 respectful
        if self._rows_cache is None:
            from datasets import load_dataset

            ds = load_dataset(MENTALMANIP_CONFIG["hf"], MENTALMANIP_CONFIG["config"], split="train")
            rows = []
            for d in ds:
                text = (d.get("dialogue") or "").strip().replace("\n", " ")
                if len(text) < 30:
                    continue
                try:
                    lab = int(d.get("manipulative"))
                except (TypeError, ValueError):
                    continue
                # spread for decorrelation
                bucket = f"mm_{int(stable_frac('bkt|' + text[:24]) * 40)}"
                rows.append((bucket, text[:800], lab))
            self._rows_cache = rows
        return self._rows_cache

    def _split(self):
        train, held = [], []
        for r in self._rows():
            (held if stable_frac("autonomy|" + r[1][:60]) < self.holdout_frac else train).append(r)
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
        for _i, (topic, text, lab) in enumerate(train):
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
        for _i, (topic, text, lab) in enumerate(held):
            jp = self._sample_diff_topic(by_lab[lab], held, topic, rng)
            opp = by_lab[1 - lab]
            jn = opp[int(rng.integers(len(opp)))] if opp else None
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
