"""PhysHarmBSE — the physical_harm dimension of the MoralVector, fed by BeaverTails.

BeaverTails QA pairs are human-labeled with `is_safe` + 14 harm categories. Five categories are
bodily/physical harm (violence, self-harm, child abuse, animal abuse, weapons/drugs). This encoder
learns the physical-harm valence, invariant to topic/wording.

invariance  -> positives = two same-valence items (both harmful, or both safe), DIFFERENT prompt
sensitivity -> negatives = the MATCHED opposite-valence response to the SAME prompt (hard negative)
label       -> BeaverTails human safety + harm-category annotation (independent of z)
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from ..admission import AdmissionCriteria
from ..pairs import PairSource, Triplet, stable_frac

PHYSHARM_CONFIG = {
    "base_model": "BAAI/bge-m3",
    "hf": "PKU-Alignment/BeaverTails",
    "split": "30k_test",
    "max_rows": 30000,
}
_PHYS = (
    "violence_aiding_and_abetting_incitement",
    "self_harm",
    "child_abuse",
    "animal_abuse",
    "drug_abuse_weapons_banned_substance",
)


class PhysHarmBSEPairSource(PairSource):
    name = "physharm"
    max_len = 192
    admission = AdmissionCriteria(
        invariant_structure="physical-harm valence of a text (bodily harm present vs safe)",
        surface_class="wording / topic of the prompt and response",
        independent_label_source="BeaverTails human safety + 14-category harm annotation",
    )

    def __init__(
        self, split: str | None = None, max_rows: int | None = None, holdout_frac: float = 0.1
    ):
        self.split = split or PHYSHARM_CONFIG["split"]
        self.max_rows = max_rows or PHYSHARM_CONFIG["max_rows"]
        self.holdout_frac = holdout_frac
        self._rows_cache = None

    def _rows(self):
        # (topic=prompt, text, harm)  harm: 1 physical-harm, 0 safe
        if self._rows_cache is None:
            from datasets import load_dataset

            ds = load_dataset(PHYSHARM_CONFIG["hf"], split=self.split, streaming=True)
            rows = []
            for i, d in enumerate(ds):
                if i >= self.max_rows:
                    break
                cat = d.get("category") or {}
                phys = any(cat.get(k) for k in _PHYS)
                safe = bool(d.get("is_safe"))
                if not (phys or safe):
                    continue  # skip non-physical unsafe (ambiguous for THIS dimension)
                text = (
                    ((d.get("prompt", "") or "") + " -> " + (d.get("response", "") or ""))
                    .strip()
                    .replace("\n", " ")
                )
                if len(text) < 30:
                    continue
                prompt = (d.get("prompt", "") or "").strip()
                rows.append((prompt, text[:800], 1 if phys else 0))
            self._rows_cache = rows
        return self._rows_cache

    def _split(self):
        train, held = [], []
        for r in self._rows():
            (held if stable_frac("physharm|" + r[0]) < self.holdout_frac else train).append(r)
        return train, held

    @staticmethod
    def _index(rows):
        by_lab = {0: [], 1: []}
        by_topic: dict[str, list[int]] = {}
        for i, (t, _x, lab) in enumerate(rows):
            by_lab[lab].append(i)
            by_topic.setdefault(t, []).append(i)
        return by_lab, by_topic

    @staticmethod
    def _sample_diff_topic(pool, rows, topic, rng, tries=12):
        for _ in range(tries):
            j = pool[int(rng.integers(len(pool)))]
            if rows[j][0] != topic:
                return j
        return None

    def _matched_or_any_neg(self, rows, by_topic, by_lab, i, topic, lab, rng):
        for k in by_topic.get(topic, []):  # matched: same prompt, opposite valence
            if rows[k][2] != lab:
                return k
        opp = by_lab[1 - lab]  # fallback: any opposite-valence item
        return opp[int(rng.integers(len(opp)))] if opp else None

    def train_triplets(self) -> Iterator[Triplet]:
        train, _ = self._split()
        by_lab, by_topic = self._index(train)
        rng = np.random.default_rng(0)
        for i, (topic, text, lab) in enumerate(train):
            jp = self._sample_diff_topic(by_lab[lab], train, topic, rng)
            jn = self._matched_or_any_neg(train, by_topic, by_lab, i, topic, lab, rng)
            if jp is None or jn is None:
                continue
            yield Triplet(anchor=text, positive=train[jp][1], negative=train[jn][1])

    def heldout_eval(self) -> dict:
        _, held = self._split()
        by_lab, by_topic = self._index(held)
        rng = np.random.default_rng(1)
        structural_pairs, surface_pairs = [], []
        for i, (topic, text, lab) in enumerate(held):
            jp = self._sample_diff_topic(by_lab[lab], held, topic, rng)
            jn = self._matched_or_any_neg(held, by_topic, by_lab, i, topic, lab, rng)
            if jp is None or jn is None:
                continue
            structural_pairs.append((text, held[jp][1], True))  # same valence, diff prompt -> near
            structural_pairs.append((text, held[jn][1], False))  # opposite valence -> far
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
