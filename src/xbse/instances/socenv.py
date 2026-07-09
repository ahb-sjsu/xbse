"""SocEnvBSE — the societal_environmental dimension (societal-harm arm), from SBIC.

Social Bias Inference Corpus posts are human-labeled offensive-to-a-target-group vs not. This
encoder learns harm-to-society valence, invariant to which group/topic. (Environmental impact has
no clean text corpus; this feeder covers the SOCIETAL arm of societal_environmental — see
moralvector_reference.md; environmental is a documented sub-gap.)

invariance  -> positives = same valence (both harmful or both benign), DIFFERENT target group
sensitivity -> negatives = opposite valence, SAME target group (topic-decorrelation control)
label       -> SBIC offensiveness + target-category annotation (independent of z)

SBIC is a script-based HF dataset (needs datasets<3 / trust_remote_code), so it is pre-exported to
a JSONL once (via the datasets<3 venv) and read from disk here — keeping training on datasets 4.x.
"""
from __future__ import annotations
import json
from typing import Iterator

import numpy as np

from ..pairs import PairSource, Triplet, stable_frac
from ..admission import AdmissionCriteria

SOCENV_CONFIG = {
    "base_model": "BAAI/bge-m3",
    "jsonl": "/archive/ethics-corpora/sbic/sbic_socenv.jsonl",  # exported: {post, offensive, target}
}


class SocEnvBSEPairSource(PairSource):
    name = "socenv"
    max_len = 192
    admission = AdmissionCriteria(
        invariant_structure="societal-harm valence (harmful to a social group vs benign)",
        surface_class="wording / which group is referenced",
        independent_label_source="SBIC offensiveness + target-group annotation",
    )

    def __init__(self, jsonl: str | None = None, holdout_frac: float = 0.1):
        self.jsonl = jsonl or SOCENV_CONFIG["jsonl"]
        self.holdout_frac = holdout_frac
        self._rows_cache = None

    def _rows(self):
        # (topic=target-group, post, offensive)  offensive: 1 harmful, 0 benign
        if self._rows_cache is None:
            seen, rows = set(), []
            with open(self.jsonl, encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    post = (d.get("post") or "").strip().replace("\n", " ")
                    if len(post) < 20 or post in seen:
                        continue
                    seen.add(post)
                    topic = (d.get("target") or "none").strip() or "none"
                    off = 1 if int(d.get("offensive", 0)) else 0
                    rows.append((topic, post[:800], off))
            self._rows_cache = rows
        return self._rows_cache

    def _split(self):
        train, held = [], []
        for r in self._rows():
            (held if stable_frac("socenv|" + r[1][:60]) < self.holdout_frac else train).append(r)
        return train, held

    @staticmethod
    def _index(rows):
        by_lab = {0: [], 1: []}
        by_topic: dict[str, list[int]] = {}
        for i, (t, _p, lab) in enumerate(rows):
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

    def _neg(self, rows, by_topic, by_lab, topic, lab, rng):
        for k in by_topic.get(topic, []):     # same target group, opposite valence
            if rows[k][2] != lab:
                return k
        opp = by_lab[1 - lab]
        return opp[int(rng.integers(len(opp)))] if opp else None

    def train_triplets(self) -> Iterator[Triplet]:
        train, _ = self._split()
        by_lab, by_topic = self._index(train)
        rng = np.random.default_rng(0)
        for i, (topic, post, lab) in enumerate(train):
            jp = self._sample_diff_topic(by_lab[lab], train, topic, rng)
            jn = self._neg(train, by_topic, by_lab, topic, lab, rng)
            if jp is None or jn is None:
                continue
            yield Triplet(anchor=post, positive=train[jp][1], negative=train[jn][1])

    def heldout_eval(self) -> dict:
        _, held = self._split()
        by_lab, by_topic = self._index(held)
        rng = np.random.default_rng(1)
        structural_pairs, surface_pairs = [], []
        for i, (topic, post, lab) in enumerate(held):
            jp = self._sample_diff_topic(by_lab[lab], held, topic, rng)
            jn = self._neg(held, by_topic, by_lab, topic, lab, rng)
            if jp is None or jn is None:
                continue
            structural_pairs.append((post, held[jp][1], True))
            structural_pairs.append((post, held[jn][1], False))
            surface_pairs.append((post, ("As posted, " + post[0].lower() + post[1:]) if post else post))
            if len(surface_pairs) >= 600:
                break
        return {"structural_pairs": structural_pairs, "surface_pairs": surface_pairs, "ood_texts": []}
