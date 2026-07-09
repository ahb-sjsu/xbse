"""PrivacyBSE — the privacy_protection dimension, from dual-judge-labeled privacy RoTs.

No corpus carries a signed privacy violation-vs-respect label (verified by research: OPP-115 /
PrivacyQA are clause classification; CI corpora are info-flow params). So privacy-keyword
Social-Chem RoTs were relabeled by a dual-judge (qwen3 + glm-5) pipeline for signed privacy
valence in [-1, +1]. This encoder learns that valence.

CAVEAT: the RoT substrate is prescriptive (mostly privacy-protective norms), so the label
distribution skews positive (~mean +0.39, ~11% violation-leaning). A scenario corpus (AITA
privacy posts) would balance it — a documented follow-up.

invariance  -> positives = same valence sign, different bucket
sensitivity -> negatives = opposite valence sign
label       -> dual-judge (qwen3+glm-5) signed privacy valence over privacy Social-Chem RoTs
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import numpy as np

from ..admission import AdmissionCriteria
from ..pairs import PairSource, Triplet, stable_frac

PRIVACY_CONFIG = {
    "base_model": "BAAI/bge-m3",
    "jsonl": "/archive/ethics-corpora/privacy/privacy_labeled.jsonl",  # {text, privacy in [-1,1]}
    "dead_band": 0.05,
}


class PrivacyBSEPairSource(PairSource):
    name = "privacy"
    max_len = 192
    admission = AdmissionCriteria(
        invariant_structure="privacy valence of a text (privacy violated vs respected)",
        surface_class="wording / topic of the rule",
        independent_label_source="dual-judge (qwen3+glm-5) signed privacy valence over privacy RoTs",
    )

    def __init__(self, jsonl: str | None = None, holdout_frac: float = 0.1):
        self.jsonl = jsonl or PRIVACY_CONFIG["jsonl"]
        self.holdout_frac = holdout_frac
        self._rows_cache = None

    def _rows(self):
        # (bucket, text, sign)  sign: '+' respected, '-' violated, '0' neutral (dropped)
        if self._rows_cache is None:
            db = PRIVACY_CONFIG["dead_band"]
            seen, rows = set(), []
            with open(self.jsonl, encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    text = (d.get("text") or "").strip()
                    if len(text) < 20 or text in seen:
                        continue
                    v = float(d.get("privacy", 0.0))
                    sign = "+" if v > db else ("-" if v < -db else "0")
                    if sign == "0":
                        continue  # keep only directional items (both classes needed)
                    seen.add(text)
                    bucket = f"pv_{int(stable_frac('pbkt|' + text[:24]) * 40)}"
                    rows.append((bucket, text[:800], sign))
            self._rows_cache = rows
        return self._rows_cache

    def _split(self):
        train, held = [], []
        for r in self._rows():
            (held if stable_frac("privacy|" + r[1][:60]) < self.holdout_frac else train).append(r)
        return train, held

    @staticmethod
    def _index(rows):
        by_sign: dict[str, list[int]] = {}
        for i, (_t, _x, s) in enumerate(rows):
            by_sign.setdefault(s, []).append(i)
        return by_sign

    @staticmethod
    def _sample_diff_topic(pool, rows, topic, rng, tries=12):
        for _ in range(tries):
            j = pool[int(rng.integers(len(pool)))]
            if rows[j][0] != topic:
                return j
        return pool[int(rng.integers(len(pool)))] if pool else None

    def train_triplets(self) -> Iterator[Triplet]:
        train, _ = self._split()
        by_sign = self._index(train)
        rng = np.random.default_rng(0)
        for _i, (topic, text, s) in enumerate(train):
            jp = self._sample_diff_topic(by_sign[s], train, topic, rng)
            opp = [x for x in by_sign if x != s and by_sign[x]]
            jn = None
            if opp:
                pool = by_sign[opp[int(rng.integers(len(opp)))]]
                jn = pool[int(rng.integers(len(pool)))]
            if jp is None or jn is None:
                continue
            yield Triplet(anchor=text, positive=train[jp][1], negative=train[jn][1])

    def heldout_eval(self) -> dict:
        _, held = self._split()
        by_sign = self._index(held)
        rng = np.random.default_rng(1)
        structural_pairs, surface_pairs = [], []
        for _i, (topic, text, s) in enumerate(held):
            jp = self._sample_diff_topic(by_sign[s], held, topic, rng)
            opp = [x for x in by_sign if x != s and by_sign[x]]
            jn = None
            if opp:
                pool = by_sign[opp[int(rng.integers(len(opp)))]]
                jn = pool[int(rng.integers(len(pool)))]
            if jp is None or jn is None:
                continue
            structural_pairs.append((text, held[jp][1], True))
            structural_pairs.append((text, held[jn][1], False))
            surface_pairs.append(
                (text, ("Generally, " + text[0].lower() + text[1:]) if text else text)
            )
            if len(surface_pairs) >= 600:
                break
        return {
            "structural_pairs": structural_pairs,
            "surface_pairs": surface_pairs,
            "ood_texts": [],
        }
