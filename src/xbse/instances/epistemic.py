"""EpistemicBSE — the epistemic_quality (honesty) dimension, from Social-Chem honesty RoTs.

Social-Chem-101 rules-of-thumb that are ABOUT honesty (keyword-filtered: lie / honest / truth /
deceive / mislead) carry a signed moral judgment. This encoder learns the honesty valence,
invariant to topic — the epistemic_quality feeder that isolated cleanly in the rank test.

invariance  -> positives = same judgment-sign (both honesty-upheld or both -violated), diff situation
sensitivity -> negatives = opposite sign, SAME situation (topic-decorrelation control)
label       -> Social-Chem-101 action-moral-judgment sign on honesty RoTs (independent of z)
"""

from __future__ import annotations

import csv
from collections.abc import Iterator

import numpy as np

from ..admission import AdmissionCriteria
from ..pairs import PairSource, Triplet, stable_frac

EPISTEMIC_CONFIG = {
    "base_model": "BAAI/bge-m3",
    "social_chem_tsv": "/archive/ethics-corpora/social-chem-101/social-chem-101/social-chem-101.v1.0.tsv",
    "keywords": (
        "lie",
        "lying",
        "lied",
        "honest",
        "dishonest",
        "truth",
        "truthful",
        "deceive",
        "deceiv",
        "mislead",
        "misleading",
    ),
}


def _jsign(v: str) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "0"
    return "+" if f > 0 else ("-" if f < 0 else "0")


class EpistemicBSEPairSource(PairSource):
    name = "epistemic"
    max_len = 192
    admission = AdmissionCriteria(
        invariant_structure="honesty valence of a rule of thumb (truthful upheld vs deceptive)",
        surface_class="wording / framing / situation topic",
        independent_label_source="Social-Chem-101 moral-judgment sign on honesty rules of thumb",
    )

    def __init__(self, tsv: str | None = None, holdout_frac: float = 0.1):
        self.tsv = tsv or EPISTEMIC_CONFIG["social_chem_tsv"]
        self.holdout_frac = holdout_frac
        self._rows_cache = None

    def _rows(self):
        # (topic=situation-short-id, rot, sign)  honesty RoTs only, rot-agree>=3
        if self._rows_cache is None:
            kw = EPISTEMIC_CONFIG["keywords"]
            seen, rows = set(), []
            with open(self.tsv, newline="", encoding="utf-8", errors="replace") as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    try:
                        agree = int(row.get("rot-agree") or 0)
                    except (TypeError, ValueError):
                        agree = 0
                    if agree < 3:
                        continue
                    rot = (row.get("rot") or "").strip()
                    low = rot.lower()
                    if not rot or not any(k in low for k in kw):
                        continue
                    topic = (row.get("situation-short-id") or "").strip()
                    if not topic or rot in seen:
                        continue
                    seen.add(rot)
                    rows.append((topic, rot, _jsign(row.get("action-moral-judgment"))))
            self._rows_cache = rows
        return self._rows_cache

    def _split(self):
        train, held = [], []
        for r in self._rows():
            (
                held
                if stable_frac("epistemic|" + r[1].strip().lower()) < self.holdout_frac
                else train
            ).append(r)
        return train, held

    @staticmethod
    def _index(rows):
        by_sign: dict[str, list[int]] = {}
        by_topic: dict[str, list[int]] = {}
        for i, (t, _rot, s) in enumerate(rows):
            by_sign.setdefault(s, []).append(i)
            by_topic.setdefault(t, []).append(i)
        return by_sign, by_topic

    @staticmethod
    def _sample(pool, rows, field, exclude, rng, tries=12):
        for _ in range(tries):
            j = pool[int(rng.integers(len(pool)))]
            if rows[j][field] != exclude:
                return j
        return None

    def train_triplets(self) -> Iterator[Triplet]:
        train, _ = self._split()
        by_sign, by_topic = self._index(train)
        rng = np.random.default_rng(0)
        for _i, (topic, rot, s) in enumerate(train):
            jp = self._sample(by_sign[s], train, 0, topic, rng)  # same sign, diff topic
            if jp is None:
                continue
            jn = self._sample(by_topic[topic], train, 2, s, rng)  # same topic, diff sign
            neg = train[jn][1] if jn is not None else None
            if neg is None:
                opp = [x for x in ("+", "-", "0") if x != s and by_sign.get(x)]
                if not opp:
                    continue
                pool = by_sign[opp[int(rng.integers(len(opp)))]]
                neg = train[pool[int(rng.integers(len(pool)))]][1]
            yield Triplet(anchor=rot, positive=train[jp][1], negative=neg)

    def heldout_eval(self) -> dict:
        _, held = self._split()
        by_sign, by_topic = self._index(held)
        rng = np.random.default_rng(1)
        structural_pairs, surface_pairs = [], []
        for _i, (topic, rot, s) in enumerate(held):
            jp = self._sample(by_sign[s], held, 0, topic, rng)
            jn = self._sample(by_topic[topic], held, 2, s, rng)  # same topic, diff sign
            if jn is None:  # honesty RoTs are topic-sparse:
                opp = [x for x in by_sign if x != s and by_sign[x]]  # fall back to any diff-sign
                if opp:
                    pool = by_sign[opp[int(rng.integers(len(opp)))]]
                    jn = pool[int(rng.integers(len(pool)))]
            if jp is None or jn is None:
                continue
            structural_pairs.append((rot, held[jp][1], True))  # same sign, diff topic -> near
            structural_pairs.append((rot, held[jn][1], False))  # diff sign, same topic -> far
            surface_pairs.append((rot, ("Generally, " + rot[0].lower() + rot[1:]) if rot else rot))
            if len(surface_pairs) >= 600:
                break
        return {
            "structural_pairs": structural_pairs,
            "surface_pairs": surface_pairs,
            "ood_texts": [],
        }
