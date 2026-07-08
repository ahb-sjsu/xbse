"""MoBSE v2 — morality *-BSE keyed on the moral FINGERPRINT, topic-decorrelated.

v1 failed (AUROC 0.571) because it keyed on valence — 1 bit, dominated by topic. v2 uses
Social-Chem-101's human labels directly: the structure is the moral FINGERPRINT
(Moral Foundation x judgment sign), which is theory-grounded and spans topics.

The whole game is TOPIC-DECORRELATION, so pairs are built to force it:
  positive = same fingerprint, DIFFERENT situation (topic)  -> near   (can't cheat via topic)
  negative = different fingerprint, SAME situation (topic)   -> far    (the decorrelation control)
If MoBSE clears the gate on THIS eval, it tracks moral structure, not topic — which is the
only version useful to the system. Label source: Social-Chem human annotation (independent of z).

Honest note: the fingerprint is coarser than SciBSE/CodeBSE's identity labels and moral labels
carry annotator disagreement (that's why `rot-agree` exists), so the AUROC bar may be genuinely
harder to hit here — a sub-bar-but-decorrelated result is itself a finding about moral structure.
"""
from __future__ import annotations
import csv
from typing import Iterator

import numpy as np

from ..pairs import PairSource, Triplet, _norm
from ..admission import AdmissionCriteria

MOBSE_CONFIG = {
    "base_model": "BAAI/bge-m3",
    "social_chem_tsv":
        "/archive/ethics-corpora/social-chem-101/social-chem-101/social-chem-101.v1.0.tsv",
    "max_rows": None,   # full corpus (~292k) — the data lever
}


def _augment(text: str, seed: int) -> str:
    prefixes = ["", "It is the case that ", "Generally, ", "As a rule, ", "In most cases, "]
    p = prefixes[seed % len(prefixes)]
    return (p + (text[0].lower() + text[1:] if p and text else text))


def _jsign(v: str) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "0"
    return "+" if f > 0 else ("-" if f < 0 else "0")


class MoBSEPairSource(PairSource):
    name = "mobse"
    admission = AdmissionCriteria(
        invariant_structure="moral fingerprint = Moral Foundation x judgment sign",
        surface_class="wording, framing, topic (situation)",
        independent_label_source="Social-Chem-101 human Moral-Foundations + moral-judgment labels",
    )

    def __init__(self, tsv: str | None = None, max_rows: int | None = None, clean: bool = True,
                 foundation: str | None = None):
        self.tsv = tsv or MOBSE_CONFIG["social_chem_tsv"]
        self.max_rows = max_rows
        self.clean = clean   # clean-label: single-foundation + agreement>=3 only (drops the 23% ambiguous)
        self.foundation = foundation   # if set: a per-foundation SUB-BSE (test 'MoBSE is too broad')
        self._rows_cache = None
        if foundation:
            self.name = "mobse_" + foundation.split("-")[0]

    def _rows(self):
        # (rot_text, fingerprint, topic_id); dedup by rot text (v1 leak fix stays)
        if self._rows_cache is None:
            seen = {}
            with open(self.tsv, newline="", encoding="utf-8", errors="replace") as f:
                for i, row in enumerate(csv.DictReader(f, delimiter="\t")):
                    if self.max_rows and i >= self.max_rows:
                        break
                    rot = (row.get("rot") or "").strip()
                    mf = (row.get("rot-moral-foundations") or "").strip()
                    found = mf.split("|")[0].strip()
                    topic = (row.get("situation-short-id") or "").strip()
                    try:
                        agree = int(row.get("rot-agree") or 0)
                    except (TypeError, ValueError):
                        agree = 0
                    if self.clean and ("|" in mf or agree < 3):
                        continue   # drop ambiguous (multi-foundation) / low-agreement labels
                    if self.foundation and found != self.foundation:
                        continue   # per-foundation sub-BSE: keep only this foundation's situations
                    key = _norm(rot)   # dedup by the SAME normalized key the circularity guard uses
                    if rot and found and topic and key not in seen:
                        # v2 fingerprint (best): Foundation x judgment-sign. v3's legality added noise.
                        seen[key] = (rot, (found, _jsign(row.get("action-moral-judgment"))), topic)
            self._rows_cache = [(rot, fp, t) for (rot, fp, t) in seen.values()]
        return self._rows_cache

    def _split(self):
        train, held = [], []
        for r in self._rows():
            (held if (hash(("mobse", _norm(r[0]))) % 1000) / 1000.0 < 0.1 else train).append(r)
        return train, held

    @staticmethod
    def _index(rows):
        by_fp, by_topic = {}, {}
        for idx, (_rot, fp, topic) in enumerate(rows):
            by_fp.setdefault(fp, []).append(idx)
            by_topic.setdefault(topic, []).append(idx)
        return by_fp, by_topic

    @staticmethod
    def _sample(pool, rows, field, exclude, rng, tries=12):
        # rejection-sample an index from pool whose rows[j][field] != exclude (O(tries), not O(pool))
        for _ in range(tries):
            j = pool[int(rng.integers(len(pool)))]
            if rows[j][field] != exclude:
                return j
        return None

    def train_triplets(self) -> Iterator[Triplet]:
        train, _ = self._split()
        by_fp, by_topic = self._index(train)
        rng = np.random.default_rng(0)
        for i, (rot, fp, topic) in enumerate(train):
            jp = self._sample(by_fp[fp], train, 2, topic, rng)          # same fp, DIFFERENT topic
            if jp is None:
                continue
            jn = self._sample(by_topic[topic], train, 1, fp, rng)       # same topic, DIFFERENT fp
            neg = train[jn][0] if jn is not None else train[int(rng.integers(len(train)))][0]
            yield Triplet(anchor=rot, positive=train[jp][0], negative=neg)

    def heldout_eval(self) -> dict:
        _, held = self._split()
        by_fp, by_topic = self._index(held)
        rng = np.random.default_rng(1)
        structural_pairs, surface_pairs = [], []
        for i, (rot, fp, topic) in enumerate(held):
            jp = self._sample(by_fp[fp], held, 2, topic, rng)
            jn = self._sample(by_topic[topic], held, 1, fp, rng)
            if jp is None or jn is None:
                continue
            structural_pairs.append((rot, held[jp][0], True))    # same fingerprint, different topic -> near
            structural_pairs.append((rot, held[jn][0], False))   # same topic, different fingerprint -> far
            surface_pairs.append((rot, _augment(rot, i)))
            if len(surface_pairs) >= 600:
                break
        return {"structural_pairs": structural_pairs, "surface_pairs": surface_pairs, "ood_texts": []}
