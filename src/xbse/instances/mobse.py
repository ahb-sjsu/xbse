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

from ..pairs import PairSource, Triplet
from ..admission import AdmissionCriteria

MOBSE_CONFIG = {
    "base_model": "BAAI/bge-m3",
    "social_chem_tsv":
        "/archive/ethics-corpora/social-chem-101/social-chem-101/social-chem-101.v1.0.tsv",
    "max_rows": 80000,
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

    def __init__(self, tsv: str | None = None, max_rows: int | None = 80000):
        self.tsv = tsv or MOBSE_CONFIG["social_chem_tsv"]
        self.max_rows = max_rows
        self._rows_cache = None

    def _rows(self):
        # (rot_text, fingerprint, topic_id); dedup by rot text (v1 leak fix stays)
        if self._rows_cache is None:
            seen = {}
            with open(self.tsv, newline="", encoding="utf-8", errors="replace") as f:
                for i, row in enumerate(csv.DictReader(f, delimiter="\t")):
                    if self.max_rows and i >= self.max_rows:
                        break
                    rot = (row.get("rot") or "").strip()
                    found = (row.get("rot-moral-foundations") or "").split("|")[0].strip()
                    topic = (row.get("situation-short-id") or "").strip()
                    legal = (row.get("action-legal") or "").strip() or "na"
                    if rot and found and topic and rot not in seen:
                        # richer fingerprint: Foundation x judgment-sign x legality -> more specific
                        # "same structure", so same-fingerprint items are genuinely more alike (v3)
                        seen[rot] = ((found, _jsign(row.get("action-moral-judgment")), legal), topic)
            self._rows_cache = [(r, fp, t) for r, (fp, t) in seen.items()]
        return self._rows_cache

    def _split(self):
        train, held = [], []
        for r in self._rows():
            (held if (hash(("mobse", r[0])) % 1000) / 1000.0 < 0.1 else train).append(r)
        return train, held

    @staticmethod
    def _index(rows):
        by_fp, by_topic = {}, {}
        for idx, (_rot, fp, topic) in enumerate(rows):
            by_fp.setdefault(fp, []).append(idx)
            by_topic.setdefault(topic, []).append(idx)
        return by_fp, by_topic

    def train_triplets(self) -> Iterator[Triplet]:
        train, _ = self._split()
        by_fp, by_topic = self._index(train)
        rng = np.random.default_rng(0)
        for i, (rot, fp, topic) in enumerate(train):
            same_fp_diff_topic = [j for j in by_fp[fp] if train[j][2] != topic]
            same_topic_diff_fp = [j for j in by_topic[topic] if train[j][1] != fp]
            if not same_fp_diff_topic:
                continue
            pos = train[same_fp_diff_topic[int(rng.integers(len(same_fp_diff_topic)))]][0]
            if same_topic_diff_fp:                                   # the decorrelation-hard negative
                neg = train[same_topic_diff_fp[int(rng.integers(len(same_topic_diff_fp)))]][0]
            else:
                k = int(rng.integers(len(train)))
                neg = train[k][0] if train[k][1] != fp else rot
            yield Triplet(anchor=rot, positive=pos, negative=neg)

    def heldout_eval(self) -> dict:
        _, held = self._split()
        by_fp, by_topic = self._index(held)
        rng = np.random.default_rng(1)
        structural_pairs, surface_pairs = [], []
        for i, (rot, fp, topic) in enumerate(held):
            same_fp_diff_topic = [j for j in by_fp[fp] if held[j][2] != topic]
            same_topic_diff_fp = [j for j in by_topic[topic] if held[j][1] != fp]
            if not same_fp_diff_topic or not same_topic_diff_fp:
                continue
            pos = held[same_fp_diff_topic[int(rng.integers(len(same_fp_diff_topic)))]][0]
            neg = held[same_topic_diff_fp[int(rng.integers(len(same_topic_diff_fp)))]][0]
            structural_pairs.append((rot, pos, True))    # same fingerprint, different topic -> near
            structural_pairs.append((rot, neg, False))   # same topic, different fingerprint -> far
            surface_pairs.append((rot, _augment(rot, i)))
            if len(surface_pairs) >= 600:
                break
        return {"structural_pairs": structural_pairs, "surface_pairs": surface_pairs, "ood_texts": []}
