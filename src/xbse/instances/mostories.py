"""MoralStoriesBSE — a moral-valence *-BSE with SURFACE-MATCHED hard negatives.

Moral Stories (Emelin et al. 2021) gives, per situation, a `moral_action` and an `immoral_action`
that share the SAME situation and intention. That yields the hardest possible negative: identical
context, opposite moral valence — forcing the encoder to track the ACTION's morality, not the
situation topic. This is the topic-decorrelation MoBSE had to engineer out of Social-Chem topics;
in Moral Stories it is native to the data.

  invariance axis  -> positives = two actions of the SAME valence, DIFFERENT story   -> near
  sensitivity axis -> negatives = the MATCHED opposite-valence action of the SAME story -> far
  independent label -> Moral Stories human-written moral/immoral action pairs

Note vs the failed MoBSE v1 (AUROC 0.571, "keyed on valence, 1 bit, dominated by topic"): valence
is still 1 bit here, but the matched same-situation negatives make topic USELESS for the decision,
which is exactly what v1 lacked. `norm` is deliberately NOT put in the item text — it states the
rule the action follows/violates, so including it would leak the label.

Data: /archive/ethics-corpora/moral_stories/moral_stories_full.jsonl (12k stories; fields
ID/norm/situation/intention/moral_action/immoral_action/consequences/label).
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import numpy as np

from ..admission import AdmissionCriteria
from ..pairs import PairSource, Triplet, stable_frac

MOSTORIES_CONFIG = {
    "base_model": "BAAI/bge-m3",
    "jsonl": "/archive/ethics-corpora/moral_stories/moral_stories_full.jsonl",
}


def _augment(text: str, seed: int) -> str:
    prefixes = ["", "In this case, ", "Consider: ", "As it happened, ", "Reportedly, "]
    p = prefixes[seed % len(prefixes)]
    return p + (text[0].lower() + text[1:] if p and text else text)


class MoralStoriesBSEPairSource(PairSource):
    name = "mostories"
    max_len = 192
    admission = AdmissionCriteria(
        invariant_structure="moral valence of an action in its situation (moral vs immoral)",
        surface_class="wording/framing of the situation and action; the situation topic",
        independent_label_source="Moral Stories (Emelin et al.) human-written moral/immoral action pairs",
    )

    def __init__(
        self, jsonl: str | None = None, max_rows: int | None = None, holdout_frac: float = 0.1
    ):
        self.jsonl = jsonl or MOSTORIES_CONFIG["jsonl"]
        self.max_rows = max_rows  # cap #stories (not #items) for quick runs
        self.holdout_frac = holdout_frac
        self._rows_cache = None

    def _rows(self):
        # each story -> two labeled items: (story_id, text, valence)  valence 1=moral, 0=immoral
        if self._rows_cache is None:
            rows = []
            with open(self.jsonl, encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f):
                    if self.max_rows and i >= self.max_rows:
                        break
                    try:
                        d = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    sid = str(d.get("ID") or i)
                    sit = (d.get("situation") or "").strip()
                    ma = (d.get("moral_action") or "").strip()
                    ia = (d.get("immoral_action") or "").strip()
                    if not sit or not ma or not ia:
                        continue
                    rows.append((sid, f"{sit} {ma}", 1))
                    rows.append((sid, f"{sit} {ia}", 0))
            self._rows_cache = rows
        return self._rows_cache

    def _split(self):
        # split by STORY id so a story's moral+immoral items land on the SAME side (no leakage)
        train, held = [], []
        for r in self._rows():
            (held if stable_frac("mostories|" + r[0]) < self.holdout_frac else train).append(r)
        return train, held

    @staticmethod
    def _index(rows):
        by_val = {0: [], 1: []}
        by_story: dict[str, dict[int, int]] = {}
        for idx, (sid, _txt, val) in enumerate(rows):
            by_val[val].append(idx)
            by_story.setdefault(sid, {})[val] = idx
        return by_val, by_story

    @staticmethod
    def _sample_other_story(pool, rows, sid, rng, tries=12):
        # rejection-sample an index from pool whose story != sid (O(tries), not O(pool))
        for _ in range(tries):
            j = pool[int(rng.integers(len(pool)))]
            if rows[j][0] != sid:
                return j
        return None

    def train_triplets(self) -> Iterator[Triplet]:
        train, _ = self._split()
        by_val, by_story = self._index(train)
        rng = np.random.default_rng(0)
        for sid, txt, val in train:
            jp = self._sample_other_story(by_val[val], train, sid, rng)  # same valence, diff story
            if jp is None:
                continue
            jn = by_story.get(sid, {}).get(1 - val)  # MATCHED opposite valence
            if jn is None:  # fallback: any opposite
                opp = by_val[1 - val]
                jn = opp[int(rng.integers(len(opp)))] if opp else None
            if jn is None:
                continue
            yield Triplet(anchor=txt, positive=train[jp][1], negative=train[jn][1])

    def heldout_eval(self) -> dict:
        _, held = self._split()
        by_val, by_story = self._index(held)
        rng = np.random.default_rng(1)
        structural_pairs, surface_pairs = [], []
        seen_neg = set()
        for idx, (sid, txt, val) in enumerate(held):
            jp = self._sample_other_story(by_val[val], held, sid, rng)  # same valence, diff story
            jn = by_story.get(sid, {}).get(1 - val)  # matched opposite valence
            if jp is None or jn is None:
                continue
            structural_pairs.append((txt, held[jp][1], True))  # same valence, diff story -> near
            key = tuple(sorted((idx, jn)))  # the matched pair is symmetric
            if key not in seen_neg:  # add each matched neg once
                structural_pairs.append(
                    (txt, held[jn][1], False)
                )  # matched opposite valence -> far
                seen_neg.add(key)
            surface_pairs.append((txt, _augment(txt, idx)))
            if len(surface_pairs) >= 600:
                break
        return {
            "structural_pairs": structural_pairs,
            "surface_pairs": surface_pairs,
            "ood_texts": [],
        }
