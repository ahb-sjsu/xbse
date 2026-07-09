"""GaBSE — gastronomy *-BSE (dish-identity proof-of-concept).

Same clean identity label as SciBSE (paper) / CodeBSE (problem): a dish's TITLE and its full
recipe describe the same dish in very different surface forms.

invariance axis  -> positives = (dish title, full recipe body) of the SAME recipe
sensitivity axis -> negatives = a DIFFERENT recipe's body
independent label -> recipe identity (authors wrote title + body as one dish)

This is the *dish-identity* version — buildable now from any recipe corpus. It is deliberately the
'boring' version (high frozen proxy: a general encoder already groups same-dish text). The
*interesting* version is flavor-structure (same flavor architecture across DIFFERENT dishes), which
needs the geometric-gastronomy flavor-pairing labels as the independent structure source.

Data: pulled from HF in-pod (NRP/Atlas have internet), like CodeBSE's MBPP.
"""
from __future__ import annotations
from typing import Iterator

from ..pairs import PairSource, Triplet, stable_frac
from ..admission import AdmissionCriteria

GABSE_CONFIG = {"base_model": "BAAI/bge-m3", "holdout_frac": 0.1, "max_recipes": 40000}
_DATASETS = ["m3hrdadfi/recipe_nlg_lite", "corbt/all-recipes", "Hieu-Pham/recipe_1M"]


def _first(d, keys):
    for k in keys:
        v = d.get(k)
        if v:
            return str(v).strip()
    return ""


def _join(d, keys):
    parts = []
    for k in keys:
        v = d.get(k)
        if v:
            parts.append(str(v) if not isinstance(v, list) else " ".join(map(str, v)))
    return " ".join(parts).strip()


class GaBSEPairSource(PairSource):
    name = "gabse"
    admission = AdmissionCriteria(
        invariant_structure="the dish (recipe identity)",
        surface_class="title vs full recipe; phrasing, units, ingredient order",
        independent_label_source="recipe identity (title <-> body, authored as one dish); "
                                 "flavor-pairing graph for the flavor-structure version",
    )

    def __init__(self, hf_dataset: str | None = None, max_recipes: int = 40000, holdout_frac: float = 0.1):
        self.hf_dataset = hf_dataset
        self.max_recipes = max_recipes
        self.holdout_frac = holdout_frac
        self._rows_cache = None

    def _rows(self):
        if self._rows_cache is None:
            from datasets import load_dataset
            ds = None
            for name in ([self.hf_dataset] if self.hf_dataset else []) + _DATASETS:
                try:
                    ds = load_dataset(name, split="train"); break
                except Exception:
                    continue
            if ds is None:
                raise RuntimeError("no recipe dataset loadable from HF")
            rows = []
            for i, d in enumerate(ds):
                if i >= self.max_recipes:
                    break
                title = _first(d, ["title", "name", "Title"])
                body = _join(d, ["ingredients", "ner", "directions", "instructions", "steps"]) or \
                    _first(d, ["input", "text"])
                if title and len(body) > 50:
                    rows.append((str(i), title, body[:1500]))
            self._rows_cache = rows
        return self._rows_cache

    def _split(self):
        train, held = [], []
        for r in self._rows():
            (held if stable_frac(self.name + "|" + str(r[0])) < self.holdout_frac else train).append(r)
        return train, held

    def train_triplets(self) -> Iterator[Triplet]:
        train, _ = self._split()
        n = len(train)
        for i, (_rid, title, body) in enumerate(train):
            neg = train[(i + n // 2) % n][2]                 # a different recipe's body
            yield Triplet(anchor=title, positive=body, negative=neg)

    def heldout_eval(self) -> dict:
        _, held = self._split()
        m = min(len(held), 800)
        structural_pairs, surface_pairs = [], []
        for i in range(m - 1):
            _rid, title, body = held[i]
            structural_pairs.append((title, body, True))              # same dish title<->body -> near
            structural_pairs.append((title, held[i + 1][2], False))   # different dish body -> far
            surface_pairs.append((title, body))                       # same dish, different surface
        return {"structural_pairs": structural_pairs, "surface_pairs": surface_pairs, "ood_texts": []}
