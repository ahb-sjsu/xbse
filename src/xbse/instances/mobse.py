"""MoBSE — the morality *-BSE instance. First instance; proves the core.

invariance axis  -> positives = same rule-of-thumb, surface-perturbed  (wording must not move z)
sensitivity axis -> negatives = OPPOSITE moral judgment                (valence must move z)

Primary source: Social-Chem-101 (292K rules-of-thumb with moral-judgment labels + a native
train/dev/test split). Scruples (dilemma verdicts) and Moral-Machine (parametric preferences)
plug in as additional PairSources the same way.

CIRCULARITY GUARD: the parametric legal thresholds and the Moral-Machine test split used for
downstream discontinuity testing are held out and never appear here. Train for invariance +
generic moral valence; test stratification on boundaries the encoder never saw.

Production TODO: replace the toy surface-augmenter with back-translation / LLM paraphrase, and
merge Scruples + Moral-Machine(train) triplets. This file is the scaffold that makes the gate
runnable, not the final data recipe.
"""
from __future__ import annotations
import csv
from typing import Iterator

from ..pairs import PairSource, Triplet
from ..admission import AdmissionCriteria

MOBSE_CONFIG = {
    "base_model": "BAAI/bge-m3",
    "proj_dim": None,                 # keep native 1024-d unless the gate motivates a bottleneck
    "temperature": 0.05,
    "adversary": None,                # optionally strip language/register later
    "social_chem_tsv":
        "/archive/ethics-corpora/social-chem-101/social-chem-101/social-chem-101.v1.0.tsv",
}

# rot-judgment strings -> coarse moral valence (the sensitivity label)
_POS = ("good", "ok", "okay", "expected", "fine", "nice", "polite", "kind")
_NEG = ("bad", "wrong", "rude", "shouldn't", "should not", "not ok", "mean", "cruel")


def _valence(judgment: str) -> int | None:
    j = (judgment or "").lower()
    if any(w in j for w in _NEG):
        return -1
    if any(w in j for w in _POS):
        return 1
    return None


def _augment(text: str, seed: int) -> str:
    """Toy surface perturbation (invariance positive). Deterministic, content-preserving.
    Replace with back-translation / LLM paraphrase for the real run."""
    prefixes = ["", "It is the case that ", "Generally, ", "As a rule, ", "In most cases, "]
    p = prefixes[seed % len(prefixes)]
    t = text[0].lower() + text[1:] if p and text else text
    return p + t


class MoBSEPairSource(PairSource):
    name = "mobse"
    admission = AdmissionCriteria(
        invariant_structure="moral judgment / valence of a rule-of-thumb",
        surface_class="paraphrase, wording, framing, language",
        independent_label_source="Social-Chem-101 human rot-judgment labels (annotated independently of z)",
    )

    def __init__(self, tsv: str | None = None, max_rows: int | None = 60000):
        self.tsv = tsv or MOBSE_CONFIG["social_chem_tsv"]
        self.max_rows = max_rows

    def _rows(self):
        with open(self.tsv, newline="", encoding="utf-8", errors="replace") as f:
            r = csv.DictReader(f, delimiter="\t")
            for i, row in enumerate(r):
                if self.max_rows and i >= self.max_rows:
                    break
                rot = (row.get("rot") or "").strip()
                val = _valence(row.get("rot-judgment", ""))
                split = (row.get("split") or "train").strip()
                if rot and val is not None:
                    yield rot, val, split

    def train_triplets(self) -> Iterator[Triplet]:
        pos, neg = [], []
        for rot, val, split in self._rows():
            if split != "train":
                continue
            (pos if val > 0 else neg).append(rot)
        n = min(len(pos), len(neg))
        for k in range(n):
            # anchor from one valence, negative from the opposite; positive = anchor paraphrase
            anchor = pos[k]
            yield Triplet(anchor=anchor, positive=_augment(anchor, k), negative=neg[k])
            yield Triplet(anchor=neg[k], positive=_augment(neg[k], k + 1), negative=pos[k])

    def heldout_eval(self) -> dict:
        pos, neg = [], []
        for rot, val, split in self._rows():
            if split not in ("dev", "test"):
                continue
            (pos if val > 0 else neg).append(rot)
        m = min(len(pos), len(neg), 500)
        structural_pairs = []
        for k in range(m - 1):
            structural_pairs.append((pos[k], pos[k + 1], True))     # same valence -> near
            structural_pairs.append((pos[k], neg[k], False))        # opposite valence -> far
        surface_pairs = [(pos[k], _augment(pos[k], k)) for k in range(m)]
        return {
            "structural_pairs": structural_pairs,
            "surface_pairs": surface_pairs,
            "ood_texts": [],   # fill with a non-moral corpus sample (e.g. arxiv) for the OOD control
        }
