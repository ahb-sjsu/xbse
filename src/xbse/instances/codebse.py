"""CodeBSE — programs *-BSE. Behaviour is externally checkable, so the label is genuinely hard.

invariance axis  -> positives = the problem description (docstring/spec) and its implementation:
                    same behaviour, very different surface (prose vs code)
sensitivity axis -> negatives = a DIFFERENT problem's implementation (different behaviour)
independent label -> problem identity (spec↔solution), with test-equivalence as the v2 upgrade

Data: MBPP (public, ~974 problems: {text: spec, code: solution}). Pod loads it from HF; locally
point at a jsonl with the same fields. Behaviour-equivalence (running the tests) is the stronger
oracle to swap in for v2.
"""
from __future__ import annotations
import json
import os
from typing import Iterator

import ast
import random

from ..pairs import PairSource, Triplet
from ..admission import AdmissionCriteria

CODEBSE_CONFIG = {"base_model": "BAAI/bge-m3", "holdout_frac": 0.1, "hf_dataset": "mbpp"}


def _augment_code(code: str, seed: int) -> str:
    """Group-augmentation for code: consistent local-variable RENAMING (the behaviour-preserving
    'group', exactly analogous to ReaBSE's color permutation). Same I/O behaviour, different surface.
    Falls back to the original on any parse/unparse failure."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return code
    locals_ = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            locals_.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            locals_.add(node.id)
    locals_ -= {"self", "cls"}
    if not locals_:
        return code
    new = [f"v{i}" for i in range(len(locals_))]
    random.Random(seed).shuffle(new)
    mapping = dict(zip(sorted(locals_), new))

    class _R(ast.NodeTransformer):
        def visit_Name(self, n):
            if n.id in mapping:
                n.id = mapping[n.id]
            return n
        def visit_arg(self, n):
            if n.arg in mapping:
                n.arg = mapping[n.arg]
            return n

    try:
        return ast.unparse(ast.fix_missing_locations(_R().visit(tree)))
    except Exception:
        return code


class CodeBSEPairSource(PairSource):
    name = "codebse"
    admission = AdmissionCriteria(
        invariant_structure="program behaviour / the algorithm",
        surface_class="syntax, identifiers, prose-vs-code",
        independent_label_source="problem identity (spec↔solution); test-equivalence oracle for v2",
    )

    def __init__(self, jsonl: str | None = None, hf_dataset: str = "mbpp", holdout_frac: float = 0.1):
        self.jsonl = jsonl
        self.hf_dataset = hf_dataset
        self.holdout_frac = holdout_frac
        self._rows = None

    def _load(self):
        if self._rows is None:
            rows = []
            if self.jsonl and os.path.exists(self.jsonl):
                for line in open(self.jsonl, encoding="utf-8"):
                    d = json.loads(line)
                    rows.append((str(d.get("task_id", len(rows))), d["text"].strip(), d["code"].strip()))
            else:
                from datasets import load_dataset             # NRP pods have internet
                # namespaced id + config (newer huggingface_hub rejects bare "mbpp")
                ds = load_dataset("google-research-datasets/mbpp", "full", split="train")
                for i, d in enumerate(ds):
                    spec, code = (d.get("text") or "").strip(), (d.get("code") or "").strip()
                    if spec and code:
                        rows.append((str(d.get("task_id", i)), spec, code))
            self._rows = rows
        return self._rows

    def _split(self):
        train, held = [], []
        for r in self._load():
            b = (hash((self.name, r[0])) % 1000) / 1000.0
            (held if b < self.holdout_frac else train).append(r)
        return train, held

    AUG_PER_PROBLEM = 8   # variable-renaming group augmentation (fixes MBPP's ~974-problem starvation)

    def train_triplets(self) -> Iterator[Triplet]:
        train, _ = self._split()
        n = len(train)
        for i, (_tid, spec, code) in enumerate(train):
            for a in range(self.AUG_PER_PROBLEM):
                aug = _augment_code(code, seed=1000 * i + a)     # same behaviour, renamed vars
                neg = train[(i + 1 + a) % n][2]                  # a different problem's code
                yield Triplet(anchor=spec, positive=aug, negative=neg)   # spec <-> behaviour
                yield Triplet(anchor=code, positive=aug, negative=neg)   # behaviour-invariance (code<->code)

    def heldout_eval(self) -> dict:
        _, held = self._split()
        m = min(len(held), 400)
        structural_pairs, surface_pairs = [], []
        for i in range(m - 1):
            _tid, spec, code = held[i]
            structural_pairs.append((spec, code, True))               # same problem spec<->code -> near
            structural_pairs.append((spec, held[i + 1][2], False))    # different problem's code -> far
            surface_pairs.append((spec, code))                        # same behaviour, different surface
        return {"structural_pairs": structural_pairs, "surface_pairs": surface_pairs, "ood_texts": []}
