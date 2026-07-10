"""Gate-math tests — offline, torch-free encoder. The gate is the framework's honesty organ;
its arithmetic (AUROC, fuzz ratio, pass/fail against a Bar) must itself be tested, not trusted."""

import numpy as np
import pytest

from xbse.bar import Bar
from xbse.validate import (
    LEBSE_BAR,
    fuzz_ratio,
    gate,
    structure_vs_surface_auroc,
    surface_invariance,
)


class VectorEncoder:
    """Maps each text to a fixed unit vector from a lookup — fully deterministic sims."""

    def __init__(self, table):
        self.table = {k: np.asarray(v, dtype="float32") for k, v in table.items()}

    def encode(self, texts):
        z = np.stack([self.table[t] for t in texts])
        return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)


def _perfect_encoder():
    # two orthogonal structure clusters; paraphrases share the anchor's vector exactly
    e1, e2 = [1.0, 0.0], [0.0, 1.0]
    return VectorEncoder({"a1": e1, "a2": e1, "a1 para": e1, "b1": e2, "b2": e2, "b1 para": e2})


STRUCT = [("a1", "a2", True), ("b1", "b2", True), ("a1", "b1", False), ("a2", "b2", False)]
SURFACE = [("a1", "a1 para"), ("b1", "b1 para")]


def test_perfect_encoder_metrics():
    enc = _perfect_encoder()
    assert structure_vs_surface_auroc(enc, STRUCT) == pytest.approx(1.0)
    assert surface_invariance(enc, SURFACE) == pytest.approx(1.0)
    assert fuzz_ratio(enc, STRUCT, SURFACE) > 1.0  # structure moves, surface doesn't


def test_auroc_known_intermediate_value():
    # sims: (x,p1)=0.9(+), (x,n1)=0.8(-), (x,p2)=0.7(+), (x,n2)=0.6(-) -> AUROC = 0.75
    def v(c):  # unit vector at angle acos(c) from x
        return [c, float(np.sqrt(1 - c * c))]

    enc = VectorEncoder({"x": [1.0, 0.0], "p1": v(0.9), "n1": v(0.8), "p2": v(0.7), "n2": v(0.6)})
    pairs = [("x", "p1", True), ("x", "n1", False), ("x", "p2", True), ("x", "n2", False)]
    assert structure_vs_surface_auroc(enc, pairs) == pytest.approx(0.75)


def test_blind_encoder_fails_any_bar():
    # every text -> the same vector: AUROC 0.5, no structural separation
    enc = VectorEncoder(
        {t: [1.0, 0.0] for pair in STRUCT for t in pair[:2]}
        | {a: [1.0, 0.0] for a, _ in SURFACE}
        | {b: [1.0, 0.0] for _, b in SURFACE}
    )
    out = gate(enc, {"structural_pairs": STRUCT, "surface_pairs": SURFACE, "ood_texts": []})
    assert out["structure_auroc"] == pytest.approx(0.5)
    assert out["passed"] is False


def test_gate_passes_and_fails_against_instance_bar():
    enc = _perfect_encoder()
    ev = {"structural_pairs": STRUCT, "surface_pairs": SURFACE, "ood_texts": []}
    lenient = Bar(0.75, 1.0, source="test", derivation="synthetic perfect-cluster fixture")
    out = gate(enc, ev, bar=lenient)
    assert out["passed"] is True
    assert out["bar_source"] == "test"  # provenance travels with the verdict
    # an unreachable bar on the same encoder must fail — pass/fail is bar-relative
    impossible = Bar(0.999999, 1e9, source="test", derivation="unreachable by construction")
    assert gate(enc, ev, bar=impossible)["passed"] is False


def test_gate_accepts_legacy_dict_bar():
    enc = _perfect_encoder()
    ev = {"structural_pairs": STRUCT, "surface_pairs": SURFACE, "ood_texts": []}
    out = gate(enc, ev, bar={"auroc_min": 0.9, "fuzz_min": 1.0})
    assert out["passed"] is True


def test_default_bar_is_legacy_lebse():
    enc = _perfect_encoder()
    ev = {"structural_pairs": STRUCT, "surface_pairs": SURFACE, "ood_texts": []}
    out = gate(enc, ev)  # no bar passed
    assert out["thresholds"] == LEBSE_BAR.as_thresholds()
    assert "LeBSE" in out["bar_source"]
