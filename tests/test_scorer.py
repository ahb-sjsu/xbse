"""Scorer readout tests — offline, no model download (a tiny fake encoder stands in for BGE-M3)."""

import numpy as np
import pytest

from xbse.report import NotValidatedError, Report
from xbse.scorer import DimensionScorer, Valence


class FakeEncoder:
    """Maps text→embedding by a keyword rule: '+' words push one way, '-' words the other.

    Lets us test the axis/calibration/score math without loading a transformer."""

    def encode(self, texts):
        out = []
        for t in texts:
            pos = t.count("good") - t.count("bad")
            out.append([float(pos), 1.0, 0.0])
        return np.asarray(out, dtype="float32")


def _scorer():
    enc = FakeEncoder()
    pos = ["good thing", "very good", "good good"]
    neg = ["bad thing", "very bad", "bad bad"]
    return DimensionScorer.fit(enc, pos, neg, name="test")


def test_fit_produces_signed_axis():
    s = _scorer()
    assert s.axis.shape == (3,)
    assert abs(np.linalg.norm(s.axis) - 1.0) < 1e-5


def test_score_direction_and_range():
    s = _scorer()
    good = s.score("good good good")
    bad = s.score("bad bad bad")
    neutral = s.score("thing")
    assert good.value > 0 and good.direction == "positive"
    assert bad.value < 0 and bad.direction == "negative"
    assert -1.0 <= good.value <= 1.0 and -1.0 <= bad.value <= 1.0
    assert neutral.direction == "neutral"
    assert isinstance(good, Valence)


def test_confidence_grows_with_margin():
    s = _scorer()
    strong = s.score("good good good good")
    weak = s.score("good")
    assert strong.confidence >= weak.confidence


def test_require_pass_gate_blocks_unvalidated():
    enc = FakeEncoder()

    class Src:
        name = "x"

        def _rows(self):
            return [(0, "good", "+"), (0, "bad", "-")]

    fail = Report(instance="x", checkpoint_hash="abc", thresholds={}, metrics={}, passed=False)
    with pytest.raises(NotValidatedError):
        DimensionScorer.from_pairsource(enc, Src(), fail, "abc")

    ok = Report(instance="x", checkpoint_hash="abc", thresholds={}, metrics={}, passed=True)
    scorer = DimensionScorer.from_pairsource(enc, Src(), ok, "abc")
    assert scorer.name == "x"
