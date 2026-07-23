"""Specificity gate (XBSE_REVIEW_1 F2/R1): matrix, gate law, demotion."""

from __future__ import annotations

import numpy as np
import pytest

from xbse.specificity import (
    REGISTERED_SPECIFICITY_MARGIN,
    SpecificityBar,
    _auroc,
    discrimination_matrix,
)


def test_auroc_basic_and_ties():
    assert _auroc([2, 3, 4], [0, 1]) == 1.0
    assert _auroc([0, 1], [2, 3, 4]) == 0.0
    assert abs(_auroc([1, 2], [1, 2]) - 0.5) < 1e-9  # symmetric ties -> 0.5


def _toy_world():
    """Two dimensions; feeder A specific to a, feeder B is a general-valence clone."""
    rng = np.random.default_rng(0)
    # texts are just floats standing in for encoded items
    pairs = {
        "a": (list(rng.normal(2, 1, 60)), list(rng.normal(-2, 1, 60))),
        "b": (list(rng.normal(0.6, 1, 60)), list(rng.normal(-0.6, 1, 60))),
    }
    scorers = {
        "a": lambda xs: np.asarray(xs, float),  # perfect on a, decent on b (shared axis)
        "b": lambda xs: np.asarray(xs, float) * 0.999,  # near-clone of a: NOT specific
    }
    return scorers, pairs


def test_matrix_shape_and_gate_demotes_the_clone():
    scorers, pairs = _toy_world()
    M = discrimination_matrix(scorers, pairs)
    assert set(M) == {"a", "b"} and set(M["a"]) == {"a", "b"}
    bar = SpecificityBar()
    out = bar.check_all(M)
    # the clone cannot beat its rival by the margin on its own pairs
    assert out["b"]["disposition"] == "DEMOTE-to-G"
    assert out["b"]["specific"] is False
    # verdict block carries everything the scorecard needs
    for k in ("own_auroc", "best_rival", "specificity_gap", "margin"):
        assert k in out["b"]


def test_truly_specific_feeder_passes():
    rng = np.random.default_rng(1)
    pairs = {
        "a": (list(rng.normal(2, 1, 80)), list(rng.normal(-2, 1, 80))),
        "b": (list(rng.normal(2, 1, 80)), list(rng.normal(-2, 1, 80))),
    }
    # feeder a scores dimension a's items by value; feeder b sees only noise there
    scorers = {
        "a": lambda xs: np.asarray(xs, float),
        "b": lambda xs: np.zeros(len(xs)) + rng.normal(0, 0.01, len(xs)),
    }
    M = discrimination_matrix(scorers, pairs)
    v = SpecificityBar().check(M, "a")
    assert v["specific"] is True and v["disposition"] == "own-axis"


def test_bar_may_tighten_never_loosen():
    SpecificityBar(margin=0.10)  # tightening is lawful
    with pytest.raises(ValueError):
        SpecificityBar(margin=0.01)  # loosening is not


def test_registered_margin_value_is_frozen():
    # the registered value is part of the public contract; changing it must
    # break a test (and therefore be a conscious, documented act)
    assert REGISTERED_SPECIFICITY_MARGIN == 0.05
