"""Calibration module (XBSE_REVIEW_1 F1/R2): fields, monotonicity, ECE behavior."""

from __future__ import annotations

import numpy as np
import pytest

from xbse.calibration import (
    calibration_fields,
    ece,
    isotonic,
    platt,
    reliability_curve,
    reliability_weight,
)


def _separable(n=400, seed=0):
    rng = np.random.default_rng(seed)
    scores = np.concatenate([rng.normal(1.0, 1.0, n // 2), rng.normal(-1.0, 1.0, n // 2)])
    labels = np.concatenate([np.ones(n // 2, int), np.zeros(n // 2, int)])
    return scores, labels


def test_platt_and_isotonic_are_monotone_probability_maps():
    scores, labels = _separable()
    for fit in (platt, isotonic):
        f = fit(scores, labels)
        grid = np.linspace(scores.min(), scores.max(), 50)
        p = f(grid)
        assert np.all(p >= 0) and np.all(p <= 1)
        assert np.all(np.diff(p) >= -1e-9)  # monotone non-decreasing
        # higher scores -> higher P(positive), decisively at the extremes
        assert f([scores.max()])[0] > f([scores.min()])[0]


def test_ece_perfect_and_broken_calibration():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)  # labels drawn AT the stated prob
    assert ece(p, y) < 0.02  # near-perfectly calibrated by construction
    assert ece(1 - p, y) > 0.3  # anti-calibrated is far from zero


def test_reliability_weight_floor_convention():
    assert reliability_weight(0.5) == 0.0
    assert reliability_weight(1.0) == 1.0
    assert reliability_weight(0.4) == 0.0  # never negative authority
    assert abs(reliability_weight(0.75) - 0.5) < 1e-12


def test_calibration_fields_ship_the_report_block():
    scores, labels = _separable()
    block = calibration_fields(scores, labels, auroc=0.85)
    for key in (
        "calibration_method",
        "calibration_ece",
        "raw_ece",
        "reliability_curve",
        "reliability_weight",
        "n_calibration_pairs",
    ):
        assert key in block
    assert block["n_calibration_pairs"] == len(labels)
    # calibrated ECE should not be worse than the raw min-max squash
    assert block["calibration_ece"] <= block["raw_ece"] + 1e-9
    assert 0 < block["reliability_weight"] <= 1
    rows = reliability_curve(np.clip(scores, 0, 1), labels)
    assert sum(r["count"] for r in rows) == len(labels)


def test_degenerate_constant_scores_do_not_crash():
    scores = np.zeros(50)
    labels = np.array([0, 1] * 25)
    block = calibration_fields(scores, labels, auroc=0.5, method="isotonic")
    assert block["reliability_weight"] == 0.0
    assert np.isfinite(block["calibration_ece"])


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_both_methods_selectable(method):
    scores, labels = _separable()
    block = calibration_fields(scores, labels, auroc=0.8, method=method)
    assert block["calibration_method"] == method


def test_report_roundtrips_calibration_block():
    """Production reports now carry the wired calibration block; Report(**json) must accept it
    (both erisml-compiler and moral-spectrum-analyzer load reports exactly that way), and
    pre-calibration reports without the key must still load."""
    import json

    from xbse.report import Report

    block = {"calibration_method": "isotonic", "calibration_ece": 0.05, "reliability_weight": 0.7}
    r = Report(
        instance="care_joint",
        checkpoint_hash="deadbeef",
        thresholds={},
        metrics={},
        passed=True,
        calibration=block,
    )
    r2 = Report(**json.loads(r.to_json()))
    assert r2.calibration == block
    legacy = json.loads(r.to_json())
    del legacy["calibration"]
    assert Report(**legacy).calibration == {}
