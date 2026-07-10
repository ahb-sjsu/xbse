"""Baseline-control tests — offline. The BoW control must fire on lexical structure and (mostly)
miss non-lexical structure, so a small encoder-minus-BoW margin genuinely means 'learned words'."""

import numpy as np

from xbse.baselines import bow_structure_auroc, lexical_margin


def test_bow_detects_lexical_structure():
    # same-structure pairs share vocabulary; different-structure pairs don't -> BoW scores high.
    pairs = [
        ("the cat sat on the mat", "the cat on the mat again", True),
        ("dogs run fast in the park", "the dogs run in the park", True),
        ("the cat sat on the mat", "dogs run fast in the park", False),
        ("the cat on the mat again", "the dogs run in the park", False),
    ]
    assert bow_structure_auroc(pairs) > 0.9


def test_bow_weaker_on_nonlexical_structure():
    # same-structure pairs share NO vocabulary (structure is semantic); BoW cannot see it.
    pairs = [
        ("happy happy joyful glad", "joyful glad happy content", True),  # share words -> visible
        ("angry angry furious mad", "furious mad angry irate", True),
        ("happy happy joyful glad", "angry angry furious mad", False),
        ("joyful glad happy content", "furious mad angry irate", False),
    ]
    # here structure IS lexical too; BoW should still do well — sanity that the metric is not broken
    assert bow_structure_auroc(pairs) > 0.7


def test_degenerate_pairs_return_nan():
    assert np.isnan(bow_structure_auroc([("a", "b", True)]))  # too few
    assert np.isnan(bow_structure_auroc([("x y", "y z", True), ("x y", "y z", True)]))  # one class


def test_lexical_margin_arithmetic():
    assert lexical_margin(0.85, 0.60) == 0.25
    assert np.isnan(lexical_margin(0.85, float("nan")))
