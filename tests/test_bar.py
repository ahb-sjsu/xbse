"""Bar tests — offline. Provenance is mandatory; malformed bars throw; legacy fallback is loud."""

import dataclasses

import pytest

from xbse.bar import LEBSE_LEGACY_BAR, Bar, BarError
from xbse.pairs import PairSource, Triplet


def _bar(**kw):
    base = dict(
        auroc_min=0.78,
        fuzz_min=1.0,
        source="noise-ceiling(rot-agree)*0.9",
        derivation="perfect-scorer AUROC vs labels flipped w.p. eps from rot-agree; *0.9",
        registered="2026-07-10",
    )
    base.update(kw)
    return Bar(**base)


def test_bar_valid_construction():
    b = _bar()
    assert b.as_thresholds() == {"auroc>": 0.78, "fuzz>": 1.0}


@pytest.mark.parametrize("field", ["source", "derivation"])
def test_bar_requires_provenance(field):
    with pytest.raises(BarError):
        _bar(**{field: "  "})


@pytest.mark.parametrize("kw", [{"auroc_min": 0.3}, {"auroc_min": 1.2}, {"fuzz_min": 0.0}])
def test_bar_rejects_out_of_range_thresholds(kw):
    with pytest.raises(BarError):
        _bar(**kw)


def test_bar_is_frozen():
    b = _bar()
    with pytest.raises(dataclasses.FrozenInstanceError):
        b.auroc_min = 0.5  # loosening in place must not be possible


class _Src(PairSource):
    name = "barsrc"

    def train_triplets(self):
        yield Triplet("t", "t v", "n")

    def heldout_eval(self):
        return {"structural_pairs": [], "surface_pairs": [], "ood_texts": []}


def test_resolve_bar_prefers_declared_bar():
    s = _Src()
    s.bar = _bar()
    assert s.resolve_bar() is s.bar


def test_resolve_bar_falls_back_to_legacy_with_warning(capsys):
    s = _Src()
    assert s.resolve_bar() is LEBSE_LEGACY_BAR
    out = capsys.readouterr().out
    assert "WARNING" in out and "legacy" in out.lower()
