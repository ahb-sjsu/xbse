"""xbse — modular framework for domain-specific BERT Sentence Embeddings.

A `*-BSE` = shared BSEEncoder + contrastive objective + a per-domain PairSource, gated by one
shared validation harness. See README and docs/MOBSE_PLAN.md for the falsification-order
discipline: build an instance -> validate (hard stop) -> only then earn downstream tools.

Import policy: the *torch-free* core — the gate math, the pre-registered `Bar`, the adversarial
`baselines`, the circularity/admission guards, `require_pass`, and the numpy-only `scorer` — imports
without torch installed. The torch-dependent names (`BSEEncoder`, `info_nce`, `discover`, `search`)
are lazy-loaded on first access, so lightweight consumers (e.g. an audit host that only needs the
gate/scorer logic) don't pay for torch.
"""

__version__ = "0.1.0"

# --- torch-free core (eager) ---
from .admission import AdmissionCriteria, AdmissionError
from .bar import LEBSE_LEGACY_BAR, Bar, BarError
from .pairs import CircularityError, PairSource, Triplet
from .report import NotValidatedError, Report, hash_checkpoint, require_pass
from .validate import LEBSE_BAR, fuzz_ratio, gate, structure_vs_surface_auroc

# --- torch-dependent names (lazy; imported on first attribute access) ---
_LAZY = {
    "BSEEncoder": ("encoder", "BSEEncoder"),
    "info_nce": ("objective", "info_nce"),
    "discover": ("discover", "discover"),
    "frozen_proxy": ("discover", "frozen_proxy"),
    "print_table": ("discover", "print_table"),
    "search": ("search", "search"),
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module, attr = _LAZY[name]
        return getattr(importlib.import_module(f".{module}", __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)


__all__ = [
    # torch-free
    "Bar",
    "BarError",
    "LEBSE_LEGACY_BAR",
    "LEBSE_BAR",
    "PairSource",
    "Triplet",
    "CircularityError",
    "AdmissionCriteria",
    "AdmissionError",
    "Report",
    "require_pass",
    "NotValidatedError",
    "hash_checkpoint",
    "fuzz_ratio",
    "structure_vs_surface_auroc",
    "gate",
    # lazy (torch)
    "BSEEncoder",
    "info_nce",
    "discover",
    "frozen_proxy",
    "print_table",
    "search",
]
