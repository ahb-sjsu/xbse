"""xbse — modular framework for domain-specific BERT Sentence Embeddings.

A `*-BSE` = shared BSEEncoder + contrastive objective + a per-domain PairSource, gated by one
shared validation harness. See README and docs/MOBSE_PLAN.md for the falsification-order
discipline: build an instance -> validate (hard stop) -> only then earn downstream tools.
"""

__version__ = "0.1.0"

from .admission import AdmissionCriteria, AdmissionError
from .bar import LEBSE_LEGACY_BAR, Bar, BarError
from .discover import discover, frozen_proxy, print_table
from .encoder import BSEEncoder
from .objective import info_nce
from .pairs import CircularityError, PairSource, Triplet
from .report import NotValidatedError, Report, hash_checkpoint, require_pass
from .search import search
from .validate import LEBSE_BAR, fuzz_ratio, gate, structure_vs_surface_auroc

__all__ = [
    "BSEEncoder",
    "PairSource",
    "Triplet",
    "Bar",
    "BarError",
    "LEBSE_LEGACY_BAR",
    "CircularityError",
    "AdmissionCriteria",
    "AdmissionError",
    "info_nce",
    "fuzz_ratio",
    "structure_vs_surface_auroc",
    "gate",
    "LEBSE_BAR",
    "Report",
    "require_pass",
    "NotValidatedError",
    "hash_checkpoint",
    "discover",
    "frozen_proxy",
    "print_table",
    "search",
]
