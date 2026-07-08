"""xbse — modular framework for domain-specific BERT Sentence Embeddings.

A `*-BSE` = shared BSEEncoder + contrastive objective + a per-domain PairSource, gated by one
shared validation harness. See README and docs/MOBSE_PLAN.md for the falsification-order
discipline: build an instance -> validate (hard stop) -> only then earn downstream tools.
"""
__version__ = "0.1.0"

from .encoder import BSEEncoder
from .pairs import PairSource, Triplet, CircularityError
from .admission import AdmissionCriteria, AdmissionError
from .objective import info_nce
from .validate import fuzz_ratio, structure_vs_surface_auroc, gate, LEBSE_BAR
from .report import Report, require_pass, NotValidatedError, hash_checkpoint
from .discover import discover, frozen_proxy, print_table

__all__ = [
    "BSEEncoder", "PairSource", "Triplet", "CircularityError",
    "AdmissionCriteria", "AdmissionError", "info_nce",
    "fuzz_ratio", "structure_vs_surface_auroc", "gate", "LEBSE_BAR",
    "Report", "require_pass", "NotValidatedError", "hash_checkpoint",
]
