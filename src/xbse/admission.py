"""Executable admission filter — the design's §3.1 as code, not a table.

A domain admits a legitimate *-BSE ONLY IF all three are concretely answerable:
  1. invariant_structure     — a definable structure that should be preserved
  2. surface_class           — the transformations it should be invariant to
  3. independent_label_source— same-structure/different-surface pairs labelable by a source
                               INDEPENDENT of the embedding itself

If "structure" is what you're trying to DISCOVER (not supervise), or the only label is the thing
you want to predict (a confound), the domain does NOT admit a *-BSE — building one embeds a
hypothesis as if it were data. A discipline that's only a doc gets violated; this throws.
"""
from __future__ import annotations
from dataclasses import dataclass


class AdmissionError(ValueError):
    """Raised when a PairSource is missing an admission criterion — it may not be registered."""


@dataclass(frozen=True)
class AdmissionCriteria:
    invariant_structure: str        # what must be preserved
    surface_class: str              # what to be invariant to
    independent_label_source: str   # where same/different-structure labels come from, independent of z

    def validate(self, name: str = "instance") -> "AdmissionCriteria":
        for field in ("invariant_structure", "surface_class", "independent_label_source"):
            if not (getattr(self, field) or "").strip():
                raise AdmissionError(
                    f"[{name}] fails admission filter: '{field}' is empty. A *-BSE needs a definable "
                    f"invariant structure, a surface class, AND an independent label source. If the "
                    f"structure is what you're trying to discover, or the label IS the target, the "
                    f"domain does not admit a *-BSE (circularity).")
        return self
