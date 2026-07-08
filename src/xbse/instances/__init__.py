"""Per-domain *-BSE instances. Each is a PairSource + config; nothing else.

- mobse: morality BSE (Social-Chem-101 / Scruples / Moral-Machine). First instance.
- (lebse: legal BSE — refactor the existing ahb-sjsu/lebse in once MoBSE proves the core.)
"""
from .mobse import MoBSEPairSource, MOBSE_CONFIG

__all__ = ["MoBSEPairSource", "MOBSE_CONFIG"]
