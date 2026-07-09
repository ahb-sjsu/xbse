"""Per-domain *-BSE instances. Each is a PairSource + config; nothing else.

- mobse: morality BSE (Social-Chem-101 / Scruples / Moral-Machine). First instance.
- (lebse: legal BSE — refactor the existing ahb-sjsu/lebse in once MoBSE proves the core.)
"""
from .mobse import MoBSEPairSource, MOBSE_CONFIG
from .scibse import SciBSEPairSource, SCIBSE_CONFIG
from .reabse import ReaBSEPairSource, REABSE_CONFIG
from .codebse import CodeBSEPairSource, CODEBSE_CONFIG
from .gabse import GaBSEPairSource, GABSE_CONFIG
from .rights import RightsBSEPairSource, RIGHTS_CONFIG
from .mostories import MoralStoriesBSEPairSource, MOSTORIES_CONFIG
from .physharm import PhysHarmBSEPairSource, PHYSHARM_CONFIG
from .epistemic import EpistemicBSEPairSource, EPISTEMIC_CONFIG
from .socenv import SocEnvBSEPairSource, SOCENV_CONFIG

__all__ = ["MoBSEPairSource", "MOBSE_CONFIG", "SciBSEPairSource", "SCIBSE_CONFIG",
           "ReaBSEPairSource", "REABSE_CONFIG", "CodeBSEPairSource", "CODEBSE_CONFIG",
           "GaBSEPairSource", "GABSE_CONFIG", "RightsBSEPairSource", "RIGHTS_CONFIG",
           "MoralStoriesBSEPairSource", "MOSTORIES_CONFIG",
           "PhysHarmBSEPairSource", "PHYSHARM_CONFIG",
           "EpistemicBSEPairSource", "EPISTEMIC_CONFIG",
           "SocEnvBSEPairSource", "SOCENV_CONFIG"]
