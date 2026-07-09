"""Per-domain *-BSE instances. Each is a PairSource + config; nothing else.

- mobse: morality BSE (Social-Chem-101 / Scruples / Moral-Machine). First instance.
- (lebse: legal BSE — refactor the existing ahb-sjsu/lebse in once MoBSE proves the core.)
"""

from .codebse import CODEBSE_CONFIG, CodeBSEPairSource
from .darkpattern import DARKPATTERN_CONFIG, AutonomyDarkBSEPairSource
from .epistemic import EPISTEMIC_CONFIG, EpistemicBSEPairSource
from .gabse import GABSE_CONFIG, GaBSEPairSource
from .mentalmanip import MENTALMANIP_CONFIG, AutonomyBSEPairSource
from .mobse import MOBSE_CONFIG, MoBSEPairSource
from .mostories import MOSTORIES_CONFIG, MoralStoriesBSEPairSource
from .physharm import PHYSHARM_CONFIG, PhysHarmBSEPairSource
from .privacybse import PRIVACY_CONFIG, PrivacyBSEPairSource
from .reabse import REABSE_CONFIG, ReaBSEPairSource
from .rights import RIGHTS_CONFIG, RightsBSEPairSource
from .scibse import SCIBSE_CONFIG, SciBSEPairSource
from .socenv import SOCENV_CONFIG, SocEnvBSEPairSource

__all__ = [
    "MoBSEPairSource",
    "MOBSE_CONFIG",
    "SciBSEPairSource",
    "SCIBSE_CONFIG",
    "ReaBSEPairSource",
    "REABSE_CONFIG",
    "CodeBSEPairSource",
    "CODEBSE_CONFIG",
    "GaBSEPairSource",
    "GABSE_CONFIG",
    "RightsBSEPairSource",
    "RIGHTS_CONFIG",
    "MoralStoriesBSEPairSource",
    "MOSTORIES_CONFIG",
    "PhysHarmBSEPairSource",
    "PHYSHARM_CONFIG",
    "EpistemicBSEPairSource",
    "EPISTEMIC_CONFIG",
    "SocEnvBSEPairSource",
    "SOCENV_CONFIG",
    "AutonomyBSEPairSource",
    "MENTALMANIP_CONFIG",
    "PrivacyBSEPairSource",
    "PRIVACY_CONFIG",
    "AutonomyDarkBSEPairSource",
    "DARKPATTERN_CONFIG",
]
