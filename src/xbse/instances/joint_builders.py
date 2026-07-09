"""Per-dimension builders that assemble a JointPairSource from >=2 real corpora on Atlas.

Each builder returns rows as (text, sign) in ONE agreed valence convention for the dimension:
    '+' = the dimension is UPHELD / respected      '-' = the dimension is VIOLATED
so that a cross-dataset positive (same sign, different corpus) is a genuine same-structure pair.
The mechanism (splitting, cross-corpus triplets, domain-adversarial labels, cross-domain gate)
all lives in JointPairSource.
"""

from __future__ import annotations

import csv
import json

from .joint import JointPairSource

DEAD_BAND = 0.05

# ----------------------------------------------------------------------------- privacy
PRIVACY_ROT = "/archive/ethics-corpora/privacy/privacy_labeled.jsonl"  # prescriptive RoTs
PRIVACY_AITA = "/archive/ethics-corpora/privacy/aita_privacy_labeled.jsonl"  # lived scenarios


def _signed_jsonl(path, field="privacy", db=DEAD_BAND):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except (ValueError, TypeError):
                continue
            text = (d.get("text") or "").strip()
            v = float(d.get(field, 0.0))
            sign = "+" if v > db else ("-" if v < -db else "0")
            if sign != "0" and len(text) >= 12:
                rows.append((text, sign))
    return rows


def build_privacy_joint(holdout_frac: float = 0.1) -> JointPairSource:
    return JointPairSource(
        name="privacy_joint",
        domains=[("rot", _signed_jsonl(PRIVACY_ROT)), ("aita", _signed_jsonl(PRIVACY_AITA))],
        holdout_frac=holdout_frac,
        invariant_structure="privacy valence (violated vs respected), shared across RoTs and scenarios",
        label_source="dual-judge signed privacy valence (RoTs) + AITA-scenario privacy valence",
    )


# ----------------------------------------------------------------------- moral foundations
SOCIAL_CHEM = "/archive/ethics-corpora/social-chem-101/social-chem-101/social-chem-101.v1.0.tsv"
ETHICS_CS = "/archive/ethics-corpora/ethics/commonsense.jsonl"

# (rot-category substring in Social-Chem, keyword set for ETHICS commonsense)
FOUNDATIONS = {
    "care": (
        "care-harm",
        (
            "hurt",
            "harm",
            "help",
            "protect",
            "care",
            "cruel",
            "abuse",
            "comfort",
            "suffer",
            "kind",
            "neglect",
            "rescue",
            "hit",
            "attack",
        ),
    ),
    "fairness": (
        "fairness-cheating",
        (
            "fair",
            "unfair",
            "cheat",
            "steal",
            "deserve",
            "equal",
            "discriminat",
            "share",
            "owe",
            "betray",
            "honest deal",
            "scam",
            "fraud",
        ),
    ),
    "legitimacy": (
        "authority-subversion",
        (
            "obey",
            "rule",
            "law",
            "permission",
            "authority",
            "boss",
            "illegal",
            "allowed",
            "disobey",
            "respect",
            "order",
            "duty",
            "supposed to",
        ),
    ),
    "epistemic": (
        None,  # honesty is keyword-filtered in Social-Chem too (no MFT category)
        (
            "lie",
            "lying",
            "lied",
            "honest",
            "dishonest",
            "truth",
            "truthful",
            "deceive",
            "deceiv",
            "mislead",
        ),
    ),
}
_EPI_KW = FOUNDATIONS["epistemic"][1]


def _sc_sign(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "0"
    return "+" if f > 0 else ("-" if f < 0 else "0")


def _social_chem_rows(category, keywords):
    rows = []
    with open(SOCIAL_CHEM, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                if int(row.get("rot-agree") or 0) < 3:
                    continue
            except (TypeError, ValueError):
                continue
            rot = (row.get("rot") or "").strip()
            if not rot:
                continue
            cats = (row.get("rot-moral-foundations") or "").lower()  # MFT foundations column
            low = rot.lower()
            ok = (category in cats) if category else any(k in low for k in keywords)
            if not ok:
                continue
            sign = _sc_sign(row.get("action-moral-judgment"))
            if sign != "0":
                rows.append((rot, sign))
    return rows


def _ethics_rows(keywords):
    # ETHICS commonsense: label 1 = morally WRONG -> '-'; label 0 = OK -> '+'
    rows = []
    with open(ETHICS_CS, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except (ValueError, TypeError):
                continue
            text = (d.get("text") or "").strip()
            low = text.lower()
            if len(text) < 12 or not any(k in low for k in keywords):
                continue
            lab = int(d.get("label", 0))
            rows.append((text, "-" if lab == 1 else "+"))
    return rows


def build_foundation_joint(name: str, holdout_frac: float = 0.1) -> JointPairSource:
    category, keywords = FOUNDATIONS[name]
    return JointPairSource(
        name=f"{name}_joint",
        domains=[
            ("socialchem", _social_chem_rows(category, keywords)),
            ("ethics", _ethics_rows(keywords)),
        ],
        holdout_frac=holdout_frac,
        invariant_structure=f"{name} valence (upheld vs violated), shared across RoTs and scenarios",
        label_source="Social-Chem action-moral-judgment sign + ETHICS commonsense label",
    )


# --------------------------------------------------------------- physical_harm & autonomy
# Both reuse existing single-corpus loaders (BeaverTails / MentalManip are already HF-cached on
# Atlas; darkpattern reads a local tsv). Convention: violation label 1 -> '-', upheld 0 -> '+'.
HARM_KW = (
    "hurt",
    "harm",
    "kill",
    "hit",
    "attack",
    "injure",
    "wound",
    "stab",
    "punch",
    "violence",
    "violent",
    "abuse",
    "assault",
    "poison",
    "choke",
    "beat",
    "weapon",
    "shoot",
    "strangle",
    "murder",
    "kick",
    "slap",
)


def _binary_neg_rows(loader_rows):
    # loader tuples are (topic/bucket, text, label) with label 1 = violation
    return [(r[1], "-" if int(r[2]) == 1 else "+") for r in loader_rows]


def build_physharm_joint(holdout_frac: float = 0.1) -> JointPairSource:
    from .physharm import PhysHarmBSEPairSource

    bt = _binary_neg_rows(PhysHarmBSEPairSource()._rows())  # BeaverTails physical-harm vs safe
    eth = _ethics_rows(HARM_KW)  # ETHICS harm-keyword scenarios
    return JointPairSource(
        name="physharm_joint",
        domains=[("beavertails", bt), ("ethics", eth)],
        holdout_frac=holdout_frac,
        invariant_structure="physical-harm valence (bodily harm vs safe), across QA pairs and scenarios",
        label_source="BeaverTails physical-harm categories + ETHICS commonsense harm scenarios",
    )


def build_autonomy_joint(holdout_frac: float = 0.1) -> JointPairSource:
    from .darkpattern import AutonomyDarkBSEPairSource
    from .mentalmanip import AutonomyBSEPairSource

    dp = _binary_neg_rows(AutonomyDarkBSEPairSource()._rows())  # dark-pattern UI text
    mm = _binary_neg_rows(AutonomyBSEPairSource()._rows())  # MentalManip dialogue
    return JointPairSource(
        name="autonomy_joint",
        domains=[("darkpattern", dp), ("mentalmanip", mm)],
        holdout_frac=holdout_frac,
        invariant_structure="manipulation valence (manipulative vs respectful of agency), across UI text and dialogue",
        label_source="ec-darkpattern binary annotation + MentalManip manipulative/non annotation",
    )


# --------------------------------------------------------------- environmental
ENV_LABELED = "/archive/ethics-corpora/environmental/env_labeled.jsonl"  # dual-judge {text, env}


def _climate_sentiment_rows():
    # ClimateBERT climate_sentiment: 0=opportunity(+), 1=neutral(drop), 2=risk(-)
    from datasets import load_dataset

    rows = []
    ds = load_dataset("climatebert/climate_sentiment")
    for sp in ds:
        for r in ds[sp]:
            lab = int(r["label"])
            sign = "+" if lab == 0 else ("-" if lab == 2 else "0")
            t = (r.get("text") or "").strip()
            if sign != "0" and len(t) >= 25:
                rows.append((t, sign))
    return rows


def build_environmental_joint(holdout_frac: float = 0.12) -> JointPairSource:
    return JointPairSource(
        name="environmental_joint",
        domains=[
            ("climate_sentiment", _climate_sentiment_rows()),
            ("env_claims", _signed_jsonl(ENV_LABELED, field="env")),
        ],
        holdout_frac=holdout_frac,
        invariant_structure="environmental-impact valence (harmed vs protected), across disclosures and claims",
        label_source="ClimateBERT risk/opportunity + dual-judge (qwen3+glm-5) env-impact valence",
    )


# --------------------------------------------------------------- rights_respect
def _echr_rows():
    # ECHR (LexGLUE ecthr_a): facts + allegedly-violated article labels -> violation '-', none '+'
    from datasets import load_dataset

    ds = load_dataset("coastalcph/lex_glue", "ecthr_a")
    rows = []
    for sp in ("train", "validation"):
        for r in ds[sp]:
            t = r["text"]
            txt = (" ".join(t) if isinstance(t, list) else str(t)).strip()[:800]
            sign = "-" if len(r["labels"]) > 0 else "+"
            if len(txt) >= 25:
                rows.append((txt, sign))
    return rows


def _ethics_justice_rows():
    # ETHICS justice: just/reasonable(1) -> '+', unjust(0) -> '-'
    from datasets import load_dataset

    ds = load_dataset("hendrycks/ethics", "justice")
    rows = []
    for sp in ds:
        for r in ds[sp]:
            s = (r.get("scenario") or "").strip()
            if len(s) >= 15:
                rows.append((s, "+" if int(r["label"]) == 1 else "-"))
    return rows


def build_rights_joint(holdout_frac: float = 0.12) -> JointPairSource:
    return JointPairSource(
        name="rights_joint",
        domains=[("echr", _echr_rows()), ("ethics_justice", _ethics_justice_rows())],
        holdout_frac=holdout_frac,
        invariant_structure="rights / just-treatment valence (violated vs respected), across ECHR cases and justice scenarios",
        label_source="ECHR article-violation labels + ETHICS justice reasonableness",
    )


BUILDERS = {
    "privacy_joint": build_privacy_joint,
    "environmental_joint": build_environmental_joint,
    "rights_joint": build_rights_joint,
    "care_joint": lambda **k: build_foundation_joint("care", **k),
    "fairness_joint": lambda **k: build_foundation_joint("fairness", **k),
    "legitimacy_joint": lambda **k: build_foundation_joint("legitimacy", **k),
    "epistemic_joint": lambda **k: build_foundation_joint("epistemic", **k),
    "physharm_joint": build_physharm_joint,
    "autonomy_joint": build_autonomy_joint,
}
