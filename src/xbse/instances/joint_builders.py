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
import os

from .joint import JointPairSource

DEAD_BAND = 0.05

# Corpus root — override with XBSE_DATA_ROOT. Defaults to the Atlas training-host location.
_DATA_ROOT = os.environ.get("XBSE_DATA_ROOT", "/archive/ethics-corpora")


def _data(*parts: str) -> str:
    return os.path.join(_DATA_ROOT, *parts)


def _open(path: str, **kw):
    """open() with a friendly not-found error pointing at the data plan (portability)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"xbse corpus not found: {path}\n"
            f"  Set XBSE_DATA_ROOT to your corpus root, or fetch the corpora "
            f"(see experiments/data_sourcing_plan.md)."
        )
    kw.setdefault("encoding", "utf-8")
    kw.setdefault("errors", "replace")
    return open(path, **kw)


# ----------------------------------------------------------------------------- privacy
PRIVACY_ROT = _data("privacy", "privacy_labeled.jsonl")  # prescriptive RoTs
PRIVACY_AITA = _data("privacy", "aita_privacy_labeled.jsonl")  # lived scenarios


def _signed_jsonl(path, field="privacy", db=DEAD_BAND):
    rows = []
    with _open(path) as f:
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
SOCIAL_CHEM = _data("social-chem-101", "social-chem-101", "social-chem-101.v1.0.tsv")
ETHICS_CS = _data("ethics", "commonsense.jsonl")

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
    # B1 (MoralVector roadmap): the two MFT "binding" foundations the DEME taxonomy misses. Social-Chem
    # carries them as first-class MFT categories (loyalty-betrayal 41k, sanctity-degradation 15k); ETHICS
    # is the independent second corpus via keywords. Convention unchanged: '+' upheld, '-' violated.
    "loyalty": (
        "loyalty-betrayal",
        (
            "loyal",
            "loyalty",
            "betray",
            "betrayal",
            "traitor",
            "treason",
            "faithful",
            "unfaithful",
            "cheat on",
            "backstab",
            "desert",
            "abandon",
            "allegiance",
            "snitch",
            "rat out",
        ),
    ),
    "purity": (
        "sanctity-degradation",
        (
            "disgusting",
            "gross",
            "obscene",
            "sacred",
            "holy",
            "sinful",
            "pure",
            "impure",
            "defile",
            "degrade",
            "vulgar",
            "indecent",
            "filthy",
            "depraved",
            "perver",
            "sanctity",
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
    with _open(SOCIAL_CHEM, newline="") as f:
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
    with _open(ETHICS_CS) as f:
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


# --------------------------------------------- Phase A: general moral valence (bifactor channel 0)
# The FAMILY-POOLED general factor G. Unlike a foundation feeder it applies NO category/keyword
# filter — it pools signed RoTs across ALL foundation categories (and uncategorized), so it learns
# the general good/bad direction, not a foundation. Second corpus = ETHICS-commonsense OVERALL
# (all rows, signed by label), a genuinely different-provenance signed valence source. Deterministic
# down-sample keeps the pool comparable in size to the specific feeders and the run tractable.
# See experiments/prereg_bifactor_readout.md (prereg 2026-07-12).
def _social_chem_all_signed(cap=40000):
    """All Social-Chem RoTs with a non-zero action-moral-judgment sign, NO category/keyword filter."""
    rows = []
    with _open(SOCIAL_CHEM, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                if int(row.get("rot-agree") or 0) < 3:
                    continue
            except (TypeError, ValueError):
                continue
            rot = (row.get("rot") or "").strip()
            if not rot:
                continue
            sign = _sc_sign(row.get("action-moral-judgment"))
            if sign != "0":
                rows.append((rot, sign))
    if cap and len(rows) > cap:
        import random

        random.Random(0).shuffle(rows)
        rows = rows[:cap]
    return rows


def _ethics_all_signed(cap=20000):
    """All ETHICS-commonsense rows, signed by label (1=wrong->'-', 0=ok->'+'), NO keyword filter."""
    rows = []
    with _open(ETHICS_CS) as f:
        for line in f:
            try:
                d = json.loads(line)
            except (ValueError, TypeError):
                continue
            text = (d.get("text") or "").strip()
            if len(text) < 12:
                continue
            lab = int(d.get("label", 0))
            rows.append((text, "-" if lab == 1 else "+"))
    if cap and len(rows) > cap:
        import random

        random.Random(0).shuffle(rows)
        rows = rows[:cap]
    return rows


def build_general_valence_joint(holdout_frac: float = 0.12) -> JointPairSource:
    """G = general moral valence (bifactor channel 0). Pooled signed Social-Chem (all categories)
    x all signed ETHICS-commonsense. Gated identically to every axis (Phase A1)."""
    return JointPairSource(
        name="general_valence_joint",
        domains=[
            ("socialchem_pooled", _social_chem_all_signed()),
            ("ethics_all", _ethics_all_signed()),
        ],
        holdout_frac=holdout_frac,
        invariant_structure="general moral valence (good vs bad), shared across all-foundation RoTs and all-topic scenarios",
        label_source="Social-Chem action-moral-judgment sign (all categories) + ETHICS commonsense label (all topics)",
    )


# ------------------------------------------ v2 independent-corpus feeders (prereg experiment)
# Replace the shared ETHICS second corpus with a DIFFERENT-provenance signed corpus, keeping
# Social-Chem as the first corpus, so the change is isolated. Tests whether corpus independence
# separates the collapsed family (see experiments/prereg_independent_corpora.md).
MS_CARE = _data("moral_stories", "care_signed.jsonl")  # Moral-Stories care-norm actions (+moral/-immoral)
MHS_FAIR = _data("mhs", "mhs_fairness.jsonl")  # Measuring Hate Speech (score>0.5 -> '-', <-1 -> '+')


def _signed_jsonl_simple(path, cap=None):
    """Read {'text','sign'} jsonl -> [(text, sign)]; optional deterministic down-sample to cap."""
    rows = []
    with _open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except (ValueError, TypeError):
                continue
            t = (d.get("text") or "").strip()
            s = d.get("sign")
            if len(t) >= 12 and s in ("+", "-"):
                rows.append((t, s))
    if cap and len(rows) > cap:
        import random

        random.Random(0).shuffle(rows)
        rows = rows[:cap]
    return rows


def build_care_v2(holdout_frac: float = 0.1) -> JointPairSource:
    """care with an INDEPENDENT second corpus (Moral-Stories care actions) replacing ETHICS."""
    return JointPairSource(
        name="care_v2_joint",
        domains=[
            ("socialchem", _social_chem_rows("care-harm", FOUNDATIONS["care"][1])),
            ("moralstories", _signed_jsonl_simple(MS_CARE)),
        ],
        holdout_frac=holdout_frac,
        invariant_structure="care valence, shared across RoTs and Moral-Stories action narratives",
        label_source="Social-Chem care-harm sign + Moral-Stories moral/immoral action (text-independent)",
    )


def build_fairness_v2(holdout_frac: float = 0.1) -> JointPairSource:
    """fairness with an INDEPENDENT second corpus (Measuring Hate Speech) replacing ETHICS."""
    return JointPairSource(
        name="fairness_v2_joint",
        domains=[
            ("socialchem", _social_chem_rows("fairness-cheating", FOUNDATIONS["fairness"][1])),
            ("mhs", _signed_jsonl_simple(MHS_FAIR, cap=12000)),
        ],
        holdout_frac=holdout_frac,
        invariant_structure="fairness valence, shared across RoTs and hate-speech comments",
        label_source="Social-Chem fairness-cheating sign + Measuring-Hate-Speech score (independent)",
    )


# ------------------------------------------ identity_attack (GTC discovered dimension, prereg 2026-07-11)
# A moral dimension the DEME-9 taxonomy MISSED: surfaced by the Moral Spectrum Analyzer's discovery
# band (+0.24 [0.20, 0.28] over the covered-category baseline) and validated here through the SAME
# pre-registered gate as the 8 passing feeders. Two genuinely independent corpora, materialised by
# scripts/build_identity_attack.py: Jigsaw civil_comments identity_attack labels (>=0.5 attack /
# clean respected) and Berkeley Measuring-Hate-Speech hate_speech_score (>0.5 attack / <-1 supportive).
CC_IDENTITY = _data("identity_attack", "civil_comments_identity.jsonl")
MHS_IDENTITY = _data("identity_attack", "mhs_identity.jsonl")


def build_identity_attack_joint(holdout_frac: float = 0.12) -> JointPairSource:
    return JointPairSource(
        name="identity_attack_joint",
        domains=[
            ("civil_comments", _signed_jsonl_simple(CC_IDENTITY)),
            ("mhs", _signed_jsonl_simple(MHS_IDENTITY)),
        ],
        holdout_frac=holdout_frac,
        invariant_structure=(
            "identity-attack valence (a person or group demeaned/attacked for their identity vs "
            "treated with dignity), shared across Jigsaw civil-comments and Berkeley hate-speech"
        ),
        label_source=(
            "Jigsaw civil_comments identity_attack label + Measuring-Hate-Speech hate_speech_score "
            "(independent corpora)"
        ),
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
ENV_LABELED = _data("environmental", "env_labeled.jsonl")  # dual-judge {text, env}


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
# LexGLUE ecthr_a label index -> ECHR article. Strata group articles by the RIGHT they protect.
ECHR_ARTICLE = {
    0: "art2_life",
    1: "art3_inhuman",
    2: "art5_liberty",
    3: "art6_fairtrial",
    4: "art8_privacy",
    5: "art9_religion",
    6: "art10_expression",
    7: "art11_assembly",
    8: "art14_discrimination",
    9: "p1_property",
}
# Physical-integrity stratum: life + inhuman treatment — the universal-human-rights core that maps
# to US 4th/8th-Amendment excessive-force, and is legitimacy-INVARIANT (torture is wrong anywhere).
ECHR_PHYSICAL_INTEGRITY = {0, 1}


def _echr_rows(articles: set[int] | None = None, exclude: set[int] | None = None):
    """ECHR (LexGLUE ecthr_a) facts. If `articles` is given, restrict the '-' class to violations
    of THOSE articles (a right-type stratum): '-' iff a listed article was violated, '+' iff no
    article was (generic no-violation). Aggregating all articles mixes incommensurable rights
    (fair-trial + torture + property), which is why the un-stratified rights joint could not learn.

    `exclude` drops any case whose violated articles intersect the set (e.g. {4} = Art-8 right to
    private life) — used for the rights→privacy contamination check (reviewer-3 point 3): Art-8 IS
    privacy, so its presence in the rights corpus contaminates a rights encoder with privacy signal.
    """
    from datasets import load_dataset

    ds = load_dataset("coastalcph/lex_glue", "ecthr_a")
    rows = []
    for sp in ("train", "validation"):
        for r in ds[sp]:
            labels = set(r["labels"])
            if exclude is not None and (labels & exclude):
                continue  # drop cases violating an excluded article (e.g. Art-8 privacy)
            if articles is not None:
                hit = bool(labels & articles)
                if labels and not hit:
                    continue  # a violation, but of a DIFFERENT stratum — drop (keeps '-' coherent)
                sign = "-" if hit else "+"
            else:
                sign = "-" if labels else "+"
            t = r["text"]
            txt = (" ".join(t) if isinstance(t, list) else str(t)).strip()[:800]
            if len(txt) >= 25:
                rows.append((txt, sign))
    return rows


# Keywords that pick US civil-rights cases in the PHYSICAL-INTEGRITY stratum (excessive force,
# bodily harm by the state) — the same right ECHR Art 2-3 protects.
_FORCE_KW = (
    "excessive force",
    "use of force",
    "deadly force",
    "physical force",
    "beaten",
    "beating",
    "assault",
    "eighth amendment",
    "cruel and unusual",
    "taser",
    "chokehold",
    "strangl",
    "punch",
    "kick",
    "baton",
    "pepper spray",
    "shot ",
    "shooting",
    "brutality",
)


def _courtlistener_force_rows():
    return [
        (t, s) for (t, s) in _courtlistener_rights_rows() if any(k in t.lower() for k in _FORCE_KW)
    ]


RIGHTS_CL = _data(
    "rights_cl", "rights_cl_labeled.jsonl"
)  # civil-rights case facts (violation-heavy)
RIGHTS_CL_PLUS = _data(
    "rights_cl", "rights_cl_plus.jsonl"
)  # mined RIGHTS-RESPECTED holdings (+ class)


def _courtlistener_rights_rows():
    # Base facts corpus skews '-' (facts describe alleged violations); the '+'-mined file adds
    # no-violation / qualified-immunity holdings so the RESPECTED class is not starved. Missing '+'
    # file is tolerated (falls back to the base corpus) so the builder works before mining is run.
    rows = _signed_jsonl(RIGHTS_CL, field="rights")
    if os.path.exists(RIGHTS_CL_PLUS):
        rows += _signed_jsonl(RIGHTS_CL_PLUS, field="rights")
    return rows


def build_rights_joint(holdout_frac: float = 0.12) -> JointPairSource:
    # Same-genre, cross-JURISDICTION pairing: European ECHR case-facts <-> US civil-rights case
    # facts. The prior ECHR<->ETHICS-justice (legal <-> everyday-scenario) pairing failed (0.475,
    # below baseline) because the two genres share no transferable structure; both corpora here are
    # court case-facts describing how a person's rights were treated. The '+' class (rights RESPECTED)
    # is under-sampled in rights litigation, so it is mined separately from no-violation holdings.
    return JointPairSource(
        name="rights_joint",
        domains=[
            ("echr", _echr_rows()),
            ("courtlistener", _courtlistener_rights_rows()),
        ],
        holdout_frac=holdout_frac,
        invariant_structure="rights-violation valence, shared across European (ECHR) and US civil-rights case facts",
        label_source="ECHR article-violation labels + dual-judge rights valence on US civil-rights case facts (+ mined no-violation holdings)",
    )


def build_rights_no_art8(holdout_frac: float = 0.12) -> JointPairSource:
    """rights_joint with ECHR Art-8 (right to private life) cases REMOVED — the contamination check
    of reviewer-3 point 3. Art-8 IS privacy; if rights→privacy=0.66 in the 9×9 is corpus
    contamination, removing Art-8 from rights training should shrink it."""
    return JointPairSource(
        name="rights_no_art8_joint",
        domains=[
            ("echr", _echr_rows(exclude={4})),  # 4 = art8_privacy
            ("courtlistener", _courtlistener_rights_rows()),
        ],
        holdout_frac=holdout_frac,
        invariant_structure="rights-violation valence with Art-8 privacy cases excluded",
        label_source="ECHR article-violation labels (Art-8 dropped) + dual-judge US civil-rights valence",
    )


def build_rights_force_joint(holdout_frac: float = 0.12) -> JointPairSource:
    # STRATIFIED rights (physical-integrity stratum): ECHR Art 2-3 (life / inhuman treatment) <->
    # US excessive-force civil-rights. Both describe the SAME universal, legitimacy-invariant right
    # (bodily integrity vs the state), so — unlike the aggregate rights joint that mixed fair-trial,
    # property, expression and could not learn — the cross-corpus positives are semantically
    # coherent. This tests whether rights transfers WITHIN a shared right-type stratum.
    return JointPairSource(
        name="rights_force_joint",
        domains=[
            ("echr_phys", _echr_rows(articles=ECHR_PHYSICAL_INTEGRITY)),
            ("courtlistener_force", _courtlistener_force_rows()),
        ],
        holdout_frac=holdout_frac,
        invariant_structure="physical-integrity rights valence (bodily harm by the state vs not), across ECHR Art 2-3 and US excessive-force cases",
        label_source="ECHR Art 2-3 violation labels + dual-judge valence on US excessive-force civil-rights facts",
    )


_RAW_BUILDERS = {
    "privacy_joint": build_privacy_joint,
    "environmental_joint": build_environmental_joint,
    "rights_joint": build_rights_joint,
    "rights_force_joint": build_rights_force_joint,
    "care_joint": lambda **k: build_foundation_joint("care", **k),
    "fairness_joint": lambda **k: build_foundation_joint("fairness", **k),
    "legitimacy_joint": lambda **k: build_foundation_joint("legitimacy", **k),
    "epistemic_joint": lambda **k: build_foundation_joint("epistemic", **k),
    "loyalty_joint": lambda **k: build_foundation_joint("loyalty", **k),      # B1 (MoralVector roadmap)
    "purity_joint": lambda **k: build_foundation_joint("purity", **k),        # B1 (MoralVector roadmap)
    "general_valence_joint": build_general_valence_joint,                     # Phase A1 (bifactor G)
    "physharm_joint": build_physharm_joint,
    "autonomy_joint": build_autonomy_joint,
    "care_v2_joint": build_care_v2,
    "fairness_v2_joint": build_fairness_v2,
    "rights_no_art8_joint": build_rights_no_art8,
    "identity_attack_joint": build_identity_attack_joint,
}

# --------------------------------------------------------------- pre-registered validation bars
# POLICY (chosen 2026-07-10, before the validation re-gate): a feeder is VALIDATED iff its
# cross-dataset held-out AUROC beats BOTH nulls — the untrained-encoder baseline AND the TF-IDF
# bag-of-words control — by >= MARGIN on the same held-out pairs. This tests the exact claim a
# cross-dataset feeder makes (real, non-lexical, transferable structure beyond the nulls) and is
# immune to the within-vs-cross ceiling mismatch that made the label-noise ceiling unfair. The two
# nulls are properties of (untrained model + corpus), independent of the trained feeder. Rejected
# the strict 0.9x-noise-ceiling policy because it compares a cross-corpus AUROC to a within-corpus
# ceiling. This block is the pre-registration record; a bar may be tightened, never loosened.
_MARGIN = 0.10
_BASELINE_NULL = {  # untrained-encoder cross-dataset AUROC, measured before training the feeder
    "privacy_joint": 0.551,
    "care_joint": 0.469,
    "fairness_joint": 0.474,
    "legitimacy_joint": 0.523,
    "epistemic_joint": 0.483,
    "physharm_joint": 0.499,
    "autonomy_joint": 0.515,
    "environmental_joint": 0.426,
    "rights_joint": 0.517,
    "identity_attack_joint": 0.552,  # prereg 2026-07-11 (discovered dimension); see _REGISTERED
    "loyalty_joint": 0.411,  # B1 (MoralVector roadmap), prereg 2026-07-12; untrained null; PASS AUROC 0.911
    # purity's BINDING null is the TF-IDF BoW (0.656 — disgust lexicon), which dominates the untrained
    # 0.404; stored here so a re-gate keeps the tight 0.756 bar (never loosen). PASS AUROC 0.811 (+0.156).
    "purity_joint": 0.656,
}
# Per-dimension pre-registration date override (default is the 2026-07-10 re-gate batch).
_REGISTERED = {
    "identity_attack_joint": "2026-07-11",
    "loyalty_joint": "2026-07-12",
    "purity_joint": "2026-07-12",
}


def _prereg_bar(name: str):
    from ..bar import Bar

    b = _BASELINE_NULL[name]
    return Bar(
        auroc_min=round(b + _MARGIN, 3),  # absolute reference floor = baseline + margin
        fuzz_min=1.0,
        policy="baseline_relative",
        margin=_MARGIN,
        baseline_auroc=b,
        source=f"baseline-relative(margin {_MARGIN}) over untrained null {b} + BoW null",
        derivation=(
            f"VALIDATED iff cross-dataset held-out AUROC beats BOTH the untrained-encoder null "
            f"({b}) and the TF-IDF bag-of-words null by >= {_MARGIN} on the same held-out pairs; "
            f"nulls are properties of (untrained model + corpus), independent of the trained feeder."
        ),
        registered=_REGISTERED.get(name, "2026-07-10"),
    )


def _attach_bar(name: str, builder):
    def build(**k):
        src = builder(**k)
        if src.bar is None and name in _BASELINE_NULL:
            src.bar = _prereg_bar(name)
        return src

    return build


BUILDERS = {name: _attach_bar(name, fn) for name, fn in _RAW_BUILDERS.items()}
