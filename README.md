# xbse — modular domain-specific sentence embeddings, validated by one shared gate

[![CI](https://github.com/ahb-sjsu/xbse/actions/workflows/ci.yml/badge.svg)](https://github.com/ahb-sjsu/xbse/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with ty](https://img.shields.io/badge/type%20checked-ty-261230.svg)](https://github.com/astral-sh/ty)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![tests](https://img.shields.io/badge/tests-pytest-0A9EDC.svg?logo=pytest&logoColor=white)](https://docs.pytest.org)

`*-BSE` models (LaBSE, LeBSE, MoBSE, …) are the *same architecture* trained to be **invariant to
one axis** and **sensitive to another**. `xbse` factors out everything they share and leaves each
domain as a small plugin — a new `*-BSE` is a `PairSource` + a config, not a new repo — and **every
instance clears the same validation gate**, so no encoder can grade itself easier than another.

Each *moral* `*-BSE` is the encoder for one **DEME MoralVector** dimension; the dimension↔feeder map
lives in `erisml-lib/docs/moralvector_reference.md`.

## The idea in one table

| model | invariance (→ positives) | sensitivity (→ negatives) |
|---|---|---|
| LaBSE | language (translation pairs) | meaning |
| LeBSE | surface / citation form | legal holding |
| **MoralVector feeders** | wording / framing / topic / **which corpus** | one moral dimension's valence |

Only the **PairSource** (what is invariant vs sensitive), an optional **domain adversary**, and the
**validation labels** differ. The encoder, contrastive objective, and gate are written once.

## Architecture

```mermaid
flowchart LR
    PS["PairSource plugin<br/>(invariance → +, sensitivity → −)"] --> ENC["BSEEncoder<br/>BGE-M3 · mean-pool · L2-norm"]
    ENC --> OBJ["InfoNCE<br/>contrastive objective"]
    OBJ --> GATE{"Shared gate<br/>structure-AUROC + fuzz-ratio"}
    GATE -->|pass| OK["validated encoder<br/>earns encode / distance / probe"]
    GATE -->|fail| STOP["HARD STOP<br/>no tools, no claims — iterate"]
    classDef s fill:#e3f2fd,stroke:#1565c0; classDef o fill:#c8e6c9,stroke:#2e7d32; classDef x fill:#ffcdd2,stroke:#c62828;
    class PS,ENC,OBJ s; class OK o; class STOP x;
```

```
xbse/
  encoder.py        BSEEncoder: base transformer → pooling → projection → L2-norm   (shared)
  pairs.py          PairSource ABC yielding (anchor, positive, negative)            (shared iface)
  objective.py      InfoNCE contrastive loss                                        (shared)
  adversarial.py    gradient-reversal domain head + polar severity readout          (shared)
  validate.py       THE GATE: structure-vs-surface AUROC + fuzz ratio               (shared)
  train.py          single-corpus training loop                                     (shared)
  train_adv.py      joint cross-dataset + domain-adversarial loop                   (shared)
  instances/
    joint.py          JointPairSource — train one dimension on ≥2 corpora at once
    joint_builders.py per-dimension corpus assembly (privacy, care, rights, …)
    mobse.py rights.py physharm.py epistemic.py socenv.py privacybse.py darkpattern.py …
  experiments/
    data_sourcing_plan.md    ≥2 independent corpora per dimension (+ legal-evidence route, APIs)
    rank_test.py             empirical moral-dimension rank test (bifactor result)
```

## Why cross-dataset validation — the correction that reshaped this project

Single-corpus contrastive fine-tuning of BGE-M3 learns a corpus's **surface**, because that is the
easiest way to separate its pairs. A feeder can score a beautiful **within-dataset** AUROC yet have
learned nothing transferable. We proved this the hard way: every moral feeder scored 0.75–0.955
within its own corpus but **collapsed to ~0.47–0.55 (random) on a second, independent corpus of the
same dimension.** The within-dataset numbers were artifacts.

**The fix — joint cross-dataset training** (`JointPairSource`): train each dimension on **≥2 corpora
at once**, drawing every same-sign positive from a *different* corpus. To pull an A-text next to a
same-valence B-text the encoder cannot use A's surface — only the shared moral structure. Held-out
AUROC is then cross-dataset **by construction**. An optional gradient-reversal domain head
(`adversarial.py`) adds explicit corpus-invariance.

```mermaid
flowchart TB
    A["Corpus A<br/>(e.g. privacy RoTs)"] --> J["JointPairSource<br/>same-sign positives ACROSS corpora"]
    B["Corpus B<br/>(e.g. AITA scenarios)"] --> J
    J --> ENC["shared BGE-M3 encoder"]
    ENC --> GRL["gradient-reversal<br/>domain head (optional)"]
    ENC --> XG["cross-dataset held-out gate<br/>= the honest number"]
    classDef s fill:#e3f2fd,stroke:#1565c0; class A,B,J,ENC,GRL,XG s;
```

## Status — honest cross-dataset scorecard

Cross-dataset held-out AUROC (evaluate on a *second* corpus), joint vs untrained baseline:

| dimension | corpora (2 independent) | baseline | **cross-dataset (honest)** |
|---|---|---:|---:|
| privacy_protection | privacy RoTs + AITA scenarios | 0.55 | **0.853** |
| epistemic_quality | Social-Chem honesty + ETHICS | 0.48 | **0.817** |
| virtue_care | Social-Chem care-harm + ETHICS | 0.47 | **0.811** |
| fairness_equity | Social-Chem fairness + ETHICS | 0.47 | **0.789** |
| autonomy_respect | ec-darkpattern + MentalManip | 0.52 | **0.747** |
| legitimacy_trust | Social-Chem authority + ETHICS | 0.52 | **0.708** |
| physical_harm | BeaverTails + ETHICS-harm | 0.50 | **0.622** |
| societal_environmental | ClimateBERT + dual-judged env-claims | 0.43 | *training* |
| rights_respect | ECHR + ETHICS-justice | — | *training* |

**Seven of nine dimensions rehabilitated from artifact to real (0.62–0.85).** Lessons banked:
(i) **within-dataset AUROC is not evidence of a real dimension** — always cross-test; (ii) the
load-bearing fix is **cross-corpus same-sign positives**, not the adversary (which only de-confounds
when the two corpora are surface-similar); (iii) `physical_harm` (0.622) is weakest — the widest
genre gap (QA-pairs vs scenarios) — and the CourtListener criminal-case route (injury-tier labels)
is its planned third corpus. Full roadmap: `experiments/data_sourcing_plan.md`.

## The non-negotiable discipline (read before adding code)

Built in **falsification order** with **hard stops** (see `docs/MOBSE_PLAN.md`):

1. **Build** an instance (a `PairSource`).
2. **Validate** with the shared gate — cross-dataset structure-vs-surface AUROC + fuzz ratio > 1.
   **HARD STOP: fail the gate → no downstream tools, no claims — iterate.**
3. Only a *validated* embedding earns `encode` / `distance` / `probe`.
4. Any geometric tool on top is a **pre-registered, falsifiable** hypothesis about the geometry.

**Language discipline:** operations are named for what they *measure*, not for retired claims — a
harm operation computes a *representation-invariant harm ledger*, never a "conserved quantity."

## Quickstart

```bash
pip install -e ".[dev]"          # torch (CPU wheel is fine for tests) + package
pytest -q                        # offline gate/objective/circularity tests
ruff check src tests
```

```python
from xbse.encoder import BSEEncoder
from xbse.instances.joint_builders import build_privacy_joint
from xbse.train_adv import train_adversarial

src = build_privacy_joint()                 # ≥2 corpora, cross-dataset positives
enc = BSEEncoder(base_model="BAAI/bge-m3", max_len=src.max_len, device="cuda")
report = train_adversarial(enc, src, max_steps=1400)   # trains, then RUNS THE GATE
print(report.metrics["structure_auroc"])    # cross-dataset held-out AUROC — the honest number
```

Training a `*-BSE` and reporting its gate number are **one operation** — a run that doesn't end in
gate output is not a finished run.

## Testing & CI

`pytest` runs offline (no model download / network): the gate math, the InfoNCE objective, and the
**circularity guard** (`assert_heldout_disjoint` fails loud if any held-out eval text leaked into
training). CI (`.github/workflows/ci.yml`) runs `ruff` + `pytest` on Python 3.10–3.12.

## Kinship (not merged, on purpose)

`*-BSE` learns **invariance** — it quotients a nuisance group out. Its equivariant cousins
([`latte`](https://github.com/ahb-sjsu/latte), [`arc-equivariant-search`](https://github.com/ahb-sjsu/arc-equivariant-search))
*covary* with a symmetry group. Same principle — a transformation group acts; you either mod it out
(here) or transform with it (there) — different codebase.

## License

MIT © Andrew H. Bond (SJSU).
