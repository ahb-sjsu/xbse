"""Materialize MFRC (USC-MOLA-Lab/MFRC) to a local jsonl so the presence builders don't hit HF each
call. Single-foundation comments only (clean identity labels). Writes {text, foundation} lines for the
5 family foundations + Non-Moral. One-off; run on Atlas."""

import json
import os

os.environ.setdefault("HF_HOME", "/archive/cache/huggingface")

from datasets import load_dataset

OUT_DIR = os.environ.get("XBSE_DATA_ROOT", "/archive/ethics-corpora") + "/mfrc"
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, "mfrc_foundations.jsonl")

# MFRC annotation label -> our foundation key
LABEL_TO_F = {
    "Care": "care",
    "Proportionality": "fairness",
    "Equality": "fairness",
    "Authority": "legitimacy",
    "Loyalty": "loyalty",
    "Purity": "purity",
}

d = load_dataset("USC-MOLA-Lab/MFRC", split="train_dedup")
by_text = {}
nonmoral = set()
for r in d:
    t = (r.get("text") or "").strip()
    if len(t) < 12:
        continue
    ann = str(r.get("annotation") or "")
    if ann in LABEL_TO_F:
        by_text.setdefault(t, set()).add(LABEL_TO_F[ann])
    elif ann == "Non-Moral":
        nonmoral.add(t)

n = 0
with open(OUT, "w", encoding="utf-8") as f:
    for t, fs in by_text.items():
        if len(fs) == 1:  # single-foundation -> clean identity label
            f.write(json.dumps({"text": t, "foundation": next(iter(fs))}) + "\n")
            n += 1
    # keep non-moral comments (not tagged with any family foundation) as an optional grounding pool
    for t in nonmoral:
        if t not in by_text:
            f.write(json.dumps({"text": t, "foundation": "non-moral"}) + "\n")
            n += 1
print(f"wrote {n} rows -> {OUT}")

from collections import Counter

c = Counter()
with open(OUT, encoding="utf-8") as f:
    for line in f:
        c[json.loads(line)["foundation"]] += 1
print(dict(c))
