"""search.py — Theory-Radar-style search over the *-BSE instance space.

Ranks candidate instances by VALUE-ADDED = (trained held-out AUROC) - (frozen held-out AUROC),
NOT by the cheap frozen proxy. Why: the frozen screen is anti-correlated with novelty. The boring
instances (dish-identity GaBSE, SciBSE) score high frozen because a general encoder already does
them; the VALUABLE ones (MoBSE, flavor-structure GaBSE) are topic-orthogonal — low frozen, high
trained. Maximizing frozen proxy finds the wrong ones. Value-added finds "structure the base is
blind to but is learnable."

Per candidate: admission filter (reject on throw) -> frozen proxy -> cheap-train (few hundred
steps) -> held-out AUROC. Honest: reports the duds and rejects too. This is a DISCOVERY screen;
survivors still go through the full gate.
"""

from __future__ import annotations

from .admission import AdmissionError
from .discover import frozen_proxy
from .encoder import BSEEncoder
from .train import train


def search(
    sources,
    base_model: str = "BAAI/bge-m3",
    steps: int = 300,
    batch: int = 32,
    device: str = "cuda",
):
    rows = []
    for s in sources:
        try:
            s.check_admission()
        except AdmissionError as e:
            rows.append({"name": s.name, "status": "REJECTED", "note": str(e)[:70]})
            continue
        try:
            max_len = getattr(s, "max_len", 192)
            enc = BSEEncoder(base_model=base_model, max_len=max_len, device=device)
            fp = frozen_proxy(s, enc)["structure_auroc_frozen"]  # BEFORE training
            rep = train(enc, s, epochs=1, batch_size=batch, max_steps=steps)  # cheap-train in place
            tr = rep.metrics["structure_auroc"]
            rows.append(
                {
                    "name": s.name,
                    "status": "ok",
                    "frozen": fp,
                    "trained": tr,
                    "value_added": tr - fp,
                    "passed": rep.passed,
                }
            )
        except Exception as e:
            rows.append({"name": s.name, "status": "error", "note": str(e)[:70]})
    rows.sort(key=lambda r: r.get("value_added", -9), reverse=True)
    return rows


def print_table(rows) -> None:
    print(f"{'instance':12s} {'status':8s} {'frozen':7s} {'trained':7s} {'value-added':11s} note")
    for r in rows:
        if r["status"] != "ok":
            print(f"  {r['name']:10s} {r['status']:8s} {'':7s} {'':7s} {'':11s} {r.get('note','')}")
        else:
            print(
                f"  {r['name']:10s} {'ok':8s} {r['frozen']:.3f}   {r['trained']:.3f}   "
                f"{r['value_added']:+.3f}      {'PASS' if r['passed'] else ''}"
            )
