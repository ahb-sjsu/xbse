"""Training loop (shared). Fits an encoder on a PairSource, then RUNS THE GATE — always.

The gate call is not optional and not skippable: training a *-BSE and reporting its validation
number are one operation. A run that doesn't end in gate() output is not a finished run.
"""
from __future__ import annotations
import json
import random
import torch
from torch.optim import AdamW

from .objective import info_nce
from .validate import gate


def train(encoder, source, epochs: int = 1, batch_size: int = 32, lr: float = 2e-5,
          temperature: float = 0.05, max_steps: int | None = None) -> dict:
    # TEETH first: fail loud if held-out eval leaked into training (rigged test), before we
    # spend a GPU-hour. Throws CircularityError.
    source.assert_heldout_disjoint()

    opt = AdamW(encoder.parameters(), lr=lr)
    triplets = list(source.train_triplets())
    print(f"[{source.name}] {len(triplets)} training triplets (circularity guard passed)", flush=True)
    encoder.train()
    step = 0
    for ep in range(epochs):
        random.shuffle(triplets)
        for i in range(0, len(triplets), batch_size):
            batch = triplets[i:i + batch_size]
            za = encoder([t.anchor for t in batch])
            zp = encoder([t.positive for t in batch])
            zn = encoder([t.negative for t in batch])
            loss = info_nce(za, zp, zn, temperature=temperature)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % 50 == 0:
                print(f"  epoch {ep} step {step} loss {loss.item():.4f}", flush=True)
            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break

    # --- MANDATORY validation gate (hard stop lives with the caller) ---
    report = gate(encoder, source.heldout_eval())
    print("=== VALIDATION GATE ===")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        print("\nHARD STOP: gate failed. Do not build downstream tools; iterate the encoder.")
    return report
