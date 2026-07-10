"""Training loop (shared). Fits an encoder on a PairSource, then RUNS THE GATE — always.

The gate call is not optional and not skippable: training a *-BSE and reporting its validation
number are one operation. A run that doesn't end in gate() output is not a finished run.
"""

from __future__ import annotations

import random

import torch
from torch.optim import AdamW

from .objective import info_nce
from .report import Report, hash_checkpoint
from .validate import gate


def train(
    encoder,
    source,
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 2e-5,
    temperature: float = 0.05,
    max_steps: int | None = None,
    checkpoint_path: str | None = None,
    report_path: str | None = None,
) -> Report:
    # TEETH first, before spending a GPU-hour:
    #   (1) admission filter — a domain without an independent structure label may not be built;
    #   (2) circularity guard — held-out eval must not leak into training.
    source.check_admission()  # throws AdmissionError
    source.assert_heldout_disjoint()  # throws CircularityError

    opt = AdamW(encoder.parameters(), lr=lr)
    triplets = list(source.train_triplets())
    print(
        f"[{source.name}] {len(triplets)} training triplets (circularity guard passed)", flush=True
    )
    encoder.train()
    step = 0
    for ep in range(epochs):
        random.shuffle(triplets)
        for i in range(0, len(triplets), batch_size):
            batch = triplets[i : i + batch_size]
            za = encoder([t.anchor for t in batch])
            zp = encoder([t.positive for t in batch])
            zn = encoder([t.negative for t in batch])
            loss = info_nce(za, zp, zn, temperature=temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 50 == 0:
                print(f"  epoch {ep} step {step} loss {loss.item():.4f}", flush=True)
            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break

    # --- MANDATORY validation gate -> signed Report (hard stop lives with the caller) ---
    if checkpoint_path:
        torch.save(encoder.state_dict(), checkpoint_path)
    metrics = gate(encoder, source.heldout_eval(), bar=source.resolve_bar())
    report = Report(
        instance=source.name,
        checkpoint_hash=hash_checkpoint(checkpoint_path) if checkpoint_path else "unsaved",
        thresholds=metrics["thresholds"],
        metrics={k: metrics[k] for k in ("surface_invariance", "fuzz_ratio", "structure_auroc")},
        passed=metrics["passed"],
        bar_source=metrics["bar_source"],
        bar_derivation=metrics["bar_derivation"],
        bar_registered=metrics["bar_registered"],
    )
    print("=== VALIDATION GATE ===")
    print(report.to_json(report_path))
    if not report.passed:
        print("\nHARD STOP: gate failed. Do not build downstream tools; iterate the encoder.")
    return report
