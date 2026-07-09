"""Domain-adversarial joint training + honest cross-dataset readout.

Extends the shared loop (train.py) with the two rehabilitation levers:
  - InfoNCE on cross-dataset triplets  (JointPairSource — the encoder must align corpora)
  - gradient-reversal domain loss       (adversarial.py — the encoder must not name the corpus)

Reports, alongside the mandatory gate:
  - structure_auroc  : cross-dataset held-out (the honest headline)
  - severity_auroc   : does the general 'severity' polar axis transfer, even if specifics don't
  - domain_acc       : adversary's final corpus-ID accuracy (want it near chance = de-confounded)
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from .adversarial import DomainHead, dann_lambda, polar_severity
from .objective import info_nce
from .report import Report, hash_checkpoint
from .validate import gate


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def train_adversarial(
    encoder,
    source,
    epochs: int = 1,
    batch_size: int = 24,
    lr: float = 2e-5,
    temperature: float = 0.05,
    max_steps: int | None = None,
    max_lambda: float = 1.0,
    checkpoint_path: str | None = None,
    report_path: str | None = None,
) -> Report:
    source.check_admission()
    source.assert_heldout_disjoint()

    tagged = list(source.train_triplets_tagged())  # (Triplet, anchor_domain)
    triplets = [t for t, _d in tagged]
    anchor_dom = [d for _t, d in tagged]
    dom_rows = source.train_rows()  # (text, domain_idx) for final probe only
    n_domains = source.n_domains
    print(
        f"[{source.name}] {len(triplets)} cross-dataset triplets, {n_domains} corpora", flush=True
    )

    dim = encoder.forward(["probe"]).shape[-1]
    domain_head = DomainHead(dim, n_domains).to(encoder.device)
    opt = AdamW(list(encoder.parameters()) + list(domain_head.parameters()), lr=lr)

    total = max_steps or (len(triplets) // batch_size) * epochs
    order = list(range(len(triplets)))
    encoder.train()
    step = 0
    for ep in range(epochs):
        random.shuffle(order)
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            batch = [triplets[j] for j in idx]
            za = encoder([t.anchor for t in batch])
            zp = encoder([t.positive for t in batch])
            zn = encoder([t.negative for t in batch])
            loss = info_nce(za, zp, zn, temperature=temperature)

            # --- adversarial domain term (gradient reversal) on the SAME anchor embeddings ---
            lambd = dann_lambda(step, total, max_lambda)
            dlabels = torch.tensor([anchor_dom[j] for j in idx], device=encoder.device)
            dlogits = domain_head(za, lambd)
            dloss = F.cross_entropy(dlogits, dlabels)
            (loss + dloss).backward()
            opt.step()
            opt.zero_grad()
            step += 1
            if step % 50 == 0:
                dacc = (dlogits.argmax(-1) == dlabels).float().mean().item()
                print(
                    f"  ep{ep} step{step} nce{loss.item():.3f} dom{dloss.item():.3f} "
                    f"lam{lambd:.2f} dacc{dacc:.2f}",
                    flush=True,
                )
            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break

    if checkpoint_path:
        torch.save(encoder.state_dict(), checkpoint_path)

    ev = source.heldout_eval()
    metrics = gate(encoder, ev)

    # --- severity (polar radius) cross-dataset sub-metric ---
    sp = ev["structural_pairs"]
    anchors = [a for a, _b, _l in sp]
    others = [b for _a, b, _l in sp]
    labels = np.array([1 if l else 0 for _a, _b, l in sp])
    za = encoder.encode(anchors)
    zb = encoder.encode(others)
    mean_dir = F.normalize(za.mean(0), dim=-1)
    sev = (polar_severity(zb, mean_dir) - polar_severity(za, mean_dir)).abs().neg().cpu().numpy()
    severity_auroc = _auroc(sev, labels)

    # adversary final accuracy on held rows (chance = 1/n_domains is the de-confounded target)
    dtexts = [t for t, _ in dom_rows[:400]]
    dlabs = np.array([d for _, d in dom_rows[:400]])
    with torch.no_grad():
        zpred = domain_head(encoder.encode(dtexts), 0.0).argmax(-1).cpu().numpy()
    domain_acc = float((zpred == dlabs).mean())

    report = Report(
        instance=source.name,
        checkpoint_hash=hash_checkpoint(checkpoint_path) if checkpoint_path else "unsaved",
        thresholds=metrics["thresholds"],
        metrics={
            **{k: metrics[k] for k in ("surface_invariance", "fuzz_ratio", "structure_auroc")},
            "severity_auroc": round(float(severity_auroc), 4),
            "domain_acc": round(domain_acc, 4),
            "domain_chance": round(1.0 / n_domains, 4),
        },
        passed=metrics["passed"],
    )
    print("=== CROSS-DATASET GATE (joint + adversarial) ===")
    print(report.to_json(report_path))
    print(
        f"severity_auroc={severity_auroc:.4f}  domain_acc={domain_acc:.4f} "
        f"(chance={1.0/n_domains:.3f})"
    )
    return report
