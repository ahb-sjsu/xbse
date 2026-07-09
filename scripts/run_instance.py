#!/usr/bin/env python3
"""Generic instance runner for NRP (self-contained, prints a signed report to stdout).

Usage: python scripts/run_instance.py <instance> [--data DIR] [--steps N]
  reabse  --data <ARC training dir>
  codebse (loads MBPP from HF)
  scibse / mobse also work if their data is reachable.

The report is printed as JSON between REPORT_BEGIN / REPORT_END so `kubectl logs` is enough to
retrieve the verdict — no PVC needed.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from xbse.encoder import BSEEncoder
from xbse.train import train


def build(inst, data):
    if inst == "reabse":
        from xbse.instances import ReaBSEPairSource

        return ReaBSEPairSource(data_dir=data), 192
    if inst == "codebse":
        from xbse.instances import CodeBSEPairSource

        return CodeBSEPairSource(), 256
    if inst == "scibse":
        from xbse.instances import SciBSEPairSource

        return SciBSEPairSource(), 192
    if inst == "mobse":
        from xbse.instances import MoBSEPairSource

        return MoBSEPairSource(), 64
    raise SystemExit(f"unknown instance {inst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("--data", default=None)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()

    src, max_len = build(a.instance, a.data)
    enc = BSEEncoder(base_model="BAAI/bge-m3", max_len=max_len, device="cuda")
    report = train(
        enc,
        src,
        epochs=1,
        batch_size=a.batch,
        lr=2e-5,
        max_steps=a.steps,
        checkpoint_path=f"/tmp/{a.instance}.pt",
    )
    print("REPORT_BEGIN")
    print(report.to_json())
    print("REPORT_END")
    print(f"{a.instance.upper()}_PASSED", report.passed)


if __name__ == "__main__":
    main()
