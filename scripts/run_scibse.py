#!/usr/bin/env python3
"""Train + validate SciBSE (first framework-proving instance). Runs on Atlas (GPU + arxiv DB).

Fails loud before training if admission or circularity is violated; ends in a signed Report and a
HARD STOP if the gate is not cleared. Nothing downstream is built here — this only answers
"is SciBSE a real instrument?".
"""

import sys

sys.path.insert(0, "/home/claude/xbse/src")

from xbse.encoder import BSEEncoder
from xbse.instances import SCIBSE_CONFIG, SciBSEPairSource
from xbse.train import train


def main():
    src = SciBSEPairSource(max_papers=SCIBSE_CONFIG["max_papers"])
    enc = BSEEncoder(base_model=SCIBSE_CONFIG["base_model"], max_len=192, device="cuda")
    report = train(
        enc,
        src,
        epochs=1,
        batch_size=32,
        lr=2e-5,
        max_steps=800,
        checkpoint_path="/home/claude/scibse.pt",
        report_path="/home/claude/scibse_report.json",
    )
    print("SCIBSE_PASSED", report.passed)


if __name__ == "__main__":
    main()
