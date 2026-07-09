"""Signed validation report — turns "falsification order" from convention into enforcement.

A validated instance emits a Report (config + checkpoint hash + pre-registered thresholds +
metrics + PASS/FAIL). Downstream tools call `require_pass(report, checkpoint_hash)` and REFUSE to
construct on anything without a matching PASS. That is what makes "tools import a validated core"
real rather than directory hygiene (design §2.4 flag).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


def hash_checkpoint(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


@dataclass
class Report:
    instance: str
    checkpoint_hash: str
    thresholds: dict
    metrics: dict
    passed: bool
    bar_source: str = "LeBSE"
    extra: dict = field(default_factory=dict)

    def to_json(self, path: str | None = None) -> str:
        s = json.dumps(asdict(self), indent=2, sort_keys=True)
        if path:
            with open(path, "w") as f:
                f.write(s)
        return s

    @property
    def signature(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]


class NotValidatedError(RuntimeError):
    """A tool/probe tried to build on an instance without a matching PASS report."""


def require_pass(report: Report, checkpoint_hash: str) -> Report:
    """Gate for downstream tools. Call before constructing anything on an instance."""
    if not report.passed:
        raise NotValidatedError(
            f"[{report.instance}] report is FAIL — no tools may be built. Iterate the encoder."
        )
    if report.checkpoint_hash != checkpoint_hash:
        raise NotValidatedError(
            f"[{report.instance}] report is for checkpoint {report.checkpoint_hash}, but you loaded "
            f"{checkpoint_hash}. A PASS report only validates the exact checkpoint it was run on."
        )
    return report
