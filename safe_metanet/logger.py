"""Structured logging for the Safe MetaNet loop.

Each test-time step is recorded as an :class:`UpdateRecord`.  The
:class:`SafeMetaNetLogger` collects these records and can print a summary
or export to a list of dicts for downstream analysis.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------

@dataclass
class UpdateRecord:
    """One test-time step record."""
    step: int
    sample_idx: int
    action: str               # "accept" | "reject"
    metric_before: float
    metric_after: float
    scope: List[str]          # layer names currently editable
    scope_expanded: bool      # True if scope was expanded this step
    gradient_steps: int       # number of gradient steps taken


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class SafeMetaNetLogger:
    """Collects and displays Safe MetaNet loop records.

    Parameters
    ----------
    level:
        Python :mod:`logging` level string, e.g. ``"INFO"`` or ``"DEBUG"``.
    name:
        Logger name; defaults to ``"safe_metanet"``.
    """

    def __init__(self, level: str = "INFO", name: str = "safe_metanet") -> None:
        self.records: List[UpdateRecord] = []
        self._log = logging.getLogger(name)
        if not self._log.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "[%(name)s %(levelname)s] %(message)s"
            )
            handler.setFormatter(formatter)
            self._log.addHandler(handler)
        self._log.setLevel(getattr(logging, level.upper(), logging.INFO))

    # ------------------------------------------------------------------
    def log_update(self, record: UpdateRecord) -> None:
        """Append *record* and emit a log line."""
        self.records.append(record)
        delta = record.metric_after - record.metric_before
        expand_tag = " [SCOPE EXPANDED]" if record.scope_expanded else ""
        msg = (
            f"step={record.step:4d}  sample={record.sample_idx:4d}  "
            f"{record.action.upper():6s}  "
            f"metric {record.metric_before:.4f} → {record.metric_after:.4f} "
            f"(Δ={delta:+.4f})  "
            f"scope={record.scope}{expand_tag}"
        )
        if record.action == "accept":
            self._log.info(msg)
        else:
            self._log.debug(msg)

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Return a human-readable summary string."""
        if not self.records:
            return "No records."
        n_accept = sum(1 for r in self.records if r.action == "accept")
        n_reject = len(self.records) - n_accept
        n_expand = sum(1 for r in self.records if r.scope_expanded)
        metrics_before = [r.metric_before for r in self.records]
        metrics_after = [r.metric_after for r in self.records]
        lines = [
            f"Total steps : {len(self.records)}",
            f"  Accepted  : {n_accept}",
            f"  Rejected  : {n_reject}",
            f"  Expanded  : {n_expand}",
            f"Metric (first): {metrics_before[0]:.4f}",
            f"Metric (last) : {metrics_after[-1]:.4f}",
        ]
        return "\n".join(lines)

    def to_dicts(self) -> List[dict]:
        """Return all records as a list of plain dicts."""
        return [asdict(r) for r in self.records]

    def reset(self) -> None:
        """Clear all recorded history."""
        self.records.clear()
