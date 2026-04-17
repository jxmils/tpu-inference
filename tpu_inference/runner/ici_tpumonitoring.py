# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project

"""Optional host-side ICI / TPU metrics via ``libtpu.sdk.tpumonitoring``.

When ``libtpu`` exposes the monitoring SDK on your VM, snapshots can be taken
immediately before and after ``model_fn()`` and merged into ``step_summary``
JSONL (same row as ``comm_bytes_proxy_*``). If the import fails or a metric is
unsupported, capture is a no-op.

See https://cloud.google.com/tpu/docs/tpu-monitoring-library for supported
metric names on your chip / runtime version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import logging

logger = logging.getLogger(__name__)


def _safe_float_list(raw: List[str]) -> List[float]:
    out: List[float] = []
    for x in raw:
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return out


def metric_delta(
    before: Dict[str, List[float]],
    after: Dict[str, List[float]],
) -> Dict[str, Any]:
    """Per-metric per-chip deltas and sums (after - before)."""
    out: Dict[str, Any] = {}
    for key in after:
        b = before.get(key)
        a = after.get(key)
        if b is None or a is None or len(b) != len(a):
            out[key] = {
                "delta_per_chip": None,
                "delta_sum": None,
                "reason": "missing_or_len_mismatch",
            }
            continue
        deltas = [float(ai) - float(bi) for bi, ai in zip(b, a)]
        out[key] = {
            "delta_per_chip": deltas,
            "delta_sum": float(sum(deltas)),
        }
    return {"ici_hw_delta": out}


@dataclass
class ICITpuMonitoringCapture:
    """Thin wrapper around ``libtpu.sdk.tpumonitoring`` (optional dependency)."""

    metric_names: Tuple[str, ...]
    _tpumonitoring: Any

    @classmethod
    def try_create(
        cls,
        metric_names: Tuple[str, ...],
    ) -> Optional["ICITpuMonitoringCapture"]:
        if not metric_names:
            return None
        try:
            from libtpu.sdk import tpumonitoring as tpm  # type: ignore[import-not-found]
        except ImportError:
            logger.info(
                "ICI tpumonitoring: libtpu.sdk.tpumonitoring not importable; "
                "skipping hardware counter capture.")
            return None
        return cls(metric_names=metric_names, _tpumonitoring=tpm)

    def snapshot(self) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {}
        for key in self.metric_names:
            try:
                m = self._tpumonitoring.get_metric(key)
                raw = m.data()
                if raw is None:
                    out[key] = []
                    continue
                out[key] = _safe_float_list(list(raw))
            except Exception as exc:  # noqa: BLE001 — best-effort per metric
                logger.debug("ICI tpumonitoring: get_metric(%r) failed: %s",
                             key, exc)
                out[key] = []
        return out
