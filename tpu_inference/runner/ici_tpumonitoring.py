# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project

"""Optional host-side ICI / TPU metrics via ``libtpu.sdk.tpumonitoring``.

When ``libtpu`` exposes the monitoring SDK on your VM, snapshots can be taken
immediately before and after ``model_fn()`` and merged into ``step_summary``
JSONL (same row as ``comm_bytes_proxy_*``). If the import fails or a metric is
unsupported, capture is a no-op.

See https://cloud.google.com/tpu/docs/tpu-monitoring-library for supported
metric names on your chip / runtime version. Defaults target
``ici_flits_tx`` / ``ici_flits_rx``; on some v6e libtpu builds those stay
empty in-process while ``ici_link_health`` may appear—use host
``tpu-info`` / ``ici_counters.sh`` for flits when tpumonitoring does not
populate them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import logging

logger = logging.getLogger(__name__)

# ``ici_link_health`` returns strings like ``tray1.chip3.ici0.int: 0`` (see metric description()).
_ICI_LINK_HEALTH_SCORE = re.compile(r":\s*(\d+)\s*$")


def _read_metric_raw(m: Any) -> Any:
    """Return the raw series from a tpumonitoring metric object.

    Google docs show both ``metric.data()`` and ``property``-style ``metric.data``.
    Calling ``data()`` when ``data`` is already a list raises ``TypeError``.
    """
    d = getattr(m, "data", None)
    if d is None:
        return None
    if callable(d):
        return d()
    return d


def _metric_values_as_floats(metric_name: str, raw: List[Any]) -> List[float]:
    """Normalize tpumonitoring ``data()`` rows to floats for delta math.

    ``ici_link_health`` is a list of ``location: score`` strings, not plain floats.
    """
    out: List[float] = []
    if metric_name == "ici_link_health":
        for x in raw:
            s = str(x).strip()
            m = _ICI_LINK_HEALTH_SCORE.search(s)
            if not m:
                out.append(float("nan"))
                continue
            try:
                out.append(float(m.group(1)))
            except ValueError:
                out.append(float("nan"))
        return out
    for x in raw:
        try:
            out.append(float(str(x)))
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
    _empty_all_metrics_logged: ClassVar[bool] = False

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
        supported: Tuple[str, ...] = ()
        try:
            names = tpm.list_supported_metrics()
            if names is not None:
                supported = tuple(str(x) for x in names)
        except Exception as exc:  # noqa: BLE001
            logger.debug("ICI tpumonitoring: list_supported_metrics failed: %s",
                         exc)
        if supported:
            unknown = [n for n in metric_names if n not in supported]
            if unknown:
                logger.warning(
                    "ICI tpumonitoring: metric name(s) not reported by "
                    "list_supported_metrics() on this runtime (will still call "
                    "get_metric): %s",
                    ", ".join(unknown))
        return cls(metric_names=metric_names, _tpumonitoring=tpm)

    def snapshot(self) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {}
        for key in self.metric_names:
            try:
                m = self._tpumonitoring.get_metric(key)
                raw = _read_metric_raw(m)
                if raw is None:
                    out[key] = []
                    continue
                out[key] = _metric_values_as_floats(key, list(raw))
            except Exception as exc:  # noqa: BLE001 — best-effort per metric
                logger.warning("ICI tpumonitoring: get_metric(%r) failed: %s",
                               key, exc)
                out[key] = []
        if (out and all(len(v) == 0 for v in out.values())
                and not ICITpuMonitoringCapture._empty_all_metrics_logged):
            ICITpuMonitoringCapture._empty_all_metrics_logged = True
            logger.warning(
                "ICI tpumonitoring: all configured metrics returned empty "
                "series. Often: wrong metric names for this chip / libtpu build "
                "(check list_supported_metrics), Docker without host-equivalent "
                "counter visibility, counters not populated until a sampling "
                "window, or no active TPU client in this process (tpumonitoring "
                "from a cold REPL can stay empty until JAX attaches devices).")
        return out
