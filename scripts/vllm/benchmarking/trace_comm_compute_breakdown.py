#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Summarize communication vs compute time from a Perfetto trace.

This script consumes a JAX profiler Perfetto trace JSON (or .json.gz) and
emits one CSV row per `execute_model` step with:
  - comm_time_ms
  - compute_time_ms
  - overlap_ms

The script is intended for traces collected with:
  - `jax.profiler.start_trace(..., create_perfetto_trace=True)`, or
  - `JAX_PROFILER_CREATE_PERFETTO_TRACE=1` in tpu-inference.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_STEP_RE = re.compile(r"^execute_model:")
_COMM_RE = re.compile(
    r"(all[-_ ]?reduce|all[-_ ]?to[-_ ]?all|reduce[-_ ]?scatter|"
    r"collective[-_ ]?permute|send|recv|cross[-_ ]?replica|cross[-_ ]?partition)",
    re.IGNORECASE,
)
_IGNORE_NAME_RE = re.compile(r"^process_name$|^thread_name$|^TraceAnnotation")


@dataclass(frozen=True)
class Event:
    name: str
    cat: str
    ts: float
    te: float


def _load_json(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_complete_events(trace: dict[str, Any]) -> list[Event]:
    out: list[Event] = []
    for ev in trace.get("traceEvents", []):
        if not isinstance(ev, dict):
            continue
        if ev.get("ph") != "X":
            continue
        if "ts" not in ev or "dur" not in ev:
            continue
        name = str(ev.get("name", ""))
        if _IGNORE_NAME_RE.search(name):
            continue
        ts = float(ev["ts"])
        te = ts + max(0.0, float(ev["dur"]))
        out.append(Event(name=name, cat=str(ev.get("cat", "")), ts=ts, te=te))
    out.sort(key=lambda e: (e.ts, e.te))
    return out


def _clip(e: Event, start: float, end: float) -> tuple[float, float] | None:
    s = max(e.ts, start)
    t = min(e.te, end)
    if t <= s:
        return None
    return (s, t)


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _sum_intervals(intervals: list[tuple[float, float]]) -> float:
    return sum(e - s for s, e in intervals)


def _intersect_intervals(
    a: list[tuple[float, float]], b: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    i = j = 0
    out: list[tuple[float, float]] = []
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if e > s:
            out.append((s, e))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return out


def _is_comm_event(ev: Event) -> bool:
    key = f"{ev.cat} {ev.name}"
    return bool(_COMM_RE.search(key))


def summarize(trace_path: Path) -> list[dict[str, Any]]:
    trace = _load_json(trace_path)
    events = _extract_complete_events(trace)
    steps = [e for e in events if _STEP_RE.match(e.name)]
    rows: list[dict[str, Any]] = []
    for i, step in enumerate(steps):
        step_start, step_end = step.ts, step.te
        all_ops: list[tuple[float, float]] = []
        comm_ops: list[tuple[float, float]] = []
        compute_ops: list[tuple[float, float]] = []
        for ev in events:
            if ev is step:
                continue
            if ev.te <= step_start or ev.ts >= step_end:
                continue
            c = _clip(ev, step_start, step_end)
            if c is None:
                continue
            all_ops.append(c)
            if _is_comm_event(ev):
                comm_ops.append(c)
            else:
                compute_ops.append(c)

        all_m = _merge_intervals(all_ops)
        comm_m = _merge_intervals(comm_ops)
        comp_m = _merge_intervals(compute_ops)
        ov_m = _intersect_intervals(comm_m, comp_m)

        total_op_us = _sum_intervals(all_m)
        comm_us = _sum_intervals(comm_m)
        comp_us = _sum_intervals(comp_m)
        overlap_us = _sum_intervals(ov_m)
        rows.append(
            {
                "step_index": i,
                "step_name": step.name,
                "step_start_us": f"{step_start:.3f}",
                "step_duration_ms": f"{(step_end - step_start) / 1000.0:.6f}",
                "comm_time_ms": f"{comm_us / 1000.0:.6f}",
                "compute_time_ms": f"{comp_us / 1000.0:.6f}",
                "overlap_ms": f"{overlap_us / 1000.0:.6f}",
                "total_op_time_ms": f"{total_op_us / 1000.0:.6f}",
            }
        )
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--trace",
        required=True,
        help="Path to perfetto_trace.json or perfetto_trace.json.gz",
    )
    p.add_argument(
        "--output",
        "-o",
        required=True,
        help="CSV output path.",
    )
    args = p.parse_args()
    rows = summarize(Path(args.trace))
    fieldnames = [
        "step_index",
        "step_name",
        "step_start_us",
        "step_duration_ms",
        "comm_time_ms",
        "compute_time_ms",
        "overlap_ms",
        "total_op_time_ms",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"Wrote {len(rows)} step rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
