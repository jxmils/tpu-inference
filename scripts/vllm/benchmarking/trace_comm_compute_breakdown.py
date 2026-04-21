#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Summarize communication vs compute time from a Perfetto / Chrome trace JSON.

Consumes trace JSON (or .json.gz) with ``traceEvents`` and emits one CSV row per
``execute_model`` step with comm_time_ms, compute_time_ms, overlap_ms.

Logic lives in ``trace_plotter/perfetto_comm_compute.py`` (shared with plot_raw_traces).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[4]
if (_REPO_ROOT / "trace_plotter" / "perfetto_comm_compute.py").is_file():
    sys.path.insert(0, str(_REPO_ROOT / "trace_plotter"))

from perfetto_comm_compute import summarize_trace_to_dataframe  # noqa: E402


def summarize(trace_path: Path) -> list[dict[str, Any]]:
    df = summarize_trace_to_dataframe(trace_path)
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        rows.append(
            {
                "step_index": int(r["step_index"]),
                "step_name": str(r["step_name"]),
                "step_start_us": f'{float(r["step_start_us"]):.3f}',
                "step_duration_ms": f'{float(r["step_duration_ms"]):.6f}',
                "comm_time_ms": f'{float(r["comm_time_ms"]):.6f}',
                "compute_time_ms": f'{float(r["compute_time_ms"]):.6f}',
                "overlap_ms": f'{float(r["overlap_ms"]):.6f}',
                "total_op_time_ms": f'{float(r["total_op_time_ms"]):.6f}',
            }
        )
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--trace",
        required=True,
        help="Path to perfetto_trace.json, Chrome trace .json, or .json.gz",
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
