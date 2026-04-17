#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Poll vLLM ``/metrics`` and append selected gauges to a CSV file.

vLLM exposes scheduler/engine statistics as Prometheus text on the OpenAI
HTTP server (see https://docs.vllm.ai/en/latest/usage/metrics.html). This script
samples two gauges commonly used for capacity dashboards:

* ``vllm:kv_cache_usage_perc`` — fraction of KV blocks in use (0–1).
* ``vllm:num_requests_running`` — requests in model execution batches.

For "active sequences" style monitoring, ``vllm:num_requests_running`` plus
``vllm:num_requests_waiting`` are the documented gauges; there is no separate
``num_active_sequences`` name in the public metrics table.

Example::

    python3 scripts/vllm/prometheus_metrics_poll_csv.py \\
      --url http://127.0.0.1:8000/metrics \\
      --output kv_stats.csv \\
      --interval-s 5
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import urllib.error
import urllib.request


# Sample line: vllm:kv_cache_usage_perc{model_name="x"} 0.12
# or without labels: vllm:kv_cache_usage_perc 0.12
_GAUGE_LINE = re.compile(
    r"^(?P<name>vllm:(?:kv_cache_usage_perc|num_requests_running))"
    r"(?:\{[^}]*\})?\s+(?P<value>-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*$")


def parse_gauges(metrics_body: str) -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "vllm:kv_cache_usage_perc": None,
        "vllm:num_requests_running": None,
    }
    for line in metrics_body.splitlines():
        m = _GAUGE_LINE.match(line.strip())
        if not m:
            continue
        name = m.group("name")
        if name in out:
            out[name] = float(m.group("value"))
    return out


def fetch_metrics(url: str, timeout_s: float) -> str:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read().decode("utf-8", errors="replace")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--url",
        default="http://127.0.0.1:8000/metrics",
        help="Full URL to the Prometheus text endpoint.",
    )
    p.add_argument(
        "--output",
        "-o",
        required=True,
        help="CSV path (created with header if missing).",
    )
    p.add_argument(
        "--interval-s",
        type=float,
        default=5.0,
        help="Seconds between samples (default: 5).",
    )
    p.add_argument(
        "--timeout-s",
        type=float,
        default=10.0,
        help="HTTP timeout per request.",
    )
    args = p.parse_args()
    fieldnames = [
        "unix_time",
        "kv_cache_usage_perc",
        "num_requests_running",
    ]
    import os

    try:
        new_file = not os.path.exists(args.output)
        while True:
            t = time.time()
            try:
                body = fetch_metrics(args.url, args.timeout_s)
            except (urllib.error.URLError, OSError) as e:
                print(f"metrics fetch failed: {e}", file=sys.stderr)
                vals = {
                    "vllm:kv_cache_usage_perc": None,
                    "vllm:num_requests_running": None,
                }
            else:
                vals = parse_gauges(body)
            row = {
                "unix_time": f"{t:.3f}",
                "kv_cache_usage_perc": vals["vllm:kv_cache_usage_perc"],
                "num_requests_running": vals["vllm:num_requests_running"],
            }
            with open(args.output, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if new_file:
                    w.writeheader()
                    new_file = False
                w.writerow(row)
            time.sleep(max(0.1, args.interval_s))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
