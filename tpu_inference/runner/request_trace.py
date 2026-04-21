# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def trace_clock_fields() -> Dict[str, Any]:
    """High-resolution host clocks for every JSONL line (and routing NPZ co-tagged).

    ``timestamp_*`` is comparable across processes/ranks; ``monotonic_time_ns`` is
    only valid for ordering within a single process.
    """
    return {
        "timestamp_unix": time.time(),
        "timestamp_unix_ns": time.time_ns(),
        "monotonic_time_ns": time.perf_counter_ns(),
        "timestamp_iso_utc": datetime.now(timezone.utc).isoformat(
            timespec="microseconds"),
    }


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def bucket_length(value: int) -> str:
    if value <= 0:
        return "0"
    bounds = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    prev = 0
    for bound in bounds:
        if value <= bound:
            return f"{prev}-{bound}"
        prev = bound + 1
    return f"{bounds[-1] + 1}+"


class RequestTraceWriter:

    def __init__(self,
                 output_dir: str,
                 *,
                 subdir: str = "request",
                 filename_prefix: str = "request_trace",
                 rank: Optional[int] = None):
        self.output_dir = output_dir
        self.subdir = subdir
        self.rank = rank
        _ensure_dir(self._trace_dir)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        rank_suffix = f"_rank{rank}" if rank is not None else ""
        self.path = os.path.join(
            self._trace_dir,
            f"{filename_prefix}{rank_suffix}_{timestamp}.jsonl",
        )

    @property
    def _trace_dir(self) -> str:
        return os.path.join(self.output_dir, self.subdir)

    def write(self, record: Dict[str, Any]) -> None:
        payload = dict(record)
        payload.update(trace_clock_fields())
        if self.rank is not None:
            payload.setdefault("rank", self.rank)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True))
            f.write("\n")
