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
from typing import Any, Dict, Optional

import numpy as np

from tpu_inference.runner.moe_routing_trace import (RoutingTraceBatchMeta,
                                                    _get_phase_per_req)
from tpu_inference.runner.request_trace import trace_clock_fields


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


class HBMTraceWriter:

    def __init__(self, output_dir: str, rank: int | None = None):
        self.output_dir = output_dir
        self.rank = rank
        _ensure_dir(self._hbm_dir)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        rank_suffix = f"_rank{rank}" if rank is not None else ""
        self.path = os.path.join(self._hbm_dir,
                                 f"hbm_trace{rank_suffix}_{timestamp}.jsonl")

    @property
    def _hbm_dir(self) -> str:
        return os.path.join(self.output_dir, "hbm")

    def write(
        self,
        *,
        event: str,
        hbm_usage: list[tuple[int, int]],
        trace_step: int,
        batch_meta: Optional[RoutingTraceBatchMeta] = None,
        request_seq_ids: Dict[str, Any] | None = None,
        phase_override: Optional[str] = None,
        extra: Dict[str, Any] | None = None,
    ) -> None:
        if not hbm_usage:
            return

        used_bytes = [int(used) for used, _ in hbm_usage]
        limit_bytes = [int(limit) for _, limit in hbm_usage]
        free_bytes = [limit - used for used, limit in hbm_usage]

        record: Dict[str, Any] = {
            "event": event,
            "trace_step": trace_step,
            "rank": -1 if self.rank is None else self.rank,
            "num_devices": len(hbm_usage),
            "per_device_used_bytes": used_bytes,
            "per_device_limit_bytes": limit_bytes,
            "per_device_free_bytes": free_bytes,
            "total_used_bytes": int(sum(used_bytes)),
            "total_limit_bytes": int(sum(limit_bytes)),
            "total_free_bytes": int(sum(free_bytes)),
        }

        if batch_meta is not None:
            req_ids = list(batch_meta.req_ids)
            record["req_ids"] = req_ids
            record["num_reqs"] = len(req_ids)
            record["num_tokens"] = int(np.sum(batch_meta.token_mask))
            record["num_scheduled_tokens"] = int(
                np.sum(batch_meta.num_scheduled_tokens_per_req))
            record["num_prompt_tokens_total"] = int(
                np.sum(batch_meta.num_prompt_tokens))
            record["num_computed_tokens_total"] = int(
                np.sum(batch_meta.num_computed_tokens))

            phase_per_req = _get_phase_per_req(batch_meta.num_computed_tokens,
                                               batch_meta.num_prompt_tokens)
            num_prefill = int(np.sum(phase_per_req))
            num_decode = len(req_ids) - num_prefill
            phase = "mixed"
            if len(req_ids) == 0:
                phase = "unknown"
            elif num_prefill == len(req_ids):
                phase = "prefill"
            elif num_decode == len(req_ids):
                phase = "decode"
            if phase_override is not None:
                phase = phase_override
            record["phase"] = phase
            record["num_prefill_reqs"] = num_prefill
            record["num_decode_reqs"] = num_decode

            if request_seq_ids is not None:
                record["sequence_ids"] = [
                    request_seq_ids.get(req_id, req_id) for req_id in req_ids
                ]

        if extra:
            record.update(extra)

        record.update(trace_clock_fields())

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True))
            f.write("\n")
