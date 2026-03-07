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
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import jax
import numpy as np

from tpu_inference.logger import init_logger

logger = init_logger(__name__)


@dataclass
class RoutingTraceBatchMeta:
    req_ids: List[str]
    token_req_indices: np.ndarray  # shape [T_padded], req index or -1 for pad
    token_positions: np.ndarray  # shape [T_padded], absolute token positions
    token_mask: np.ndarray  # shape [T_padded], True for real tokens
    num_scheduled_tokens_per_req: np.ndarray  # shape [R]
    num_prompt_tokens: np.ndarray  # shape [R]
    num_computed_tokens: np.ndarray  # shape [R]


def _to_numpy_pytree(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_numpy_pytree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_numpy_pytree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_numpy_pytree(v) for v in obj)
    if isinstance(obj, jax.Array):
        return np.asarray(jax.device_get(obj))
    return obj


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _unique_token_counts(topk_experts: np.ndarray,
                         num_experts: int) -> np.ndarray:
    counts = np.zeros(num_experts, dtype=np.int32)
    for row in topk_experts:
        counts[np.unique(row)] += 1
    return counts


def _get_phase_per_req(num_computed_tokens: np.ndarray,
                       num_prompt_tokens: np.ndarray) -> np.ndarray:
    # Prefill if we haven't computed all prompt tokens yet.
    return (num_computed_tokens < num_prompt_tokens)


class RoutingTraceWriter:

    def __init__(self,
                 output_dir: str,
                 save_raw: bool = True,
                 save_summary: bool = True):
        self.output_dir = output_dir
        self.save_raw = save_raw
        self.save_summary = save_summary
        _ensure_dir(self.output_dir)
        _ensure_dir(self._raw_dir)
        _ensure_dir(self._summary_dir)

    @property
    def _raw_dir(self) -> str:
        return os.path.join(self.output_dir, "raw")

    @property
    def _summary_dir(self) -> str:
        return os.path.join(self.output_dir, "summary")

    def write(self,
              routing_stats: Dict[str, Any],
              batch_meta: RoutingTraceBatchMeta,
              *,
              trace_step: int,
              batch_id: int,
              rank: int | None = None,
              request_seq_ids: Dict[str, Any] | None = None,
              phase_override: Optional[str] = None) -> None:
        per_layer = routing_stats.get("per_layer", [])
        aggregate = routing_stats.get("aggregate", {})
        per_layer_np = _to_numpy_pytree(per_layer)
        aggregate_np = _to_numpy_pytree(aggregate)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        rank_suffix = f"_rank{rank}" if rank is not None else ""
        suffix = f"step{trace_step}_batch{batch_id}{rank_suffix}_{timestamp}"

        if self.save_raw:
            self._write_raw_npz(per_layer_np, aggregate_np, batch_meta, suffix,
                                trace_step, batch_id, rank)
        if self.save_summary:
            self._write_summary_jsonl(per_layer_np, batch_meta, suffix,
                                      request_seq_ids, phase_override,
                                      trace_step, batch_id, rank)

        self._run_invariants(per_layer_np, aggregate_np, batch_meta)

    def _write_raw_npz(self, per_layer: List[dict | None],
                       aggregate: Dict[str, Any],
                       batch_meta: RoutingTraceBatchMeta, suffix: str,
                       trace_step: int, batch_id: int,
                       rank: int | None) -> None:
        raw_path = os.path.join(self._raw_dir,
                                f"routing_raw_{suffix}.npz")

        raw: Dict[str, Any] = {
            "req_ids":
            np.array(batch_meta.req_ids, dtype=object),
            "token_req_indices":
            batch_meta.token_req_indices.astype(np.int32),
            "token_positions":
            batch_meta.token_positions.astype(np.int32),
            "token_mask":
            batch_meta.token_mask.astype(np.bool_),
            "num_scheduled_tokens_per_req":
            batch_meta.num_scheduled_tokens_per_req.astype(np.int32),
            "num_prompt_tokens":
            batch_meta.num_prompt_tokens.astype(np.int32),
            "num_computed_tokens":
            batch_meta.num_computed_tokens.astype(np.int32),
            "trace_step":
            np.asarray([trace_step], dtype=np.int64),
            "batch_id":
            np.asarray([batch_id], dtype=np.int64),
            "rank":
            np.asarray([-1 if rank is None else rank], dtype=np.int32),
        }

        if aggregate:
            for key, value in aggregate.items():
                raw[f"aggregate_{key}"] = value

        for layer_idx, layer_stats in enumerate(per_layer):
            if layer_stats is None:
                continue
            for key, value in layer_stats.items():
                raw[f"layer_{layer_idx}_{key}"] = value

        np.savez_compressed(raw_path, **raw)

    def _write_summary_jsonl(self, per_layer: List[dict | None],
                             batch_meta: RoutingTraceBatchMeta, suffix: str,
                             request_seq_ids: Dict[str, Any] | None,
                             phase_override: Optional[str],
                             trace_step: int, batch_id: int,
                             rank: int | None) -> None:
        summary_path = os.path.join(self._summary_dir,
                                    f"routing_summary_{suffix}.jsonl")

        req_ids = batch_meta.req_ids
        token_req_indices = batch_meta.token_req_indices
        token_positions = batch_meta.token_positions
        token_mask = batch_meta.token_mask
        num_scheduled_tokens_per_req = batch_meta.num_scheduled_tokens_per_req
        num_prompt_tokens = batch_meta.num_prompt_tokens
        num_computed_tokens = batch_meta.num_computed_tokens
        phase_per_req = _get_phase_per_req(num_computed_tokens,
                                           num_prompt_tokens)

        if request_seq_ids is None:
            request_seq_ids = {}

        with open(summary_path, "w", encoding="utf-8") as f:
            for layer_idx, layer_stats in enumerate(per_layer):
                if layer_stats is None:
                    continue

                routing_is_exact = int(layer_stats["routing_is_exact"])
                num_experts = int(layer_stats["num_experts"])
                k = int(layer_stats["k"])
                bytes_per_token = int(layer_stats["bytes_per_token"])

                topk_experts = layer_stats["topk_experts"]
                topk_scores = layer_stats["topk_scores"]

                if topk_experts.shape[0] != token_req_indices.shape[0]:
                    logger.warning(
                        "Routing trace token length mismatch: %s vs %s",
                        topk_experts.shape[0], token_req_indices.shape[0])

                valid_mask = token_mask
                topk_experts = topk_experts[valid_mask]
                topk_scores = topk_scores[valid_mask]
                token_req_indices_valid = token_req_indices[valid_mask]
                token_positions_valid = token_positions[valid_mask]

                for req_idx, req_id in enumerate(req_ids):
                    req_mask = token_req_indices_valid == req_idx
                    if not np.any(req_mask):
                        continue
                    req_topk = topk_experts[req_mask]
                    req_scores = topk_scores[req_mask]

                    expert_counts = np.bincount(req_topk.reshape(-1),
                                                minlength=num_experts)
                    unique_counts = _unique_token_counts(req_topk, num_experts)
                    dispatch_bytes = expert_counts.astype(
                        np.int64) * bytes_per_token
                    return_bytes = dispatch_bytes

                    is_prefill = bool(phase_per_req[req_idx])
                    phase = "prefill" if is_prefill else "decode"
                    if phase_override is not None:
                        phase = phase_override

                    token_positions_req = token_positions_valid[req_mask]
                    decode_step = None
                    token_position = None
                    prefill_token_count = None
                    token_start = None
                    token_end = None
                    if is_prefill:
                        prefill_token_count = int(
                            num_scheduled_tokens_per_req[req_idx])
                        token_start = int(token_positions_req.min())
                        token_end = int(token_positions_req.max())
                    else:
                        token_position = int(token_positions_req[0])
                        decode_step = int(token_position -
                                          num_prompt_tokens[req_idx])

                    seq_id = request_seq_ids.get(req_id, req_id)

                    for expert_id in range(num_experts):
                        record = {
                            "trace_step":
                            trace_step,
                            "batch_id":
                            batch_id,
                            "rank":
                            rank,
                            "phase":
                            phase,
                            "request_id":
                            req_id,
                            "sequence_id":
                            seq_id,
                            "batch_index":
                            int(req_idx),
                            "layer_idx":
                            int(layer_idx),
                            "expert_id":
                            int(expert_id),
                            "decode_step":
                            decode_step,
                            "token_position":
                            token_position,
                            "prefill_token_count":
                            prefill_token_count,
                            "token_start":
                            token_start,
                            "token_end":
                            token_end,
                            "num_tokens":
                            int(req_topk.shape[0]),
                            "num_experts":
                            num_experts,
                            "k":
                            k,
                            "expert_count":
                            int(expert_counts[expert_id]),
                            "unique_token_count":
                            int(unique_counts[expert_id]),
                            "estimated_dispatch_bytes":
                            int(dispatch_bytes[expert_id]),
                            "estimated_return_bytes":
                            int(return_bytes[expert_id]),
                            "routing_is_exact":
                            routing_is_exact,
                        }
                        f.write(json.dumps(record) + "\n")

    def _run_invariants(self, per_layer: List[dict | None],
                        aggregate: Dict[str, Any],
                        batch_meta: RoutingTraceBatchMeta) -> None:
        token_mask = batch_meta.token_mask
        total_tokens = int(token_mask.sum())

        for layer_stats in per_layer:
            if layer_stats is None:
                continue
            routing_is_exact = int(layer_stats["routing_is_exact"]) == 1
            k = int(layer_stats["k"])
            bytes_per_token = int(layer_stats["bytes_per_token"])
            num_tokens_stats = int(layer_stats["num_tokens"])
            if num_tokens_stats != total_tokens:
                logger.warning(
                    "Invariant note: stats num_tokens=%s differs from token_mask sum=%s",
                    num_tokens_stats,
                    total_tokens,
                )
            topk_experts = layer_stats["topk_experts"][token_mask]

            expert_counts = np.bincount(topk_experts.reshape(-1),
                                        minlength=int(
                                            layer_stats["num_experts"]))
            if routing_is_exact:
                if expert_counts.sum() != total_tokens * k:
                    logger.warning(
                        "Invariant failed: sum(expert_counts)=%s != num_tokens*k=%s",
                        expert_counts.sum(),
                        total_tokens * k,
                    )
                if "expert_counts" in layer_stats:
                    full_counts = layer_stats["expert_counts"]
                    if full_counts.sum() != num_tokens_stats * k:
                        logger.warning(
                            "Invariant failed: full expert_counts sum %s != num_tokens*k=%s",
                            full_counts.sum(),
                            num_tokens_stats * k,
                        )

            expected_bytes = expert_counts.astype(
                np.int64) * bytes_per_token
            expected_total = int(expected_bytes.sum())
            if "estimated_dispatch_bytes_total" in layer_stats:
                dispatch_total = int(layer_stats["estimated_dispatch_bytes_total"])
                if dispatch_total != expected_total:
                    logger.warning(
                        "Invariant failed: dispatch bytes total %s != expected %s",
                        dispatch_total,
                        expected_total,
                    )
            if "estimated_return_bytes_total" in layer_stats:
                return_total = int(layer_stats["estimated_return_bytes_total"])
                if return_total != expected_total:
                    logger.warning(
                        "Invariant failed: return bytes total %s != expected %s",
                        return_total,
                        expected_total,
                    )

        if aggregate:
            try:
                per_layer_counts = [
                    layer_stats["expert_counts"] for layer_stats in per_layer
                    if layer_stats is not None
                ]
                if per_layer_counts:
                    summed = np.sum(per_layer_counts, axis=0)
                    if not np.array_equal(summed,
                                          aggregate["expert_counts"]):
                        logger.warning(
                            "Invariant failed: aggregate counts do not match sum of layer counts."
                        )
            except Exception as exc:
                logger.warning("Invariant check failed: %s", exc)
