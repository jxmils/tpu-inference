# Copyright 2025 Google LLC
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

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional

import jax
from jax.sharding import Mesh


@dataclass
class VllmModelWrapperContext:
    kv_caches: List[jax.Array]
    mesh: Mesh
    layer_name_to_kvcache_index: Dict[str, int]
    routing_stats: Optional[List[Optional[dict]]] = None
    routing_stats_next_index: int = 0


_vllm_model_wrapper_context: Optional[VllmModelWrapperContext] = None


def get_vllm_model_wrapper_context() -> VllmModelWrapperContext:
    assert _vllm_model_wrapper_context is not None, (
        "VllmModelWrapperContext is not set. "
        "Please use `set_vllm_model_wrapper_context` to set the VllmModelWrapperContext."
    )
    return _vllm_model_wrapper_context


def record_vllm_routing_stats(layer_idx: Optional[int],
                              stats: dict) -> None:
    if _vllm_model_wrapper_context is None:
        return
    ctx = _vllm_model_wrapper_context
    if ctx.routing_stats is None:
        return
    if layer_idx is None or layer_idx < 0:
        idx = ctx.routing_stats_next_index
        if idx >= len(ctx.routing_stats):
            ctx.routing_stats.append(stats)
        else:
            ctx.routing_stats[idx] = stats
        ctx.routing_stats_next_index = idx + 1
        return
    if layer_idx >= len(ctx.routing_stats):
        ctx.routing_stats.extend([None] * (layer_idx + 1 - len(ctx.routing_stats)))
    ctx.routing_stats[layer_idx] = stats
    ctx.routing_stats_next_index = max(ctx.routing_stats_next_index, layer_idx + 1)


@contextmanager
def set_vllm_model_wrapper_context(
    *,
    kv_caches: List[jax.Array],
    mesh: Mesh,
    layer_name_to_kvcache_index: Dict[str, int] = None,
    routing_stats: Optional[List[Optional[dict]]] = None,
):
    global _vllm_model_wrapper_context
    prev_context = _vllm_model_wrapper_context
    _vllm_model_wrapper_context = VllmModelWrapperContext(
        kv_caches=kv_caches,
        mesh=mesh,
        layer_name_to_kvcache_index=layer_name_to_kvcache_index,
        routing_stats=routing_stats,
        routing_stats_next_index=0,
    )

    try:
        yield
    finally:
        _vllm_model_wrapper_context = prev_context
