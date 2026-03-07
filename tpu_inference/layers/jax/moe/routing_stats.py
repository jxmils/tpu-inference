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

from typing import Dict, Optional

import jax
import jax.numpy as jnp

from tpu_inference.layers.common.fused_moe_gmm import apply_scoring_fn


def _as_i32(value) -> jax.Array:
    return jnp.asarray(value, dtype=jnp.int32)


def _as_i64(value) -> jax.Array:
    return jnp.asarray(value, dtype=jnp.int64)


def _compute_unique_token_counts(topk_experts: jax.Array,
                                 num_experts: int) -> jax.Array:
    # topk_experts: [T, K]
    one_hot = jax.nn.one_hot(topk_experts,
                             num_classes=num_experts,
                             dtype=jnp.int32)
    token_has_expert = jnp.any(one_hot, axis=1)
    return jnp.sum(token_has_expert, axis=0)


def _assemble_stats(
    *,
    topk_experts: jax.Array,
    topk_scores: jax.Array,
    num_experts: int,
    hidden_size: int,
    bytes_per_element: int,
    layer_idx: Optional[int],
    router_logits: Optional[jax.Array] = None,
    router_probs: Optional[jax.Array] = None,
    include_router_logits: bool = True,
    include_router_probs: bool = False,
    include_unique_token_counts: bool = True,
    routing_is_exact: bool = True,
) -> Dict[str, jax.Array]:
    num_tokens = topk_experts.shape[0]
    flat_experts = topk_experts.reshape(-1)
    expert_counts = jnp.bincount(flat_experts, length=num_experts)
    if include_unique_token_counts:
        unique_token_counts = _compute_unique_token_counts(
            topk_experts, num_experts)
    else:
        unique_token_counts = jnp.zeros((num_experts, ), dtype=jnp.int32)

    bytes_per_token = _as_i64(hidden_size * bytes_per_element)
    expert_counts_i64 = expert_counts.astype(jnp.int64)
    estimated_dispatch_bytes = expert_counts_i64 * bytes_per_token
    estimated_return_bytes = expert_counts_i64 * bytes_per_token

    stats: Dict[str, jax.Array] = {
        "layer_idx": _as_i32(-1 if layer_idx is None else layer_idx),
        "num_tokens": _as_i32(num_tokens),
        "num_experts": _as_i32(num_experts),
        "k": _as_i32(topk_experts.shape[1]),
        "hidden_size": _as_i32(hidden_size),
        "bytes_per_element": _as_i32(bytes_per_element),
        "num_assignments": _as_i32(num_tokens * topk_experts.shape[1]),
        "router_logits_shape": _as_i32((num_tokens, num_experts)),
        "topk_experts": topk_experts,
        "topk_scores": topk_scores,
        "expert_counts": expert_counts,
        "unique_token_counts": unique_token_counts,
        "bytes_per_token": bytes_per_token,
        "estimated_dispatch_bytes": estimated_dispatch_bytes,
        "estimated_return_bytes": estimated_return_bytes,
        "estimated_dispatch_bytes_total": _as_i64(
            jnp.sum(estimated_dispatch_bytes)),
        "estimated_return_bytes_total": _as_i64(
            jnp.sum(estimated_return_bytes)),
        "routing_is_exact": _as_i32(1 if routing_is_exact else 0),
    }

    if include_router_logits and router_logits is not None:
        stats["router_logits"] = router_logits
    if include_router_probs and router_probs is not None:
        stats["router_probs"] = router_probs

    return stats


def compute_routing_stats_from_logits(
    router_logits: jax.Array,
    *,
    k: int,
    scoring_func: str,
    renormalize: bool,
    hidden_size: int,
    bytes_per_element: int,
    layer_idx: Optional[int] = None,
    include_router_logits: bool = True,
    include_router_probs: bool = False,
    include_unique_token_counts: bool = True,
    routing_is_exact: bool = True,
) -> Dict[str, jax.Array]:
    router_probs = apply_scoring_fn(scoring_func, router_logits)
    topk_scores, topk_experts = jax.lax.top_k(router_probs, k)
    if renormalize:
        denom = jnp.sum(topk_scores, axis=-1, keepdims=True)
        topk_scores = jnp.where(denom > 0, topk_scores / denom, topk_scores)

    return _assemble_stats(
        topk_experts=topk_experts,
        topk_scores=topk_scores,
        num_experts=router_logits.shape[1],
        hidden_size=hidden_size,
        bytes_per_element=bytes_per_element,
        layer_idx=layer_idx,
        router_logits=router_logits,
        router_probs=router_probs,
        include_router_logits=include_router_logits,
        include_router_probs=include_router_probs,
        include_unique_token_counts=include_unique_token_counts,
        routing_is_exact=routing_is_exact,
    )


def compute_routing_stats_from_topk(
    topk_scores: jax.Array,
    topk_experts: jax.Array,
    *,
    num_experts: int,
    hidden_size: int,
    bytes_per_element: int,
    layer_idx: Optional[int] = None,
    include_unique_token_counts: bool = True,
    routing_is_exact: bool = True,
) -> Dict[str, jax.Array]:
    return _assemble_stats(
        topk_experts=topk_experts,
        topk_scores=topk_scores,
        num_experts=num_experts,
        hidden_size=hidden_size,
        bytes_per_element=bytes_per_element,
        layer_idx=layer_idx,
        router_logits=None,
        router_probs=None,
        include_router_logits=False,
        include_router_probs=False,
        include_unique_token_counts=include_unique_token_counts,
        routing_is_exact=routing_is_exact,
    )
