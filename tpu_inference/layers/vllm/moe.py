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
import re
import torch
from torchax.interop import jax_view, torch_view
from vllm.model_executor.layers.fused_moe import FusedMoE, FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig

from tpu_inference import envs
from tpu_inference.layers.common.moe import MoEBackend, moe_apply
from tpu_inference.layers.common.process_weights.moe_weights import \
    FusedMoEWeights
from tpu_inference.layers.jax.moe.routing_stats import \
    compute_routing_stats_from_logits
from tpu_inference.logger import init_logger
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    record_vllm_routing_stats

logger = init_logger(__name__)


def _extract_layer_idx(layer: FusedMoE) -> int | None:
    for name in ("layer_idx", "layer_id", "layer_number", "layer_index"):
        value = getattr(layer, name, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    layer_name = getattr(layer, "layer_name", None) or getattr(layer, "name",
                                                               None)
    if isinstance(layer_name, str):
        match = re.search(r"(?:layers?|blocks?)\\.(\\d+)", layer_name)
        if match is not None:
            try:
                return int(match.group(1))
            except (TypeError, ValueError):
                pass
    return None


def _get_layer_attr(layer: FusedMoE,
                    name: str,
                    default=None):
    value = getattr(layer, name, None)
    if value is not None:
        return value
    moe_config = getattr(layer, "moe_config", None)
    return getattr(moe_config, name, default)


def select_moe_backend_from_fused_moe_config(
        moe: FusedMoEConfig) -> MoEBackend:
    """
    Select the MoE backend based on the FusedMoEConfig.

    NOTE (jacobplatin): we don't currently support DENSE_MAT or MEGABLX_GMM
    backends on the vLLM path for now.

    Args:
        moe: The FusedMoEConfig.

    Returns:
        The selected MoE backend.
    """

    if envs.USE_MOE_EP_KERNEL:
        if moe.use_ep:
            logger.info_once("[MoE]: Using fused MoE EP kernel")
            return MoEBackend.FUSED_MOE
        logger.warning_once(
            "USE_MOE_EP_KERNEL=1 but expert parallelism is not "
            "enabled. Falling back to gmm implementation.")

    if moe.use_ep:
        logger.info_once("[MoE]: Using GMM EP kernel")
        return MoEBackend.GMM_EP

    # Use default implementation.
    logger.info_once("[MoE]: Using GMM TP kernel")
    return MoEBackend.GMM_TP


def vllm_moe_apply(layer: FusedMoE, weights: FusedMoEWeights,
                   quant_method_instance: FusedMoEMethodBase, x: torch.Tensor,
                   router_logits: torch.Tensor) -> torch.Tensor:
    """
    Shared function for applying a FusedMoE layer for the TorchAX/vLLM backend.

    Args:
        layer: The FusedMoE layer.
        weights: The FusedMoE weights.
        quant_method_instance: The quantization method instance.
        x: The input tensor.
        router_logits: The router logits.

    Returns:
        The output tensor from the MoE fowrard pass.
    """
    assert isinstance(layer, FusedMoE)
    assert isinstance(quant_method_instance, FusedMoEMethodBase)
    assert isinstance(weights, FusedMoEWeights)

    if envs.moe_routing_stats_enabled():
        try:
            k = _get_layer_attr(layer, "top_k", None)
            if k is None:
                k = _get_layer_attr(layer, "num_experts_per_tok", None)
            if k is None:
                logger.warning_once(
                    "MoE routing stats skipped: missing top_k for layer %s",
                    type(layer))
            else:
                scoring_func = _get_layer_attr(layer, "scoring_func",
                                               "softmax")
                renormalize = bool(
                    _get_layer_attr(layer, "renormalize", False))
                layer_idx = _extract_layer_idx(layer)
                hidden_size = int(x.shape[-1])
                bytes_per_element = int(x.element_size())
                routing_stats = compute_routing_stats_from_logits(
                    router_logits=jax_view(router_logits),
                    k=int(k),
                    scoring_func=str(scoring_func),
                    renormalize=renormalize,
                    hidden_size=hidden_size,
                    bytes_per_element=bytes_per_element,
                    layer_idx=layer_idx,
                    include_router_logits=True,
                    include_router_probs=envs.moe_router_probs_enabled(),
                    include_unique_token_counts=True,
                    routing_is_exact=quant_method_instance.moe_backend
                    in (MoEBackend.GMM_EP, MoEBackend.GMM_TP),
                )
                record_vllm_routing_stats(layer_idx, routing_stats)
        except Exception as exc:
            logger.warning_once("MoE routing stats capture failed: %s", exc)

    return torch_view(
        moe_apply(
            layer=layer,
            x=jax_view(x),
            gating_output=jax_view(router_logits),
            weights=weights,
            moe_backend=quant_method_instance.moe_backend,
            mesh=quant_method_instance.mesh,
            extra_backend_kwargs=quant_method_instance.extra_backend_kwargs,
        ))
