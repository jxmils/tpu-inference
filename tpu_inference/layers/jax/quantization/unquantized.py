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

from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference import envs
from tpu_inference.layers.common.moe import MoEBackend, moe_apply
from tpu_inference.profiler_trace import tpu_trace_region
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, UnfusedMoEWeights)
from tpu_inference.layers.common.quantization import unquantized as jax_common
from tpu_inference.layers.common.quantization.configs import QuantLinearConfig
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.linear import JaxEinsum
from tpu_inference.layers.jax.moe.moe import JaxMoE
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.layers.jax.quantization.configs import QuantizationConfig
from tpu_inference.models.jax.utils.weight_utils import shard_put
from tpu_inference.layers.jax.moe.routing_stats import (
    compute_routing_stats_from_logits,
    compute_routing_stats_from_topk,
)


class UnquantizedLinearMethod(QuantizeMethodBase,
                              jax_common.UnquantizedLinearMethod):
    """Unquantized method for JAX Linear layer.
    """

    def apply_jax(self, layer: JaxModule, x: jax.Array) -> jax.Array:
        assert isinstance(layer, JaxEinsum)

        with jax.named_scope(layer._get_name()):
            if self.linear_config.fuse_matmuls:
                out = self._apply_fused(
                    x,
                    layer.weight.value,
                    layer.bias.value if layer.bias else None,
                    einsum_str=layer.einsum_str)
            else:
                raise NotImplementedError(
                    "Non-fused matmuls not implemented yet.")

        return out


class UnquantizedFusedMoEMethod(QuantizeMethodBase):
    """
    Unquantized method for JAXMoE layer.

    TODO (jacobplatin): support weight loading -- currently, model-dependent.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_backend_kwargs = {}

    def process_weights_after_loading(self, layer: JaxMoE, *args,
                                      **kwargs) -> bool:
        """
        Process weights after loading.

        Please see https://github.com/vllm-project/tpu-inference/blob/bb1a88/tpu_inference/layers/common/moe.py#L39
        for more information on the expected weights per MoE backend.

        Args:
            layer: The layer to process.
        """
        if layer.moe_backend == MoEBackend.FUSED_MOE:
            if layer.edf_sharding:
                e2df_sharding = (layer.edf_sharding[0], None,
                                 layer.edf_sharding[1], layer.edf_sharding[2])
            # fuse the weights into w13: [Gate, Up]
            w_gate = layer.kernel_gating_EDF.value
            w_up = layer.kernel_up_proj_EDF.value

            # stack to create a 4d array
            w13_val = jnp.stack([w_gate, w_up], axis=1)

            layer.kernel_gating_upproj_E2DF = nnx.Param(
                shard_put(w13_val, shardings=e2df_sharding))

            del layer.kernel_gating_EDF
            del layer.kernel_up_proj_EDF

            ep_axis_name = layer.efd_sharding[0]

            self.extra_backend_kwargs = {
                "ep_axis_name": ep_axis_name,
                "bt": 32,
                "bf": 512,
                "bd1": 512,
                "bd2": 512,
                "btc": 64,
                "bfc": 256,
                "bd1c": 256,
                "bd2c": 256,
            }

        elif layer.moe_backend in [MoEBackend.GMM_EP, MoEBackend.GMM_TP]:
            if any(
                    any(w is None for w in param._weights_to_load) for param in
                [layer.kernel_gating_EDF, layer.kernel_up_proj_EDF]):
                return False
            w_gate = layer.kernel_gating_EDF.value
            w_up = layer.kernel_up_proj_EDF.value

            # Fuse the weights into w13: [Gate, Up]
            w13_val = jnp.concatenate([w_gate, w_up], axis=1)

            # TODO (jacobplatin): we probably want to make the sharding configurable
            layer.kernel_gating_upproj_EDF = nnx.Param(
                shard_put(w13_val, shardings=layer.edf_sharding))

            del layer.kernel_gating_EDF
            del layer.kernel_up_proj_EDF

        return True

    def apply_jax(self,
                  layer: JaxModule,
                  x: jax.Array,
                  *,
                  capture_routing_stats: bool = False,
                  layer_idx: int | None = None) -> jax.Array:
        assert isinstance(layer, JaxMoE)

        x_TD = jnp.asarray(x, layer.dtype)
        x_TD = jax.lax.with_sharding_constraint(
            x_TD, NamedSharding(layer.mesh, P(*layer.activation_ffw_td)))

        router_logits = None
        routing_stats = None
        # Fused weight backends
        if layer.moe_backend in MoEBackend.fused_moe_backends():
            # of shape TE, only 1D in this case
            with tpu_trace_region("gate"):
                router_logits = layer.router(x_TD)

            w13_weight = layer.kernel_gating_upproj_E2DF.value if layer.moe_backend == MoEBackend.FUSED_MOE else layer.kernel_gating_upproj_EDF.value
            w2_weight = layer.kernel_down_proj_EFD.value
            w13_weight = jnp.swapaxes(w13_weight, 1, 2)
            w2_weight = jnp.swapaxes(w2_weight, 1, 2)
            # TODO (jacobplatin/bzgoogle): we should support bias
            weights = FusedMoEWeights(
                w13_weight=w13_weight,
                w13_weight_scale=None,
                w13_bias=None,
                w2_weight=w2_weight,
                w2_weight_scale=None,
                w2_bias=None,
            )
        elif layer.moe_backend in [
                MoEBackend.DENSE_MAT, MoEBackend.MEGABLX_GMM
        ]:
            # Composed of weights_TX and indices_TX, so 2D in this case
            with tpu_trace_region("gate"):
                router_logits = layer.router(x_TD)
            # TODO (jacobplatin/bzgoogle): we should support bias
            weights = UnfusedMoEWeights(
                w1_weight=layer.kernel_gating_EDF.value,
                w1_weight_scale=None,
                w1_bias=None,
                w2_weight=layer.kernel_up_proj_EDF.value,
                w2_weight_scale=None,
                w2_bias=None,
                w3_weight=layer.kernel_down_proj_EFD.value,
                w3_weight_scale=None,
                w3_bias=None,
            )

        else:
            raise ValueError(f"Unsupported moe backend {layer.moe_backend}")
        if capture_routing_stats:
            bytes_per_element = x_TD.dtype.itemsize
            if isinstance(router_logits, tuple):
                # Router returned (weights, indices) directly.
                weights_TX, indices_TX = router_logits
                routing_stats = compute_routing_stats_from_topk(
                    topk_scores=weights_TX,
                    topk_experts=indices_TX,
                    num_experts=layer.num_local_experts,
                    hidden_size=layer.hidden_size,
                    bytes_per_element=bytes_per_element,
                    layer_idx=layer_idx,
                    data_axis_name=layer.data_axis_name,
                    num_expert_parallelism=layer.num_expert_parallelism,
                    num_local_experts=layer.num_local_experts,
                    capture_a2a=envs.moe_routing_a2a_enabled(),
                    routing_is_exact=True,
                )
            else:
                routing_stats = compute_routing_stats_from_logits(
                    router_logits=router_logits,
                    k=layer.top_k,
                    scoring_func=layer.scoring_func,
                    renormalize=layer.renormalize,
                    hidden_size=layer.hidden_size,
                    bytes_per_element=bytes_per_element,
                    layer_idx=layer_idx,
                    data_axis_name=layer.data_axis_name,
                    num_expert_parallelism=layer.num_expert_parallelism,
                    num_local_experts=layer.num_local_experts,
                    capture_a2a=envs.moe_routing_a2a_enabled(),
                    include_router_logits=True,
                    include_router_probs=envs.moe_router_probs_enabled(),
                    include_unique_token_counts=True,
                    routing_is_exact=layer.moe_backend
                    in (MoEBackend.GMM_EP, MoEBackend.GMM_TP),
                )

        capture_ep_ragged = (
            capture_routing_stats
            and layer.moe_backend == MoEBackend.MEGABLX_GMM
            and envs.moe_ep_ragged_a2a_matrix_enabled()
        )
        if layer.moe_backend == MoEBackend.MEGABLX_GMM:
            moe_out = moe_apply(
                layer,
                x_TD,
                router_logits,
                weights,
                layer.moe_backend,
                layer.mesh,
                self.extra_backend_kwargs,
                capture_ep_ragged_a2a=capture_ep_ragged,
            )
        else:
            with tpu_trace_region("moe_kernel"):
                moe_out = moe_apply(
                    layer,
                    x_TD,
                    router_logits,
                    weights,
                    layer.moe_backend,
                    layer.mesh,
                    self.extra_backend_kwargs,
                    capture_ep_ragged_a2a=capture_ep_ragged,
                )
        if capture_ep_ragged and isinstance(moe_out, tuple):
            output, ep_extra = moe_out
            routing_stats = {**routing_stats, **ep_extra}
        else:
            output = moe_out
        if capture_routing_stats:
            return output, routing_stats
        return output


class UnquantizedConfig(QuantizationConfig):

    def get_quant_method(self, layer: JaxModule,
                         prefix: str) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, JaxEinsum):
            # Derive output's last dim from the einsum string.
            einsum_str = layer.einsum_str.replace(" ", "")
            _, w_axis = einsum_str.split("->")[0].split(",")
            last_out_char = einsum_str.split("->")[1][-1]
            out_size = layer.kernel_shape[w_axis.index(last_out_char)]

            linear_config = QuantLinearConfig(enable_sp=False,
                                              output_sizes=[out_size])
            return UnquantizedLinearMethod(linear_config)
        if isinstance(layer, JaxMoE):
            return UnquantizedFusedMoEMethod()
        return None
