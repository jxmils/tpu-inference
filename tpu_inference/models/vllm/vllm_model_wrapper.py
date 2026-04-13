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

import copy
import inspect
import time
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any, List, Optional, Tuple
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn
import torchax
import vllm.envs as vllm_envs
from flax.typing import PRNGKey
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import TORCH_DTYPE_TO_JAX
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.lora.layers import BaseLayerWithLoRA
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.pooler import Pooler
from vllm.model_executor.model_loader import get_model as vllm_get_model
from vllm.model_executor.models import supports_lora, supports_multimodal
from vllm.model_executor.models.interfaces_base import is_pooling_model
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata

from tpu_inference.distributed.jax_parallel_state import \
    get_pp_group as jax_get_pp_group
from tpu_inference import envs
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.vllm.mla_attention import \
    VllmTPUMultiHeadLatentAttentionWrapper
from tpu_inference.layers.vllm.process_weights.cleanup_sharding import \
    shard_model_to_tpu
from tpu_inference.layers.vllm.quantization import get_tpu_quantization_config
from tpu_inference.logger import init_logger
from tpu_inference.models.common.interface import PoolerFunc
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.models.vllm.vllm_model_wrapper_context import (
    get_vllm_model_wrapper_context, set_vllm_model_wrapper_context)
from tpu_inference.runner.lora_utils import replace_lora_metadata

logger = init_logger(__name__)


def _patch_pallas_attention_output_block_scale() -> None:
    """Make vLLM/Pallas attention forward compatible with older torchax."""
    try:
        import importlib
        import pkgutil

        modules = []
        candidate_modules = (
            "vllm.model_executor.layers.attention.backends.pallas",
            "vllm.model_executor.layers.attention.backends.pallas_attn",
            "vllm.model_executor.layers.attention.backends.pallas_attention",
        )
        for mod_name in candidate_modules:
            try:
                modules.append(importlib.import_module(mod_name))
            except Exception:
                continue

        try:
            import vllm.model_executor.layers.attention.backends as backends
        except Exception:
            backends = None

        if backends is not None and hasattr(backends, "__path__"):
            for modinfo in pkgutil.walk_packages(
                    backends.__path__, backends.__name__ + "."):
                if "pallas" not in modinfo.name:
                    continue
                try:
                    modules.append(importlib.import_module(modinfo.name))
                except Exception:
                    continue

        cls = None
        for mod in modules:
            if hasattr(mod, "PallasAttentionBackendImpl"):
                cls = getattr(mod, "PallasAttentionBackendImpl")
                break
        if cls is None:
            return
        if "output_block_scale" in inspect.signature(cls.forward).parameters:
            return

        original_forward = cls.forward

        def _forward(self, *args, output_block_scale=None, **kwargs):
            return original_forward(self, *args, **kwargs)

        cls.forward = _forward
        logger.warning(
            "Patched PallasAttentionBackendImpl.forward to ignore "
            "output_block_scale for compatibility with older torchax.")
    except Exception:
        logger.warning(
            "Failed to apply Pallas attention compatibility patch.",
            exc_info=True)


_patch_pallas_attention_output_block_scale()


def _patch_unified_attention_output_block_scale() -> None:
    """Drop output_block_scale when the backend doesn't accept it."""
    try:
        import importlib

        attn_mod = importlib.import_module(
            "vllm.model_executor.layers.attention.attention")
        if getattr(attn_mod, "_tpu_inference_obs_patch", False):
            return

        def _supports_output_block_scale(impl_forward) -> bool:
            sig = inspect.signature(impl_forward)
            if "output_block_scale" in sig.parameters:
                return True
            return any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in sig.parameters.values())

        def _wrapped(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            output: torch.Tensor,
            layer_name: Any,
            output_scale: torch.Tensor | None = None,
            output_block_scale: torch.Tensor | None = None,
            kv_cache_dummy_dep: torch.Tensor | None = None,
        ) -> None:
            del kv_cache_dummy_dep
            layer_name = attn_mod._resolve_layer_name(layer_name)
            attn_metadata, self, kv_cache, _ = attn_mod.get_attention_context(
                layer_name)
            kwargs = {
                "output": output,
                "output_scale": output_scale,
            }
            if _supports_output_block_scale(self.impl.forward):
                kwargs["output_block_scale"] = output_block_scale
            self.impl.forward(
                self,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                **kwargs,
            )

        attn_mod.unified_attention_with_output = _wrapped
        attn_mod._tpu_inference_obs_patch = True
        logger.warning(
            "Patched unified_attention_with_output to drop output_block_scale "
            "when unsupported by the backend.")
    except Exception:
        logger.warning(
            "Failed to apply unified attention compatibility patch.",
            exc_info=True)


_patch_unified_attention_output_block_scale()


def _aggregate_routing_stats(per_layer_stats: List[dict | None]) -> dict:
    present = [stats for stats in per_layer_stats if stats is not None]
    if not present:
        return {}
    total_expert_counts = sum(stats["expert_counts"] for stats in present)
    total_unique_token_counts = sum(
        stats["unique_token_counts"] for stats in present)
    total_dispatch_bytes = sum(stats["estimated_dispatch_bytes"]
                               for stats in present)
    total_return_bytes = sum(stats["estimated_return_bytes"]
                             for stats in present)
    total_a2a_bytes = None
    total_a2a_counts = None
    if any("a2a_bytes" in stats for stats in present):
        total_a2a_bytes = sum(
            stats.get("a2a_bytes", 0) for stats in present)
        total_a2a_counts = sum(
            stats.get("a2a_counts", 0) for stats in present)
    ep_ragged_dispatch_bytes = None
    ep_ragged_return_bytes = None
    if any("ep_ragged_a2a_dispatch_bytes" in stats for stats in present):
        ep_ragged_dispatch_bytes = sum(
            stats["ep_ragged_a2a_dispatch_bytes"]
            for stats in present
            if "ep_ragged_a2a_dispatch_bytes" in stats)
        ep_ragged_return_bytes = sum(
            stats["ep_ragged_a2a_return_bytes"]
            for stats in present
            if "ep_ragged_a2a_return_bytes" in stats)
    return {
        "num_layers": jnp.asarray(len(present), dtype=jnp.int32),
        "num_experts": present[0]["num_experts"],
        "k": present[0]["k"],
        "expert_counts": total_expert_counts,
        "unique_token_counts": total_unique_token_counts,
        "estimated_dispatch_bytes": total_dispatch_bytes,
        "estimated_return_bytes": total_return_bytes,
        "estimated_dispatch_bytes_total": jnp.sum(total_dispatch_bytes),
        "estimated_return_bytes_total": jnp.sum(total_return_bytes),
        "a2a_bytes_total": total_a2a_bytes,
        "a2a_counts_total": total_a2a_counts,
        "a2a_bytes_total_sum": (jnp.sum(total_a2a_bytes)
                                if total_a2a_bytes is not None else None),
        "ep_ragged_a2a_dispatch_bytes_total": ep_ragged_dispatch_bytes,
        "ep_ragged_a2a_return_bytes_total": ep_ragged_return_bytes,
    }


class _VllmRunner(torch.nn.Module):

    def __init__(self, vllm_model: torch.nn.Module):
        super().__init__()
        self.vllm_model = vllm_model

        has_pooler = is_pooling_model(vllm_model)
        self.pooler = vllm_model.pooler if has_pooler else None

    def forward(self, **kwargs) -> torch.Tensor:
        if "hidden_state" in kwargs:
            return self.compute_logits(kwargs["hidden_state"])
        else:
            return self.compute_hidden_state(
                kwargs["input_ids"],
                kwargs["positions"],
                kwargs["intermediate_tensors"],
                kwargs["inputs_embeds"],
            )

    def compute_hidden_state(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        hidden_state = self.vllm_model(input_ids, positions,
                                       intermediate_tensors, inputs_embeds)
        return hidden_state

    def compute_logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.vllm_model.compute_logits(hidden_state)


class VllmModelWrapper:
    """ Wraps a vLLM Pytorch model and let it run on the JAX engine. """

    rng: PRNGKey
    mesh: Mesh
    model: _VllmRunner

    def __init__(self, vllm_config: VllmConfig, rng: PRNGKey, mesh: Mesh):
        self.vllm_config = vllm_config
        self.rng = rng
        self.mesh = mesh

        self.vllm_config.quant_config = get_tpu_quantization_config(
            self.vllm_config, self.mesh)
        self._apply_pp_patch()

        MultiHeadLatentAttentionWrapper.register_oot(
            VllmTPUMultiHeadLatentAttentionWrapper)

    def _apply_pp_patch(self):
        # patch `get_pp_group` in vLLM to jax's get_pp_group.
        import sys

        import vllm.distributed as vllm_dist
        import vllm.distributed.parallel_state as vllm_ps

        vllm_ps.get_pp_group = jax_get_pp_group
        vllm_dist.get_pp_group = jax_get_pp_group

        for module_name, module in sys.modules.items():
            if module_name.startswith("vllm.model_executor.models"):
                if hasattr(module, "get_pp_group"):
                    setattr(module, "get_pp_group", jax_get_pp_group)

    def load_weights(self):
        loading_start = time.time()
        # Set up to load the model into CPU first.
        # Cache device slice config since device config cannot be deepcopied
        modified_slice_config = False
        if hasattr(
                self.vllm_config.device_config,
                'slice') and self.vllm_config.device_config.slice is not None:
            slice_config = self.vllm_config.device_config.slice
            modified_slice_config = True
            self.vllm_config.device_config.slice = None
        self.vllm_config.compilation_config.static_forward_context.clear()

        vllm_config_for_load = copy.deepcopy(self.vllm_config)
        if modified_slice_config:
            self.vllm_config.device_config.slice = slice_config
        assert self.vllm_config.model_config.dtype in TORCH_DTYPE_TO_JAX, "The model_config.dtype must be a PyTorch dtype."
        vllm_config_for_load.device_config.device = "cpu"
        # Remove the dynamically added sharding_config attribute to avoid errors
        # when vLLM's replace() function checks for dataclass fields.
        # This is safe because vllm_config_for_load is only used for model loading
        # which doesn't need sharding_config, and self.vllm_config still has it.
        if hasattr(vllm_config_for_load, 'sharding_config'):
            delattr(vllm_config_for_load, 'sharding_config')
        # Clearing the cached compilation config, otherwise vllm model init will fail

        # When expert parallelism is enabled, vLLM loads weight in sharding
        # aware manner. Since tpu-inference has its own sharding logic, this
        # may casue errors. Therefore, we disable it during weight loading.
        vllm_config_for_load.parallel_config.enable_expert_parallel = False

        use_random_weights = (
            vllm_config_for_load.load_config.load_format == "dummy")
        if use_random_weights:
            logger.info(
                "Initializing vLLM model with random weights, weight loading skipped."
            )
        # The DummyModelLoader in vLLM calls torch._sync for torch_xla path when
        # it detects the tpu platform, but we don't need it and it causes crash
        # without proper setup.
        load_context = patch(
            "torch._sync",
            return_value=None) if use_random_weights else nullcontext()

        # By default load weights to the CPU device first. If we are running
        # under Pathways, this would cause weights to be loaded on a CPU-only
        # node, so we'll need to remove this context.
        jax_context = jax.default_device(
            jax.devices("cpu")
            [0]) if not vllm_envs.VLLM_TPU_USING_PATHWAYS else nullcontext()
        # Load the vLLM model and wrap it into a new model whose forward
        # function can calculate the hidden_state and logits.
        with set_current_vllm_config(vllm_config_for_load), load_context, jax_context:
            vllm_model = vllm_get_model(vllm_config=vllm_config_for_load)
        lora_manager = None
        if vllm_config_for_load.lora_config is not None:
            # Replace layers in the model with LoRA layers.
            with torchax.default_env():
                # Argument "device" in load_lora_model is used to set the device
                # used in punica wrapper.
                lora_manager, vllm_model = load_lora_model(
                    vllm_model, vllm_config_for_load, device="jax")
            replace_set_lora(vllm_model)

        static_forward_context = vllm_config_for_load.compilation_config.static_forward_context
        self.vllm_config.compilation_config.static_forward_context = static_forward_context
        self.vllm_config.compilation_config.static_all_moe_layers = vllm_config_for_load.compilation_config.static_all_moe_layers

        self.model = _VllmRunner(vllm_model)
        params_and_buffers = shard_model_to_tpu(self.model, self.mesh)

        self._pooler: Pooler | None = self.model.pooler

        loading_end = time.time()
        total_loading_time = loading_end - loading_start
        logger.info(
            f"Total time to load model weights from storage to TPU: {total_loading_time:.2f} seconds."
        )
        # Returning to the jax land, so we need to wrap it into a JaxValue.
        return jax_view(params_and_buffers), lora_manager

    def jit_step_func(self):

        @jax.jit(
            donate_argnames=("kv_caches", ),
            out_shardings=(
                None,  # kv_caches - keep original sharding
                NamedSharding(self.mesh,
                              PartitionSpec(ShardingAxisName.ATTN_DATA, None)),
                None,  # empty list
            ),
            compiler_options={
                "xla_tpu_all_gather_collective_matmul_mode":
                "post_spmd_conservative",
                "xla_tpu_reduce_scatter_collective_matmul_mode":
                "post_spmd_conservative"
            },
            static_argnames=(
                "layer_name_to_kvcache_index",
                "is_first_rank",
                "is_last_rank",
            ),
        )
        def step_fun(
            params_and_buffers,  # This has been wrapped into torchax TorchValue
            kv_caches: List[jax.Array],
            input_ids: jax.Array,
            attn_metadata: AttentionMetadata,
            input_embeds: jax.Array,
            input_positions: jax.Array,
            layer_name_to_kvcache_index: Sequence[Tuple[str, int]],
            lora_metadata,
            intermediate_tensors: JaxIntermediateTensors = None,
            is_first_rank: bool = True,
            is_last_rank: bool = True,
            *args,
        ) -> Tuple[List[jax.Array], jax.Array, object]:
            layer_name_to_kvcache_index = dict(layer_name_to_kvcache_index)
            lora_metadata = torch_view(lora_metadata)
            capture_routing_stats = envs.moe_routing_stats_enabled()
            num_layers = getattr(
                getattr(self.vllm_config, "model_config", None), "hf_config",
                None)
            num_layers = getattr(num_layers, "num_hidden_layers", None)
            routing_stats = None
            if capture_routing_stats:
                if num_layers is None:
                    routing_stats = []
                else:
                    routing_stats = [None] * int(num_layers)
            with torchax.default_env(), set_vllm_model_wrapper_context(
                    kv_caches=kv_caches,
                    mesh=self.mesh,
                    layer_name_to_kvcache_index=layer_name_to_kvcache_index,
                    routing_stats=routing_stats,
            ), set_forward_context(attn_metadata=attn_metadata,
                                   vllm_config=self.vllm_config):
                # We need to wrap args from jax land into TorchValue with
                # torch_view in order to call the Torch function.
                original_lora_metadata = replace_lora_metadata(
                    self.model, lora_metadata, self.vllm_config.lora_config)
                if not is_first_rank:
                    intermediate_tensors = intermediate_tensors.to_torch()
                output_from_torch = torch.func.functional_call(
                    self.model,
                    torch_view(params_and_buffers),
                    kwargs={
                        "input_ids": torch_view(input_ids),
                        "positions": torch_view(input_positions),
                        "intermediate_tensors": intermediate_tensors,
                        "inputs_embeds": None,
                    },
                    tie_weights=False,
                )
                replace_lora_metadata(self.model, original_lora_metadata,
                                      self.vllm_config.lora_config)
                vllm_model_wrapper_context = get_vllm_model_wrapper_context()
                new_kv_caches = vllm_model_wrapper_context.kv_caches
                routing_stats = vllm_model_wrapper_context.routing_stats
            # Wrap the output(hidden states or intermediate tensor)
            # from torch land into a JaxValue for the jax code to consume.
            if not is_last_rank:
                output = JaxIntermediateTensors.from_torch(output_from_torch)
            else:
                output = jax_view(output_from_torch)
            if capture_routing_stats:
                routing_payload = {
                    "per_layer": routing_stats or [],
                    "aggregate": _aggregate_routing_stats(routing_stats or []),
                }
                return new_kv_caches, output, ([], routing_payload)
            return new_kv_caches, output, []

        return step_fun

    def jit_compute_logits_func(self):

        @jax.jit(out_shardings=(NamedSharding(
            self.mesh,
            PartitionSpec(ShardingAxisName.MLP_DATA,
                          ShardingAxisName.MLP_TENSOR))))
        def compute_logits_func(
            params_and_buffers: Any,
            hidden_states: jax.Array,
            lora_metadata,
        ) -> jax.Array:
            lora_metadata = torch_view(lora_metadata)
            with torchax.default_env(), set_vllm_model_wrapper_context(
                    kv_caches=None, mesh=self.mesh):
                original_lora_metadata = replace_lora_metadata(
                    self.model, lora_metadata, self.vllm_config.lora_config)
                logits = torch.func.functional_call(
                    self.model,
                    torch_view(params_and_buffers),
                    kwargs={
                        "hidden_state": torch_view(hidden_states),
                    },
                    tie_weights=False,
                )
                replace_lora_metadata(self.model, original_lora_metadata,
                                      self.vllm_config.lora_config)
            return jax_view(logits)

        return compute_logits_func

    def build_pooler_func(self) -> PoolerFunc:

        def compute_pooler_output(
            hidden_states: jax.Array,
            pooling_metadata: PoolingMetadata,
            seq_lens: np.ndarray,
        ) -> PoolerOutput:
            assert self._pooler is not None, "Model does not support pooling"

            torch_states: torch.Tensor = torch_view(hidden_states)
            with torchax.default_env():
                torch_states = torch_states.to('cpu', non_blocking=True)
                pooling_metadata.build_pooling_cursor(
                    seq_lens,
                    torch.tensor(seq_lens),
                    device=torch_states.device,
                )
                outputs: list[torch.Tensor] = self._pooler(
                    torch_states,
                    pooling_metadata,
                )
                return outputs

        return compute_pooler_output


def load_lora_model(model: torch.nn.Module, vllm_config: VllmConfig,
                    device: str) -> torch.nn.Module:
    if not supports_lora(model):
        raise ValueError(
            f"{model.__class__.__name__} does not support LoRA yet.")

    if supports_multimodal(model):
        logger.warning("Regarding multimodal models, vLLM currently "
                       "only supports adding LoRA to language model.")

    # Add LoRA Manager to the Model Runner
    lora_manager = LRUCacheWorkerLoRAManager(
        vllm_config,
        device,
        model.embedding_modules,
    )
    return lora_manager, lora_manager.create_lora_manager(model)


# The reason why replace the method is that the set_lora and reset_lora need to
# run under torchax env.
def replace_set_lora(model):

    def _tpu_set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
    ):
        with torchax.default_env():
            self._original_set_lora(index, lora_a, lora_b)

    def _tpu_reset_lora(self, index: int):
        with torchax.default_env():
            self._original_reset_lora(index)

    for _, module in model.named_modules():
        if isinstance(module, BaseLayerWithLoRA):
            module._original_set_lora = module.set_lora
            module._original_reset_lora = module.reset_lora
            module.set_lora = _tpu_set_lora.__get__(module, module.__class__)
            module.reset_lora = _tpu_reset_lora.__get__(
                module, module.__class__)
