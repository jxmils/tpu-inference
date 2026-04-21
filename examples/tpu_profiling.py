# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Implements profiling for vLLM on TPU VMs using the JAX profiler.
# With --profiler-config profiler=torch, the worker uses jax.profiler.start_trace
# / stop_trace (not PyTorch autograd). Traces are XPlane-compatible and show
# on-device work on the TPU ICI, including tensor-parallel all-reduce and MoE
# expert-parallel all-to-all.
# NOTE: you will need the tensorboard-plugin-profile python package to
# visualize the results in TensorBoard.
# Please see docs/profiling.md for more details.
#
# If you see "unrecognized arguments: --profile-scenario", this file on the VM
# is stale — run `git pull` in the repo (or use the explicit --input-len lines).
# Same presets via env: export TPU_PROFILING_SCENARIO=prefill|decode
#
# Presets (recommended): pass --model and any EngineArgs (TP size, etc.).
#   Prefill-heavy — 1 request, 1024 prompt tokens, 1 generated token:
#     python3 examples/tpu_profiling.py --profile-scenario prefill --model google/gemma-2b
#   Decode-heavy — 256 requests, 1 prompt + 1 output token each:
#     python3 examples/tpu_profiling.py --profile-scenario decode --model google/gemma-2b
#
# Same shapes without --profile-scenario:
#   python3 examples/tpu_profiling.py --model google/gemma-2b --input-len 1024 --output-len 1 --batch-size 1
#   python3 examples/tpu_profiling.py --model google/gemma-2b --input-len 1 --output-len 1 --batch-size 256

import argparse
import dataclasses
import os
import time
from typing import Any

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.utils.argparse_utils import FlexibleArgumentParser

DURATION_MS = int(os.getenv("VLLM_TPU_PROFILE_DURATION_MS", 3000))
DELAY_MS = int(os.getenv("VLLM_TPU_PROFILE_DELAY_MS", 0))


def _normalize_compilation_config_for_llm_init(llm_kwargs: dict[str, Any]) -> None:
    """Argparse/EngineArgs can leave None where vLLM's Pydantic CompilationConfig expects values.

    Seen on TPU: ``cudagraph_capture_sizes`` must be a list; ``pass_config.fuse_minimax_qk_norm``
    must be a bool (not None).
    """
    cc = llm_kwargs.get("compilation_config")
    if not isinstance(cc, dict):
        return
    if cc.get("cudagraph_capture_sizes") is None:
        cc["cudagraph_capture_sizes"] = []
    pc = cc.get("pass_config")
    if not isinstance(pc, dict):
        pc = {}
        cc["pass_config"] = pc
    if pc.get("fuse_minimax_qk_norm") is None:
        pc["fuse_minimax_qk_norm"] = False


def _apply_profile_scenario(args: argparse.Namespace) -> None:
    """Match docs/profiling.md canonical prefill vs decode profiling shapes."""
    scenario = getattr(args, "profile_scenario", None)
    if scenario is None:
        env_scenario = os.environ.get("TPU_PROFILING_SCENARIO")
        if env_scenario in ("prefill", "decode"):
            scenario = env_scenario
            args.profile_scenario = env_scenario
    if scenario == "prefill":
        args.input_len = 1024
        args.output_len = 1
        args.batch_size = 1
    elif scenario == "decode":
        args.input_len = 1
        args.output_len = 1
        args.batch_size = 256


def main(args: argparse.Namespace):
    _apply_profile_scenario(args)
    print(args)

    # Profile
    profile_dir = args.profile_result_dir
    print(f"Profiling (results will be saved to '{profile_dir}')...")

    profiler_config = args.profiler_config
    profiler_config.profiler = "torch"
    profiler_config.torch_profiler_dir = profile_dir
    args.profiler_config = profiler_config

    engine_args = EngineArgs.from_cli_args(args)
    llm_kwargs = dataclasses.asdict(engine_args)
    _normalize_compilation_config_for_llm_init(llm_kwargs)
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    print(sampling_params)
    dummy_prompt_token_ids = np.random.randint(10000,
                                               size=(args.batch_size,
                                                     args.input_len))
    dummy_prompts: list[PromptType] = [{
        "prompt_token_ids": batch
    } for batch in dummy_prompt_token_ids.tolist()]

    def run_to_completion():
        start_time = time.perf_counter()
        llm.generate(dummy_prompts,
                     sampling_params=sampling_params,
                     use_tqdm=False)
        end_time = time.perf_counter()
        latency = end_time - start_time
        return latency

    # Warmup
    print("Warming up...")
    warmup_latencies = []
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        warmup_latencies.append(run_to_completion())
    print(f"Average warmup latency: {np.mean(warmup_latencies):.4f}s")

    # Enable tracing on server
    llm.start_profile()
    if DELAY_MS == 0:
        time.sleep(1.0)
    profile_latencies = []
    for _ in tqdm(range(args.num_iters), desc="Profile iterations"):
        profile_latencies.append(run_to_completion())
    llm.stop_profile()
    print(f"Average profile latency: {np.mean(profile_latencies):.4f}s")

    return


def parse_args():
    parser = FlexibleArgumentParser(
        description="Benchmark the latency of processing a single batch of "
        "requests till completion.")
    parser.add_argument("--input-len", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--num-iters-warmup",
        type=int,
        default=5,
        help="Number of iterations to run for warmup.",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=1,
        help="Number of iterations to run for profiling.",
    )
    parser.add_argument(
        "--profile-result-dir",
        type=str,
        default="profiles",
        help=("path to save the JAX profiler output. Can be visualized "
              "with ui.perfetto.dev, Tensorboard, or XProf"),
    )

    parser = EngineArgs.add_cli_args(parser)
    # Register after EngineArgs so vLLM's parser keeps owning standard flags.
    parser.add_argument(
        "--profile-scenario",
        choices=("prefill", "decode"),
        default=None,
        help=(
            "Workload preset: prefill => input_len=1024, output_len=1, batch_size=1; "
            "decode => input_len=1, output_len=1, batch_size=256. "
            "Overrides --input-len, --output-len, and --batch-size. "
            "Same as env TPU_PROFILING_SCENARIO=prefill|decode when unset."),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
