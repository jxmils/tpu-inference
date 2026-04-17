# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project

import os

# Disable CUDA-specific shared experts stream for TPU
# This prevents errors when trying to create CUDA streams on TPU hardware
# The issue was introduced by vllm-project/vllm#26440
os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] = "1"

# XLA HLO dumps (memory plan / buffer sizes): merge XLA_DUMP_TO into XLA_FLAGS
# before JAX or vLLM import any code that compiles HLO.
from tpu_inference.xla_env import apply_xla_dump_from_env

apply_xla_dump_from_env()
