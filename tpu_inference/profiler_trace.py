# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Host-visible XProf / Perfetto regions for JAX forwards.

Regions use ``jax.profiler.TraceAnnotation`` with names prefixed by ``tpu_trace:``
so post-processors can attribute time to attention / gate / experts / collectives
without relying on low-level kernel name heuristics.

Disable all regions (e.g. microbenchmarks) with::

    export DISABLE_TPU_TRACE_ANNOTATIONS=1
"""

from __future__ import annotations

import contextlib
import os

import jax


def tpu_trace_regions_enabled() -> bool:
    v = os.getenv("DISABLE_TPU_TRACE_ANNOTATIONS", "")
    return v.lower() not in ("1", "true", "yes", "on")


@contextlib.contextmanager
def tpu_trace_region(name: str):
    """``with tpu_trace_region("gate"):`` records ``tpu_trace:gate`` in Chrome / XProf."""
    if not tpu_trace_regions_enabled():
        yield
        return
    with jax.profiler.TraceAnnotation(f"tpu_trace:{name}"):
        yield


__all__ = ["tpu_trace_region", "tpu_trace_regions_enabled"]
