# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[2] / "tpu_inference" / "runner" / "step_trace_proxies.py"
    spec = importlib.util.spec_from_file_location("step_trace_proxies", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_comm = _load().comm_bytes_proxy_for_step


def test_comm_bytes_proxy_prefers_positive_a2a():
    assert _comm(100, 100, 50) == (50, "a2a")


def test_comm_bytes_proxy_zero_a2a_uses_dispatch_return():
    assert _comm(30, 70, 0) == (100, "dispatch_plus_return")


def test_comm_bytes_proxy_dispatch_symmetric_fallback():
    assert _comm(40, None, None) == (80, "dispatch_times_two")


def test_comm_bytes_proxy_none_when_no_signal():
    assert _comm(None, None, None) == (None, "none")
