# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "vllm" / "prometheus_metrics_poll_csv.py"
    spec = importlib.util.spec_from_file_location("prometheus_metrics_poll_csv",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_parse_gauges_with_labels():
    m = _load_module()
    body = """
# HELP vllm:kv_cache_usage_perc KV-cache usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="m"} 0.25
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="m"} 4
""".strip()
    g = m.parse_gauges(body)
    assert g["vllm:kv_cache_usage_perc"] == 0.25
    assert g["vllm:num_requests_running"] == 4.0


def test_parse_gauges_without_labels():
    m = _load_module()
    body = "vllm:kv_cache_usage_perc 0.1\nvllm:num_requests_running 2\n"
    g = m.parse_gauges(body)
    assert g["vllm:kv_cache_usage_perc"] == 0.1
    assert g["vllm:num_requests_running"] == 2.0
