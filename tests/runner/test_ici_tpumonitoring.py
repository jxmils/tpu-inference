# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path


def _load_metric_delta():
    path = (Path(__file__).resolve().parents[2] / "tpu_inference" / "runner" /
            "ici_tpumonitoring.py")
    name = "ici_tpumonitoring_test_load"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.metric_delta


def test_metric_delta_per_chip_and_sum():
    metric_delta = _load_metric_delta()
    before = {"ici_flits_tx": [100.0, 200.0]}
    after = {"ici_flits_tx": [103.0, 206.0]}
    out = metric_delta(before, after)
    assert "ici_hw_delta" in out
    d = out["ici_hw_delta"]["ici_flits_tx"]
    assert d["delta_per_chip"] == [3.0, 6.0]
    assert d["delta_sum"] == 9.0


def test_metric_delta_len_mismatch():
    metric_delta = _load_metric_delta()
    before = {"a": [1.0]}
    after = {"a": [1.0, 2.0]}
    out = metric_delta(before, after)
    assert out["ici_hw_delta"]["a"]["delta_sum"] is None
