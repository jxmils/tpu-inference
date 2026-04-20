# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
from pathlib import Path


def _load_mod():
    path = (Path(__file__).resolve().parents[1] / "scripts" / "vllm" /
            "benchmarking" / "trace_comm_compute_breakdown.py")
    name = "trace_comm_compute_breakdown_test_load"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_summarize_basic(tmp_path):
    mod = _load_mod()
    # Timestamps are microseconds (Perfetto convention).
    trace = {
        "traceEvents": [
            {
                "name": "execute_model: 2 reqs, 16 toks",
                "ph": "X",
                "ts": 0,
                "dur": 1000
            },
            {
                "name": "dot_general",
                "cat": "xla",
                "ph": "X",
                "ts": 100,
                "dur": 400
            },
            {
                "name": "all-reduce",
                "cat": "xla",
                "ph": "X",
                "ts": 300,
                "dur": 400
            },
        ]
    }
    p = tmp_path / "perfetto_trace.json"
    p.write_text(json.dumps(trace), encoding="utf-8")
    rows = mod.summarize(p)
    assert len(rows) == 1
    r = rows[0]
    # comm = [300,700] => 0.4 ms
    assert float(r["comm_time_ms"]) == 0.4
    # compute = [100,500] => 0.4 ms
    assert float(r["compute_time_ms"]) == 0.4
    # overlap = [300,500] => 0.2 ms
    assert float(r["overlap_ms"]) == 0.2
    # union(all ops) = [100,700] => 0.6 ms
    assert float(r["total_op_time_ms"]) == 0.6
