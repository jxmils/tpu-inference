# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path


def _tp():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "trace_plotter"))
    import perfetto_op_breakdown as pob  # noqa: PLC0415

    return pob


def test_breakdown_buckets_single_step(tmp_path):
    pob = _tp()
    trace = {
        "traceEvents": [
            {"name": "execute_model: a", "ph": "X", "ts": 0, "dur": 1000},
            {
                "name": "flash_attention_forward",
                "cat": "xla",
                "ph": "X",
                "ts": 50,
                "dur": 100,
            },
            {"name": "router_logits", "cat": "xla", "ph": "X", "ts": 200, "dur": 80},
            {"name": "megablox_dispatch", "cat": "xla", "ph": "X", "ts": 300, "dur": 120},
            {"name": "all-reduce", "cat": "xla", "ph": "X", "ts": 400, "dur": 90},
            {"name": "rms_norm", "cat": "xla", "ph": "X", "ts": 520, "dur": 60},
            {"name": "dot_general", "cat": "xla", "ph": "X", "ts": 700, "dur": 50},
        ]
    }
    p = tmp_path / "t.json"
    p.write_text(json.dumps(trace), encoding="utf-8")
    df = pob.summarize_execute_model_breakdown(p)
    assert len(df) == 1
    r = df.iloc[0]
    assert float(r["attention_ms"]) > 0
    assert float(r["gate_ms"]) > 0
    assert float(r["experts_ms"]) > 0
    assert float(r["collective_ms"]) > 0
    assert float(r["add_norm_ms"]) > 0
    assert float(r["other_compute_ms"]) > 0


def test_tpu_instrumentation_overrides_heuristic(tmp_path):
    pob = _tp()
    trace = {
        "traceEvents": [
            {"name": "execute_model: a", "ph": "X", "ts": 0, "dur": 1000},
            {
                "name": "tpu_trace:gate",
                "cat": "python",
                "ph": "X",
                "ts": 100,
                "dur": 50,
            },
        ]
    }
    p = tmp_path / "t3.json"
    p.write_text(json.dumps(trace), encoding="utf-8")
    df = pob.summarize_execute_model_breakdown(p)
    assert float(df.iloc[0]["gate_ms"]) > 0


def test_collective_ops_shape_hint(tmp_path):
    pob = _tp()
    trace = {
        "traceEvents": [
            {
                "name": "all-to-all [1024,4096]",
                "cat": "xla",
                "ph": "X",
                "ts": 10,
                "dur": 500,
            },
        ]
    }
    p = tmp_path / "t2.json"
    p.write_text(json.dumps(trace), encoding="utf-8")
    ops = pob.extract_collective_ops(p)
    assert len(ops) == 1
    assert ops.iloc[0]["shape_hint"] == "[1024,4096]"
