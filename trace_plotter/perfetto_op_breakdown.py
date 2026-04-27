# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Chrome / Perfetto trace → per-step time in coarse buckets (attention, gate, …).

When the model is built with ``tpu_inference.profiler_trace.tpu_trace_region``,
events include ``tpu_trace:*`` names and are classified **deterministically** first.
Otherwise buckets fall back to kernel / op name heuristics. Overlapping work can
make bucket sums exceed wall time; treat as indicative, not mutually exclusive wall time.
"""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from perfetto_comm_compute import _is_execute_model_step  # noqa: PLC2701

_COMM_RE = re.compile(
    r"(all[-_ ]?reduce|all[-_ ]?to[-_ ]?all|reduce[-_ ]?scatter|"
    r"collective[-_ ]?permute|send|recv|cross[-_ ]?replica|cross[-_ ]?partition)",
    re.IGNORECASE,
)
_IGNORE_NAME_RE = re.compile(r"^process_name$|^thread_name$|^TraceAnnotation$")

# Heuristic matchers (lowered string). Order: collective first, then fine-grained compute.
_ATTENTION = re.compile(
    r"(attention|attn|mha|mqa|gqa|flash|sdpa|scaled_dot|softmax|"
    r"dot_general/Dot.*[QKVO]|qkv|query|key_proj|value_proj|o_proj|"
    r"rope|rotary|alibi|sliding.window|kv_cache|kv_|paged_attn)",
    re.IGNORECASE,
)
_GATE = re.compile(
    r"(router|gating|\bgate\b|routing_logits|topk.*expert|expert.*logit|"
    r"mixture.*router|switch_mlp)",
    re.IGNORECASE,
)
_EXPERTS = re.compile(
    r"(megablox|sparse_moe|moe\b|mixture.of.experts|expert_parallel|"
    r"expert_matmul|experts\.|expert_fc|dispatch|combine|einsum.*expert)",
    re.IGNORECASE,
)
_ADD_NORM = re.compile(
    r"(rms.?norm|layernorm|layer_norm|\bln\b|add_norm|residual|"
    r"group_norm|bias_add)",
    re.IGNORECASE,
)


def traffic_heatmap_colormap():
    """White (no traffic) → green → dark blue (heavy traffic)."""
    try:
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib required") from exc

    return LinearSegmentedColormap.from_list(
        "ici_traffic_wgb",
        [
            (1.0, 1.0, 1.0),
            (0.15, 0.72, 0.38),
            (0.05, 0.12, 0.42),
        ],
        N=256,
    )


def _bucket_from_tpu_instrumentation(name: str, cat: str) -> str | None:
    """Map ``jax.profiler.TraceAnnotation`` names emitted by ``tpu_trace_region``."""
    hay = f"{cat} {name}".lower()
    if "tpu_trace:" not in hay:
        return None
    if "tpu_trace:norm_pre_attn" in hay or "tpu_trace:norm_pre_mlp" in hay:
        return "add_norm"
    if "tpu_trace:add_post_attn" in hay or "tpu_trace:add_post_mlp" in hay:
        return "add_norm"
    if "tpu_trace:add_shared_experts_residual" in hay:
        return "add_norm"
    if "tpu_trace:attention" in hay:
        return "attention"
    if "tpu_trace:gate" in hay:
        return "gate"
    if any(
        x in hay for x in (
            "tpu_trace:moe_permute_global",
            "tpu_trace:moe_all2all_dispatch",
            "tpu_trace:moe_post_dispatch_permute",
            "tpu_trace:moe_all2all_return",
            "tpu_trace:moe_local_route",
            "tpu_trace:moe_unpermute",
        )
    ):
        return "collective"
    if "tpu_trace:moe_experts" in hay:
        return "experts"
    if any(
        x in hay for x in (
            "tpu_trace:moe_kernel",
            "tpu_trace:moe_fused_ep",
            "tpu_trace:moe_gmm",
            "tpu_trace:moe_dense_mat",
            "tpu_trace:mlp",
            "tpu_trace:mlp_moe",
            "tpu_trace:mlp_dense",
            "tpu_trace:shared_experts",
        )
    ):
        return "experts"
    return None


def _bucket_for_event(name: str, cat: str) -> str:
    inst = _bucket_from_tpu_instrumentation(name, cat)
    if inst is not None:
        return inst
    key = f"{cat} {name}"
    if _COMM_RE.search(key):
        return "collective"
    kl = key.lower()
    if _ATTENTION.search(kl):
        return "attention"
    if _GATE.search(kl):
        return "gate"
    if _EXPERTS.search(kl):
        return "experts"
    if _ADD_NORM.search(kl):
        return "add_norm"
    return "other_compute"


def _load_trace_json(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class _Ev:
    name: str
    cat: str
    ts: float
    te: float
    args: dict[str, Any] | None


def _iter_complete_events(trace: dict[str, Any]) -> list[_Ev]:
    out: list[_Ev] = []
    for raw in trace.get("traceEvents", []):
        if not isinstance(raw, dict):
            continue
        if raw.get("ph") != "X":
            continue
        if "ts" not in raw or "dur" not in raw:
            continue
        name = str(raw.get("name", ""))
        if _IGNORE_NAME_RE.search(name):
            continue
        ts = float(raw["ts"])
        te = ts + max(0.0, float(raw["dur"]))
        args = raw.get("args")
        args_d = args if isinstance(args, dict) else None
        out.append(_Ev(name=name, cat=str(raw.get("cat", "")), ts=ts, te=te, args=args_d))
    out.sort(key=lambda e: (e.ts, e.te))
    return out


def _clip(ts: float, te: float, start: float, end: float) -> tuple[float, float] | None:
    s = max(ts, start)
    t = min(te, end)
    if t <= s:
        return None
    return (s, t)


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _sum_us(intervals: list[tuple[float, float]]) -> float:
    return sum(e - s for s, e in intervals)


def summarize_execute_model_breakdown(trace_path: Path) -> pd.DataFrame:
    """One row per ``execute_model`` slice with milliseconds per coarse bucket."""
    trace = _load_trace_json(trace_path)
    events = _iter_complete_events(trace)
    steps = [e for e in events if _is_execute_model_step(e.name)]
    rows: list[dict[str, Any]] = []
    for i, step in enumerate(steps):
        s0, s1 = step.ts, step.te
        dur_us = max(0.0, s1 - s0)
        acc: dict[str, list[tuple[float, float]]] = {
            "attention": [],
            "gate": [],
            "collective": [],
            "experts": [],
            "add_norm": [],
            "other_compute": [],
        }
        for ev in events:
            if ev is step:
                continue
            if ev.te <= s0 or ev.ts >= s1:
                continue
            clip = _clip(ev.ts, ev.te, s0, s1)
            if clip is None:
                continue
            bname = _bucket_for_event(ev.name, ev.cat)
            acc[bname].append(clip)

        row: dict[str, Any] = {
            "step_index": i,
            "step_name": step.name,
            "step_start_us": s0,
            "step_wall_ms": dur_us / 1000.0,
        }
        total_bucket_us = 0.0
        for short in ("attention", "gate", "collective", "experts", "add_norm", "other_compute"):
            merged = _merge_intervals(acc[short])
            us = _sum_us(merged)
            total_bucket_us += us
            row[f"{short}_ms"] = us / 1000.0
        row["bucket_sum_ms"] = total_bucket_us / 1000.0
        rows.append(row)
    return pd.DataFrame(rows)


_SHAPE_BRACKET_RE = re.compile(r"\[[^\]]{1,120}\]")


def _shape_hint(name: str, args: dict[str, Any] | None) -> str | None:
    m = _SHAPE_BRACKET_RE.search(name)
    if m:
        return m.group(0).strip()
    if args:
        for k in ("shape", "dims", "long_name", "metadata", "expression"):
            v = args.get(k)
            if isinstance(v, str) and "[" in v:
                snip = v.strip()
                return snip[:160]
    return None


def extract_collective_ops(trace_path: Path) -> pd.DataFrame:
    """All collective-like ``X`` events with duration; optional shape hint from name/args."""
    trace = _load_trace_json(trace_path)
    events = _iter_complete_events(trace)
    step_marks = [e for e in events if _is_execute_model_step(e.name)]
    rows: list[dict[str, Any]] = []
    for ev in events:
        key = f"{ev.cat} {ev.name}"
        if not _COMM_RE.search(key):
            continue
        step_index: int | None = None
        for i, st in enumerate(step_marks):
            if ev.ts < st.te and ev.te > st.ts:
                step_index = i
                break
        rows.append(
            {
                "name": ev.name,
                "cat": ev.cat,
                "ts_us": ev.ts,
                "te_us": ev.te,
                "duration_ms": max(0.0, ev.te - ev.ts) / 1000.0,
                "step_index": step_index,
                "shape_hint": _shape_hint(ev.name, ev.args),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "extract_collective_ops",
    "summarize_execute_model_breakdown",
    "traffic_heatmap_colormap",
]
