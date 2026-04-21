#!/usr/bin/env python3
"""
Read traces/raw_traces JSONL (step, hbm, summary) and optional raw NPZ routing
dumps; write many PDF figures under ./results/ by default. When ``--traces-dir``
ends with ``raw_traces`` inside a ``tpu_trace_bundle``, also plots Chrome/JAX
trace comm vs compute, HLO module family counts, HBM execute brackets, and
ICI overlays (unless ``--skip-bundle-extras``).

Includes paper-style figures:
  (a) Stacked expert dispatch (MB) vs trace_step from routing summary.
  (b) Expert×expert heatmaps from raw/routing_raw_*.npz top-k tensors (routing
      proxy, not hardware all-to-all). Enable CAPTURE raw NPZ to populate raw/.
  (c) Optional EP shard×shard heatmaps from layer_*_ep_ragged_a2a_* keys when
      inference sets CAPTURE_MOE_EP_RAGGED_A2A=1 (Megablox send_sizes schedule).
  (d) Multi-page PDF: per-layer bar histogram of expert load (summed expert_count).

Usage:
  python plot_raw_traces.py
  python plot_raw_traces.py --traces-dir /path/to/raw_traces --out-dir /path/to/out

By default this generates **all** standard figures (routing, paper stacks, expert×expert
heatmaps for **every** MoE layer found in ``raw/routing_raw_*.npz``, EP-ragged A2A
attempts when keys exist, and bundle extras when ``raw_traces`` sits under ``tpu_trace_bundle``).
Override with ``--paper-layer N`` for a single layer only, or ``--skip-ep-ragged-a2a`` /
``--skip-bundle-extras`` to save time or disk.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping

_plotter_dir = Path(__file__).resolve().parent
_mpl_dir = _plotter_dir / ".mplconfig"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_bundle_extras as _bundle


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_traces_dir() -> Path:
    return _repo_root() / "traces" / "raw_traces"


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_step_frames(traces_dir: Path) -> pd.DataFrame:
    paths = sorted(traces_dir.glob("step/*.jsonl"))
    if not paths:
        return pd.DataFrame()
    frames = []
    for p in paths:
        df = pd.read_json(p, lines=True)
        df["_source_file"] = p.name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if "record_type" in out.columns:
        out = out[out["record_type"] == "step_summary"]
    out = out.sort_values("trace_step")
    if "trace_step" in out.columns and out["trace_step"].duplicated().any():
        out = out.drop_duplicates(subset=["trace_step"], keep="last")
    return out.reset_index(drop=True)


def _load_hbm_frames(traces_dir: Path) -> pd.DataFrame:
    paths = sorted(traces_dir.glob("hbm/*.jsonl"))
    if not paths:
        return pd.DataFrame()
    frames = []
    for p in paths:
        df = pd.read_json(p, lines=True)
        df["_source_file"] = p.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _summary_step_from_name(name: str) -> int | None:
    m = re.search(r"routing_summary_step(\d+)_batch", name)
    return int(m.group(1)) if m else None


def _raw_npz_step_from_name(name: str) -> int | None:
    m = re.search(r"routing_raw_step(\d+)_batch", name)
    return int(m.group(1)) if m else None


def _discover_layer_keys(z: Any) -> list[int]:
    layers: set[int] = set()
    for k in z.files:
        m = re.match(r"layer_(\d+)_topk_experts$", k)
        if m:
            layers.add(int(m.group(1)))
    return sorted(layers)


def _discover_moe_layer_indices(traces_dir: Path) -> list[int]:
    """MoE layer indices present in ``raw/routing_raw_*.npz`` (first file that loads)."""
    paths = sorted(
        traces_dir.glob("raw/routing_raw_*.npz"),
        key=lambda p: (_raw_npz_step_from_name(p.name) or -1, p.name),
    )
    for p in paths:
        z = None
        try:
            z = np.load(p, mmap_mode="r")
            layers = _discover_layer_keys(z)
            if layers:
                return layers
        except (OSError, ValueError):
            continue
        finally:
            if z is not None:
                z.close()
    return []


def _paper_heatmap_layer_list(traces_dir: Path, paper_layer: int) -> list[int]:
    """``paper_layer < 0`` → all layers from NPZ; else single layer index."""
    if paper_layer >= 0:
        return [paper_layer]
    found = _discover_moe_layer_indices(traces_dir)
    return found if found else [0]


def _pair_matrix_topk_slots(
    topk: np.ndarray,
    *,
    bytes_per_token: float,
    num_experts: int,
) -> np.ndarray:
    """Directed expert×expert weights from top-k slot pairs (paper-style heatmap proxy).

    For each token and ordered slot pair (a,b), a≠b, add
    bytes_per_token / (k*(k-1)) to M[expert[a], expert[b]]. Summing M equals
    T*k*bpt in the k-regular case (matches assignment-weight intuition).
    """
    if topk.ndim != 2 or topk.shape[0] == 0:
        return np.zeros((num_experts, num_experts), dtype=np.float64)
    t, k = topk.shape
    if k < 2:
        return np.zeros((num_experts, num_experts), dtype=np.float64)
    w = float(bytes_per_token) / float(k * (k - 1))
    m = np.zeros((num_experts, num_experts), dtype=np.float64)
    # Vectorized over tokens
    for a in range(k):
        for b in range(k):
            if a == b:
                continue
            ea = topk[:, a].astype(np.int64, copy=False)
            eb = topk[:, b].astype(np.int64, copy=False)
            np.add.at(m, (ea, eb), w)
    return m


def _load_pair_matrix_from_npz(
    path: Path, layer_idx: int
) -> tuple[np.ndarray, int] | None:
    z: Any = None
    try:
        z = np.load(path, allow_pickle=True)
        tk = f"layer_{layer_idx}_topk_experts"
        if tk not in z.files:
            return None
        topk_full = np.asarray(z[tk])
        bpt_key = f"layer_{layer_idx}_bytes_per_token"
        ne_key = f"layer_{layer_idx}_num_experts"
        bpt = float(np.asarray(z[bpt_key]).reshape(())) if bpt_key in z.files else 0.0
        num_experts = int(np.asarray(z[ne_key]).reshape(())) if ne_key in z.files else 0
        if "token_mask" in z.files:
            mask = np.asarray(z["token_mask"], dtype=bool)
            if mask.shape[0] == topk_full.shape[0]:
                topk = topk_full[mask]
            else:
                topk = topk_full
        else:
            topk = topk_full
        if num_experts <= 0:
            num_experts = int(topk.max()) + 1 if topk.size else 1
        mat = _pair_matrix_topk_slots(topk, bytes_per_token=bpt, num_experts=num_experts)
        return mat, num_experts
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None
    finally:
        if z is not None:
            z.close()


def _aggregate_raw_pair_by_trace_step(
    traces_dir: Path, layer_idx: int
) -> dict[int, np.ndarray]:
    paths = sorted(
        traces_dir.glob("raw/routing_raw_*.npz"),
        key=lambda p: (_raw_npz_step_from_name(p.name) or -1, p.name),
    )
    buckets: dict[int, list[np.ndarray]] = defaultdict(list)
    for p in paths:
        st = _raw_npz_step_from_name(p.name)
        if st is None:
            continue
        got = _load_pair_matrix_from_npz(p, layer_idx)
        if got is None:
            continue
        mat, _ = got
        buckets[st].append(mat)
    merged: dict[int, np.ndarray] = {}
    for st, mats in buckets.items():
        if not mats:
            continue
        e = max(m.shape[0] for m in mats)
        acc = np.zeros((e, e), dtype=np.float64)
        for m in mats:
            acc[: m.shape[0], : m.shape[1]] += m
        merged[st] = acc
    return merged


def _crop_expert_indices(matrices: list[np.ndarray], max_experts: int) -> np.ndarray:
    """Return sorted expert indices to keep (by total in+out mass).

    ``max_experts <= 0`` keeps **all** experts (no crop).
    """
    if not matrices:
        return np.arange(0)
    e = max(m.shape[0] for m in matrices)
    if max_experts <= 0 or max_experts >= e:
        return np.arange(e, dtype=int)
    mass = np.zeros(e, dtype=np.float64)
    for m in matrices:
        ee = m.shape[0]
        mass[:ee] += m.sum(axis=0) + m.sum(axis=1)
    order = np.argsort(-mass)
    keep = order[: min(max_experts, e)]
    return np.sort(keep)


def _row_normalize_2d(mat: np.ndarray) -> np.ndarray:
    """Row-normalize a 2D matrix; zero rows remain zero."""
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return np.asarray(arr, dtype=np.float64)
    den = arr.sum(axis=1, keepdims=True)
    out = np.zeros_like(arr, dtype=np.float64)
    np.divide(arr, den, out=out, where=den > 0)
    return out


def _read_one_summary(p: Path) -> pd.DataFrame:
    df = pd.read_json(p, lines=True)
    if not df.empty and "record_type" in df.columns:
        df = df[df["record_type"].astype("string") != "routing_write_meta"]
    df["_source_file"] = p.name
    return df


def _list_routing_summary_paths(traces_dir: Path, max_files: int | None) -> list[Path]:
    paths = sorted(
        traces_dir.glob("summary/routing_summary_*.jsonl"),
        key=lambda p: (_summary_step_from_name(p.name) or -1, p.name),
    )
    if not paths:
        return []
    if max_files is not None and len(paths) > max_files:
        idx = np.linspace(0, len(paths) - 1, max_files, dtype=int)
        paths = [paths[i] for i in sorted(set(idx))]
    return paths


def _load_routing_summary(traces_dir: Path, max_files: int | None) -> pd.DataFrame:
    paths = _list_routing_summary_paths(traces_dir, max_files)
    if not paths:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    workers = min(16, len(paths))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_read_one_summary, p): p for p in paths}
        for fut in as_completed(futs):
            frames.append(fut.result())
    return pd.concat(frames, ignore_index=True)


def _iter_jsonl_chunks(path: Path, chunk_lines: int) -> Iterable[pd.DataFrame]:
    """Yield DataFrames of at most ``chunk_lines`` rows from a JSONL file.

    Parses each line with :func:`json.loads` instead of ``pd.read_json`` on a
    joined buffer, so ujson/pandas bulk decoding does not fail on unusual
    string escaping inside fields.
    """
    buf: list[dict[str, Any]] = []
    bad = 0
    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                bad += 1
                if bad <= 3:
                    print(
                        f"Warning: bad JSONL {path.name}:{line_no}: {e}",
                        file=sys.stderr,
                    )
                continue
            if not isinstance(obj, dict):
                bad += 1
                continue
            buf.append(obj)
            if len(buf) >= chunk_lines:
                yield pd.DataFrame(buf)
                buf.clear()
    if buf:
        yield pd.DataFrame(buf)
    if bad > 3:
        print(
            f"Warning: skipped {bad} bad JSONL line(s) in {path.name}",
            file=sys.stderr,
        )


EC_HIST_MAX = 512


def _paper_a_pivot_all_experts(pivot: pd.DataFrame) -> tuple[pd.DataFrame, list[Any]]:
    """Reorder routing pivot columns by expert id; every expert stays its own series (no ``other``)."""
    cols = list(pivot.columns)

    def sort_key(c: Any) -> tuple[int, float, str]:
        n = pd.to_numeric(c, errors="coerce")
        if pd.isna(n):
            return (1, float("inf"), str(c))
        return (0, float(n), str(c))

    ordered = sorted(cols, key=sort_key)
    return pivot[ordered], ordered


def _paper_a_expert_legend_label(c: Any) -> str:
    n = pd.to_numeric(c, errors="coerce")
    if pd.notna(n):
        return f"E{int(n)}"
    return f"E{c}"


def _expert_id_sort_key(c: Any) -> tuple[int, float, str]:
    n = pd.to_numeric(c, errors="coerce")
    if pd.isna(n):
        return (1, float("inf"), str(c))
    return (0, float(n), str(c))


def _plot_bar_dispatch_bytes_per_expert(
    expert_bytes: Mapping[Any, float],
    out_dir: Path,
    fname: str,
    *,
    title_suffix: str,
) -> Path | None:
    """One bar per expert: total ``estimated_dispatch_bytes`` (routing summary)."""
    if not expert_bytes:
        return None
    items = sorted(expert_bytes.items(), key=lambda kv: _expert_id_sort_key(kv[0]))
    eids = [k for k, _ in items]
    vals = np.asarray([float(v) for _, v in items], dtype=float)
    n = len(eids)
    fig_w = min(28.0, max(9.0, 0.11 * n))
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))
    pos = np.arange(n)
    ax.bar(pos, vals / 1e9, width=0.88, align="center", alpha=0.88, color="steelblue")
    ax.set_xticks(pos)
    fs = max(4, min(8, 220 // max(n, 1)))
    ax.set_xticklabels([_paper_a_expert_legend_label(e) for e in eids], rotation=90, fontsize=fs)
    ax.set_title(
        "Total estimated_dispatch_bytes per expert (sum over all routing-summary rows)"
        + title_suffix
    )
    ax.set_xlabel("expert_id")
    ax.set_ylabel("GB (estimated_dispatch_bytes)")
    ax.grid(True, alpha=0.3, axis="y")
    return _save(fig, out_dir, fname)


def _sort_layer_keys_numeric(keys: Iterable[Any]) -> list[Any]:
    keyed: list[tuple[Any, Any, Any]] = []
    for x in keys:
        try:
            keyed.append((0, int(float(x)), x))
        except (TypeError, ValueError):
            keyed.append((1, str(x), x))
    keyed.sort(key=lambda t: (t[0], t[1]))
    return [t[2] for t in keyed]


def _plot_layer_expert_load_histogram_pdf(
    layer_expert_load: Mapping[Any, Mapping[Any, float]],
    out_dir: Path,
    fname: str,
    *,
    title_suffix: str,
) -> Path | None:
    """Multi-page PDF: one page per layer, bar chart expert_id vs summed expert_count."""
    if not layer_expert_load:
        return None
    from matplotlib.backends.backend_pdf import PdfPages

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{fname}.pdf"
    layers = _sort_layer_keys_numeric(layer_expert_load.keys())
    n_saved = 0
    with PdfPages(path) as pdf:
        for li in layers:
            d = layer_expert_load[li]
            if not d:
                continue
            items = sorted(d.items(), key=lambda kv: _expert_id_sort_key(kv[0]))
            eids = [k for k, _ in items]
            vals = np.asarray([float(v) for _, v in items], dtype=float)
            n = len(eids)
            if n == 0:
                continue
            fig_w = min(28.0, max(9.0, 0.11 * n))
            fig, ax = plt.subplots(figsize=(fig_w, 5.2))
            pos = np.arange(n)
            ax.bar(pos, vals, width=0.88, align="center", alpha=0.88, color="teal")
            ax.set_xticks(pos)
            fs = max(4, min(8, 220 // max(n, 1)))
            ax.set_xticklabels(
                [_paper_a_expert_legend_label(e) for e in eids], rotation=90, fontsize=fs
            )
            ax.set_title(
                f"Expert load (summed expert_count), layer_idx={li}" + title_suffix
            )
            ax.set_xlabel("expert_id")
            ax.set_ylabel("summed expert_count (token × top-k assignments)")
            ax.grid(True, alpha=0.3, axis="y")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            n_saved += 1
    if n_saved == 0:
        path.unlink(missing_ok=True)
        return None
    return path


def _routing_phase_nz_bytes_and_rows(
    routing: pd.DataFrame, phcol: pd.Series, phase: str
) -> tuple[float, int]:
    """Sum(dispatch+return) and row count for rows with expert_count > 0 in one phase."""
    sub = routing.loc[phcol == phase]
    if sub.empty or "expert_count" not in sub.columns:
        return 0.0, 0
    ec = pd.to_numeric(sub["expert_count"], errors="coerce").fillna(0)
    nz = ec > 0
    if not nz.any():
        return 0.0, 0
    s = sub.loc[nz]
    d = float(
        pd.to_numeric(s["estimated_dispatch_bytes"], errors="coerce").fillna(0).sum()
    )
    if "estimated_return_bytes" in s.columns:
        d += float(
            pd.to_numeric(s["estimated_return_bytes"], errors="coerce").fillna(0).sum()
        )
    return d, int(nz.sum())


def _plot_routing_prefill_decode_summary(
    *,
    phase_dispatch: Mapping[str, float],
    phase_return: Mapping[str, float],
    n_prefill_rows: int,
    n_decode_rows: int,
    n_mixed_rows: int = 0,
    phase_nz_bytes_sum: Mapping[str, float] | None = None,
    phase_nz_row_count: Mapping[str, int] | None = None,
    out_dir: Path,
    fname: str,
    title_suffix: str,
) -> Path | None:
    """Three panels: total GB by phase; mean KB/row (all rows); mean KB/row where expert_count > 0."""

    def _tot_bytes(ph: str) -> float:
        return float(phase_dispatch.get(ph, 0.0)) + float(phase_return.get(ph, 0.0))

    pre_b = _tot_bytes("prefill")
    dec_b = _tot_bytes("decode")
    mix_b = _tot_bytes("mixed")
    if pre_b + dec_b + mix_b <= 0.0:
        return None

    nz_sum = phase_nz_bytes_sum or {}
    nz_cnt = phase_nz_row_count or {}

    n_pr = max(int(n_prefill_rows), 0)
    n_dr = max(int(n_decode_rows), 0)
    n_mr = max(int(n_mixed_rows), 0)
    # Mean "message" size = total estimated MoE bytes (dispatch+return) / number of summary rows.
    avg_pre_kb = (pre_b / n_pr / 1e3) if n_pr else 0.0
    avg_dec_kb = (dec_b / n_dr / 1e3) if n_dr else 0.0
    avg_mix_kb = (mix_b / n_mr / 1e3) if n_mr and mix_b > 0.0 else 0.0

    n_pr_nz = max(int(nz_cnt.get("prefill", 0)), 0)
    n_dr_nz = max(int(nz_cnt.get("decode", 0)), 0)
    n_mr_nz = max(int(nz_cnt.get("mixed", 0)), 0)
    nz_pre_b = float(nz_sum.get("prefill", 0.0))
    nz_dec_b = float(nz_sum.get("decode", 0.0))
    nz_mix_b = float(nz_sum.get("mixed", 0.0))
    avg_nz_pre_kb = (nz_pre_b / n_pr_nz / 1e3) if n_pr_nz else 0.0
    avg_nz_dec_kb = (nz_dec_b / n_dr_nz / 1e3) if n_dr_nz else 0.0
    avg_nz_mix_kb = (nz_mix_b / n_mr_nz / 1e3) if n_mr_nz and nz_mix_b > 0.0 else 0.0

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.8), constrained_layout=True)
    ax0, ax1, ax2 = axes

    labels_tot = ["prefill", "decode"]
    vals_gb = [pre_b / 1e9, dec_b / 1e9]
    colors = ["tab:blue", "tab:orange"]
    if mix_b > 0.0:
        labels_tot.append("mixed")
        vals_gb.append(mix_b / 1e9)
        colors.append("tab:gray")
    x0 = np.arange(len(labels_tot))
    ax0.bar(x0, vals_gb, color=colors, alpha=0.88, width=0.65)
    ax0.set_xticks(x0)
    ax0.set_xticklabels(labels_tot)
    ax0.set_ylabel("GB (dispatch + return, estimated)")
    ax0.set_title("Total MoE traffic by phase" + title_suffix)
    ax0.grid(True, alpha=0.3, axis="y")

    labels_msg = [
        f"prefill\n({n_pr} rows)",
        f"decode\n({n_dr} rows)",
    ]
    vals_kb = [avg_pre_kb, avg_dec_kb]
    colors_msg = ["tab:blue", "tab:orange"]
    if mix_b > 0.0 and n_mr > 0:
        labels_msg.append(f"mixed\n({n_mr} rows)")
        vals_kb.append(avg_mix_kb)
        colors_msg.append("tab:gray")

    ax1.bar(
        labels_msg,
        vals_kb,
        color=colors_msg,
        alpha=0.88,
        width=0.55,
    )
    ax1.set_ylabel("KB per routing-summary row (mean)")
    ax1.set_title(
        "Mean KB/row (all rows)\n"
        "(dispatch+return / row count)"
        + title_suffix
    )
    ax1.grid(True, alpha=0.3, axis="y")

    labels_nz = [
        f"prefill\n({n_pr_nz} rows)",
        f"decode\n({n_dr_nz} rows)",
    ]
    vals_nz_kb = [avg_nz_pre_kb, avg_nz_dec_kb]
    colors_nz = ["tab:blue", "tab:orange"]
    if mix_b > 0.0 and n_mr_nz > 0:
        labels_nz.append(f"mixed\n({n_mr_nz} rows)")
        vals_nz_kb.append(avg_nz_mix_kb)
        colors_nz.append("tab:gray")

    ax2.bar(
        labels_nz,
        vals_nz_kb,
        color=colors_nz,
        alpha=0.88,
        width=0.55,
    )
    ax2.set_ylabel("KB per row (mean)")
    ax2.set_title(
        "Mean KB/row (expert_count > 0 only)\n"
        "(dispatch+return / nonzero expert rows)"
        + title_suffix
    )
    ax2.grid(True, alpha=0.3, axis="y")

    return _save(fig, out_dir, fname)


class RoutingStreamAgg:
    """Incremental aggregates over routing summary JSONL (memory-safe for huge runs)."""

    def __init__(self, scatter_sample_size: int = 50_000) -> None:
        self.scatter_sample_size = scatter_sample_size
        self._step: dict[Any, dict[str, float]] = defaultdict(
            lambda: {"sum_dispatch": 0.0, "sum_return": 0.0, "sum_exact": 0.0, "n": 0}
        )
        self._phase_hist: dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(EC_HIST_MAX + 1, dtype=np.int64)
        )
        self._layer_moments: dict[Any, dict[str, float]] = {}
        self._layer_dispatch_sum: dict[Any, float] = defaultdict(float)
        self._expert_hits: dict[Any, float] = defaultdict(float)
        self._expert_dispatch_bytes: dict[Any, float] = defaultdict(float)
        self._layer_expert_load: dict[Any, dict[Any, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._phase_dispatch_sum: dict[str, float] = defaultdict(float)
        self._phase_return_sum: dict[str, float] = defaultdict(float)
        self._phase_row_count: dict[str, int] = defaultdict(int)
        self._phase_nz_bytes_sum: dict[str, float] = defaultdict(float)
        self._phase_nz_row_count: dict[str, int] = defaultdict(int)
        self._prefill_req_keys: set[tuple[Any, Any]] = set()
        self._decode_req_keys: set[tuple[Any, Any]] = set()
        self._decode_step: dict[Any, list[float]] = defaultdict(lambda: [0.0, 0.0])
        self._stack: dict[tuple[Any, Any], float] = defaultdict(float)
        self._stack_layer0: dict[tuple[Any, Any], float] = defaultdict(float)
        self._min_layer: int | None = None
        self._frac: dict[tuple[Any, Any], dict[Any, float]] = defaultdict(lambda: defaultdict(float))
        self._k_hist: dict[int, np.ndarray] = defaultdict(
            lambda: np.zeros(EC_HIST_MAX + 1, dtype=np.int64)
        )
        self._reservoir: list[tuple[float, float]] = []
        self._reservoir_n = 0
        self.total_rows = 0

    def _ensure_layer_moment(self, layer: Any) -> dict[str, float]:
        if layer not in self._layer_moments:
            self._layer_moments[layer] = {
                "n": 0.0,
                "mean": 0.0,
                "M2": 0.0,
                "maxv": float("-inf"),
            }
        return self._layer_moments[layer]

    def _welford_update_batch(self, layer: Any, vals: np.ndarray) -> None:
        if vals.size == 0:
            return
        st = self._ensure_layer_moment(layer)
        n, mean, M2 = int(st["n"]), float(st["mean"]), float(st["M2"])
        maxv = float(st["maxv"])
        for x in vals.astype(np.float64, copy=False).ravel():
            if not math.isfinite(x):
                continue
            n += 1
            d = x - mean
            mean += d / n
            d2 = x - mean
            M2 += d * d2
            if x > maxv:
                maxv = x
        st["n"] = float(n)
        st["mean"] = mean
        st["M2"] = M2
        st["maxv"] = maxv

    def _reservoir_add(self, ux: np.ndarray, ec: np.ndarray) -> None:
        k = self.scatter_sample_size
        for a, b in zip(ux.tolist(), ec.tolist()):
            if not (math.isfinite(float(a)) and math.isfinite(float(b))):
                continue
            self._reservoir_n += 1
            pair = (float(a), float(b))
            if len(self._reservoir) < k:
                self._reservoir.append(pair)
            else:
                j = np.random.randint(0, self._reservoir_n)
                if j < k:
                    self._reservoir[j] = pair

    def add_chunk(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        self.total_rows += len(df)

        if "layer_idx" in df.columns:
            li = pd.to_numeric(df["layer_idx"], errors="coerce")
            valid = li.notna()
            if valid.any():
                m = int(pd.Series(li[valid]).min())
                self._min_layer = m if self._min_layer is None else min(self._min_layer, m)

        if "trace_step" in df.columns:
            g = df.groupby("trace_step", as_index=True)
            if "estimated_dispatch_bytes" in df.columns:
                sd = g["estimated_dispatch_bytes"].sum()
                for ts, v in sd.items():
                    self._step[ts]["sum_dispatch"] += float(v)
            if "estimated_return_bytes" in df.columns:
                sr = g["estimated_return_bytes"].sum()
                for ts, v in sr.items():
                    self._step[ts]["sum_return"] += float(v)
            if "routing_is_exact" in df.columns:
                ri = pd.to_numeric(df["routing_is_exact"], errors="coerce").fillna(0)
                tmp = df.assign(_ri=ri)
                for ts, part in tmp.groupby("trace_step"):
                    self._step[ts]["sum_exact"] += float(part["_ri"].sum())
                    self._step[ts]["n"] += float(len(part))

        if "phase" in df.columns and "expert_count" in df.columns:
            ec = pd.to_numeric(df["expert_count"], errors="coerce").fillna(0).astype(np.int64)
            for ph, part in df.groupby("phase"):
                phs = str(ph)
                h = self._phase_hist[phs]
                v = part["expert_count"].to_numpy()
                v = np.clip(np.nan_to_num(v, nan=0).astype(np.int64), 0, EC_HIST_MAX)
                bc = np.bincount(v, minlength=EC_HIST_MAX + 1)
                h[: len(bc)] += bc.astype(np.int64)

        if "phase" in df.columns:
            for ph, part in df.groupby("phase"):
                phs = str(ph)
                self._phase_row_count[phs] += int(len(part))
                if "estimated_dispatch_bytes" in part.columns:
                    self._phase_dispatch_sum[phs] += float(
                        pd.to_numeric(part["estimated_dispatch_bytes"], errors="coerce")
                        .fillna(0)
                        .sum()
                    )
                if "estimated_return_bytes" in part.columns:
                    self._phase_return_sum[phs] += float(
                        pd.to_numeric(part["estimated_return_bytes"], errors="coerce")
                        .fillna(0)
                        .sum()
                    )
                if (
                    "expert_count" in part.columns
                    and "estimated_dispatch_bytes" in part.columns
                ):
                    ec = pd.to_numeric(part["expert_count"], errors="coerce").fillna(0)
                    nz_mask = ec > 0
                    if bool(nz_mask.any()):
                        sub = part.loc[nz_mask]
                        dnz = pd.to_numeric(
                            sub["estimated_dispatch_bytes"], errors="coerce"
                        ).fillna(0).sum()
                        if "estimated_return_bytes" in sub.columns:
                            rnz = pd.to_numeric(
                                sub["estimated_return_bytes"], errors="coerce"
                            ).fillna(0).sum()
                        else:
                            rnz = 0.0
                        self._phase_nz_bytes_sum[phs] += float(dnz + rnz)
                        self._phase_nz_row_count[phs] += int(nz_mask.sum())
                if (
                    "trace_step" in part.columns
                    and "request_id" in part.columns
                    and phs in ("prefill", "decode")
                ):
                    sub = part[["trace_step", "request_id"]].drop_duplicates()
                    keys = zip(sub["trace_step"].tolist(), sub["request_id"].tolist())
                    if phs == "prefill":
                        self._prefill_req_keys.update(keys)
                    else:
                        self._decode_req_keys.update(keys)

        if "layer_idx" in df.columns and "expert_count" in df.columns:
            for layer, part in df.groupby("layer_idx"):
                vals = pd.to_numeric(part["expert_count"], errors="coerce").to_numpy()
                self._welford_update_batch(layer, vals)
            if "estimated_dispatch_bytes" in df.columns:
                ld = df.groupby("layer_idx")["estimated_dispatch_bytes"].sum()
                for layer, v in ld.items():
                    self._layer_dispatch_sum[layer] += float(v)

        if "expert_id" in df.columns and "expert_count" in df.columns:
            sub = df.groupby("expert_id")["expert_count"].sum()
            for eid, v in sub.items():
                self._expert_hits[eid] += float(v)

        if (
            "layer_idx" in df.columns
            and "expert_id" in df.columns
            and "expert_count" in df.columns
        ):
            le = df.groupby(["layer_idx", "expert_id"], as_index=False)[
                "expert_count"
            ].sum()
            for _, row in le.iterrows():
                li, ei = row["layer_idx"], row["expert_id"]
                v = float(pd.to_numeric(row["expert_count"], errors="coerce") or 0.0)
                self._layer_expert_load[li][ei] += v

        if "expert_id" in df.columns and "estimated_dispatch_bytes" in df.columns:
            sub = df.groupby("expert_id")["estimated_dispatch_bytes"].sum()
            for eid, v in sub.items():
                self._expert_dispatch_bytes[eid] += float(v)

        if (
            "phase" in df.columns
            and "decode_step" in df.columns
            and "estimated_dispatch_bytes" in df.columns
        ):
            dec = df[df["phase"].astype(str) == "decode"]
            if not dec.empty:
                ds = dec.groupby("decode_step")["estimated_dispatch_bytes"]
                for dsk, part in ds:
                    s, c = self._decode_step[dsk]
                    partv = pd.to_numeric(part, errors="coerce")
                    s += float(partv.sum())
                    c += float(partv.count())
                    self._decode_step[dsk] = [s, c]

        if (
            "trace_step" in df.columns
            and "expert_id" in df.columns
            and "estimated_dispatch_bytes" in df.columns
        ):
            agg = df.groupby(["trace_step", "expert_id"], as_index=False)[
                "estimated_dispatch_bytes"
            ].sum()
            for _, row in agg.iterrows():
                ts, eid = row["trace_step"], row["expert_id"]
                self._stack[(ts, eid)] += float(row["estimated_dispatch_bytes"])
            if self._min_layer is not None and "layer_idx" in df.columns:
                m = self._min_layer
                sub = df[pd.to_numeric(df["layer_idx"], errors="coerce") == m]
                if not sub.empty:
                    agg0 = sub.groupby(["trace_step", "expert_id"], as_index=False)[
                        "estimated_dispatch_bytes"
                    ].sum()
                    for _, row in agg0.iterrows():
                        self._stack_layer0[(row["trace_step"], row["expert_id"])] += float(
                            row["estimated_dispatch_bytes"]
                        )

        if (
            "trace_step" in df.columns
            and "layer_idx" in df.columns
            and "expert_id" in df.columns
            and "expert_count" in df.columns
        ):
            for _, row in df.iterrows():
                ts, li, ei = row["trace_step"], row["layer_idx"], row["expert_id"]
                v = float(pd.to_numeric(row["expert_count"], errors="coerce") or 0.0)
                self._frac[(ts, li)][ei] += v

        if "k" in df.columns and "expert_count" in df.columns:
            for kval, part in df.groupby("k"):
                try:
                    ki = int(kval)
                except (TypeError, ValueError):
                    continue
                v = (
                    pd.to_numeric(part["expert_count"], errors="coerce")
                    .fillna(0)
                    .astype(np.int64)
                    .to_numpy()
                )
                v = np.clip(v, 0, EC_HIST_MAX)
                bc = np.bincount(v, minlength=EC_HIST_MAX + 1)
                self._k_hist[ki][: len(bc)] += bc.astype(np.int64)

        if "unique_token_count" in df.columns and "expert_count" in df.columns:
            ux = pd.to_numeric(df["unique_token_count"], errors="coerce").to_numpy()
            ec = pd.to_numeric(df["expert_count"], errors="coerce").to_numpy()
            self._reservoir_add(ux, ec)

    def to_step_agg_dataframe(self) -> pd.DataFrame:
        if not self._step:
            return pd.DataFrame()
        rows = []
        for ts, st in sorted(self._step.items(), key=lambda x: x[0]):
            n = max(st["n"], 1.0)
            rows.append(
                {
                    "trace_step": ts,
                    "sum_dispatch": st["sum_dispatch"],
                    "sum_return": st["sum_return"],
                    "mean_routing_exact": st["sum_exact"] / n,
                    "rows": st["n"],
                }
            )
        return pd.DataFrame(rows)

    def build_routing_step_merge_df(self) -> pd.DataFrame:
        """Minimal df for plot_step_vs_routing (sum_dispatch per trace_step)."""
        df = self.to_step_agg_dataframe()
        if df.empty:
            return df
        return df.rename(columns={"sum_dispatch": "r_sum_dispatch"})


def _stream_aggregate_routing(
    traces_dir: Path,
    max_files: int | None,
    chunk_lines: int,
) -> RoutingStreamAgg:
    paths = _list_routing_summary_paths(traces_dir, max_files)
    agg = RoutingStreamAgg()
    for p in paths:
        for chunk in _iter_jsonl_chunks(p, chunk_lines):
            chunk["_source_file"] = p.name
            agg.add_chunk(chunk)
    return agg


def _heatmap_across_paths_chunked(
    paths: list[Path],
    value_col: str,
    chunk_lines: int,
    *,
    phase: str | None = None,
) -> pd.DataFrame | None:
    """Layer × expert means from JSONL paths, reading at most ``chunk_lines`` rows at a time."""
    sum_p: dict[tuple[Any, Any], float] = defaultdict(float)
    cnt_p: dict[tuple[Any, Any], int] = defaultdict(int)
    for path in paths:
        for chunk in _iter_jsonl_chunks(path, chunk_lines):
            if value_col not in chunk.columns or "layer_idx" not in chunk.columns:
                return None
            if "expert_id" not in chunk.columns:
                return None
            work = chunk
            if phase is not None:
                if "phase" not in chunk.columns:
                    return None
                work = chunk[chunk["phase"].astype(str) == phase]
            if work.empty:
                continue
            g = work.groupby(["layer_idx", "expert_id"], as_index=False).agg(
                _sum=(value_col, "sum"),
                _cnt=(value_col, "count"),
            )
            for _, row in g.iterrows():
                key = (row["layer_idx"], row["expert_id"])
                sum_p[key] += float(row["_sum"])
                cnt_p[key] += int(row["_cnt"])
    if not sum_p:
        return None
    rows = []
    for key in sum_p:
        li, ei = key
        c = max(cnt_p[key], 1)
        rows.append({"layer_idx": li, "expert_id": ei, value_col: sum_p[key] / c})
    return pd.DataFrame(rows)


def _heatmap_from_file_chunked(
    path: Path, value_col: str, chunk_lines: int, *, phase: str | None = None
) -> pd.DataFrame | None:
    """Mean pivot layer × expert for one summary file (optional phase filter), chunked."""
    return _heatmap_across_paths_chunked([path], value_col, chunk_lines, phase=phase)


def plot_routing_from_agg(
    agg: RoutingStreamAgg,
    traces_dir: Path,
    out_dir: Path,
    *,
    chunk_lines: int,
    max_summary_files: int | None = None,
    prefix: str = "routing",
) -> list[Path]:
    """Plots from RoutingStreamAgg + chunked heatmap reads."""
    saved: list[Path] = []
    g = agg.to_step_agg_dataframe()
    if not g.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(g["trace_step"], g["sum_dispatch"] / 1e9, label="sum dispatch", lw=1)
        ax.plot(g["trace_step"], g["sum_return"] / 1e9, label="sum return", lw=1, alpha=0.7)
        ax.set_title("Routing summary: summed estimated MoE bytes per trace_step (streamed)")
        ax.set_xlabel("trace_step")
        ax.set_ylabel("GB (sum over all rows in summary files)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        saved.append(_save(fig, out_dir, f"{prefix}_sum_bytes_per_trace_step"))

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(g["trace_step"], g["mean_routing_exact"], marker=".", ms=3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Mean routing_is_exact per trace_step (streamed)")
        ax.set_xlabel("trace_step")
        ax.grid(True, alpha=0.3)
        saved.append(_save(fig, out_dir, f"{prefix}_routing_is_exact_mean"))

    for ph, h in sorted(agg._phase_hist.items()):
        if h.sum() == 0:
            continue
        edges = np.arange(len(h))
        fig, ax = plt.subplots(figsize=(8, 4))
        nz = h > 0
        ax.bar(edges[nz], h[nz], log=True, alpha=0.85, width=1.0)
        ax.set_title(f"Histogram of expert_count (phase={ph}, log y, streamed)")
        ax.set_xlabel("expert_count")
        ax.set_ylabel("count (log)")
        ax.grid(True, alpha=0.3)
        saved.append(_save(fig, out_dir, f"{prefix}_hist_expert_count_{ph}"))

    if agg._layer_moments:
        layers = sorted(agg._layer_moments.keys(), key=lambda x: (isinstance(x, float) and math.isnan(x), x))
        means = [agg._layer_moments[L]["mean"] for L in layers]
        stds = [
            math.sqrt(agg._layer_moments[L]["M2"] / max(agg._layer_moments[L]["n"] - 1, 1))
            if agg._layer_moments[L]["n"] > 1
            else 0.0
            for L in layers
        ]
        maxs = [agg._layer_moments[L]["maxv"] for L in layers]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(layers, means, label="mean expert_count")
        ax.plot(layers, maxs, label="max expert_count", alpha=0.8)
        ax.set_title("Expert token load vs layer (streamed aggregates)")
        ax.set_xlabel("layer_idx")
        ax.legend()
        ax.grid(True, alpha=0.3)
        saved.append(_save(fig, out_dir, f"{prefix}_layer_mean_max_expert_count"))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(layers, stds, color="darkred")
        ax.set_title("Std of expert_count across rows per layer (streamed)")
        ax.set_xlabel("layer_idx")
        ax.grid(True, alpha=0.3)
        saved.append(_save(fig, out_dir, f"{prefix}_layer_std_expert_count"))

    paths = _list_routing_summary_paths(traces_dir, max_summary_files)
    steps = sorted(
        {_summary_step_from_name(p.name) for p in paths if _summary_step_from_name(p.name) is not None}
    )
    if steps:
        pick_idx = sorted({0, len(steps) // 2, len(steps) - 1})
        picks = [steps[i] for i in pick_idx]
        for ts in picks:
            cands = [p for p in paths if _summary_step_from_name(p.name) == ts]
            if not cands:
                continue
            path_one = cands[0]
            for col, tag in (("expert_count", "expert_count_mean"), ("estimated_dispatch_bytes", "dispatch_bytes_mean")):
                sub = _heatmap_from_file_chunked(path_one, col, chunk_lines)
                if sub is None or sub.empty:
                    continue
                p = _heatmap_layer_expert(
                    sub,
                    f"{col} heatmap (trace_step={ts}, streamed)",
                    col,
                    out_dir,
                    f"{prefix}_heatmap_{tag}_step{ts}",
                )
                if p:
                    saved.append(p)

        ts_last = int(steps[-1])
        cands = [p for p in paths if _summary_step_from_name(p.name) == ts_last]
        if cands:
            for ph in ("prefill", "decode", "mixed"):
                sub = _heatmap_across_paths_chunked(
                    cands, "expert_count", chunk_lines, phase=ph
                )
                if sub is None or sub.empty:
                    continue
                p = _heatmap_layer_expert(
                    sub,
                    f"expert_count heatmap step={ts_last} phase={ph} (streamed)",
                    "expert_count",
                    out_dir,
                    f"{prefix}_heatmap_expert_count_step{ts_last}_{ph}",
                )
                if p:
                    saved.append(p)

    if agg._reservoir:
        sample = np.array(agg._reservoir)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(sample[:, 0], sample[:, 1], s=2, alpha=0.25)
        ax.set_title("expert_count vs unique_token_count (reservoir sample, streamed)")
        ax.set_xlabel("unique_token_count")
        ax.set_ylabel("expert_count")
        ax.grid(True, alpha=0.3)
        saved.append(_save(fig, out_dir, f"{prefix}_scatter_unique_vs_expert"))

    if agg._decode_step:
        rows = []
        for ds, (s, c) in sorted(agg._decode_step.items(), key=lambda x: x[0]):
            rows.append({"decode_step": ds, "m": s / max(c, 1.0)})
        g2 = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(g2["decode_step"], g2["m"] / 1e6, marker=".", ms=3)
        ax.set_title("Decode: mean estimated_dispatch_bytes vs decode_step (streamed)")
        ax.set_xlabel("decode_step")
        ax.set_ylabel("MB (mean over rows)")
        ax.grid(True, alpha=0.3)
        saved.append(_save(fig, out_dir, f"{prefix}_decode_mean_dispatch_vs_decode_step"))

    if agg._layer_dispatch_sum:
        ld = sorted(agg._layer_dispatch_sum.items(), key=lambda x: x[0])
        xs, ys = zip(*ld)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(xs, np.asarray(ys, dtype=float) / 1e9, width=0.9, alpha=0.85)
        ax.set_title("Sum of estimated_dispatch_bytes by layer (streamed)")
        ax.set_xlabel("layer_idx")
        ax.set_ylabel("GB")
        ax.grid(True, alpha=0.3, axis="y")
        saved.append(_save(fig, out_dir, f"{prefix}_layer_sum_dispatch_bar"))

    if agg._expert_hits:
        top = sorted(agg._expert_hits.items(), key=lambda x: -x[1])[:32]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar([str(t[0]) for t in top], [t[1] for t in top], alpha=0.85)
        ax.set_title("Top 32 experts by summed expert_count (streamed)")
        ax.set_xlabel("expert_id")
        ax.set_ylabel("sum expert_count")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3, axis="y")
        saved.append(_save(fig, out_dir, f"{prefix}_top32_experts_by_expert_count"))

    p_ex = _plot_bar_dispatch_bytes_per_expert(
        agg._expert_dispatch_bytes,
        out_dir,
        f"{prefix}_histogram_dispatch_bytes_per_expert",
        title_suffix="\n(streamed aggregate)",
    )
    if p_ex:
        saved.append(p_ex)

    p_layers = _plot_layer_expert_load_histogram_pdf(
        {k: dict(v) for k, v in agg._layer_expert_load.items()},
        out_dir,
        f"{prefix}_layer_expert_load_histograms",
        title_suffix="\n(streamed aggregate)",
    )
    if p_layers:
        saved.append(p_layers)

    p_ph = _plot_routing_prefill_decode_summary(
        phase_dispatch=agg._phase_dispatch_sum,
        phase_return=agg._phase_return_sum,
        n_prefill_rows=int(agg._phase_row_count.get("prefill", 0)),
        n_decode_rows=int(agg._phase_row_count.get("decode", 0)),
        n_mixed_rows=int(agg._phase_row_count.get("mixed", 0)),
        phase_nz_bytes_sum=agg._phase_nz_bytes_sum,
        phase_nz_row_count=agg._phase_nz_row_count,
        out_dir=out_dir,
        fname=f"{prefix}_prefill_decode_traffic_summary",
        title_suffix="\n(streamed aggregate)",
    )
    if p_ph:
        saved.append(p_ph)

    if agg._phase_hist:
        fig, ax = plt.subplots(figsize=(8, 5))
        for ph in sorted(agg._phase_hist.keys()):
            h = agg._phase_hist[ph].astype(np.float64)
            tot = h.sum()
            if tot <= 0:
                continue
            cdf = np.cumsum(h) / tot
            xs = np.arange(len(h))
            ax.plot(xs[h > 0], cdf[h > 0], label=str(ph), lw=1.5)
        ax.set_title("CDF of expert_count by phase (from streamed histograms)")
        ax.set_xlabel("expert_count")
        ax.set_ylabel("CDF")
        ax.legend()
        ax.grid(True, alpha=0.3)
        saved.append(_save(fig, out_dir, f"{prefix}_cdf_expert_count_by_phase"))

    if agg._frac:
        frac_rows = []
        for (ts, li), ed in agg._frac.items():
            if not ed:
                continue
            vals = list(ed.values())
            frac_rows.append(
                {
                    "trace_step": ts,
                    "layer_idx": li,
                    "frac_active": float(np.mean([float(v) > 0 for v in vals])),
                }
            )
        if frac_rows:
            u = pd.DataFrame(frac_rows)
            u_mean = u.groupby("trace_step", as_index=False)["frac_active"].mean()
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(u_mean["trace_step"], u_mean["frac_active"], marker=".", ms=4)
            ax.set_ylim(0, 1.02)
            ax.set_title("Mean fraction of experts with load > 0 (streamed)")
            ax.set_xlabel("trace_step")
            ax.set_ylabel("fraction")
            ax.grid(True, alpha=0.3)
            saved.append(_save(fig, out_dir, f"{prefix}_mean_frac_active_experts_per_step"))

    if agg._k_hist and len(agg._k_hist) <= 8:
        fig, ax = plt.subplots(figsize=(8, 4))
        for ki in sorted(agg._k_hist.keys()):
            h = agg._k_hist[ki]
            if h.sum() == 0:
                continue
            nz = h > 0
            ax.plot(np.arange(len(h))[nz], h[nz], alpha=0.6, label=f"k={ki}", lw=1)
        ax.set_yscale("log")
        ax.set_title("expert_count histogram by k (streamed, log y)")
        ax.legend()
        ax.set_xlabel("expert_count")
        ax.grid(True, alpha=0.3)
        saved.append(_save(fig, out_dir, f"{prefix}_hist_expert_count_by_k"))

    return saved


def plot_paper_stacked_from_agg(agg: RoutingStreamAgg, out_dir: Path, prefix: str = "paper_a") -> list[Path]:
    saved: list[Path] = []

    def _from_stack(stack: dict[tuple[Any, Any], float], title_suffix: str, fname_suffix: str) -> None:
        if not stack:
            return
        rows = [{"trace_step": a, "expert_id": b, "estimated_dispatch_bytes": v} for (a, b), v in stack.items()]
        df = pd.DataFrame(rows)
        agg2 = df.groupby(["trace_step", "expert_id"], as_index=False)["estimated_dispatch_bytes"].sum()
        pivot = agg2.pivot(index="trace_step", columns="expert_id", values="estimated_dispatch_bytes").fillna(0.0)
        pivot = pivot.sort_index()
        if pivot.shape[1] == 0:
            return
        pivot, cols = _paper_a_pivot_all_experts(pivot)
        n_exp = len(cols)
        x = pivot.index.to_numpy()
        series = [pivot[c].to_numpy(dtype=float) / 1e6 for c in cols]
        labels = [_paper_a_expert_legend_label(c) for c in cols]
        fig_w = min(22.0, 11.0 + n_exp / 40.0)
        fig, ax = plt.subplots(figsize=(fig_w, 5.5))
        ax.stackplot(x, *series, labels=labels, alpha=0.9)
        ax.set_title(
            "Stacked estimated dispatch (MB) by expert vs trace_step "
            + title_suffix
            + "\n(proxy for paper-style expert traffic share; from routing summary)"
        )
        ax.set_xlabel("trace_step")
        ax.set_ylabel("MB (sum of estimated_dispatch_bytes)")
        ncol = max(4, min(16, (n_exp + 7) // 8))
        fs = max(4, min(7, 140 // max(n_exp, 1)))
        ax.legend(loc="upper left", fontsize=fs, ncol=ncol, framealpha=0.92)
        ax.grid(True, alpha=0.25, axis="y")
        saved.append(_save(fig, out_dir, f"{prefix}_stacked_dispatch_mb_{fname_suffix}"))

    # Same basenames as plot_paper_style_stacked_expert_volume (legacy full load).
    _from_stack(agg._stack, "(all layers)", "all_layers")
    if agg._stack_layer0 and agg._min_layer is not None:
        _from_stack(
            agg._stack_layer0,
            f"(layer_idx == {agg._min_layer} only)",
            "first_layer_only",
        )
    return saved


def plot_step_vs_routing_agg(step: pd.DataFrame, agg: RoutingStreamAgg, out_dir: Path) -> list[Path]:
    saved: list[Path] = []
    rs = agg.build_routing_step_merge_df()
    if step.empty or rs.empty:
        return saved
    m = step.merge(rs, on="trace_step", how="inner")
    if m.empty or "estimated_dispatch_bytes_total" not in m.columns:
        return saved
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        m["estimated_dispatch_bytes_total"] / 1e9,
        m["r_sum_dispatch"] / 1e9,
        s=12,
        alpha=0.6,
    )
    lim = max(
        float(np.nanmax(m["estimated_dispatch_bytes_total"])),
        float(np.nanmax(m["r_sum_dispatch"])),
    )
    lim = lim / 1e9 * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5, label="y=x")
    ax.set_title("Step summary dispatch total vs sum of routing summary dispatch (streamed)")
    ax.set_xlabel("step: estimated_dispatch_bytes_total (GB)")
    ax.set_ylabel("routing: sum(estimated_dispatch_bytes) (GB)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    saved.append(_save(fig, out_dir, "compare_step_vs_routing_dispatch_total"))
    return saved


def _per_device_spread(row: pd.Series) -> float | np.floating:
    vals = row.get("per_device_used_bytes")
    if not isinstance(vals, (list, tuple)) or len(vals) == 0:
        return np.nan
    arr = np.asarray(vals, dtype=float)
    return float(np.nanmax(arr) - np.nanmin(arr))


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.pdf"
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def _find_first_scalar(obj: Any, key: str) -> Any | None:
    if isinstance(obj, Mapping):
        if key in obj and not isinstance(obj[key], (dict, list, tuple)):
            return obj[key]
        for v in obj.values():
            got = _find_first_scalar(v, key)
            if got is not None:
                return got
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            got = _find_first_scalar(v, key)
            if got is not None:
                return got
    return None


def _find_first_vector_len(obj: Any, key: str) -> int | None:
    if isinstance(obj, Mapping):
        if key in obj and isinstance(obj[key], (list, tuple)):
            return len(obj[key])
        for v in obj.values():
            got = _find_first_vector_len(v, key)
            if got is not None:
                return got
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            got = _find_first_vector_len(v, key)
            if got is not None:
                return got
    return None


def _infer_expert_dims_from_summary(
    traces_dir: Path,
) -> tuple[int | None, int | None, int | None]:
    """Infer (num_experts, hidden_size, bytes_per_element) from routing summary JSONL."""
    summary_paths = sorted(
        traces_dir.glob("summary/routing_summary_*.jsonl"),
        key=lambda p: (_summary_step_from_name(p.name) or -1, p.name),
    )
    for p in summary_paths:
        try:
            for row in _iter_jsonl(p):
                ne = _find_first_scalar(row, "num_experts")
                hs = _find_first_scalar(row, "hidden_size")
                bpe = _find_first_scalar(row, "bytes_per_element")
                if ne is None:
                    ne = _find_first_vector_len(row, "expert_counts")
                if ne is not None and hs is not None:
                    try:
                        ne_i = int(ne)
                        hs_i = int(hs)
                        bpe_i = int(bpe) if bpe is not None else None
                        return ne_i, hs_i, bpe_i
                    except (TypeError, ValueError):
                        continue
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return None, None, None


def _load_runtime_expert_capacity(traces_dir: Path) -> dict[str, Any] | None:
    paths = sorted(traces_dir.glob("summary/expert_capacity_*.jsonl"))
    if not paths:
        return None
    latest: dict[str, Any] | None = None
    latest_ts = -1.0
    for p in paths:
        try:
            for row in _iter_jsonl(p):
                if row.get("record_type") != "expert_capacity_summary":
                    continue
                ts = float(row.get("timestamp_unix", 0.0) or 0.0)
                if ts >= latest_ts:
                    latest = row
                    latest_ts = ts
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return latest


def plot_expert_hbm_capacity_from_runtime(out_dir: Path,
                                          runtime: Mapping[str, Any],
                                          *,
                                          prefix: str = "expert_hbm_capacity"
                                          ) -> list[Path]:
    saved: list[Path] = []
    vals = runtime.get("per_expert_weight_bytes_list")
    if not isinstance(vals, list) or len(vals) == 0:
        return saved
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return saved
    fig, ax = plt.subplots(figsize=(max(10, min(22, arr.size * 0.08)), 4.8))
    ax.bar(np.arange(arr.size), arr / 1e9, width=0.9, color="seagreen")
    ax.set_title("Per-expert HBM footprint from runtime model tensors")
    ax.set_xlabel("Expert index")
    ax.set_ylabel("GB per expert")
    ax.grid(True, alpha=0.25, axis="y")
    arch = runtime.get("architectures", [])
    arch_s = ",".join(arch) if isinstance(arch, list) else str(arch)
    info = (
        f"model={runtime.get('model','?')} arch={arch_s}\n"
        f"tp={runtime.get('tp_size','?')} ep={runtime.get('ep_size','?')} "
        f"layers_with_experts={runtime.get('num_layers_with_experts','?')}\n"
        f"representative_layer={runtime.get('representative_layer_idx','?')}, "
        f"num_experts={arr.size}, per_expert={np.nanmean(arr)/1e6:.2f} MB"
    )
    ax.text(0.01, 0.98, info, transform=ax.transAxes, va="top", ha="left", fontsize=9)
    saved.append(_save(fig, out_dir, f"{prefix}_per_expert_bar_runtime"))
    return saved


def plot_expert_hbm_capacity(
    out_dir: Path,
    *,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    bytes_per_element: int,
    prefix: str = "expert_hbm_capacity",
) -> list[Path]:
    """Static MoE expert weight memory (no traffic), one bar per expert."""
    saved: list[Path] = []
    if num_experts <= 0 or hidden_size <= 0 or intermediate_size <= 0 or bytes_per_element <= 0:
        return saved

    gate_bytes = hidden_size * intermediate_size * bytes_per_element
    up_bytes = hidden_size * intermediate_size * bytes_per_element
    down_bytes = intermediate_size * hidden_size * bytes_per_element
    per_expert_bytes = gate_bytes + up_bytes + down_bytes
    total_bytes = per_expert_bytes * num_experts

    x = np.arange(num_experts, dtype=int)
    per_expert_gb = np.full(num_experts, per_expert_bytes / 1e9, dtype=float)
    fig, ax = plt.subplots(figsize=(max(10, min(22, num_experts * 0.08)), 4.8))
    ax.bar(x, per_expert_gb, width=0.9, color="steelblue")
    ax.set_title("Per-expert static HBM footprint (weights only; no traffic)")
    ax.set_xlabel("Expert index")
    ax.set_ylabel("GB per expert")
    ax.grid(True, alpha=0.25, axis="y")
    info = (
        f"E={num_experts}, D={hidden_size}, F={intermediate_size}, bpe={bytes_per_element}\n"
        f"per expert = 3*D*F*bpe = {per_expert_bytes/1e6:.2f} MB, "
        f"all experts = {total_bytes/1e9:.2f} GB"
    )
    ax.text(0.01, 0.98, info, transform=ax.transAxes, va="top", ha="left", fontsize=9)
    saved.append(_save(fig, out_dir, f"{prefix}_per_expert_bar"))
    return saved


def plot_step_series(df: pd.DataFrame, out_dir: Path, prefix: str = "step") -> list[Path]:
    saved: list[Path] = []
    if df.empty or "trace_step" not in df.columns:
        return saved
    s = df.sort_values("trace_step")
    x = s["trace_step"].to_numpy()

    def line(ykey: str, title: str, ylabel: str, fname: str):
        if ykey not in s.columns:
            return
        y = pd.to_numeric(s[ykey], errors="coerce")
        if y.notna().sum() == 0:
            return
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, y, marker=".", ms=3)
        ax.set_title(title)
        ax.set_xlabel("trace_step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        saved.append(_save(fig, out_dir, f"{prefix}_{fname}"))

    line("num_tokens", "Scheduled tokens per step", "num_tokens", "num_tokens")
    line("num_reqs", "Active requests per step", "num_reqs", "num_reqs")
    line("num_prefill_reqs", "Prefill requests per step", "count", "num_prefill_reqs")
    line("num_decode_reqs", "Decode requests per step", "count", "num_decode_reqs")
    line(
        "estimated_dispatch_bytes_total",
        "Estimated MoE dispatch bytes (step summary)",
        "bytes",
        "dispatch_bytes_total",
    )
    line(
        "estimated_return_bytes_total",
        "Estimated MoE return bytes (step summary)",
        "bytes",
        "return_bytes_total",
    )
    line(
        "kv_external_load_reqs",
        "KV external-load requests per step",
        "count",
        "kv_external_load_reqs",
    )
    line(
        "kv_local_hit_reqs",
        "KV local-hit requests per step",
        "count",
        "kv_local_hit_reqs",
    )
    line(
        "kv_pull_done_reqs",
        "KV pull-complete notifications per step",
        "count",
        "kv_pull_done_reqs",
    )
    line(
        "kv_external_load_tokens",
        "KV external-load tokens per step",
        "tokens",
        "kv_external_load_tokens",
    )
    line(
        "kv_hot_block_ids_gt1",
        "Hot KV block IDs shared by >1 request (per step)",
        "count",
        "kv_hot_block_ids_gt1",
    )
    line(
        "kv_hot_block_max_fanout",
        "Max KV block fanout per step",
        "refs / block",
        "kv_hot_block_max_fanout",
    )
    line(
        "kv_hot_block_ref_ratio",
        "Hot KV block reference ratio per step",
        "ratio",
        "kv_hot_block_ref_ratio",
    )
    line(
        "kv_miss_hit_ratio",
        "KV external-load ratio over (external + local-hit)",
        "ratio",
        "kv_miss_hit_ratio",
    )
    line(
        "kv_pull_to_hit_ratio",
        "KV pull-to-hit ratio (external / local-hit)",
        "ratio",
        "kv_pull_to_hit_ratio",
    )

    if "a2a_bytes_total_sum" in s.columns and s["a2a_bytes_total_sum"].notna().any():
        line(
            "a2a_bytes_total_sum",
            "All-to-all bytes sum (when recorded)",
            "bytes",
            "a2a_bytes_total_sum",
        )

    if "num_tokens" in s.columns and "estimated_dispatch_bytes_total" in s.columns:
        nt = pd.to_numeric(s["num_tokens"], errors="coerce").replace(0, np.nan)
        disp = pd.to_numeric(s["estimated_dispatch_bytes_total"], errors="coerce")
        bpt = disp / nt
        if bpt.notna().sum() > 0:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(x, bpt, marker=".", ms=3, color="darkgreen")
            ax.set_title("Estimated dispatch bytes per token")
            ax.set_xlabel("trace_step")
            ax.set_ylabel("bytes / token")
            ax.grid(True, alpha=0.3)
            saved.append(_save(fig, out_dir, f"{prefix}_dispatch_bytes_per_token"))

    if "timestamp_unix" in s.columns:
        ts = pd.to_numeric(s["timestamp_unix"], errors="coerce")
        dt = ts.diff()
        if dt.notna().sum() > 1:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(x[1:], dt.iloc[1:].to_numpy(), marker=".", ms=3, color="purple")
            ax.set_title("Wall time between consecutive step summaries (Δ timestamp)")
            ax.set_xlabel("trace_step")
            ax.set_ylabel("seconds")
            ax.grid(True, alpha=0.3)
            saved.append(_save(fig, out_dir, f"{prefix}_inter_step_wall_seconds"))

    # Stacked area: prefill vs decode reqs
    if "num_prefill_reqs" in s.columns and "num_decode_reqs" in s.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.stackplot(
            x,
            s["num_prefill_reqs"].to_numpy(),
            s["num_decode_reqs"].to_numpy(),
            labels=("prefill", "decode"),
            alpha=0.85,
        )
        ax.legend(loc="upper right")
        ax.set_title("Request mix per step (stacked)")
        ax.set_xlabel("trace_step")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.3)
        saved.append(_save(fig, out_dir, f"{prefix}_req_mix_stacked"))

    if "num_tokens" in s.columns and "estimated_dispatch_bytes_total" in s.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(
            pd.to_numeric(s["num_tokens"], errors="coerce"),
            pd.to_numeric(s["estimated_dispatch_bytes_total"], errors="coerce") / 1e9,
            s=14,
            alpha=0.5,
            c=pd.to_numeric(s["trace_step"], errors="coerce"),
            cmap="viridis",
        )
        ax.set_title("Dispatch total vs num_tokens (color=trace_step)")
        ax.set_xlabel("num_tokens")
        ax.set_ylabel("estimated_dispatch_bytes_total (GB)")
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("trace_step")
        saved.append(_save(fig, out_dir, f"{prefix}_scatter_dispatch_vs_num_tokens"))

    if "num_tokens" in s.columns and "estimated_dispatch_bytes_total" in s.columns:
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(x, s["num_tokens"], color="C0", marker=".", ms=3, label="num_tokens")
        ax1.set_xlabel("trace_step")
        ax1.set_ylabel("num_tokens", color="C0")
        ax1.tick_params(axis="y", labelcolor="C0")
        ax2 = ax1.twinx()
        ax2.plot(
            x,
            pd.to_numeric(s["estimated_dispatch_bytes_total"], errors="coerce") / 1e9,
            color="C1",
            marker=".", ms=3,
            label="dispatch GB",
        )
        ax2.set_ylabel("estimated_dispatch_bytes_total (GB)", color="C1")
        ax2.tick_params(axis="y", labelcolor="C1")
        ax1.set_title("Tokens and MoE dispatch vs trace_step (twin axis)")
        ax1.grid(True, alpha=0.3)
        saved.append(_save(fig, out_dir, f"{prefix}_twin_tokens_and_dispatch"))

    if "model_forward_wall_time_s" in s.columns:
        line(
            "model_forward_wall_time_s",
            "Host wall time inside model_fn() (per step)",
            "seconds",
            "model_forward_wall_time_s",
        )

    if "model_forward_wall_time_s" in s.columns:
        t_fwd = pd.to_numeric(s["model_forward_wall_time_s"], errors="coerce").replace(0, np.nan)
        if t_fwd.notna().any():
            for byte_col, tag, fname in (
                (
                    "a2a_bytes_total_sum",
                    "a2a_bytes_total_sum / model_forward_wall_time_s (GB/s)",
                    "eff_bw_proxy_a2a_gbps",
                ),
                (
                    "estimated_dispatch_bytes_total",
                    "(dispatch+return)/2 proxy / model_forward_wall_time_s (GB/s)",
                    "eff_bw_proxy_dispatch_gbps",
                ),
            ):
                if byte_col not in s.columns:
                    continue
                b = pd.to_numeric(s[byte_col], errors="coerce")
                if byte_col == "estimated_dispatch_bytes_total":
                    if "estimated_return_bytes_total" in s.columns:
                        ret = pd.to_numeric(s["estimated_return_bytes_total"], errors="coerce")
                        b = (b + ret) * 0.5
                    else:
                        continue
                gbps = (b / 1e9) / t_fwd
                if gbps.notna().sum() == 0:
                    continue
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(x, gbps, marker=".", ms=3)
                ax.set_title(
                    f"Effective-bytes proxy: {tag}\n"
                    "(host model_fn wall time; includes non-collective work — see docs/profiling.md)"
                )
                ax.set_xlabel("trace_step")
                ax.set_ylabel("GB/s")
                ax.grid(True, alpha=0.3)
                saved.append(_save(fig, out_dir, f"{prefix}_{fname}"))

    return saved


def plot_hbm(hbm: pd.DataFrame, out_dir: Path, prefix: str = "hbm") -> list[Path]:
    saved: list[Path] = []
    if hbm.empty:
        return saved

    if "timestamp_unix" in hbm.columns and "total_used_bytes" in hbm.columns:
        t = pd.to_numeric(hbm["timestamp_unix"], errors="coerce")
        u = pd.to_numeric(hbm["total_used_bytes"], errors="coerce")
        m = t.notna() & u.notna()
        if m.sum() > 0:
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(t[m], u[m] / 1e9, lw=0.8, alpha=0.85)
            for ev in hbm.loc[m, "event"].unique():
                if ev in ("before_execute_model", "after_execute_model"):
                    continue
                sub = hbm[m & (hbm["event"] == ev)]
                if len(sub) <= 3:
                    ax.scatter(
                        pd.to_numeric(sub["timestamp_unix"], errors="coerce"),
                        pd.to_numeric(sub["total_used_bytes"], errors="coerce") / 1e9,
                        s=22,
                        label=str(ev)[:40],
                    )
            ax.set_title("Total HBM used vs time (lifecycle markers as scatter)")
            ax.set_xlabel("unix timestamp")
            ax.set_ylabel("total_used_bytes (GB)")
            ax.legend(loc="best", fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)
            saved.append(_save(fig, out_dir, f"{prefix}_total_used_vs_time"))
        # Same plot but split by TPU rank/node when available.
        if "rank" in hbm.columns:
            ranks = pd.to_numeric(hbm["rank"], errors="coerce")
            mr = m & ranks.notna()
            uniq_ranks = sorted(set(int(r) for r in ranks[mr].tolist()))
            if len(uniq_ranks) > 1:
                fig, ax = plt.subplots(figsize=(11, 4.5))
                for rk in uniq_ranks:
                    sub = hbm[mr & (ranks == rk)]
                    tx = pd.to_numeric(sub["timestamp_unix"], errors="coerce")
                    ux = pd.to_numeric(sub["total_used_bytes"], errors="coerce")
                    mm = tx.notna() & ux.notna()
                    if mm.sum() == 0:
                        continue
                    ax.plot(tx[mm], ux[mm] / 1e9, lw=0.9, alpha=0.9, label=f"rank{rk}")
                ax.set_title("Total HBM used vs time by TPU rank/node")
                ax.set_xlabel("unix timestamp")
                ax.set_ylabel("total_used_bytes (GB)")
                ax.legend(loc="best", fontsize=8, ncol=3)
                ax.grid(True, alpha=0.3)
                saved.append(_save(fig, out_dir, f"{prefix}_total_used_vs_time_by_rank"))

    exec_rows = hbm[hbm["event"] == "before_execute_model"].copy()
    if not exec_rows.empty and "trace_step" in exec_rows.columns:
        exec_rows = exec_rows.sort_values("trace_step")
        x = pd.to_numeric(exec_rows["trace_step"], errors="coerce")
        if "total_used_bytes" in exec_rows.columns:
            y = pd.to_numeric(exec_rows["total_used_bytes"], errors="coerce")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(x, y / 1e9, marker=".", ms=4)
            ax.set_title("HBM total used before_execute_model vs trace_step")
            ax.set_xlabel("trace_step")
            ax.set_ylabel("GB")
            ax.grid(True, alpha=0.3)
            saved.append(_save(fig, out_dir, f"{prefix}_before_exec_used_vs_step"))

        for col, ylab, fname in (
            ("num_scheduled_tokens", "tokens", "scheduled_tokens_vs_step"),
            ("num_tokens", "tokens", "num_tokens_vs_step"),
            ("num_active_reqs", "requests", "active_reqs_vs_step"),
            ("num_reqs", "requests", "num_reqs_vs_step"),
            ("num_finished_reqs", "finished", "finished_reqs_vs_step"),
        ):
            if col not in exec_rows.columns:
                continue
            yv = pd.to_numeric(exec_rows[col], errors="coerce")
            if yv.notna().sum() == 0:
                continue
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(x, yv, marker=".", ms=4)
            ax.set_title(f"{col} (before_execute_model)")
            ax.set_xlabel("trace_step")
            ax.set_ylabel(ylab)
            ax.grid(True, alpha=0.3)
            saved.append(_save(fig, out_dir, f"{prefix}_{fname}"))

        spreads = exec_rows.apply(_per_device_spread, axis=1)
        if spreads.notna().sum() > 0:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(x, spreads / 1e6, marker=".", ms=4, color="brown")
            ax.set_title("Per-device HBM used imbalance (max − min) before execute")
            ax.set_xlabel("trace_step")
            ax.set_ylabel("MB")
            ax.grid(True, alpha=0.3)
            saved.append(_save(fig, out_dir, f"{prefix}_device_used_spread_mb"))

        # Per-device lines
        if exec_rows["per_device_used_bytes"].notna().any():
            fig, ax = plt.subplots(figsize=(10, 5))
            for i in range(8):
                ys = []
                for _, row in exec_rows.iterrows():
                    v = row.get("per_device_used_bytes")
                    if isinstance(v, (list, tuple)) and len(v) > i:
                        ys.append(float(v[i]))
                    else:
                        ys.append(np.nan)
                arr = np.asarray(ys, dtype=float)
                finite = arr[np.isfinite(arr)]
                if finite.size and float(np.max(np.abs(finite))) > 0:
                    ax.plot(x, arr / 1e9, marker=".", ms=2, label=f"dev{i}")
            ax.set_title("per_device_used_bytes (before_execute_model)")
            ax.set_xlabel("trace_step")
            ax.set_ylabel("GB")
            ax.legend(ncol=4, fontsize=8)
            ax.grid(True, alpha=0.3)
            saved.append(_save(fig, out_dir, f"{prefix}_per_device_used_lines"))

    # Delta used across single step (after - before)
    bef = hbm[hbm["event"] == "before_execute_model"].set_index("trace_step")
    aft = hbm[hbm["event"] == "after_execute_model"].set_index("trace_step")
    common = bef.index.intersection(aft.index)
    if len(common) > 0:
        du = []
        for st in common:
            ub = pd.to_numeric(bef.loc[st, "total_used_bytes"], errors="coerce")
            ua = pd.to_numeric(aft.loc[st, "total_used_bytes"], errors="coerce")
            if np.isscalar(ub) and np.isscalar(ua):
                du.append(float(ua - ub))
            else:
                du.append(float(ua.iloc[0] - ub.iloc[0]))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(common.to_numpy(), np.asarray(du) / 1e6, width=0.85, alpha=0.8)
        ax.set_title("Δ total_used_bytes (after − before) execute_model per trace_step")
        ax.set_xlabel("trace_step")
        ax.set_ylabel("MB")
        ax.grid(True, alpha=0.3, axis="y")
        saved.append(_save(fig, out_dir, f"{prefix}_delta_used_after_minus_before_mb"))

    if "event" in hbm.columns:
        vc = hbm["event"].value_counts().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, max(3, 0.22 * len(vc))))
        ax.barh(vc.index.astype(str), vc.to_numpy(), alpha=0.85)
        ax.set_title("HBM trace: row counts by event type")
        ax.set_xlabel("count")
        ax.grid(True, alpha=0.3, axis="x")
        saved.append(_save(fig, out_dir, f"{prefix}_event_counts_barh"))

    return saved


def _heatmap_layer_expert(
    sub: pd.DataFrame, title: str, value_col: str, out_dir: Path, fname: str
) -> Path | None:
    if sub.empty or "layer_idx" not in sub.columns or "expert_id" not in sub.columns:
        return None
    pivot = sub.pivot_table(
        index="layer_idx",
        columns="expert_id",
        values=value_col,
        aggfunc="mean",
        fill_value=0.0,
    )
    if pivot.size == 0:
        return None
    fig, ax = plt.subplots(figsize=(14, max(4, 0.35 * len(pivot.index))))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_ylabel("layer_idx")
    ax.set_xlabel("expert_id")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    return _save(fig, out_dir, fname)


def plot_routing(routing: pd.DataFrame, out_dir: Path, prefix: str = "routing") -> list[Path]:
    saved: list[Path] = []
    if routing.empty:
        return saved

    # Aggregate vs step
    if "trace_step" in routing.columns:
        g = routing.groupby("trace_step", as_index=False).agg(
            sum_dispatch=("estimated_dispatch_bytes", "sum"),
            sum_return=("estimated_return_bytes", "sum"),
            mean_routing_exact=("routing_is_exact", "mean"),
            rows=("expert_id", "size"),
        )
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(g["trace_step"], g["sum_dispatch"] / 1e9, label="sum dispatch", lw=1)
        ax.plot(g["trace_step"], g["sum_return"] / 1e9, label="sum return", lw=1, alpha=0.7)
        ax.set_title("Routing summary: summed estimated MoE bytes per trace_step")
        ax.set_xlabel("trace_step")
        ax.set_ylabel("GB (sum over all rows in summary files)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        saved.append(_save(fig, out_dir, f"{prefix}_sum_bytes_per_trace_step"))

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(g["trace_step"], g["mean_routing_exact"], marker=".", ms=3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Mean routing_is_exact per trace_step")
        ax.set_xlabel("trace_step")
        ax.grid(True, alpha=0.3)
        saved.append(_save(fig, out_dir, f"{prefix}_routing_is_exact_mean"))

    # Per phase histograms
    if "phase" in routing.columns and "expert_count" in routing.columns:
        for ph in routing["phase"].dropna().unique():
            sub = routing[routing["phase"] == ph]
            ec = pd.to_numeric(sub["expert_count"], errors="coerce").dropna()
            if len(ec) == 0:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(ec, bins=min(80, max(10, int(ec.max()) + 1)), log=True, alpha=0.85)
            ax.set_title(f"Histogram of expert_count (phase={ph}, log y)")
            ax.set_xlabel("expert_count")
            ax.set_ylabel("count (log)")
            ax.grid(True, alpha=0.3)
            saved.append(_save(fig, out_dir, f"{prefix}_hist_expert_count_{ph}"))

    # Layer imbalance metrics
    if "layer_idx" in routing.columns and "expert_count" in routing.columns:
        lay = routing.groupby("layer_idx")["expert_count"].agg(["mean", "std", "max"]).reset_index()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(lay["layer_idx"], lay["mean"], label="mean expert_count")
        ax.plot(lay["layer_idx"], lay["max"], label="max expert_count", alpha=0.8)
        ax.set_title("Expert token load vs layer (aggregated over loaded summary rows)")
        ax.set_xlabel("layer_idx")
        ax.legend()
        ax.grid(True, alpha=0.3)
        saved.append(_save(fig, out_dir, f"{prefix}_layer_mean_max_expert_count"))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(lay["layer_idx"], lay["std"], color="darkred")
        ax.set_title("Std of expert_count across experts per layer")
        ax.set_xlabel("layer_idx")
        ax.grid(True, alpha=0.3)
        saved.append(_save(fig, out_dir, f"{prefix}_layer_std_expert_count"))

    # Heatmaps for selected trace_steps
    if "trace_step" in routing.columns:
        steps = sorted(routing["trace_step"].unique())
        picks: list[int] = []
        if steps:
            picks.append(int(steps[0]))
            picks.append(int(steps[len(steps) // 2]))
            picks.append(int(steps[-1]))
        for ts in picks:
            sub = routing[routing["trace_step"] == ts]
            for col, tag in (
                ("expert_count", "expert_count_mean"),
                ("estimated_dispatch_bytes", "dispatch_bytes_mean"),
            ):
                if col not in sub.columns:
                    continue
                p = _heatmap_layer_expert(
                    sub,
                    f"{col} heatmap (trace_step={ts})",
                    col,
                    out_dir,
                    f"{prefix}_heatmap_{tag}_step{ts}",
                )
                if p:
                    saved.append(p)

        # Phase-specific heatmap at last step
        if steps:
            ts = int(steps[-1])
            for ph in ("prefill", "decode", "mixed"):
                sub = routing[(routing["trace_step"] == ts) & (routing["phase"] == ph)]
                if sub.empty:
                    continue
                p = _heatmap_layer_expert(
                    sub,
                    f"expert_count heatmap step={ts} phase={ph}",
                    "expert_count",
                    out_dir,
                    f"{prefix}_heatmap_expert_count_step{ts}_{ph}",
                )
                if p:
                    saved.append(p)

    # Scatter sample
    if "expert_count" in routing.columns and "unique_token_count" in routing.columns:
        sample = routing.sample(min(50000, len(routing)), random_state=0)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(
            pd.to_numeric(sample["unique_token_count"], errors="coerce"),
            pd.to_numeric(sample["expert_count"], errors="coerce"),
            s=2,
            alpha=0.25,
        )
        ax.set_title("expert_count vs unique_token_count (sample)")
        ax.set_xlabel("unique_token_count")
        ax.set_ylabel("expert_count")
        ax.grid(True, alpha=0.3)
        saved.append(_save(fig, out_dir, f"{prefix}_scatter_unique_vs_expert"))

    # Decode-only: decode_step vs mean dispatch
    dec = routing[routing["phase"] == "decode"]
    if not dec.empty and "decode_step" in dec.columns and "estimated_dispatch_bytes" in dec.columns:
        g2 = (
            dec.groupby("decode_step", as_index=False)["estimated_dispatch_bytes"]
            .mean()
            .sort_values("decode_step")
        )
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(g2["decode_step"], g2["estimated_dispatch_bytes"] / 1e6, marker=".", ms=3)
        ax.set_title("Decode: mean estimated_dispatch_bytes vs decode_step")
        ax.set_xlabel("decode_step")
        ax.set_ylabel("MB (mean over rows)")
        ax.grid(True, alpha=0.3)
        saved.append(_save(fig, out_dir, f"{prefix}_decode_mean_dispatch_vs_decode_step"))

    if "layer_idx" in routing.columns and "estimated_dispatch_bytes" in routing.columns:
        ld = routing.groupby("layer_idx", as_index=False)["estimated_dispatch_bytes"].sum()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(ld["layer_idx"], ld["estimated_dispatch_bytes"] / 1e9, width=0.9, alpha=0.85)
        ax.set_title("Sum of estimated_dispatch_bytes by layer (all loaded summary rows)")
        ax.set_xlabel("layer_idx")
        ax.set_ylabel("GB")
        ax.grid(True, alpha=0.3, axis="y")
        saved.append(_save(fig, out_dir, f"{prefix}_layer_sum_dispatch_bar"))

    if "expert_id" in routing.columns and "expert_count" in routing.columns:
        top = routing.groupby("expert_id", as_index=False)["expert_count"].sum().nlargest(32, "expert_count")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(top["expert_id"].astype(str), top["expert_count"], alpha=0.85)
        ax.set_title("Top 32 experts by summed expert_count")
        ax.set_xlabel("expert_id")
        ax.set_ylabel("sum expert_count")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3, axis="y")
        saved.append(_save(fig, out_dir, f"{prefix}_top32_experts_by_expert_count"))

    if "expert_id" in routing.columns and "estimated_dispatch_bytes" in routing.columns:
        tot = routing.groupby("expert_id")["estimated_dispatch_bytes"].sum()
        p_ex = _plot_bar_dispatch_bytes_per_expert(
            dict(tot.items()),
            out_dir,
            f"{prefix}_histogram_dispatch_bytes_per_expert",
            title_suffix="",
        )
        if p_ex:
            saved.append(p_ex)

    if (
        "layer_idx" in routing.columns
        and "expert_id" in routing.columns
        and "expert_count" in routing.columns
    ):
        lel = defaultdict(lambda: defaultdict(float))
        g_le = routing.groupby(["layer_idx", "expert_id"], as_index=False)[
            "expert_count"
        ].sum()
        for _, row in g_le.iterrows():
            lel[row["layer_idx"]][row["expert_id"]] += float(row["expert_count"])
        p_le = _plot_layer_expert_load_histogram_pdf(
            {k: dict(v) for k, v in lel.items()},
            out_dir,
            f"{prefix}_layer_expert_load_histograms",
            title_suffix="",
        )
        if p_le:
            saved.append(p_le)

    if (
        "phase" in routing.columns
        and "estimated_dispatch_bytes" in routing.columns
        and "trace_step" in routing.columns
        and "request_id" in routing.columns
    ):
        phcol = routing["phase"].astype(str)
        phase_dispatch = {
            str(k): float(v)
            for k, v in routing.groupby(phcol)["estimated_dispatch_bytes"]
            .sum()
            .items()
        }
        phase_return: dict[str, float] = {}
        if "estimated_return_bytes" in routing.columns:
            phase_return = {
                str(k): float(v)
                for k, v in routing.groupby(phcol)["estimated_return_bytes"]
                .sum()
                .items()
            }
        pre = routing[phcol == "prefill"]
        dec = routing[phcol == "decode"]
        mix = routing[phcol == "mixed"]
        n_pre_rows = int(len(pre)) if not pre.empty else 0
        n_dec_rows = int(len(dec)) if not dec.empty else 0
        n_mix_rows = int(len(mix)) if not mix.empty else 0
        nz_b: dict[str, float] = {}
        nz_n: dict[str, int] = {}
        if "expert_count" in routing.columns:
            for ph_key in ("prefill", "decode", "mixed"):
                b, n = _routing_phase_nz_bytes_and_rows(routing, phcol, ph_key)
                if n > 0:
                    nz_b[ph_key] = b
                    nz_n[ph_key] = n
        p_ph = _plot_routing_prefill_decode_summary(
            phase_dispatch=phase_dispatch,
            phase_return=phase_return,
            n_prefill_rows=n_pre_rows,
            n_decode_rows=n_dec_rows,
            n_mixed_rows=n_mix_rows,
            phase_nz_bytes_sum=nz_b if nz_b else None,
            phase_nz_row_count=nz_n if nz_n else None,
            out_dir=out_dir,
            fname=f"{prefix}_prefill_decode_traffic_summary",
            title_suffix="",
        )
        if p_ph:
            saved.append(p_ph)

    if "phase" in routing.columns and "expert_count" in routing.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        for ph in sorted(routing["phase"].dropna().unique()):
            ec = pd.to_numeric(routing.loc[routing["phase"] == ph, "expert_count"], errors="coerce").dropna()
            if len(ec) == 0:
                continue
            xs = np.sort(ec.to_numpy())
            ys = np.linspace(0, 1, len(xs), endpoint=True)
            ax.plot(xs, ys, label=str(ph), lw=1.5)
        ax.set_title("CDF of expert_count by phase")
        ax.set_xlabel("expert_count")
        ax.set_ylabel("CDF")
        ax.legend()
        ax.grid(True, alpha=0.3)
        saved.append(_save(fig, out_dir, f"{prefix}_cdf_expert_count_by_phase"))

    if (
        "trace_step" in routing.columns
        and "expert_id" in routing.columns
        and "expert_count" in routing.columns
        and "layer_idx" in routing.columns
    ):
        frac_rows: list[dict[str, Any]] = []
        for (ts, li), g in routing.groupby(["trace_step", "layer_idx"]):
            sub = g.groupby("expert_id")["expert_count"].sum()
            frac_rows.append(
                {
                    "trace_step": ts,
                    "layer_idx": li,
                    "frac_active": float((sub > 0).mean()) if len(sub) else float("nan"),
                }
            )
        u = pd.DataFrame(frac_rows)
        u_mean = u.groupby("trace_step", as_index=False)["frac_active"].mean()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(u_mean["trace_step"], u_mean["frac_active"], marker=".", ms=4)
        ax.set_ylim(0, 1.02)
        ax.set_title("Mean fraction of experts with load > 0 (averaged over layers)")
        ax.set_xlabel("trace_step")
        ax.set_ylabel("fraction")
        ax.grid(True, alpha=0.3)
        saved.append(_save(fig, out_dir, f"{prefix}_mean_frac_active_experts_per_step"))

    if "k" in routing.columns and "expert_count" in routing.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        kvals = pd.to_numeric(routing["k"], errors="coerce").dropna().unique()
        if len(kvals) <= 8:
            for k in sorted(kvals):
                ec = pd.to_numeric(routing.loc[routing["k"] == k, "expert_count"], errors="coerce").dropna()
                if len(ec):
                    ax.hist(ec, bins=40, alpha=0.4, label=f"k={int(k)}", log=True)
            ax.set_title("expert_count histogram by k (log y)")
            ax.legend()
            ax.set_xlabel("expert_count")
            ax.grid(True, alpha=0.3)
            saved.append(_save(fig, out_dir, f"{prefix}_hist_expert_count_by_k"))

    return saved


def plot_paper_style_stacked_expert_volume(
    routing: pd.DataFrame, out_dir: Path, prefix: str = "paper_a"
) -> list[Path]:
    """(a)-style stacked area: per trace_step, share of estimated_dispatch_bytes per expert.

    This matches the *spirit* of the paper's stacked expert traffic plot: volume
    attributed to each expert over time. It uses routing *summary* (destination
    expert bytes), not a measured NIC/ICI all-to-all tensor.
    """
    saved: list[Path] = []
    if routing.empty or "trace_step" not in routing.columns:
        return saved
    if "expert_id" not in routing.columns or "estimated_dispatch_bytes" not in routing.columns:
        return saved

    def _one(df: pd.DataFrame, title_suffix: str, fname_suffix: str) -> None:
        agg = df.groupby(["trace_step", "expert_id"], as_index=False)["estimated_dispatch_bytes"].sum()
        pivot = agg.pivot(index="trace_step", columns="expert_id", values="estimated_dispatch_bytes").fillna(
            0.0
        )
        pivot = pivot.sort_index()
        if pivot.shape[1] == 0:
            return
        pivot, cols = _paper_a_pivot_all_experts(pivot)
        n_exp = len(cols)
        x = pivot.index.to_numpy()
        series = [pivot[c].to_numpy(dtype=float) / 1e6 for c in cols]
        labels = [_paper_a_expert_legend_label(c) for c in cols]
        fig_w = min(22.0, 11.0 + n_exp / 40.0)
        fig, ax = plt.subplots(figsize=(fig_w, 5.5))
        ax.stackplot(x, *series, labels=labels, alpha=0.9)
        ax.set_title(
            "Stacked estimated dispatch (MB) by expert vs trace_step "
            + title_suffix
            + "\n(proxy for paper-style expert traffic share; from routing summary)"
        )
        ax.set_xlabel("trace_step")
        ax.set_ylabel("MB (sum of estimated_dispatch_bytes)")
        ncol = max(4, min(16, (n_exp + 7) // 8))
        fs = max(4, min(7, 140 // max(n_exp, 1)))
        ax.legend(loc="upper left", fontsize=fs, ncol=ncol, framealpha=0.92)
        ax.grid(True, alpha=0.25, axis="y")
        saved.append(_save(fig, out_dir, f"{prefix}_stacked_dispatch_mb_{fname_suffix}"))

    _one(routing, "(all layers)", "all_layers")
    if "layer_idx" in routing.columns:
        li0 = int(pd.to_numeric(routing["layer_idx"], errors="coerce").min())
        sub0 = routing[pd.to_numeric(routing["layer_idx"], errors="coerce") == li0]
        if not sub0.empty:
            _one(sub0, f"(layer_idx == {li0} only)", "first_layer_only")

    return saved


def plot_paper_style_expert_pair_heatmaps(
    traces_dir: Path,
    out_dir: Path,
    *,
    layer_idx: int,
    max_experts_display: int,
    prefix: str = "paper_b",
) -> list[Path]:
    """(b)-style expert×expert heatmaps at a few trace_steps.

    Built from ``raw/routing_raw_*.npz`` ``layer_*_topk_experts`` when present.
    Cell (i,j) accumulates bytes from ordered top-k *slot* pairs (see
    :func:`_pair_matrix_topk_slots`); this is a **routing proxy**, not a measured
    per-link all-to-all log from hardware.
    """
    saved: list[Path] = []
    merged = _aggregate_raw_pair_by_trace_step(traces_dir, layer_idx)
    if not merged:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No raw/routing_raw_*.npz files found — expert×expert heatmaps need raw NPZ captures.",
            ha="center",
            va="center",
            fontsize=11,
        )
        saved.append(_save(fig, out_dir, f"{prefix}_no_npz_placeholder"))
        return saved

    steps_sorted = sorted(merged.keys())
    n = len(steps_sorted)
    pick_idx = sorted({0, n // 3, (2 * n) // 3, max(0, n - 1)})
    pick_steps = [steps_sorted[i] for i in pick_idx]
    while len(pick_steps) < 4:
        pick_steps.append(pick_steps[-1])
    pick_steps = pick_steps[:4]

    mats = [merged[s] for s in pick_steps]
    keep = _crop_expert_indices(mats, max_experts_display)
    if keep.size == 0:
        keep = np.arange(mats[0].shape[0])

    cropped = [m[np.ix_(keep, keep)] / 1e6 for m in mats]
    vmax = max(float(np.max(c)) if c.size else 0.0 for c in cropped)
    if vmax <= 0:
        vmax = 1.0

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for ax, st, mat in zip(axes, pick_steps, cropped):
        im = ax.imshow(mat, aspect="equal", interpolation="nearest", vmin=0.0, vmax=vmax)
        ax.set_title(f"step {st}")
        ax.set_xlabel("expert j")
        ax.set_ylabel("expert i")
        tick = np.arange(len(keep))
        step_t = max(1, len(keep) // 16)
        ax.set_xticks(tick[::step_t])
        ax.set_xticklabels([str(keep[i]) for i in tick[::step_t]], rotation=45, fontsize=7)
        ax.set_yticks(tick[::step_t])
        ax.set_yticklabels([str(keep[i]) for i in tick[::step_t]], fontsize=7)
    fig.suptitle(
        f"Expert×expert routing proxy (MB), layer_idx={layer_idx}\n"
        "Directed weights from top-k slot pairs; cropped to busiest experts",
        fontsize=11,
    )
    fig.colorbar(im, ax=list(axes.flat), shrink=0.55, label="MB")
    saved.append(_save(fig, out_dir, f"{prefix}_expert_pair_heatmaps_layer{layer_idx}"))

    # Normalized view (row-wise share by source expert)
    fig_n, axes_n = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    normed = [_row_normalize_2d(m) for m in cropped]
    for ax, st, mat_n in zip(axes_n, pick_steps, normed):
        im_n = ax.imshow(
            mat_n, aspect="equal", interpolation="nearest", vmin=0.0, vmax=1.0
        )
        ax.set_title(f"step {st} (row-norm)")
        ax.set_xlabel("expert j")
        ax.set_ylabel("expert i")
    fig_n.suptitle(
        f"Expert×expert routing proxy (row-normalized), layer_idx={layer_idx}",
        fontsize=11,
    )
    fig_n.colorbar(im_n, ax=list(axes_n.flat), shrink=0.55, label="fraction of src expert mass")
    saved.append(_save(fig_n, out_dir, f"{prefix}_expert_pair_heatmaps_norm_layer{layer_idx}"))

    # Symmetrized view (closer visual to some MoE analyses)
    fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    sym_max = 0.0
    syms = []
    for mat in cropped:
        sym = (mat + mat.T) * 0.5
        syms.append(sym)
        sym_max = max(sym_max, float(np.max(sym)) if sym.size else 0.0)
    if sym_max <= 0:
        sym_max = 1.0
    for ax, st, sym in zip(axes2, pick_steps, syms):
        im2 = ax.imshow(sym, aspect="equal", interpolation="nearest", vmin=0.0, vmax=sym_max)
        ax.set_title(f"step {st} (sym)")
        ax.set_xlabel("expert j")
        ax.set_ylabel("expert i")
    fig2.suptitle(
        f"Symmetrized (i,j)+(j,i)/2, layer_idx={layer_idx}",
        fontsize=11,
    )
    fig2.colorbar(im2, ax=list(axes2.flat), shrink=0.55, label="MB")
    saved.append(_save(fig2, out_dir, f"{prefix}_expert_pair_sym_heatmaps_layer{layer_idx}"))

    return saved


def plot_step_vs_routing(step: pd.DataFrame, routing: pd.DataFrame, out_dir: Path) -> list[Path]:
    saved: list[Path] = []
    if step.empty or routing.empty:
        return saved
    rs = routing.groupby("trace_step", as_index=False).agg(
        r_sum_dispatch=("estimated_dispatch_bytes", "sum"),
    )
    m = step.merge(rs, on="trace_step", how="inner")
    if m.empty or "estimated_dispatch_bytes_total" not in m.columns:
        return saved
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        m["estimated_dispatch_bytes_total"] / 1e9,
        m["r_sum_dispatch"] / 1e9,
        s=12,
        alpha=0.6,
    )
    lim = max(
        float(np.nanmax(m["estimated_dispatch_bytes_total"])),
        float(np.nanmax(m["r_sum_dispatch"])),
    )
    lim = lim / 1e9 * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5, label="y=x")
    ax.set_title("Step summary dispatch total vs sum of routing summary dispatch")
    ax.set_xlabel("step: estimated_dispatch_bytes_total (GB)")
    ax.set_ylabel("routing: sum(estimated_dispatch_bytes) (GB)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    saved.append(_save(fig, out_dir, "compare_step_vs_routing_dispatch_total"))
    return saved


def plot_ep_ragged_a2a_heatmaps(traces_dir: Path, out_dir: Path) -> list[Path]:
    """Shard×shard byte volumes from ragged all-to-all send_sizes (Megablox EP)."""
    saved: list[Path] = []
    raw_dir = traces_dir / "raw"
    if not raw_dir.is_dir():
        return saved
    npz_paths = sorted(raw_dir.glob("routing_raw_*.npz"))
    if not npz_paths:
        return saved
    pat = re.compile(r"layer_(\d+)_ep_ragged_a2a_dispatch_bytes$")
    for npz_path in npz_paths[-8:]:
        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception:
            continue
        stem = npz_path.stem.replace("routing_raw_", "")
        layer_ids: list[int] = []
        for k in data.files:
            m = pat.match(k)
            if m:
                layer_ids.append(int(m.group(1)))
        for lid in sorted(set(layer_ids)):
            dk = f"layer_{lid}_ep_ragged_a2a_dispatch_bytes"
            rk = f"layer_{lid}_ep_ragged_a2a_return_bytes"
            if dk not in data.files:
                continue
            d = np.asarray(data[dk], dtype=float)
            r = (np.asarray(data[rk], dtype=float)
                 if rk in data.files else None)
            ncols = 2 if r is not None else 1
            fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.2))
            ax_list = np.atleast_1d(axes)
            im0 = ax_list[0].imshow(d, aspect="auto", interpolation="nearest")
            ax_list[0].set_title(
                "dispatch bytes\n(send_sizes × hidden × dtype; not NIC)"
            )
            ax_list[0].set_xlabel("dst expert shard")
            ax_list[0].set_ylabel("src expert shard")
            fig.colorbar(im0, ax=ax_list[0], fraction=0.046, pad=0.04)
            if r is not None and ncols == 2:
                im1 = ax_list[1].imshow(r, aspect="auto", interpolation="nearest")
                ax_list[1].set_title("return bytes")
                ax_list[1].set_xlabel("dst expert shard")
                ax_list[1].set_ylabel("src expert shard")
                fig.colorbar(im1, ax=ax_list[1], fraction=0.046, pad=0.04)
            fig.suptitle(f"{npz_path.name}  layer {lid}", fontsize=10)
            fig.tight_layout()
            saved.append(
                _save(fig, out_dir, f"ep_ragged_a2a_{stem}_layer{lid}"))
            # Also save normalized (row-wise) variants to highlight shard traffic shares.
            d_norm = _row_normalize_2d(d)
            r_norm = _row_normalize_2d(r) if r is not None else None
            fig_n, axes_n = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.2))
            axn = np.atleast_1d(axes_n)
            imn0 = axn[0].imshow(d_norm, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)
            axn[0].set_title("dispatch share (row-normalized)")
            axn[0].set_xlabel("dst expert shard")
            axn[0].set_ylabel("src expert shard")
            fig_n.colorbar(imn0, ax=axn[0], fraction=0.046, pad=0.04)
            if r_norm is not None and ncols == 2:
                imn1 = axn[1].imshow(r_norm, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)
                axn[1].set_title("return share (row-normalized)")
                axn[1].set_xlabel("dst expert shard")
                axn[1].set_ylabel("src expert shard")
                fig_n.colorbar(imn1, ax=axn[1], fraction=0.046, pad=0.04)
            fig_n.suptitle(f"{npz_path.name}  layer {lid}  (normalized)", fontsize=10)
            fig_n.tight_layout()
            saved.append(
                _save(fig_n, out_dir, f"ep_ragged_a2a_norm_{stem}_layer{lid}"))
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot raw_traces JSONL to PDFs.")
    parser.add_argument(
        "--traces-dir",
        type=Path,
        default=_default_traces_dir(),
        help="Directory containing step/, hbm/, summary/",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Output directory for PDFs",
    )
    parser.add_argument(
        "--max-summary-files",
        type=int,
        default=None,
        help="Cap number of routing_summary jsonl files (default: all)",
    )
    parser.add_argument(
        "--paper-layer",
        type=int,
        default=-1,
        help="Expert×expert NPZ heatmaps: layer index, or -1 (default) for **all** MoE layers.",
    )
    parser.add_argument(
        "--paper-heatmap-max-experts",
        type=int,
        default=0,
        help="Crop heatmaps to this many busiest experts; 0 (default) = show all experts.",
    )
    parser.add_argument(
        "--routing-chunk-lines",
        type=int,
        default=50_000,
        help="JSONL rows per chunk when streaming routing summaries (default: 50000)",
    )
    parser.add_argument(
        "--legacy-full-routing-load",
        action="store_true",
        help="Load entire routing summary into RAM (fast for small runs; can OOM on huge summary/)",
    )
    parser.add_argument(
        "--skip-ep-ragged-a2a",
        action="store_true",
        help="Skip shard×shard EP ragged A2A heatmaps (on by default when NPZ keys exist).",
    )
    parser.add_argument(
        "--expert-count",
        type=int,
        default=None,
        help="Number of experts (optional; inferred from routing summary if omitted)",
    )
    parser.add_argument(
        "--expert-hidden-size",
        type=int,
        default=None,
        help="MoE hidden size D (optional; inferred from routing summary if omitted)",
    )
    parser.add_argument(
        "--expert-intermediate-size",
        type=int,
        default=None,
        help="MoE intermediate size F. Required for static expert HBM-capacity bars.",
    )
    parser.add_argument(
        "--expert-weight-bytes-per-element",
        type=int,
        default=None,
        help="Bytes/element for expert weights (default: infer from traces else 2)",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help="Directory containing raw_traces/, vllm_jax_profiles/, hlo_dumps/. "
        "Default: parent of --traces-dir when it is named raw_traces.",
    )
    parser.add_argument(
        "--skip-bundle-extras",
        action="store_true",
        help="Skip Chrome trace, HLO family counts, HBM brackets, ICI overlay plots.",
    )
    parser.add_argument(
        "--benchmark-json",
        type=Path,
        default=None,
        help="Optional JSON: benchmark_duration_s or duration_s; successful_requests or num_prompts.",
    )
    parser.add_argument(
        "--hlo-max-dirs",
        type=int,
        default=50_000,
        help="Max HLO dump subdirs to classify for bundle-extra plots (default: high cap).",
    )
    args = parser.parse_args()
    traces_dir: Path = args.traces_dir.expanduser().resolve()
    out_dir: Path = args.out_dir.expanduser().resolve()

    if not traces_dir.is_dir():
        print(f"Missing traces dir: {traces_dir}", file=sys.stderr)
        return 1

    print(f"Loading from {traces_dir}")
    step = _load_step_frames(traces_dir)
    hbm = _load_hbm_frames(traces_dir)

    all_saved: list[Path] = []
    all_saved.extend(plot_step_series(step, out_dir))
    all_saved.extend(plot_hbm(hbm, out_dir))

    if not args.skip_bundle_extras:
        bundle: Path | None = None
        if args.bundle_dir is not None:
            bundle = args.bundle_dir.expanduser().resolve()
        else:
            bundle = _bundle.resolve_bundle_root(traces_dir)
        if bundle is not None and bundle.is_dir():
            has_prof = (bundle / "vllm_jax_profiles").is_dir()
            has_hlo = (bundle / "hlo_dumps").is_dir()
            if has_prof or has_hlo:
                print(f"Bundle extras under {bundle}")
            tp = _bundle.discover_chrome_trace(bundle) if has_prof else None
            if tp is not None:
                print(f"  profile trace: {tp}")
                all_saved.extend(_bundle.plot_perfetto_comm_compute(tp, out_dir))
            if has_hlo:
                all_saved.extend(
                    _bundle.plot_hlo_module_families(
                        bundle / "hlo_dumps",
                        out_dir,
                        max_dirs=args.hlo_max_dirs,
                    )
                )
            all_saved.extend(_bundle.plot_hbm_execute_brackets(hbm, out_dir))
            all_saved.extend(_bundle.plot_ici_from_step(step, out_dir))
            all_saved.extend(_bundle.plot_step_comm_ici_overlay(step, out_dir))
            if args.benchmark_json is not None:
                pbj = args.benchmark_json.expanduser().resolve()
                if pbj.is_file():
                    bench = json.loads(pbj.read_text(encoding="utf-8"))
                    all_saved.extend(
                        _bundle.plot_benchmark_engine_utilization(
                            step, bench, out_dir
                        )
                    )

    runtime_expert_capacity = _load_runtime_expert_capacity(traces_dir)
    if runtime_expert_capacity is not None:
        all_saved.extend(
            plot_expert_hbm_capacity_from_runtime(out_dir, runtime_expert_capacity)
        )

    inf_e, inf_d, inf_bpe = _infer_expert_dims_from_summary(traces_dir)
    expert_count = args.expert_count if args.expert_count is not None else inf_e
    expert_hidden = (
        args.expert_hidden_size if args.expert_hidden_size is not None else inf_d
    )
    expert_bpe = (
        args.expert_weight_bytes_per_element
        if args.expert_weight_bytes_per_element is not None
        else (inf_bpe if inf_bpe is not None else 2)
    )
    if (
        runtime_expert_capacity is None
        and
        expert_count is not None
        and expert_hidden is not None
        and args.expert_intermediate_size is not None
    ):
        all_saved.extend(
            plot_expert_hbm_capacity(
                out_dir,
                num_experts=expert_count,
                hidden_size=expert_hidden,
                intermediate_size=args.expert_intermediate_size,
                bytes_per_element=expert_bpe,
            )
        )
    elif runtime_expert_capacity is None:
        print(
            "Skipping expert HBM-capacity bar plot "
            "(need --expert-intermediate-size; expert count/hidden size can be inferred)."
        )

    heatmap_layers = _paper_heatmap_layer_list(traces_dir, args.paper_layer)
    print(
        f"  expert×expert heatmap layers: {heatmap_layers} "
        f"(use --paper-layer N for a single layer)"
    )

    if args.legacy_full_routing_load:
        routing = _load_routing_summary(traces_dir, args.max_summary_files)
        print(
            f"  step rows: {len(step)}, hbm rows: {len(hbm)}, "
            f"routing rows (full load): {len(routing)}"
        )
        all_saved.extend(plot_routing(routing, out_dir))
        all_saved.extend(plot_paper_style_stacked_expert_volume(routing, out_dir))
        for lid in heatmap_layers:
            all_saved.extend(
                plot_paper_style_expert_pair_heatmaps(
                    traces_dir,
                    out_dir,
                    layer_idx=lid,
                    max_experts_display=args.paper_heatmap_max_experts,
                )
            )
        all_saved.extend(plot_step_vs_routing(step, routing, out_dir))
        if not args.skip_ep_ragged_a2a:
            all_saved.extend(plot_ep_ragged_a2a_heatmaps(traces_dir, out_dir))
    else:
        print(
            f"  step rows: {len(step)}, hbm rows: {len(hbm)}; "
            f"streaming routing summaries ({args.routing_chunk_lines} rows/chunk)"
        )
        agg = _stream_aggregate_routing(
            traces_dir, args.max_summary_files, args.routing_chunk_lines
        )
        print(f"  routing rows processed (streamed): {agg.total_rows}")
        all_saved.extend(
            plot_routing_from_agg(
                agg,
                traces_dir,
                out_dir,
                chunk_lines=args.routing_chunk_lines,
                max_summary_files=args.max_summary_files,
            )
        )
        all_saved.extend(plot_paper_stacked_from_agg(agg, out_dir))
        for lid in heatmap_layers:
            all_saved.extend(
                plot_paper_style_expert_pair_heatmaps(
                    traces_dir,
                    out_dir,
                    layer_idx=lid,
                    max_experts_display=args.paper_heatmap_max_experts,
                )
            )
        all_saved.extend(plot_step_vs_routing_agg(step, agg, out_dir))
        if not args.skip_ep_ragged_a2a:
            all_saved.extend(plot_ep_ragged_a2a_heatmaps(traces_dir, out_dir))

    print(f"Wrote {len(all_saved)} PDFs under {out_dir}")
    for p in sorted(all_saved)[:30]:
        print(f"  {p.name}")
    if len(all_saved) > 30:
        print(f"  ... and {len(all_saved) - 30} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
