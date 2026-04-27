# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Plots from vLLM/JAX Chrome traces, HLO dump trees, HBM brackets, ICI JSONL, optional bench JSON."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from perfetto_comm_compute import summarize_trace_to_dataframe
from perfetto_op_breakdown import traffic_heatmap_colormap


def _save(fig: matplotlib.figure.Figure, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def resolve_bundle_root(traces_dir: Path) -> Path | None:
    """If ``traces_dir`` is ``.../raw_traces``, return parent bundle directory."""
    if traces_dir.name == "raw_traces" and traces_dir.parent.is_dir():
        return traces_dir.parent
    return None


def discover_chrome_trace(bundle_root: Path) -> Path | None:
    """Prefer JAX profile under ``plugins/profile``; else any ``*.trace.json``."""
    prof = bundle_root / "vllm_jax_profiles"
    if not prof.is_dir():
        return None
    candidates: list[Path] = []
    for p in prof.rglob("*.trace.json"):
        if "plugins/profile" in str(p):
            candidates.append(p)
    if not candidates:
        for p in prof.rglob("*.trace.json"):
            candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


def plot_perfetto_comm_compute(trace_path: Path, out_dir: Path, prefix: str = "profile") -> list[Path]:
    saved: list[Path] = []
    df = summarize_trace_to_dataframe(trace_path)
    if df.empty:
        return saved

    n = np.arange(len(df))
    comm = pd.to_numeric(df["comm_time_ms"], errors="coerce").fillna(0.0).to_numpy()
    comp = pd.to_numeric(df["compute_time_ms"], errors="coerce").fillna(0.0).to_numpy()
    ov = pd.to_numeric(df["overlap_ms"], errors="coerce").fillna(0.0).to_numpy()
    rest = np.maximum(
        0.0,
        pd.to_numeric(df["step_duration_ms"], errors="coerce").fillna(0.0).to_numpy() - comm - comp + ov,
    )

    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.15), 5))
    ax.bar(n, comp, label="compute (merged)", color="steelblue")
    ax.bar(n, comm, bottom=comp, label="comm (merged)", color="darkorange")
    ax.bar(n, ov, bottom=comp + comm, label="overlap (comm∩compute)", color="mediumpurple")
    ax.bar(n, rest, bottom=comp + comm + ov, label="other / unclassified in step", color="lightgray")
    ax.set_title(
        f"Chrome trace execute_model steps — comm vs compute vs overlap\n{trace_path.name}"
    )
    ax.set_xlabel("execute_model index")
    ax.set_ylabel("ms (trace clock)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25, axis="y")
    saved.append(_save(fig, out_dir, f"{prefix}_comm_compute_stacked"))

    # Mean fraction pie (excluding "rest" for clarity when small)
    tot_c, tot_m, tot_o = float(np.nansum(comp)), float(np.nansum(comm)), float(np.nansum(ov))
    pie_vals = [tot_c, tot_m, tot_o]
    pie_labels = ["compute Σ", "comm Σ", "overlap Σ"]
    if sum(pie_vals) > 1e-9:
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        ax2.pie(pie_vals, labels=pie_labels, autopct="%1.1f%%", startangle=90)
        ax2.set_title("Share of merged comm/compute/overlap time (all steps)")
        saved.append(_save(fig2, out_dir, f"{prefix}_comm_compute_pie"))

    csv_path = out_dir / f"{prefix}_comm_compute_per_step.csv"
    df.to_csv(csv_path, index=False)
    return saved


def _hlo_family(name: str) -> str:
    if not name.startswith("jit_"):
        return "non_jit"
    body = re.sub(r"_\d+$", "", name[4:])
    return body[:72] or "jit"


def plot_hlo_module_families(hlo_root: Path, out_dir: Path, max_dirs: int = 800, prefix: str = "hlo") -> list[Path]:
    saved: list[Path] = []
    if not hlo_root.is_dir():
        return saved
    dirs = [p for p in hlo_root.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.name)
    dirs = dirs[:max_dirs]
    if not dirs:
        return saved
    counts: dict[str, int] = {}
    for d in dirs:
        fam = _hlo_family(d.name)
        counts[fam] = counts.get(fam, 0) + 1
    items = sorted(counts.items(), key=lambda kv: -kv[1])[:40]
    if not items:
        return saved
    labels, vals = zip(*items)
    fig, ax = plt.subplots(figsize=(12, max(4, len(labels) * 0.18)))
    y = np.arange(len(labels))
    ax.barh(y, vals, color="teal")
    ax.set_yticks(y, labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(f"compiled module count (first {len(dirs)} dirs under hlo_dumps)")
    ax.set_title("XLA HLO dump directories by JIT family (topology: what shapes of code exist)")
    ax.grid(True, alpha=0.25, axis="x")
    saved.append(_save(fig, out_dir, f"{prefix}_module_family_counts"))
    return saved


def plot_hbm_execute_brackets(hbm: pd.DataFrame, out_dir: Path, prefix: str = "hbm") -> list[Path]:
    """Wall time between before_execute_model and after_execute_model per trace_step."""
    saved: list[Path] = []
    if hbm.empty or "event" not in hbm.columns:
        return saved
    before = hbm[hbm["event"] == "before_execute_model"].copy()
    after = hbm[hbm["event"] == "after_execute_model"].copy()
    if before.empty or after.empty or "trace_step" not in before.columns:
        return saved
    before = before.sort_values("trace_step")
    after = after.sort_values("trace_step")
    merged = pd.merge(
        before[["trace_step", "timestamp_unix"]],
        after[["trace_step", "timestamp_unix"]],
        on="trace_step",
        suffixes=("_before", "_after"),
    )
    if merged.empty:
        return saved
    tb = pd.to_numeric(merged["timestamp_unix_before"], errors="coerce")
    ta = pd.to_numeric(merged["timestamp_unix_after"], errors="coerce")
    dt = ta - tb
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(merged["trace_step"], dt, marker=".", ms=4, color="brown")
    ax.set_title("HBM-bracketed execute window (after − before execute_model) per trace_step")
    ax.set_xlabel("trace_step")
    ax.set_ylabel("seconds (host wall)")
    ax.grid(True, alpha=0.3)
    saved.append(_save(fig, out_dir, f"{prefix}_execute_wall_seconds"))

    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    ax2.hist(dt.dropna().to_numpy(), bins=min(40, max(5, len(dt) // 2)), color="peru", edgecolor="white")
    ax2.set_title("Distribution of HBM-bracketed execute windows (after − before)")
    ax2.set_xlabel("seconds")
    ax2.set_ylabel("count")
    ax2.grid(True, alpha=0.25, axis="y")
    saved.append(_save(fig2, out_dir, f"{prefix}_execute_wall_hist"))
    return saved


def _extract_ici_rows(step: pd.DataFrame) -> pd.DataFrame | None:
    if step.empty or "ici_hw_delta" not in step.columns:
        return None
    rows = []
    for _, r in step.iterrows():
        ts = r.get("trace_step")
        blob = r.get("ici_hw_delta")
        if not isinstance(blob, dict):
            continue
        for metric, payload in blob.items():
            if not isinstance(payload, dict):
                continue
            dlist = payload.get("delta_per_chip")
            dsum = payload.get("delta_sum")
            if dlist is None and dsum is None:
                continue
            if isinstance(dlist, list) and len(dlist) > 0:
                for chip_i, v in enumerate(dlist):
                    try:
                        fv = float(v)
                    except (TypeError, ValueError):
                        fv = float("nan")
                    rows.append(
                        {
                            "trace_step": ts,
                            "metric": metric,
                            "chip": chip_i,
                            "delta": fv,
                        }
                    )
            elif dsum is not None:
                try:
                    rows.append(
                        {
                            "trace_step": ts,
                            "metric": metric,
                            "chip": -1,
                            "delta": float(dsum),
                        }
                    )
                except (TypeError, ValueError):
                    pass
    if not rows:
        return None
    return pd.DataFrame(rows)


def plot_ici_from_step(step: pd.DataFrame, out_dir: Path, prefix: str = "ici") -> list[Path]:
    saved: list[Path] = []
    ici = _extract_ici_rows(step)
    if ici is None or ici.empty:
        return saved
    # Sum per step per metric (chips already summed when chip=-1)
    g = ici.groupby(["trace_step", "metric"], as_index=False)["delta"].sum()
    if g.empty:
        return saved
    for metric in sorted(g["metric"].unique()):
        sub = g[g["metric"] == metric]
        if not sub["delta"].abs().gt(0).any():
            continue
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(sub["trace_step"], sub["delta"], marker=".", ms=3)
        ax.set_title(f"ICI hardware delta — {metric} (per trace_step)")
        ax.set_xlabel("trace_step")
        ax.set_ylabel("Δ counter (runtime units)")
        ax.grid(True, alpha=0.3)
        safe = re.sub(r"[^\w\-]+", "_", metric)[:60]
        saved.append(_save(fig, out_dir, f"{prefix}_{safe}_per_step"))

    # Heatmap chip × step when multiple chips
    chip_pos = ici[ici["chip"] >= 0]
    if chip_pos.empty:
        return saved
    pivot = chip_pos.pivot_table(
        index="chip", columns="trace_step", values="delta", aggfunc="sum", fill_value=0.0
    )
    if pivot.shape[1] > 200:
        pivot = pivot.iloc[:, :: max(1, pivot.shape[1] // 200)]
    fig, ax = plt.subplots(figsize=(min(14, 0.06 * pivot.shape[1] + 4), min(8, 0.25 * pivot.shape[0] + 2)))
    im = ax.imshow(
        pivot.to_numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap=traffic_heatmap_colormap(),
    )
    ax.set_title("ICI Δ per chip (rows) vs trace_step (columns, subsampled if long)")
    ax.set_ylabel("chip index")
    ax.set_xlabel("trace_step column index (ordered)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    saved.append(_save(fig, out_dir, f"{prefix}_chip_step_heatmap"))
    return saved


def plot_step_comm_ici_overlay(step: pd.DataFrame, out_dir: Path, prefix: str = "step") -> list[Path]:
    """Comm proxy and host forward time; ICI on a second panel when present."""
    saved: list[Path] = []
    if step.empty or "trace_step" not in step.columns:
        return saved
    s = step.sort_values("trace_step")
    x = s["trace_step"].to_numpy()
    ici = _extract_ici_rows(s)
    two = ici is not None and not ici.empty
    if two:
        fig, (ax1, axb) = plt.subplots(
            2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
        )
    else:
        fig, ax1 = plt.subplots(figsize=(11, 4))
        axb = None
    if "comm_bytes_proxy_total" in s.columns:
        y = pd.to_numeric(s["comm_bytes_proxy_total"], errors="coerce") / 1e9
        if y.notna().any():
            ax1.plot(x, y, color="C0", marker=".", ms=3, label="comm_bytes_proxy_total (GB)")
    ax1.set_ylabel("comm proxy (GB)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax2 = ax1.twinx()
    if "model_forward_wall_time_s" in s.columns:
        t = pd.to_numeric(s["model_forward_wall_time_s"], errors="coerce")
        if t.notna().any():
            ax2.plot(x, t * 1000, color="C1", marker=".", ms=3, label="model_forward (ms)")
    ax2.set_ylabel("model_forward_wall_time_s (ms)", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax1.set_title("Topology proxies: comm bytes proxy vs host model_fn wall time")
    ax1.grid(True, alpha=0.25)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7)
    if axb is not None:
        g = ici.groupby("trace_step", as_index=False)["delta"].sum()
        mg = pd.DataFrame({"trace_step": x}).merge(g, on="trace_step", how="left")
        axb.plot(x, mg["delta"].fillna(0.0), color="C2", marker="x", ms=3, ls="--")
        axb.set_ylabel("ICI Δ (Σ metrics)")
        axb.set_xlabel("trace_step")
        axb.grid(True, alpha=0.25)
    else:
        ax1.set_xlabel("trace_step")
    saved.append(_save(fig, out_dir, f"{prefix}_comm_forward_ici_overlay"))
    return saved


def plot_benchmark_engine_utilization(
    step: pd.DataFrame,
    bench: Mapping[str, Any],
    out_dir: Path,
    prefix: str = "bench",
) -> list[Path]:
    """Rough engine busy fraction: sum(model_forward) / benchmark_duration."""
    saved: list[Path] = []
    dur = float(bench.get("benchmark_duration_s") or bench.get("duration_s") or 0.0)
    if dur <= 0 or step.empty or "model_forward_wall_time_s" not in step.columns:
        return saved
    fwd = pd.to_numeric(step["model_forward_wall_time_s"], errors="coerce")
    busy = float(fwd.sum())
    # step_summary is only on TRACE_STEP_STRIDE; Σ forward can exceed wall clock — cap for display.
    if busy > dur:
        busy = dur
    idle_proxy = max(0.0, dur - busy)
    nreq = int(bench.get("successful_requests") or bench.get("num_prompts") or 0)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        [busy, idle_proxy],
        labels=[f"Σ model_forward ({busy:.1f}s)", f"remainder of bench window ({idle_proxy:.1f}s)"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#2ca02c", "#dddddd"],
    )
    ax.set_title(
        "Bench window utilization (proxy)\n"
        f"duration={dur:.1f}s prompts={nreq}\n"
        "Note: remainder includes client queueing, scheduler, gaps between traced steps"
    )
    saved.append(_save(fig, out_dir, f"{prefix}_engine_time_vs_window"))
    return saved


def plot_perfetto_execute_model_breakdown(trace_path: Path, out_dir: Path, prefix: str = "profile") -> list[Path]:
    """Stacked bars per execute_model step: attention / gate / collective / experts / add_norm / other."""
    saved: list[Path] = []
    try:
        from perfetto_op_breakdown import extract_collective_ops, summarize_execute_model_breakdown
    except ImportError:
        return saved

    df = summarize_execute_model_breakdown(trace_path)
    if df.empty:
        return saved

    cols = [
        ("attention_ms", "attention", "#1f77b4"),
        ("gate_ms", "gate", "#9467bd"),
        ("collective_ms", "collective", "#ff7f0e"),
        ("experts_ms", "experts", "#2ca02c"),
        ("add_norm_ms", "add_norm", "#17becf"),
        ("other_compute_ms", "other_compute", "#bdbdbd"),
    ]
    x = np.arange(len(df))
    bottoms = np.zeros(len(df))
    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.18), 5.5))
    for key, lab, color in cols:
        vals = pd.to_numeric(df[key], errors="coerce").fillna(0.0).to_numpy()
        ax.bar(x, vals, bottom=bottoms, label=lab, width=0.82, color=color)
        bottoms += vals
    ax.set_title(
        "execute_model steps — heuristic compute/comm buckets (from trace names)\n"
        f"{trace_path.name}"
    )
    ax.set_xlabel("execute_model index")
    ax.set_ylabel("ms (merged within bucket; overlap can double-count)")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.22, axis="y")
    saved.append(_save(fig, out_dir, f"{prefix}_execute_model_breakdown_stacked"))

    csv_path = out_dir / f"{prefix}_execute_model_breakdown.csv"
    df.to_csv(csv_path, index=False)

    ops = extract_collective_ops(trace_path)
    if not ops.empty:
        ops_path = out_dir / f"{prefix}_collective_ops.csv"
        ops.to_csv(ops_path, index=False)
        du = pd.to_numeric(ops["duration_ms"], errors="coerce").fillna(0.0)
        if du.sum() > 0:
            arr = du.to_numpy(dtype=float)
            hi = float(np.nanpercentile(arr, 99))
            if not np.isfinite(hi) or hi <= 0:
                hi = float(np.nanmax(arr)) if arr.size else 1.0
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.hist(
                np.clip(arr, 0.0, hi),
                bins=40,
                color="#ff7f0e",
                edgecolor="white",
            )
            ax2.set_title(
                "Collective-like op durations (histogram, clipped at 99p to limit long tails)"
            )
            ax2.set_xlabel("duration (ms)")
            ax2.set_ylabel("count")
            ax2.grid(True, alpha=0.25, axis="y")
            saved.append(_save(fig2, out_dir, f"{prefix}_collective_duration_hist"))

    return saved
