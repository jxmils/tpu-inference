#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Aggregate step_summary CSV rows across multiple trace bundles (models / TP / EP).

Manifest JSON schema::

    {
      "runs": [
        {
          "label": "Qwen TP8 EP1",
          "model": "org/name",
          "tp": 8,
          "ep": 1,
          "traces_dir": "/abs/path/raw_traces"
        }
      ]
    }

Writes comparison PDFs + summary CSV under ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _load_step_summary(traces_dir: Path) -> pd.DataFrame:
    paths = sorted(traces_dir.glob("step/*.jsonl"))
    if not paths:
        return pd.DataFrame()
    frames = []
    for p in paths:
        df = pd.read_json(p, lines=True)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if "record_type" in out.columns:
        out = out[out["record_type"] == "step_summary"]
    return out.sort_values("trace_step")


def _summarize_run(step: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    if step.empty:
        return out
    if "comm_bytes_proxy_total" in step.columns:
        c = pd.to_numeric(step["comm_bytes_proxy_total"], errors="coerce")
        out["mean_comm_proxy_gb"] = float((c / 1e9).mean())
        out["sum_comm_proxy_gb"] = float((c / 1e9).sum())
    if "estimated_dispatch_bytes_total" in step.columns:
        d = pd.to_numeric(step["estimated_dispatch_bytes_total"], errors="coerce")
        out["mean_dispatch_gb"] = float((d / 1e9).mean())
        if "comm_bytes_proxy_total" in step.columns:
            tot_c = pd.to_numeric(step["comm_bytes_proxy_total"], errors="coerce")
            mix = tot_c.fillna(0) + d.fillna(0)
            positive = mix > 0
            if positive.any():
                frac = tot_c[positive] / mix[positive]
                out["mean_comm_share_of_comm_plus_dispatch"] = float(frac.mean())
    if "num_tokens" in step.columns:
        out["mean_num_tokens"] = float(
            pd.to_numeric(step["num_tokens"], errors="coerce").mean()
        )
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True, help="JSON manifest path.")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory.")
    args = p.parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = raw.get("runs")
    if not isinstance(runs, list) or not runs:
        print("manifest must contain non-empty runs[]", file=sys.stderr)
        return 2

    rows: list[dict[str, object]] = []
    saved: list[Path] = []

    for run in runs:
        if not isinstance(run, dict):
            continue
        label = str(run.get("label") or run.get("id") or "run")
        td = run.get("traces_dir")
        if not td:
            print(f"skip run missing traces_dir: {run}", file=sys.stderr)
            continue
        traces_dir = Path(str(td)).expanduser().resolve()
        step = _load_step_summary(traces_dir)
        stats = _summarize_run(step)
        row = {
            "label": label,
            "model": run.get("model", ""),
            "tp": run.get("tp", ""),
            "ep": run.get("ep", ""),
            **stats,
        }
        rows.append(row)

    if not rows:
        print("no runs processed", file=sys.stderr)
        return 3

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    csv_path = out_dir / "multi_run_step_summary.csv"
    summary.to_csv(csv_path, index=False)

    # Stacked bar: mean comm proxy vs mean dispatch per run label (traffic split proxy).
    labels = [str(r["label"]) for r in rows]
    comm_g = [
        float(summary.loc[i, "mean_comm_proxy_gb"])
        if "mean_comm_proxy_gb" in summary.columns
        and pd.notna(summary.loc[i, "mean_comm_proxy_gb"])
        else 0.0
        for i in range(len(summary))
    ]
    disp_g = [
        float(summary.loc[i, "mean_dispatch_gb"])
        if "mean_dispatch_gb" in summary.columns
        and pd.notna(summary.loc[i, "mean_dispatch_gb"])
        else 0.0
        for i in range(len(summary))
    ]
    x_idx = np.arange(len(labels))
    if sum(comm_g) + sum(disp_g) > 1e-12:
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.85), 5))
        ax.bar(x_idx, comm_g, label="mean comm proxy (GB/step)", color="#ff7f0e")
        ax.bar(x_idx, disp_g, bottom=comm_g, label="mean dispatch total (GB/step)", color="#2ca02c")
        ax.set_xticks(x_idx)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("GB (mean per step_summary row)")
        ax.set_title(
            "Traffic mix proxy across runs — mean comm_bytes_proxy vs MoE dispatch\n"
            "(same yardstick only when TRACE_STEP_STRIDE matches)"
        )
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.25, axis="y")
        fig.tight_layout()
        pdf_path = out_dir / "multi_run_comm_vs_dispatch_stacked.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        saved.append(pdf_path)

    # Share % pie proxies where defined.
    if "mean_comm_share_of_comm_plus_dispatch" in summary.columns:
        fig2, ax2 = plt.subplots(figsize=(max(10, len(labels) * 0.45), 4))
        shares = pd.to_numeric(
            summary["mean_comm_share_of_comm_plus_dispatch"], errors="coerce"
        ).fillna(0.0)
        ax2.bar(x_idx, shares * 100.0, color="steelblue")
        ax2.set_xticks(x_idx)
        ax2.set_xticklabels(labels, rotation=35, ha="right")
        ax2.set_ylabel("mean % comm / (comm + dispatch)")
        ax2.set_title("Communication share of combined proxy (per run)")
        ax2.grid(True, alpha=0.25, axis="y")
        fig2.tight_layout()
        p2 = out_dir / "multi_run_comm_fraction_bar.pdf"
        fig2.savefig(p2, bbox_inches="tight")
        plt.close(fig2)
        saved.append(p2)

    print(f"Wrote {csv_path}")
    for s in saved:
        print(f"  {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
