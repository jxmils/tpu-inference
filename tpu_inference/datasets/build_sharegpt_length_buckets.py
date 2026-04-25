#!/usr/bin/env python3
"""Build ShareGPT prompt-length buckets for serving experiments.

Creates JSON files with prompts whose tokenized input length is near target
bucket centers (default: 128, 256, 512, 1024, 2048, 4096).

Example:
  python3 tpu_inference/datasets/build_sharegpt_length_buckets.py \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --out-dir tpu_inference/datasets/sharegpt_buckets \
    --max-per-bucket 1000
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


DEFAULT_TARGETS = [128, 256, 512, 1024, 2048, 4096]


@dataclass(frozen=True)
class BucketSpec:
    target: int
    lower: int
    upper: int


def _bucket_specs(targets: list[int], tolerance_pct: float) -> list[BucketSpec]:
    out: list[BucketSpec] = []
    for t in sorted(set(targets)):
        half = max(8, int(round(t * tolerance_pct)))
        out.append(BucketSpec(target=t, lower=max(1, t - half), upper=t + half))
    return out


def _extract_first_user_prompt(row: dict) -> str:
    conv = row.get("conversations")
    if isinstance(conv, list):
        for turn in conv:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("from", "")).lower()
            if role in ("human", "user"):
                text = turn.get("value", "")
                if isinstance(text, str) and text.strip():
                    return text
    prompt = row.get("prompt", "")
    if isinstance(prompt, str):
        return prompt
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Tokenizer model id used for token-length bucketing.",
    )
    parser.add_argument(
        "--dataset-name",
        default="Aeala/ShareGPT_Vicuna_unfiltered",
        help="HF dataset id for ShareGPT-style rows.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to read (default: train).",
    )
    parser.add_argument(
        "--targets",
        default="128,256,512,1024,2048,4096",
        help="Comma-separated token-length bucket centers.",
    )
    parser.add_argument(
        "--tolerance-pct",
        type=float,
        default=0.12,
        help="Half-window as percent of target (default: 0.12).",
    )
    parser.add_argument(
        "--max-source-rows",
        type=int,
        default=200_000,
        help="Max rows read from source split (default: 200000).",
    )
    parser.add_argument(
        "--max-per-bucket",
        type=int,
        default=1000,
        help="Maximum prompts written per bucket (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Sampling seed (default: 100).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tpu_inference/datasets/sharegpt_buckets"),
        help="Output directory for generated JSON files.",
    )
    args = parser.parse_args()

    targets = [int(x.strip()) for x in args.targets.split(",") if x.strip()]
    specs = _bucket_specs(targets, args.tolerance_pct)
    rng = random.Random(args.seed)

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    ds = load_dataset(args.dataset_name, split=args.split)
    n_rows = min(args.max_source_rows, len(ds))

    selected: dict[int, list[dict]] = {s.target: [] for s in specs}

    # Single pass through source rows; assign prompt to first matching bucket.
    # We keep going until source exhausted or all buckets are full.
    for ex in ds.select(range(n_rows)):
        if all(len(v) >= args.max_per_bucket for v in selected.values()):
            break
        prompt = _extract_first_user_prompt(ex)
        if not prompt.strip():
            continue
        n_tok = len(tok.encode(prompt, add_special_tokens=False))
        for s in specs:
            if len(selected[s.target]) >= args.max_per_bucket:
                continue
            if s.lower <= n_tok <= s.upper:
                selected[s.target].append(
                    {
                        "prompt": prompt,
                        "prompt_tokens": n_tok,
                        "target_bucket": s.target,
                        "bucket_lower": s.lower,
                        "bucket_upper": s.upper,
                    }
                )
                break

    for t in sorted(selected):
        rows = selected[t]
        rng.shuffle(rows)
        rows = rows[: args.max_per_bucket]
        out_path = out_dir / f"sharegpt_sp{t}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)
        print(f"wrote {out_path} rows={len(rows)}")

    manifest = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "model": args.model,
        "targets": sorted(selected.keys()),
        "tolerance_pct": args.tolerance_pct,
        "max_source_rows": args.max_source_rows,
        "max_per_bucket": args.max_per_bucket,
        "seed": args.seed,
        "counts": {str(k): len(v) for k, v in selected.items()},
    }
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
