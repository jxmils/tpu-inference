#!/usr/bin/env python3
"""Patch vLLM compilation_time aggregation for mixed formats."""

from __future__ import annotations

from pathlib import Path


def _patch_text(text: str) -> str:
    # Keep this patch expression-only to avoid indentation/syntax regressions
    # across upstream formatting variants.
    replacements = (
        (
            "t.language_model for t in compilation_times",
            "getattr(t, \"language_model\", t) for t in compilation_times",
        ),
        (
            "t.encoder for t in compilation_times",
            "getattr(t, \"encoder\", 0.0) for t in compilation_times",
        ),
    )
    out = text
    for old, new in replacements:
        out = out.replace(old, new)
    return out


def main() -> None:
    path = Path("/workspace/vllm/vllm/v1/executor/abstract.py")
    if not path.exists():
        return
    text = path.read_text()
    patched = _patch_text(text)
    if patched != text:
        path.write_text(patched)


if __name__ == "__main__":
    main()
