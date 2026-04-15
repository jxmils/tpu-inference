#!/usr/bin/env python3
"""Patch vLLM compilation_time aggregation for mixed formats.

Some vLLM/tpu_inference commit combinations produce `compilation_times` as
floats, while newer code expects objects with a `.language_model` attribute.
This patch makes the aggregation tolerant to both.
"""

from __future__ import annotations

from pathlib import Path


def _patch_text(text: str) -> str:
    marker = "# TPU_INFERENCE_COMPILATION_TIME_COMPAT_PATCH"
    if marker in text:
        return text

    old = (
        "self.vllm_config.compilation_config.compilation_time = max(\n"
        "    t.language_model for t in compilation_times\n"
        ")"
    )
    if old in text:
        new = (
            f"{marker}\n"
            "def _compilation_time_language_model_value(item):\n"
            "    return getattr(item, \"language_model\", item)\n\n"
            "self.vllm_config.compilation_config.compilation_time = max(\n"
            "    _compilation_time_language_model_value(t)\n"
            "    for t in compilation_times\n"
            ")"
        )
        return text.replace(old, new, 1)

    # Fallback: insert helper and rewrite just the generator expression if style differs.
    needle = "t.language_model for t in compilation_times"
    if needle not in text:
        return text

    helper = (
        f"{marker}\n"
        "def _compilation_time_language_model_value(item):\n"
        "    return getattr(item, \"language_model\", item)\n\n"
    )
    text = text.replace(needle, "_compilation_time_language_model_value(t) for t in compilation_times", 1)

    assign_anchor = "self.vllm_config.compilation_config.compilation_time = max("
    idx = text.find(assign_anchor)
    if idx == -1:
        return text
    return text[:idx] + helper + text[idx:]


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
