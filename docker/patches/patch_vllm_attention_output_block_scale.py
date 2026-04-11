#!/usr/bin/env python3
"""Patch vLLM attention to ignore output_block_scale when unsupported."""

from __future__ import annotations

from pathlib import Path


def _ensure_import_inspect(text: str) -> str:
    if "\nimport inspect\n" in text or "\nfrom inspect " in text:
        return text

    # Prefer inserting right before TYPE_CHECKING to avoid splitting
    # parenthesized multiline imports.
    marker = "\nif TYPE_CHECKING:\n"
    if marker in text:
        return text.replace(marker, "\nimport inspect" + marker, 1)

    lines = text.splitlines()
    insert_at = 0
    in_imports = False
    paren_balance = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not in_imports and (line.startswith("import ") or line.startswith("from ")):
            in_imports = True
        if in_imports:
            paren_balance += line.count("(") - line.count(")")
            insert_at = idx + 1
            if paren_balance == 0 and not (line.startswith("import ") or line.startswith("from ")):
                break

    # Keep a visual separation from subsequent code.
    if insert_at < len(lines) and lines[insert_at].strip():
        lines.insert(insert_at, "")
        insert_at += 1
    lines.insert(insert_at, "import inspect")
    return "\n".join(lines) + "\n"


def _patch_unified_attention(text: str) -> str:
    marker = "# TPU_INFERENCE_OUTPUT_BLOCK_SCALE_PATCH"
    if marker in text:
        return text

    start = text.find("def unified_attention_with_output(")
    if start == -1:
        return text
    end = text.find("def unified_attention_with_output_fake", start)
    if end == -1:
        return text

    before = text[:start]
    body = text[start:end]
    after = text[end:]

    lines = body.splitlines()
    call_start = None
    call_end = None
    for i, line in enumerate(lines):
        if "self.impl.forward(" in line:
            call_start = i
            indent = line[: len(line) - len(line.lstrip())]
            for j in range(i + 1, len(lines)):
                if lines[j].strip() == ")":
                    call_end = j
                    break
            if call_end is None:
                return text
            break
    if call_start is None:
        return text

    replacement = [
        indent + marker,
        indent + "kwargs = {\"output\": output, \"output_scale\": output_scale}",
        indent + "sig = inspect.signature(self.impl.forward)",
        indent + "if \"output_block_scale\" in sig.parameters or any(",
        indent + "    p.kind == inspect.Parameter.VAR_KEYWORD",
        indent + "    for p in sig.parameters.values()",
        indent + "):",
        indent + "    kwargs[\"output_block_scale\"] = output_block_scale",
        indent + "self.impl.forward(",
        indent + "    self,",
        indent + "    query,",
        indent + "    key,",
        indent + "    value,",
        indent + "    kv_cache,",
        indent + "    attn_metadata,",
        indent + "    **kwargs,",
        indent + ")",
    ]

    new_lines = lines[:call_start] + replacement + lines[call_end + 1 :]
    return before + "\n".join(new_lines) + "\n" + after


def main() -> None:
    path = Path("/workspace/vllm/vllm/model_executor/layers/attention/attention.py")
    if not path.exists():
        return
    text = path.read_text()
    if "output_block_scale" not in text:
        return
    text = _ensure_import_inspect(text)
    text = _patch_unified_attention(text)
    path.write_text(text)


if __name__ == "__main__":
    main()
