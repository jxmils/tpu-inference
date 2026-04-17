#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# List or stream TPU ICI-related raw counters when `tpu-info` is on PATH (VM / image specific).
# Usage:
#   ./scripts/tpu/ici_counters.sh list
#   ICI_POLL_INTERVAL=200ms ./scripts/tpu/ici_counters.sh poll ici_flits_tx ici_flits_rx
# See docs/profiling.md (TPU ICI hardware counters section).

set -euo pipefail

if ! command -v tpu-info >/dev/null 2>&1; then
  echo "error: tpu-info not found on PATH." >&2
  echo "Install or use the Google TPU VM accelerator diagnostics bundle, then retry." >&2
  exit 127
fi

cmd="${1:-}"
shift || true

case "$cmd" in
  list)
    echo "=== tpu-info --help (excerpt) ==="
    tpu-info --help 2>&1 | head -40 || true
    echo ""
    echo "=== raw counters (if supported) ==="
    if tpu-info --list_raw_counters 2>&1; then
      :
    else
      echo "(flag --list_raw_counters not supported on this build; try tpu-info --help for alternatives.)" >&2
    fi
    ;;
  poll)
    if [[ $# -lt 1 ]]; then
      echo "usage: $0 poll <counter_name> [counter_name ...]" >&2
      echo "example: $0 poll ici_flits_tx ici_flits_rx" >&2
      exit 2
    fi
    interval="${ICI_POLL_INTERVAL:-100ms}"
    # shellcheck disable=SC2068
    exec tpu-info --metric_raw "$@" --interval "$interval"
    ;;
  *)
    echo "usage: $0 list | poll <counter> [counter ...]" >&2
    exit 2
    ;;
esac
