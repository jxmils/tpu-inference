#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Poll ``libtpu.sdk.tpumonitoring`` for ICI flit metrics after JAX attaches devices.

Run inside the same environment as vLLM (e.g. container ``python3``). On many v6e
builds ``ici_flits_*`` stay empty in-process; use ``scripts/tpu/ici_counters.sh``
on the host for flits when this script shows no data.

  python3 scripts/tpu/smoke_tpumonitoring_flits.py
  python3 scripts/tpu/smoke_tpumonitoring_flits.py --metrics ici_flits_tx,ici_flits_rx --sleep 1 --iter 5
"""

from __future__ import annotations

import argparse
import time


def _read_data(m: object) -> object:
    d = getattr(m, "data", None)
    if d is None:
        return None
    return d() if callable(d) else d


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--metrics",
        default="ici_flits_tx,ici_flits_rx",
        help="Comma-separated tpumonitoring metric names (default: flit counters).",
    )
    p.add_argument("--iter", type=int, default=5, help="Poll iterations.")
    p.add_argument("--sleep", type=float, default=1.0, help="Seconds between reads.")
    args = p.parse_args()
    names = tuple(x.strip() for x in args.metrics.split(",") if x.strip())

    import jax

    jax.devices()

    from libtpu.sdk import tpumonitoring

    supported = tpumonitoring.list_supported_metrics()
    flit_like = [x for x in supported if "flit" in str(x).lower()]
    ici_like = [x for x in supported if "ici" in str(x).lower()]
    print("supported (flit substring):", flit_like)
    print("supported (ici substring):", ici_like)

    for name in names:
        if name not in supported:
            print(f"warn: {name!r} not in list_supported_metrics()", flush=True)

    for i in range(args.iter):
        parts = []
        for name in names:
            raw = _read_data(tpumonitoring.get_metric(name))
            n = len(raw) if raw is not None else 0
            sample = (list(raw)[:3] if raw else None)
            parts.append(f"{name} n={n} sample={sample}")
        print(i, "|", " | ".join(parts), flush=True)
        if i + 1 < args.iter:
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
