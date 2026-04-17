# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project

"""Host-side step trace helpers (no JAX import)."""


def comm_bytes_proxy_for_step(
    dispatch_total: int | None,
    return_total: int | None,
    a2a_total: int | None,
) -> tuple[int | None, str]:
    """Single communication-byte proxy for step JSONL (pair with XPlane for time).

    Prefer measured EP all-to-all bytes when present; else routing-informed
    dispatch+return totals; else a symmetric fallback from dispatch alone.
    """
    if a2a_total is not None and a2a_total > 0:
        return a2a_total, "a2a"
    if dispatch_total is not None and return_total is not None:
        if dispatch_total + return_total > 0:
            return dispatch_total + return_total, "dispatch_plus_return"
    if dispatch_total is not None and dispatch_total > 0:
        return 2 * dispatch_total, "dispatch_times_two"
    return None, "none"
