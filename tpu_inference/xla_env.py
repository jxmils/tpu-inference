# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project

"""XLA compiler flags from environment (stdlib only; safe before ``import jax``)."""

from __future__ import annotations

import os


def apply_xla_dump_from_env() -> None:
    """If ``XLA_DUMP_TO`` is set, append ``--xla_dump_to=...`` to ``XLA_FLAGS``.

    This must run before JAX triggers the first XLA compilation in the process
    (ideally before ``import jax``). HLO dumps include buffer assignment and
    sizes useful for activation and KV-cache planning.

    You can instead set ``XLA_FLAGS`` yourself, e.g.::

        export XLA_FLAGS="--xla_dump_to=/tmp/hlo_dumps"

    If ``XLA_FLAGS`` already contains ``--xla_dump_to=``, this function does
    nothing.
    """
    dump_to = os.environ.get("XLA_DUMP_TO", "").strip()
    if not dump_to:
        return
    try:
        os.makedirs(dump_to, exist_ok=True)
    except OSError:
        pass
    token = f"--xla_dump_to={dump_to}"
    existing = os.environ.get("XLA_FLAGS", "").strip()
    if token in existing:
        return
    os.environ["XLA_FLAGS"] = f"{existing} {token}".strip() if existing else token
