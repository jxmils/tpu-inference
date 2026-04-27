# Profiling

There are several complementary ways to profile your workload (in-process JAX, step JSONL, host ICI counters when available):

## Using `examples/tpu_profiling.py`

### vLLM TPU Profiling Script

This script is a utility for profiling the performance of the vLLM engine on TPU VMs. It uses the JAX profiler to capture detailed performance traces.

The profiling results can be visualized using tools like TensorBoard (with the `tensorboard-plugin-profile` package) or Perfetto UI.

### How to Use

#### Prerequisites
You must install the TensorBoard profile plugin to visualize the results:

```bash
pip install tensorboard-plugin-profile
```

#### Basic Command
The script is run from the command line, specifying the workload parameters and any necessary vLLM engine arguments.

```bash
python3 examples/tpu_profiling.py --model <your-model-name> [OPTIONS]
```

#### Key Arguments
* `--model`: (Required) The name or path of the model to profile.
* `--input-len`: The length of the input prompt tokens per request
* `--output-len`: The number of tokens to generate per request.
* `--batch-size`: The number of requests.
* `--profile-result-dir`: The directory where the JAX profiler output will be saved.
* The script also accepts all standard vLLM `EngineArgs` (e.g., `--tensor-parallel-size`, `--dtype`).

#### Examples

**1. Profile a Prefill Operation:**
To profile a single request with a long input prompt (e.g., 1024 tokens), set `--input-len` high and `--batch-size` to 1.

```bash
python3 examples/tpu_profiling.py \
  --model google/gemma-2b \
  --input-len 1024 \
  --output-len 1 \
  --batch-size 1
```

**2. Profile a Decoding Operation:**
To profile a large batch of single-token decoding steps, set `--input-len` and `--output-len` to 1 and use a large `--batch-size`.

```bash
python3 examples/tpu_profiling.py \
  --model google/gemma-2b \
  --input-len 1 \
  --output-len 1 \
  --batch-size 256
```

## XLA HLO dumps (buffer assignment, activations, KV)

To inspect XLA memory plans and buffer sizes (including large activations and
KV-related buffers), enable HLO dumps to a writable directory.

**Option A — environment only (matches upstream XLA):**

```bash
export XLA_FLAGS="--xla_dump_to=/tmp/hlo_dumps"
mkdir -p /tmp/hlo_dumps
```

**Option B — tpu-inference helper (merged before JAX compiles):**

```bash
export XLA_DUMP_TO=/tmp/hlo_dumps
```

When `tpu_inference` is imported normally, `env_override.py` merges
`--xla_dump_to=<path>` into `XLA_FLAGS` if it is not already present. Set this in
the same environment as `vllm serve` before the process starts. Dumps can be
large and slow compilation; use a fast local disk and remove old trees between
runs.

## vLLM production metrics (Prometheus / Grafana / CSV)

vLLM publishes engine statistics as Prometheus text on the OpenAI-compatible
HTTP server at **`/metrics`** (no separate “StatsCollector” HTTP endpoint; the
collector feeds these gauges). See
[Production metrics](https://docs.vllm.ai/en/latest/usage/metrics.html).

Relevant gauges for KV and load:

| Gauge | Meaning |
| --- | --- |
| `vllm:kv_cache_usage_perc` | KV block utilization (0–1; 1 = full). |
| `vllm:num_requests_running` | Requests in model execution batches. |
| `vllm:num_requests_waiting` | Requests waiting to be scheduled. |

**Grafana / Prometheus:** scrape `http://<api-host>:<port>/metrics` with your
usual Prometheus job config.

**Local CSV:** sample the endpoint into a spreadsheet-friendly file:

```bash
python3 scripts/vllm/prometheus_metrics_poll_csv.py \
  --url http://127.0.0.1:8000/metrics \
  --output /tmp/vllm_kv_stats.csv \
  --interval-s 5
```

Stop with Ctrl+C. The table uses the public metric names above; there is no
separate Prometheus series named `num_active_sequences` — use
`vllm:num_requests_running` (and waiting) for concurrency.

## Using `PHASED_PROFILING_DIR`
If you set the following environment variable:

```

PHASED_PROFILING_DIR=<DESIRED PROFILING OUTPUT DIR>

```

we will automatically capture profiles during three phases of your workload (assuming they are encountered):
1. Prefill-heavy (the quotient of prefill / total scheduled tokens for the given batch is => 0.9)
2. Decode-heavy (the quotient of prefill / total scheduled tokens for the given batch is <= 0.2)
3. Mixed (the quotient of prefill / total scheduled tokens for the given batch is between 0.4 and 0.6)

To aid in your analysis, we will also log the batch composition for the profiled batches.

### Prefill *and* decode: bandwidth-oriented traces

**Yes — this is already what phased profiling is for.** Each phase gets its own `jax.profiler.start_trace` … `stop_trace` window under:

- `<PHASED_PROFILING_DIR>/prefill_heavy/` — batches where ≥ **90%** of scheduled tokens are prefill (`PREFILL_HEAVY_RATIO_THRESHOLD` in `runner/utils.py`).
- `<PHASED_PROFILING_DIR>/decode_heavy/` — batches where ≤ **20%** of scheduled tokens are prefill.
- `<PHASED_PROFILING_DIR>/balanced/` — prefill fraction between **40%** and **60%**.

So you get **separate XLA/JAX trace directories** for prefill-dominated vs decode-dominated steps (plus mixed). Open each folder in **TensorBoard → Profile** or **Perfetto** and inspect collective / communication ops and timeline duration — that is the right place to reason about **effective interconnect / memory bandwidth per phase**, not the lightweight MoE JSONL traces.

**Tuning:**

- `PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR` (default **15**): how many engine steps to include per phase after a phase first triggers.
- Phased profiling only starts after the first “real” batch (`num_reqs > 1` and `total_num_scheduled_tokens > 1`) so a trivial first step does not consume a phase slot.
- Batches whose prefill ratio falls in the **ambiguous** bands (e.g. between 20% and 40%, or 60% and 90%) do **not** start a new phase profile; design your load (bench mix, concurrency) so you actually hit prefill-heavy and decode-heavy windows during the run.

**Alternative (offline, two runs):** run `examples/tpu_profiling.py` twice — once with large `--input-len` and small `--output-len` (prefill-shaped), once with `--input-len 1` and large `--batch-size` (decode-shaped). Each run writes one profile tree under `--profile-result-dir` (add subdirs or different dirs per run).

**Live server:** with `USE_JAX_PROFILER_SERVER=True`, you can manually **Capture Profile** twice: once while the server is doing mostly prefill, and again during steady decode (e.g. after prompts are loaded).

## Using `USE_JAX_PROFILER_SERVER`
If you set the following environment variable:

```

USE_JAX_PROFILER_SERVER=True

```

you can instead manually decide when to capture a profile and for how long, which can helpful if your workload (e.g. E2E benchmarking) is
large and taking a profile of the entire workload (i.e. using the above method) will generate a massive tracing file.

You can additionally set the desired profiling port (default is `9999`):

```

JAX_PROFILER_SERVER_PORT=XXXX

```

In order to use this approach, you can do the following:

1. Run your typical `vllm serve` or `offline_inference` command (making sure to set `USE_JAX_PROFILER_SERVER=True`)
2. Run your benchmarking command (`python benchmark_serving.py...`)
3. Once the warmup has completed and your benchmark is running, start a new tensorboard instance with your `logdir` set to the desired output location of your profiles (e.g. `tensorboard --logdir=profiles/llama3-mmlu/`)
4. Open the tensorboard instance and navigate to the `profile` page (e.g. `http://localhost:6006/#profile`)
5. Click `Capture Profile` and, in the `Profile Service URL(s) or TPU name` box, enter `localhost:XXXX` where `XXXX` is your `JAX_PROFILER_SERVER_PORT` (default is `9999`)

6. Enter the desired amount of time (in ms)

## Compute vs communication breakdown

**Time (device-level):** Open the JAX/XPlane trace (TensorBoard **Profile** or **Perfetto**). The runner adds nested profiler scopes so you can align timelines:

- `execute_model: …` — full scheduling step (existing).
- `vllm:model_fn` — transformer `model_fn()` (attention, MoE, all collectives emitted inside that compiled region).
- `vllm:compute_logits` — LM head / logits after hidden states.
- `vllm:mm_encoder` — multimodal encoder path only.

**First-class layer scopes (`tpu_trace:*`):** the JAX forward also emits `jax.profiler.TraceAnnotation` regions via `tpu_inference.profiler_trace.tpu_trace_region` (for example `tpu_trace:attention`, `tpu_trace:gate`, `tpu_trace:moe_experts`, `tpu_trace:moe_all2all_dispatch`). `trace_plotter/perfetto_op_breakdown.py` prefers these names over kernel-string heuristics when building stacked breakdown plots. Disable all such regions with `DISABLE_TPU_TRACE_ANNOTATIONS=1`.

**Fused collective + matmul (XLA):** the TPU runner’s main `jax.jit` step in `VllmModelWrapper` sets `compiler_options` with `xla_tpu_all_gather_collective_matmul_mode` and `xla_tpu_reduce_scatter_collective_matmul_mode` both to `post_spmd_conservative`. That fusion is good for performance but Chrome / Perfetto exports often show the fused region as opaque names such as `$<unknown> reduce` instead of separate dot and collective slices. For debugging or apples-to-apples comparison with a minimal JAX workload, set **`DISABLE_TPU_COLLECTIVE_MATMUL_FUSION=1`** before process start so those options are omitted (expect a **recompile** and different performance).

In XProf, collective ops (all-reduce, all-to-all, reduce-scatter, etc.) are **communication**; matmuls, activations, softmax, etc. are **compute**. **Overlap** appears when the critical path is shorter than the sum of isolated collective durations—use the trace critical path, not summed kernel labels alone.

**Host wall splits (approximate):** `step/step_trace_*.jsonl` `step_summary` records include `model_forward_wall_time_s` and `compute_logits_wall_time_s` (body vs logits head only; they do **not** split MoE matmul vs MoE all-to-all).

**Communication byte proxy:** `comm_bytes_proxy_total` and `comm_bytes_proxy_source` unify A2A vs dispatch/return estimates for GB/s numerators (see list under *MoE / interconnect* below). Pair with `model_forward_wall_time_s` only as a **rough** effective rate—the denominator still includes non-communication work.

## MoE / interconnect effective bandwidth from traces

There is no single counter that reports “ICI GB/s” in Cloud Monitoring. You combine **payload estimates** with **time**:

### A) Step JSONL (`step/step_trace_*.jsonl`) + routing aggregate

When MoE routing capture is enabled, each `step_summary` line includes:

- `estimated_dispatch_bytes_total`, `estimated_return_bytes_total` — token×expert routing byte models.
- `a2a_bytes_total_sum` — populated only when **`CAPTURE_MOE_ROUTING_A2A=1`** (default on) **and** the model uses expert parallelism with a named data axis (`capture_a2a` path in `routing_stats.py`). Otherwise this field is `null`.
- **`comm_bytes_proxy_total`**, **`comm_bytes_proxy_source`** — unified communication-byte estimate for the step (prefers A2A sum when positive, else dispatch+return, else a dispatch-only fallback; see section *Compute vs communication breakdown* above).
- **`model_forward_wall_time_s`** — host-side wall time for the `model_fn()` call on that step (added for bandwidth proxies). This spans the full forward, not an isolated all-to-all kernel.
- **`compute_logits_wall_time_s`** — host-side wall time for the logits head on that step (when the generative path runs `compute_logits_fn`).

Set **`MOE_ROUTING_STATS_DIR`** and **`REQUEST_STATS_DIR`** to the same trace root (e.g. `.../raw_traces`) so routing summaries and step summaries land together.

**Rough proxies** (interpret with care):

- `(a2a_bytes_total_sum / model_forward_wall_time_s) / 1e9` → GB/s if A2A bytes are present.
- `(comm_bytes_proxy_total / model_forward_wall_time_s) / 1e9` → unified comm-byte / time proxy when `comm_bytes_proxy_total` is non-null.
- `((dispatch + return) / 2) / model_forward_wall_time_s` → another byte/time ratio when A2A is absent; denominator still includes attention, MLP, etc.

For **kernel-level** collective time and sizes, use the JAX profiler (`examples/tpu_profiling.py`, `PHASED_PROFILING_DIR`, or `USE_JAX_PROFILER_SERVER`) and inspect the trace in TensorBoard / Perfetto.

### Automatic per-step comm vs compute table (Perfetto post-processing)

If your trace directory contains `perfetto_trace.json` (or `.json.gz`), you can
generate a per-step CSV with:

- `comm_time_ms`
- `compute_time_ms`
- `overlap_ms`

using:

```bash
python3 scripts/vllm/benchmarking/trace_comm_compute_breakdown.py \
  --trace /workspace/vllm_jax_profiles/perfetto_trace.json.gz \
  --output /workspace/vllm_jax_profiles/step_comm_compute.csv
```

This script uses the `execute_model: ...` trace scope as a step boundary and
classifies communication events by name/category matches for collectives (for
example `all-reduce`, `all-to-all`, `reduce-scatter`, `collective-permute`,
`send`, `recv`). It is a practical approximation for paper plots; verify edge
cases by checking the underlying trace lanes in TensorBoard/Perfetto.

## TPU ICI hardware counters and host diagnostics (v6e / Trillium)

JAX/XPlane (above) shows **what the program expressed** (collectives, fusion, timeline). On **TPU VMs**, Google’s host-side tooling can additionally expose **hardware performance counters** for the **Inter-Chip Interconnect (ICI)**—useful for a **traffic-matrix / hot-link** story during MoE all-to-all versus steady TP all-reduce.

### In-process ICI snapshots (`libtpu.sdk.tpumonitoring`)

**Traffic story (flits):** the useful fabric counters for “how much moved on ICI” are typically **`ici_flits_tx`** / **`ici_flits_rx`** (or names your **`tpu-info --list_raw_counters`** prints). The runner defaults **`ICI_TPUMONITORING_METRICS`** to those flit names so **`step_summary`** can carry per-step deltas when libtpu exposes them.

**No sidecar:** set **`CAPTURE_ICI_TPUMONITORING=1`**. The runner snapshots selected counters **immediately before** and **immediately after** **`model_fn()`**, with **`jax.block_until_ready`** on **`(kv_caches, hidden_states)`** before the post-forward read so the delta aligns with completed device work. Deltas are written into the same **`step_summary`** JSONL record as **`comm_bytes_proxy_*`**, nested under **`ici_hw_delta`**: for each metric, **`delta_per_chip`** (list) and **`delta_sum`**.

- **`ICI_TPUMONITORING_METRICS`**: comma-separated names (default **`ici_flits_tx,ici_flits_rx`**). On **v6e / Trillium** we have often seen **`list_supported_metrics()`** omit flit names and **`get_metric("ici_flits_*").data()`** stay **`[]`** — in-process tpumonitoring is then **not** your flit source. For **actual flit telemetry** on those stacks, use **host** **`tpu-info`** / **`scripts/tpu/ici_counters.sh`** (next subsection) in parallel with the benchmark; correlate by wall clock or trace windows. Optional **`ici_link_health`** is a **link diagnostic** (scores in strings), not a substitute for flit volume; you can add it to the comma-separated list if you want both. Confirm any name with **`tpumonitoring.list_supported_metrics()`** in the **same Python as vLLM** (inside the container), or [Cloud TPU monitoring docs](https://cloud.google.com/tpu/docs/tpu-monitoring-library).
- **Prerequisites for the row:** **`CAPTURE_REQUEST_STATS`** + **`REQUEST_STATS_DIR`** (step trace writer), and the **MoE routing** path that calls **`_persist_step_trace`** — i.e. enable **`CAPTURE_MOE_ROUTING_STATS`** and **`MOE_ROUTING_STATS_DIR`** so **`step_summary`** lines are emitted for those steps.

If **`libtpu.sdk.tpumonitoring`** is not importable, capture is skipped (no error).

**Empty `data()` in a REPL:** `get_metric("ici_link_health").data()` can return **`[]`** in a standalone Python process that has **not** initialized JAX / attached TPU devices. Try again **after** `jax.devices()` (or from code paths that already run inside **`vllm serve`**). The runner snapshots only around **`model_fn`**, so production captures use a process where the TPU client is already live.

**Important:** package names, CLI flags, and **counter spellings change by image generation and diagnostics bundle version**. For host-only workflows, treat the **`tpu-info`** commands below as a **workflow**; always run `tpu-info --help` (or the tool your image ships) and **`--list_raw_counters`** (or equivalent) on **your** node before scripting a paper pipeline.

### 1) Raw ICI counters (“flit” or link-level streams)

On many TPU VM images, **`tpu-info`** is provided as part of Google’s accelerator diagnostics stack (naming varies, e.g. **cloud-accelerator-diagnostics**). Typical workflow:

1. Discover what your build exports (examples only—**verify locally**):

   ```bash
   tpu-info --help
   tpu-info --list_raw_counters    # if supported; flag name may differ
   ```

2. Stream a small set of **ICI-related** counters while you drive load (second shell or `tmux`), e.g. flit transmit/receive, stalls, or congestion—**pick exact metric names from the list**, not from a blog table:

   ```bash
   # From the tpu-inference repo root (inside the container if the repo is mounted):
   ./scripts/tpu/ici_counters.sh list
   # Illustrative metric names only—substitute names printed by the list command:
   ./scripts/tpu/ici_counters.sh poll ici_flits_tx ici_flits_rx
   ```

   If `tpu-info` flags differ on your image, run `tpu-info --help` and adjust `scripts/tpu/ici_counters.sh` locally.

3. **Correlate** counter CSV or logs with:

   - wall-clock from your benchmark (`benchmark_serving.py` timestamps),
   - engine **`step_summary`** lines (`trace_step`, `comm_bytes_proxy_*`, MoE routing),
   - JAX **`jax.profiler.start_trace`** windows.

That gives a **per-link or per-chip aggregate** view of whether certain ICI directions saturate during **EP-heavy** steps (skewed expert routing) versus flatter **TP AR-heavy** phases.

### 2) XProf / low-level operation (LLO) bundles (optional)

Some VM + **vLLM + libtpu** combinations support **extra low-level** profiling (sometimes referred to as **LLO** bundles or extended **XProf** capture). This is **not wired inside tpu-inference**; it depends on the **exact** `vllm`, `libtpu`, and Cloud Diagnostics packages on your **`moe-trace`** image.

- Set **`VLLM_TPU_LLO_PROFILING=1`** in the environment **before** starting `vllm serve` **if** your stack documents that variable (confirm with `vllm serve --help` / release notes for your pin).
- If your image installs **Cloud Diagnostics XProf** Python APIs, you may be able to trigger a **snapshot** to a directory alongside your existing `jax.profiler` output—again, follow the **version-specific** docs bundled with that package (module names and APIs move between releases).

Use the same **output directory discipline** as JAX profiling (large files; fast disk).

### 3) Relating XLA / HLO collectives to ICI behavior (TP vs EP)

When you enable **`XLA_FLAGS=--xla_dump_to=...`** (or **`XLA_DUMP_TO`**, see earlier sections), HLO text and buffer assignment help you **name** the collective pattern per compiled region. At runtime, **XPlane** shows those collectives on the timeline. **ICI counters** (when available) show **fabric-level** stress. Together:

| XLA / trace pattern | Typical role in vLLM TPU stacks | ICI intuition (paper framing) |
| --- | --- | --- |
| **All-reduce** (and related reductions across the **model** axis) | Tensor-parallel partial sums / norms | Steady, structured traffic; often similar **per layer** in dense or MoE blocks. |
| **All-to-all** (and permutes / ragged variants) | Expert-parallel **dispatch / combine** | Volume and **skew** follow **router** choices; good fit for **hot-spot** counter analysis. |

There is **no guaranteed 1:1** string mapping from a single HLO opcode name to a single hardware counter bucket on every build; use **HLO + XPlane + counters** as **triangulation**, not a single counter name in isolation.
