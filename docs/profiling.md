# Profiling

There are currently three ways to profile your workload:

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

## MoE / interconnect effective bandwidth from traces

There is no single counter that reports “ICI GB/s” in Cloud Monitoring. You combine **payload estimates** with **time**:

### A) Step JSONL (`step/step_trace_*.jsonl`) + routing aggregate

When MoE routing capture is enabled, each `step_summary` line includes:

- `estimated_dispatch_bytes_total`, `estimated_return_bytes_total` — token×expert routing byte models.
- `a2a_bytes_total_sum` — populated only when **`CAPTURE_MOE_ROUTING_A2A=1`** (default on) **and** the model uses expert parallelism with a named data axis (`capture_a2a` path in `routing_stats.py`). Otherwise this field is `null`.
- **`model_forward_wall_time_s`** — host-side wall time for the `model_fn()` call on that step (added for bandwidth proxies). This spans the full forward, not an isolated all-to-all kernel.

Set **`MOE_ROUTING_STATS_DIR`** and **`REQUEST_STATS_DIR`** to the same trace root (e.g. `.../raw_traces`) so routing summaries and step summaries land together.

**Rough proxies** (interpret with care):

- `(a2a_bytes_total_sum / model_forward_wall_time_s) / 1e9` → GB/s if A2A bytes are present.
- `((dispatch + return) / 2) / model_forward_wall_time_s` → another byte/time ratio when A2A is absent; denominator still includes attention, MLP, etc.

For **kernel-level** collective time and sizes, use the JAX profiler (`examples/tpu_profiling.py`, `PHASED_PROFILING_DIR`, or `USE_JAX_PROFILER_SERVER`) and inspect the trace in TensorBoard / Perfetto.
