---
title: Benchmarking
---

# Benchmarking

## Prerequisites

Use a source checkout of Project SuperDex. The benchmarking tools are scripts in the
source checkout, not part of the installable package. From the `project_superdex` root,
install the core workspace and the benchmark's additional dependency (see
[Installation and Setup](./setup.md)):

```bash
uv sync --extra core
uv pip install "tqdm>=4.67.1"
```

## Working Directory

From the `project_superdex` repository root, change to the directory containing the
benchmark script:

```bash
cd superdex_lab/apps/envs
```

## Run a Benchmark

Run the smallest environment as a smoke test:

```bash
# Smoke test. Any environment name works; cart_pole is the smallest.
uv run python benchmark.py \
  --env cart_pole --num_workers 1 --num_envs_per_worker 1
```

## Expected Output

Numeric values and the logical CPU count depend on the host. The output first lists the
planned configuration and system resources, then reports initialization and memory
measurements, progress, and a profiler summary with this structure:

```text
Benchmarking env 'cart_pole' for 1 combinations:
- 1 workers × 1 envs/worker = 1 envs

Logical CPUs: <logical CPU count>
Memory: <used> /  <total> GB

Initialization took <seconds> s.
Approx. memory used by the environments: <gigabytes> GB
System-wide memory usage: <percent>%
- Performed <iterations> iterations (<steps> steps), elapsed <seconds>s

Section                                      Last (ms)    Total (ms)    Total (%)    Count    Mean (ms)    Std.Dev (ms)    Info
1 workers × 1 envs/worker = 1 envs                                                        init_time=<seconds>s, sim_fps=<value>, control_fps=<value>, samples=<steps>, mem_usage=<gigabytes> GB
```

The profiler table's border style depends on the terminal. It contains the timing
columns shown above and the `init_time`, `sim_fps`, `control_fps`, `samples`, and
`mem_usage` fields. The script prints a cumulative summary after each configuration
and once more at the end, so a one-configuration smoke test prints the same summary
twice.

## Other Runs

```bash
# Sweep worker/environment combinations (3 x 4 = 12 configurations).
uv run python benchmark.py --env cart_pole --num_workers 1,4,8 --num_envs_per_worker 1,5,25,125

# Benchmark every CLI-visible environment with 8 workers and 20 envs/worker.
uv run python benchmark.py --all --num_workers 8 --num_envs_per_worker 20

# Custom timing constraints for a longer run.
uv run python benchmark.py --env cart_pole --num_workers 8 --num_envs_per_worker 20 \
  --min_time 10.0 --max_time 120.0 --min_iterations 50

# Export results to JSON.
uv run python benchmark.py --env cart_pole --num_workers 8 --num_envs_per_worker 20 \
  --write_to_file
```

## What Gets Benchmarked

The main benchmarking script, `apps/envs/benchmark.py`, measures throughput across
worker and per-worker environment counts. Select one environment with `--env`, or
every CLI-visible environment with `--all`. Run with `--help` to list the available
names.

It supports:

- **Work-distribution inputs**: sweeps the Cartesian product of worker counts and
  environments per worker
- **Performance outputs**: initialization time, memory usage, control FPS, and
  simulation FPS metrics
- **JSON export**: optional detailed results export for further analysis

## How Results Are Measured

The script overrides three environment configuration fields regardless of what
the environment or its variant JSON specifies:

| Field | Forced value | Effect on the numbers |
| --- | --- | --- |
| `num_worker_threads` | `0` | The physics solver is single-threaded; parallelism comes only from the vectorization layers |
| `use_shared_scenes` | `True` | Environments within a worker share one physics scene |
| `render_mode` | `None` | No rendering cost is included |

Each configuration builds `num_workers × num_envs_per_worker` environments and times
batched `env.step()` calls. **When that product is 1, the script constructs a bare
environment and bypasses `HybridVectorEnv` entirely**, so the single-env row is an
unvectorized reference point rather than a 1-worker vectorized measurement.

The metrics are defined below, where `mean_step_ms` is the mean wall time of one
batched `step()` in milliseconds:

```
control_fps = 1000 * num_envs / mean_step_ms
sim_fps     = control_fps * simulation_frequency / control_frequency
init_time   = wall time to construct all environments and
              perform the first reset()                    [s]
mem_usage   = system RAM used after construction and the first
              reset() minus before construction              [GB]
```

`control_fps` is control steps per second summed across all environments;
`sim_fps` is physics substeps per second. Action sampling happens inside the timed
region, so its cost is included.

**Stopping rule.** The timed stepping loop runs until both minimums (`min_time` and
`min_iterations`) are met, then stops when either maximum is reached. A minimum can
therefore extend a run past the other metric's maximum.

## Command-Line Options

| Option | Default | Notes |
| --- | --- | --- |
| `--env` | required unless `--all` | Environment CLI name. Validated against discovery. |
| `--all` | off | Benchmark every environment available to the CLI. Mutually exclusive with `--env`. |
| `--num_workers` | required | Accepts one or more positive integers, separated by spaces and/or commas. |
| `--num_envs_per_worker` | required | Accepts one or more positive integers, separated by spaces and/or commas. |
| `--min_iterations` | `10` | Minimum batched `step()` calls per configuration |
| `--max_iterations` | `100` | Maximum batched `step()` calls per configuration |
| `--min_time` | `5.0` | Minimum timed stepping duration per configuration, in seconds |
| `--max_time` | `60.0` | Maximum timed stepping duration per configuration, in seconds |
| `--write_to_file` | off | Writes `apps/envs/output/benchmark.json` — a script-relative path. With `--all`, writes one `benchmark_<env>.json` file per environment. Each file is rewritten after every configuration and accumulates all results so far. |

`--num_workers` and `--num_envs_per_worker` are combined as a full Cartesian
product, so `--num_workers 1,4,8 --num_envs_per_worker 1,5,25,125` runs twelve
configurations. Both options require at least one value.

:::tip Choosing a configuration
Each batched step waits for every worker to finish. Increasing
`--num_envs_per_worker` can reduce relative variation in worker completion times
and amortize fixed overhead, often improving aggregate throughput toward a plateau.
More environments per worker can also increase cache and memory pressure, so
throughput may eventually decline.

Sweep `--num_envs_per_worker 1,5,25,125`; `1` is the
one-environment-per-worker baseline. Use `--num_workers 1
--num_envs_per_worker 1` for the single-environment, unvectorized reference.
Start the `--num_workers` sweep at or below the physical CPU core count, then
optionally extend it up to the logical CPU count. Tune both values for the
environment and hardware.
:::
