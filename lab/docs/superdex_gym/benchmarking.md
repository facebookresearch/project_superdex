---
title: Benchmarking
---

# Benchmarking

SuperDex Lab's benchmarking tools are scripts in the source checkout, not part
of the installable package. The main one is `apps/envs/benchmark.py`, which
measures throughput across a sweep of worker and per-worker environment counts.
Select one environment with `--env`, or explicitly select every discovered
environment with `--all` (run with `--help` to list them).

## Configurable Settings

- **Work distribution**: worker counts and environments per worker, swept as a
  Cartesian product
- **Performance metrics**: initialization time, memory usage, control FPS and
  simulation FPS
- **Timing constraints**: minimum and maximum iterations and time limits
- **JSON export**: optional detailed results export for further analysis

## Usage

After `uv sync --extra core` from the repository root (see
[Installation and Setup](./setup.md)):

:::note Extra packages these scripts need
`benchmark.py` imports `psutil` and `tqdm`, both at module level, so the failure lands
before any flag is parsed. `psutil` comes from `uv.lock`, so the one that actually
bites on a synced environment is `tqdm`: `uv run --no-sync python benchmark.py
--help` exits with `ModuleNotFoundError: No module named 'tqdm'`. It is not a declared
dependency of `superdex-lab`, so install it even if you never touch training:

```bash
uv pip install "tqdm>=4.67.1"
```

Note that a later `uv sync` removes it again. The commands below use `--no-sync`,
which is safe after the initial sync and preserves `tqdm` plus the RLlib dependency
versions if you install that stack later. See
[the caveats in Installation and Setup](./setup.md#dependencies-for-apps).
:::

```bash
cd superdex_lab/apps/envs
```

```bash
# Smoke test. Any of the environment names works; cart_pole is the smallest.
uv run --no-sync python benchmark.py --env cart_pole --num_workers 1 --num_envs_per_worker 1

# Sweep worker/environment combinations (3 x 4 = 12 configurations).
uv run --no-sync python benchmark.py --env cart_pole --num_workers 1,4,8 --num_envs_per_worker 1,5,25,125

# Benchmark every discovered environment with 8 workers and 20 envs/worker.
uv run --no-sync python benchmark.py --all --num_workers 8 --num_envs_per_worker 20

# Custom timing constraints for a longer run.
uv run --no-sync python benchmark.py --env cart_pole --num_workers 8 --num_envs_per_worker 20 \
  --min_time 10.0 --max_time 120.0 --min_iterations 50

# Export results to JSON.
uv run --no-sync python benchmark.py --env cart_pole --num_workers 8 --num_envs_per_worker 20 \
  --write_to_file
```

## How results are measured

The script **overrides three environment configuration fields** regardless of what
the environment or its variant JSON specifies:

| Field | Forced value | Effect on the numbers |
| --- | --- | --- |
| `num_worker_threads` | `0` | The physics solver is single-threaded; parallelism comes only from the vectorization layers |
| `use_shared_scenes` | `True` | Environments within a worker share one physics scene |
| `render_mode` | `None` | No rendering cost is included |

Each configuration builds `num_workers × num_envs_per_worker` environments and
times batched `env.step()` calls. **When that product is 1, the script constructs a
bare environment and bypasses `HybridVectorEnv` entirely**, so the single-env row is
an unvectorized reference point rather than a 1-worker vectorized measurement.

Metrics, where `mean_step_ms` is the mean wall time of one batched `step()` in
milliseconds:

```
control_fps = 1000 * num_envs / mean_step_ms
sim_fps     = control_fps * simulation_frequency / control_frequency
init_time   = wall time to construct all environments and
              perform the first reset()                    [s]
mem_usage   = system RAM used after construction minus before   [GB]
```

`control_fps` is control steps per second summed across all environments;
`sim_fps` is physics substeps per second. Action sampling happens inside the timed
region, so its cost is included.

**Stopping rule.** `min_iterations` and `min_time` are both hard floors — the loop
cannot exit until both are satisfied. Once they are, it exits as soon as *either*
`max_iterations` or `max_time` is reached. If `min_time` exceeds the time needed for
`max_iterations`, the run continues past the iteration cap.

## Command-Line Options

| Option | Default | Notes |
| --- | --- | --- |
| `--env` | required unless `--all` | Environment CLI name. Validated against discovery. |
| `--all` | off | Benchmark every discovered environment. Mutually exclusive with `--env`. |
| `--num_workers` | required | Accepts one or more positive integers, separated by spaces and/or commas. |
| `--num_envs_per_worker` | required | Accepts one or more positive integers, separated by spaces and/or commas. |
| `--min_iterations` | `10` | Minimum batched `step()` calls per configuration |
| `--max_iterations` | `100` | Maximum batched `step()` calls per configuration |
| `--min_time` | `5.0` | Minimum runtime per configuration, in seconds |
| `--max_time` | `60.0` | Maximum runtime per configuration, in seconds |
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
Initially set `--num_workers` no higher than the physical CPU core count, then
tune both values for the environment and hardware.
:::
