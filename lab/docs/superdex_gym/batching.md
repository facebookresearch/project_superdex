---
title: Training with Large Batches
---

# Training with Large Batches

SuperDex Gym scales RL training by combining two-tier vectorization with scene sharing
within each worker. See
[`run_with_gymnasium_vectorization.py`](./running_examples.md#2-run_with_gymnasium_vectorizationpy)
for a runnable example and [Benchmarking](./benchmarking.md) for measuring throughput
on your hardware.

## Example Pattern

This minimal example shows how to batch training safely with `HybridVectorEnv`.

```python
from superdex.lab.gym.envs.benchmarks.cartpole_env import CartPoleEnv
from superdex.lab.gym.utils.vector import HybridVectorEnv

num_training_steps = 1000


def main():
    # Environment factory functions, one per environment.
    env_creators = [lambda: CartPoleEnv({"render_mode": None}) for _ in range(1000)]

    # 8 async workers, each stepping 125 environments sequentially.
    with HybridVectorEnv(env_creators, num_envs_per_worker=125) as env:
        print(f"Created {env.num_envs} environments")
        print(f"Using {env.num_workers} async workers")
        print(f"Each worker manages {env.num_envs_per_worker} environments")

        observations, _ = env.reset(seed=list(range(env.num_envs)))
        for step in range(num_training_steps):
            actions = ...  # your policy
            observations, rewards, terminated, truncated, info = env.step(actions)


# Required: the wrapper spawns its workers, so each child re-imports this module.
if __name__ == "__main__":
    main()
```

Using `HybridVectorEnv` as a context manager closes all workers if any worker raises
an exception during the training loop.

:::warning The `__main__` guard is not optional
Construct `HybridVectorEnv` inside a function called from an
`if __name__ == "__main__":` block, or spawned workers may fail to start.
:::

### Output and batch shapes

The example prints:

```text
Created 1000 environments
Using 8 async workers
Each worker manages 125 environments
```

Inside the wrapper, the batch shape is
`(num_workers, num_envs_per_worker, ...)`; the public API flattens it to
`(num_envs, ...)`. For the example above, the internal leading dimensions are
`(8, 125, ...)`, and the public batch dimension is `1000`:

| Value | Expected shape or result |
| --- | --- |
| `observations` | `env.observation_space.shape`, beginning with `(1000, ...)` |
| `actions` | Exactly `env.action_space.shape`, beginning with `(1000, ...)` |
| `rewards` | `(1000,)` |
| `terminated` | `(1000,)` |
| `truncated` | `(1000,)` |
| `info` | A dict of per-environment arrays each indexed by environment ID, keyed by information type, such as truncation or termination reasons. |

### Environment properties

| Property | Meaning |
| --- | --- |
| `num_envs` | Total environments (`num_workers × num_envs_per_worker`) |
| `num_workers` | Async subprocesses |
| `num_envs_per_worker` | Environments stepped sequentially per worker |
| `observation_space` / `action_space` | Batched, shape `(num_envs, ...)` |
| `single_observation_space` / `single_action_space` | Per-environment shape |

`step()` requires a NumPy action array with shape `action_space.shape` and raises
`ValueError` for any other input. It returns a `BatchedStepResult` named tuple of
`(observation, reward, terminated, truncated, info)`.

`info` includes arrays keyed by "terminated_reason" and "truncated_reason", and each array is indexed by global environment index. To read termination or truncation reasons from `info`:

```python
observations, rewards, terminated, truncated, info = env.step(actions)
for i in range(env.num_envs):
    if terminated[i] or truncated[i]:
        key = "terminated_reason" if terminated[i] else "truncated_reason"
        print(i, info[key][i])
```

## How Batching Works

Large-scale training uses two mechanisms:

1. **`HybridVectorEnv`** combines async and sync execution.
2. **Scene sharing** reduces memory use and startup time through shared physics
   scenes.

### Two-tier vectorization

`HybridVectorEnv` has two execution layers:

- **Outer layer**: an `AsyncVectorEnv` distributes work across `num_workers`
  subprocesses.
- **Inner layer**: each worker runs a `SyncVectorEnv` that steps
  `num_envs_per_worker` environments sequentially in-process.

The wrapper derives the worker count as
`num_envs // num_envs_per_worker`; it does not choose a count based on available CPU
cores. More workers increase parallelism and inter-process communication, while more
environments per worker can reduce load imbalance and improve scene sharing.

### Common constructor options

```python
HybridVectorEnv(
    env_fns,                     # Sequence[Callable[[], Env]], one per environment
    num_envs_per_worker=1,
    autoreset_mode=AutoresetMode.NEXT_STEP,
)
```

`env_fns` contains one factory per environment. Autoreset defaults to `NEXT_STEP` and
runs at the inner layer. The outer layer disables autoreset so the two layers do not
compound it.

### Shared-scene state isolation

Within each process, `SceneManager` keeps one instance per environment class name. Each
instance holds scenes by name with a reference count. Two environment classes never
share a scene, even when both use the same scene name.

Each environment that shares a scene keeps its own state snapshot. Before that
environment steps, it restores its snapshot into the shared scene; afterward, it
captures the updated state. Sequential stepping prevents one environment's state from
leaking into another.

`Scene.capture_state()` captures dynamic state only: poses, velocities, controller
and constraint targets, external forces, the scene clock, and the step counter.
Restoring a snapshot also rewinds simulation time. It does not capture topology,
materials, joint limits, or gravity. Every configuration field that changes scene
construction must be part of the scene name. Otherwise, environments with different
structures incorrectly share one scene.

Enable sharing through the environment configuration:

```python
config = {
    "use_shared_scenes": True,  # Default.
    "control_frequency": 20,
    "simulation_frequency": 100,
    # ... other config options
}

env_creators = [lambda: YourCustomEnv(config) for _ in range(1000)]
```

### Scene sharing in custom environments

Route scene creation through `_load_scene()`. It reuses a scene with the requested
name or registers a new scene. Include every configuration field that changes scene
construction in that name. See
[Authoring a Custom Environment](./environments.mdx#building-a-scene-from-a-prefab)
for a complete example.

The three built-in benchmark environments use this pattern. `CartPoleEnv` uses
`uid_fields = (cfg.use_damping, cfg.use_gravity, cfg.free_pole)`, the three fields
that alter its prefab. It excludes fields such as `actuate_on_pole` that change only
how actions are applied.

## Failure Modes

### Construction and reset inputs

- **Divisibility is required.** `len(env_fns) % num_envs_per_worker != 0` raises
  `ValueError: The total number of environments must be divisible by the number of
  environments per process.`
- **The multiprocessing context is always `"spawn"`.** Passing another explicit
  context emits a `RuntimeWarning`, and the wrapper uses `"spawn"` instead.
- **Spaces must be `Box`.** The wrapper reshapes `low` and `high` arrays directly. A
  `Dict` or `Discrete` space fails during space flattening. Every SuperDex Gym
  environment satisfies this contract because `MochiEnv` flattens its structured
  spaces to a single `Box`.
- **There are no reset masks.** Passing `options={"reset_mask": ...}` raises
  `ValueError: Reset mask is not supported for this vectorization.` Use Gymnasium's
  `AsyncVectorEnv` or `SyncVectorEnv` directly for selective resets.

:::warning Use per-environment seeds for independent rollouts
A scalar `int` seed is expanded twice: once by the outer `AsyncVectorEnv` and again by
each inner `SyncVectorEnv`. Environment `(w, k)` receives `seed + w + k`. Therefore,
`W × K` environments receive only `W + K - 1` distinct seeds. With 8 workers and 125
environments per worker, 1000 environments receive only 132 distinct seeds. The
wrapper emits no error or warning, and rollouts become silently correlated.

```python
# Wrong: 1000 environments, 132 distinct seeds.
env.reset(seed=42)

# Right: one seed per environment.
env.reset(seed=list(range(env.num_envs)))
```

Pass a list or tuple containing one seed per environment. `reset()` validates its
length and distributes the values across workers so global environment `i` receives
`seed[i]`. NumPy arrays do not take this sequence-chunking path and are unsupported.
:::

### Shared-scene correctness

:::caution Scene sharing assumes sequential stepping
`use_shared_scenes` defaults to `True`, so every environment opts in. Sharing is
correct only when environments in the same process run one at a time.
`HybridVectorEnv` provides this guarantee through each worker's inner `SyncVectorEnv`.
Do not use shared scenes with a vectorizer that steps multiple environments
concurrently in the same process; their state will be corrupted.
:::

State capture does not support scenes containing ROM actors. Attempting to capture
one raises `"State capture is not supported for scenes with ROM actors."`

If scene sharing is enabled but the scene was not created through `_load_scene()`,
the environment emits a `RuntimeWarning` on reset.

## Performance

### Choose the async/sync split

For a fixed total environment count, the worker split has two main tradeoffs:

- **More workers with fewer environments each** increase CPU parallelism. Each worker
  adds a subprocess and a full physics context. Because every worker builds its own
  scene, this split reduces scene-sharing benefits.
- **Fewer workers with more environments each** amortize scene construction and
  memory across each worker. Each worker steps environments sequentially, so its wall
  time generally increases with `num_envs_per_worker`.

Use [the benchmark script](./benchmarking.md) to tune the split for the target
environment and hardware. Start `num_workers` at up to the physical CPU core count,
then optionally sweep up to the logical CPU count. Increase `num_envs_per_worker` from
1 until throughput plateaus. The script sweeps `--num_workers` and
`--num_envs_per_worker` combinations and reports control and simulation FPS.

### Scene-sharing savings

When environments use an identical scene configuration, sharing reduces memory use
and startup time. Sequential stepping within each worker makes the optimization safe.
Within each worker process, the SuperDex Physics file cache also avoids repeated
model-loading work.
