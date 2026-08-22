---
title: Training with Large Batches
---

# Training with Large Batches

SuperDex Gym scales RL training to large numbers of parallel environments through
two mechanisms: a two-tier vectorization wrapper and shared physics scenes. This
guide explains both, and the correctness constraints each one carries. See
[`run_with_gymnasium_vectorization.py`](./running_examples.md#2-run_with_gymnasium_vectorizationpy)
for a runnable example, and [Benchmarking](./benchmarking.md) for measured
throughput on your hardware.

## Overview

Large-scale training relies on:

1. **`HybridVectorEnv`**: a vectorization wrapper that combines async and sync
   execution.
2. **Scene sharing**: memory and startup savings through shared physics scenes.
3. **Reference-counted resource management**: automatic scene cleanup.

## HybridVectorEnv: Hybrid Parallelization Strategy

`HybridVectorEnv` uses a two-tiered approach:

- **Outer layer**: an `AsyncVectorEnv` distributes work across `num_workers`
  subprocesses.
- **Inner layer**: each worker runs a `SyncVectorEnv` that steps
  `num_envs_per_worker` environments sequentially in-process.

Worker count is pure arithmetic — `num_workers = num_envs // num_envs_per_worker`.
Nothing in the wrapper consults `os.cpu_count()`, `psutil` or CPU affinity, so the
split is entirely yours to choose.

This gives:

- **Parallelism**: multiple cores are used via async workers.
- **Efficiency**: fewer, larger inter-process messages than one worker per
  environment.
- **Memory optimization**: sequential stepping within a worker is what makes scene
  sharing safe.

### Basic Usage

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
            observations, rewards, terminated, truncated, infos = env.step(actions)


# Required: the wrapper spawns its workers, so each child re-imports this module.
if __name__ == "__main__":
    main()
```

:::warning The `__main__` guard is not optional
`HybridVectorEnv` forces the `"spawn"` start method, and a spawned child re-imports
the module that created it. Constructing the wrapper at module level therefore makes
every child try to build its own set of workers. Each child dies on Python's standard
guard —

```
RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.
        ...
```

The real message continues for several more lines and wraps as shown, so search for a
fragment rather than the whole sentence — but that error never reaches the parent,
which does not raise and does not exit. The
script hangs, with the children's tracebacks interleaved on stderr. Keep the
construction inside a function called from an
`if __name__ == "__main__":` block.
:::

`HybridVectorEnv` is a context manager, so `with` closes it even if the training
loop raises. A bare `env.close()` also works.

:::warning Seed with a list, never a scalar
A scalar `int` seed is expanded **twice** — once by `HybridVectorEnv`'s outer
`AsyncVectorEnv` and again by each inner `SyncVectorEnv`. Environment `(w, k)` ends
up with `seed + w + k`, so `W × K` environments share only `W + K - 1` distinct
seeds. At 8 workers × 125 environments that is **132 distinct seeds for 1000
environments**, with heavy duplication and no error or warning. The rollouts become
correlated silently.

```python
# Wrong: 1000 environments, 132 distinct seeds.
env.reset(seed=42)

# Right: one seed per environment.
env.reset(seed=list(range(env.num_envs)))
```

`reset()` handles a `Sequence` correctly — it validates the length against
`num_envs` and chunks it per worker, so global environment `i` receives `seed[i]`.
The test is `isinstance(seed, Sequence)` against `typing.Sequence`, so a **`list` or
`tuple` takes the chunking path but a `numpy.ndarray` does not** — an array is handed
straight to the outer `AsyncVectorEnv`, which fails loudly rather than silently. An
array of length `num_envs` raises `AssertionError: If seeds are passed as a list the
length must match num_envs=<num_workers> but got length=<num_envs>` — the message says
`num_envs` but is reporting the *worker* count — and an array of length `num_workers`
raises `TypeError: object of type 'numpy.int64' has no len()` from the inner
`SyncVectorEnv`. Note that the wrapper's own docstring advertises the scalar form;
prefer a list.
:::

### Constructor and constraints

```python
HybridVectorEnv(
    env_fns,                     # Sequence[Callable[[], Env]], one per environment
    num_envs_per_worker=1,
    *args,
    autoreset_mode=AutoresetMode.NEXT_STEP,
    inner_env_cls=None,
    inner_env_kwargs=None,
    **kwargs,
)
```

- **Divisibility is required.** `len(env_fns) % num_envs_per_worker != 0` raises
  `ValueError: The total number of environments must be divisible by the number of
  environments per process.`
- **The multiprocessing context is forced to `"spawn"`.** Anything else is replaced
  and warned about with a `RuntimeWarning`.
- **Spaces must be `Box`.** The wrapper reshapes `low`/`high` arrays directly, so a
  `Dict` or `Discrete` space fails during space flattening. Every SuperDex Gym
  environment satisfies this, because `MochiEnv` flattens its structured spaces to
  a single `Box`.
- **Autoreset is on by default** (`NEXT_STEP`), applied at the inner layer. The
  outer layer has autoreset disabled so the two do not compound.
- **There are no reset masks.** Passing `options={"reset_mask": ...}` raises
  `ValueError: Reset mask is not supported for this vectorization.` Use Gymnasium's
  `AsyncVectorEnv` or `SyncVectorEnv` directly if you need selective resets.

### Shapes and info

Internally the batch is nested `(num_workers, num_envs_per_worker, ...)`; the public
API is flat `(num_envs, ...)`.

| Property | Meaning |
| --- | --- |
| `num_envs` | Total environments (`num_workers × num_envs_per_worker`) |
| `num_workers` | Async subprocesses |
| `num_envs_per_worker` | Environments stepped sequentially per worker |
| `observation_space` / `action_space` | Batched, shape `(num_envs, ...)` |
| `single_observation_space` / `single_action_space` | Per-environment shape |

`step()` validates that the action is a NumPy array whose shape matches
`action_space.shape`, raising `ValueError` otherwise. It returns a
`BatchedStepResult` named tuple of `(observation, reward, terminated, truncated,
info)`.

`info` is a **flat dict of per-environment arrays** indexed by the global
environment index, not a list of per-environment dicts:

```python
observations, rewards, terminated, truncated, infos = env.step(actions)
for i in range(env.num_envs):
    if terminated[i] or truncated[i]:
        key = "terminated_reason" if terminated[i] else "truncated_reason"
        print(i, infos[key][i])
```

### Choosing the async/sync split

The two knobs trade the same total environment count against each other:

- **More workers, fewer environments each** maximizes CPU parallelism but costs one
  subprocess and one full physics context per worker, and defeats scene sharing —
  each worker builds its own copy of the scene.
- **Fewer workers, more environments each** amortizes scene construction and memory
  across a whole worker, but each worker steps its environments sequentially, so a
  worker's wall time grows linearly with `num_envs_per_worker`.

Use [the benchmark script](./benchmarking.md) to tune this split for the target
environment and hardware. It requires explicit `--num_workers` and
`--num_envs_per_worker` values and benchmarks their Cartesian product. For an initial
sweep, keep worker counts at or below the physical CPU core count and try
`--num_envs_per_worker 1,5,25,125`; then refine both axes around the best observed
throughput. (`benchmark.py` imports `psutil` and `tqdm` at module level and neither is
a declared dependency — see
[Installation and Setup](./setup.md) if it fails to start.)

## Scene Sharing for Memory Optimization

When many environments share an identical scene configuration, scene sharing cuts
both memory use and startup time.

### How scene sharing works

`SceneManager` keeps **one instance per environment class name**, and within that
instance a dictionary of scenes keyed by scene name, each with a reference count.
It is not a global singleton: two different environment classes never share a
scene, even if they choose the same scene name.

Each environment keeps its **own** state snapshot. Before simulating, it restores
its snapshot into the shared scene; after simulating, it re-captures. That is what
makes interleaved stepping correct — the shared scene holds only whichever
environment is currently running.

`Scene.capture_state()` covers **dynamic state only**: poses, velocities,
controller and constraint targets, external forces, and the scene clock and step
counter — so restoring a snapshot also rewinds simulation time. It does **not** cover
topology, materials, joint limits or gravity. This is the correctness constraint
behind scene naming: any configuration field that changes how the scene is *built*
must be part of the scene name, or environments with different structure will
wrongly share one scene.

State capture is not supported for scenes containing ROM actors; capturing one raises
`"State capture is not supported for scenes with ROM actors."`

SuperDex Physics's file cache is also enabled for every process, which is the other half of
the startup-time saving.

:::caution Scene sharing assumes sequential stepping
`use_shared_scenes` defaults to `True`, so every environment is opted in. The
mechanism is only correct when environments in the same process are stepped **one
at a time**. `HybridVectorEnv` guarantees this, because each worker's inner
`SyncVectorEnv` steps sequentially. Other vectorization strategies do not, and
stepping shared-scene environments concurrently in one process will corrupt state.
:::

### Enabling scene sharing

```python
config = {
    "use_shared_scenes": True,  # Default.
    "control_frequency": 20,
    "simulation_frequency": 100,
    # ... other config options
}

env_creators = [lambda: YourCustomEnv(config) for _ in range(1000)]
```

### Verifying sharing kicked in

```python
from superdex.lab.gym.envs.scene_manager import SceneManager

# Use the exact class name of the environment - the manager is keyed on
# type(env).__name__.
manager = SceneManager.get_instance(type(env).__name__)
print(manager.scene_count)  # Number of distinct scenes held.
print(manager.scene_info)   # {scene_name: ref_count}
```

If `scene_count` grows with the environment count, the scene names are encoding
something that varies per environment and sharing is not happening.

:::caution `get_instance()` creates the manager it cannot find
`get_instance(name)` is a create-or-get, not a lookup: a misspelled class name
silently registers a brand-new empty manager and reports `scene_count == 0`, which is
indistinguishable from "everything is shared". Check `name in SceneManager._instances`
first, or read `env._scene_manager` off an environment you already hold.
:::

### Scene sharing in custom environments

Route scene creation through `_load_scene()`. It looks the name up in the manager,
reuses an existing scene when it finds one, and registers a new one otherwise.

`_init_scene` is a convention, not a `MochiEnv` hook — nothing in the base class calls
it, so your `__init__` has to.

```python
from typing import Any

import superdex.physics as physics
from superdex.lab.gym.envs import MochiEnv
from superdex.lab.gym.utils import mochi_helpers

# Your own config class, subclassing MochiEnvCfg and adding the fields this
# environment reads (use_gravity and use_damping below).
from my_package import MyCustomEnvCfg


class MyCustomEnv(MochiEnv):
    def __init__(self, cfg: MyCustomEnvCfg | dict[str, Any]):
        if not isinstance(cfg, MyCustomEnvCfg):
            cfg = MyCustomEnvCfg(**cfg)
        super().__init__(cfg)

        # Nothing in MochiEnv calls _init_scene - do it here.
        self._init_scene(cfg)

    def _init_scene(self, cfg: MyCustomEnvCfg):
        def scene_builder():
            scene = physics.create_scene("MyScene")

            agent_params = physics.ArticulatedActorParams()
            # Configure agent_params here...
            agent = scene.create_articulated_actor(agent_params)

            return scene, agent

        # The name must encode every config field that changes scene construction.
        # A constant name would share one scene across incompatible configurations.
        uid_fields = (cfg.use_gravity, cfg.use_damping)
        self._load_scene(f"scene_{hash(uid_fields)}", scene_builder)

        self._initial_pose = mochi_helpers.get_articulated_pose(self._agent)
        self._initial_velocity = mochi_helpers.get_articulated_joint_velocities(
            self._agent
        )
```

All three shipped environments follow this idiom. `CartPoleEnv`, for example, uses
`uid_fields = (cfg.use_damping, cfg.use_gravity, cfg.free_pole)` — the three fields
that alter its prefab — and deliberately excludes fields such as `actuate_on_pole`
that only affect how actions are applied.

If scene sharing is enabled but the scene was not created through `_load_scene()`,
the environment emits a `RuntimeWarning` on reset.
