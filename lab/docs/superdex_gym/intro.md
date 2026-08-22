---
title: Introduction
---

# SuperDex Gym

SuperDex Gym is a reinforcement learning framework for training AI agent
controllers, built on top of the SuperDex Physics engine. It provides a
Gymnasium-compatible interface that enables researchers and developers to train
RL agents in high-fidelity physics simulations.

SuperDex Lab has three components — the environments, the RLlib training
integration and the benchmarking tools — and they reach you in two different
ways:

1. **SuperDex Physics environments**, in the installable package: A suite of
   Python classes that implement the Gymnasium paradigm for common robot agents,
   and manage the low-level communication to the SuperDex Physics C++ simulation
   backend.
2. **Training and benchmarking scripts**, in the source checkout only: the
   Ray/RLlib integration at `apps/rllib/`, which allows fast adoption of RL
   algorithms with efficient distributed training, and the benchmark entry point at
   `apps/envs/benchmark.py`. None of these are part of the installable
   package, and they need packages a core sync does not provide - `tqdm`, plus
   the training stack for RLlib work - so install those separately; see
   [dependencies for `apps/`](./setup.md#dependencies-for-apps).

SuperDex Gym provides several example benchmark environments (classic-control and
locomotion tasks), all simulated
with high-fidelity physics through the SuperDex Physics engine.

## Key Features

- **High-fidelity Physics**: Powered by the SuperDex Physics engine for accurate
  simulations
- **Gymnasium Compatible**: Standard RL interface for easy integration with
  existing workflows
- **Ray/RLlib Integration**: Distributed training with PPO and SAC
- **Environment Suite**: Classic-control and locomotion
  benchmark environments
- **Flexible Configuration**: Extensive configuration options for environments
  and training
- **Visualization**: Built-in Polyscope rendering and video generation
- **Cross-platform Support**: Runs on Linux, macOS and Windows

## Available Environments

Environments and their config variants are discovered by scanning the files that sit
beside the environment modules, so the list below reflects a source checkout — the
install route [setup](./setup.md) documents. Run
`uv run python apps/envs/run_sample.py --help` from the `superdex_lab` project root to
print the exact
set for your install — Python puts the script's own directory on `sys.path`, so
its sibling imports resolve wherever you invoke it from.

| Environment | CLI name | Gymnasium ID | Description |
| --- | --- | --- | --- |
| **Ant** | `ant` | `superdex_gym/Ant-v0` | Quadrupedal locomotion task benchmark |
| **CartPole** | `cart_pole` | `superdex_gym/CartPole-v0` | Classic pole balancing task benchmark |
| **HalfCheetah** | `half_cheetah` | `superdex_gym/HalfCheetah-v0` | Planar running task benchmark |

Each of these also ships one or more config *variants* — alternative configurations of
the same environment class, each registered under its own Gymnasium id:

| Variant CLI name | Gymnasium ID |
| --- | --- |
| `ant_full_observation` | `superdex_gym/AntFullObservation-v0` |
| `ant_no_contact` | `superdex_gym/AntNoContact-v0` |
| `ant_rotation_vector` | `superdex_gym/AntRotationVector-v0` |
| `cart_pole_actuate_on_pole` | `superdex_gym/CartPoleActuateOnPole-v0` |
| `half_cheetah_full_observation` | `superdex_gym/HalfCheetahFullObservation-v0` |

See [Environment file naming](./rllib.md#environment-file-naming) for how variants are
declared. Variants whose name contains a `test` segment are smoke-tested but never
registered, so they never appear in a CLI listing.

:::caution Two parallel naming schemes
Command-line tools take the snake_case **CLI name** (`cart_pole`), while
`gym.make()` takes the PascalCase **Gymnasium ID** (`superdex_gym/CartPole-v0`).
The source filenames follow a third convention — `cartpole_env.py`,
`halfcheetah_env.py` — so do not derive one from another.
:::

:::note Environment prerequisites
All three environments run from a public checkout: `cart_pole`, `ant` and
`half_cheetah` load their scenes and meshes from `assets/benchmarks/`, all of which
ship.

`superdex-robotics` is required by *all* environments, CartPole and
Ant included: `superdex/lab/gym/utils/mochi_helpers.py` imports
`superdex.robotics` at module level and every `*_env` module imports those
helpers. It is a declared dependency of `superdex-lab`, so a supported install
always has it — but if its native extension was not built, discovery fails
before any `--env` selection takes effect, and choosing a different environment
is not a workaround. See
[Verifying the Install](./setup.md#verifying-the-install).
:::
