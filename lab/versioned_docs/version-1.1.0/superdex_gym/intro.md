---
title: Introduction
---

# SuperDex Gym

SuperDex Gym is a reinforcement-learning (RL) framework built on the SuperDex Physics
engine. It provides an environment Python API that presents a simulated task through
Gymnasium, a standard Python interface for reinforcement learning. You can use these
environments to train AI agent controllers in high-fidelity physics simulations.

## What to Install

SuperDex Lab delivers installable environments and source-only app scripts in
different ways. An app script is an executable Python program for training or
benchmarking.

1. **To use an environment, install the `superdex-lab` package.** The package
   provides Python environment classes for common robot agents. These classes
   implement the Gymnasium interface and communicate with the SuperDex Physics C++
   simulation backend.
2. **To train with Ray/RLlib or run benchmarks, use the source checkout.** A
   **source checkout** is a local copy of the Project SuperDex repository;
   `superdex_lab` is a subproject within it. The app scripts are not part of the
   installable package. The Ray/RLlib integration in `apps/rllib/` provides quick
   access to reinforcement learning algorithms and efficient distributed training.
   The benchmark entry point is `apps/envs/benchmark.py`.

A core source sync does not provide all dependencies for the source-only app scripts.
Install `tqdm` for the app scripts, plus the training stack for RLlib work. See
[dependencies for `apps/`](./setup.md#dependencies-for-apps).

SuperDex Gym includes classic-control and locomotion benchmark environments. All of
them use the SuperDex Physics engine for high-fidelity simulation.

## Key Features

- **High-fidelity Physics**: Powered by the SuperDex Physics engine for accurate
  simulations
- **Gymnasium Compatible**: Standard reinforcement-learning interface for easy
  integration with existing workflows
- **Ray/RLlib Integration**: Distributed training with Proximal Policy Optimization
  (PPO) and Soft Actor-Critic (SAC)
- **Environment Suite**: Classic-control and locomotion benchmark environments
- **Flexible Configuration**: Extensive configuration options for environments and
  training
- **Visualization**: Built-in Filament-based rendering and video generation
- **Cross-platform Support**: Runs on Linux, macOS and Windows

## Available Environments

Each environment is identified by its canonical **Gymnasium ID**, such as
`superdex_gym/CartPole-v0`. Pass the same ID to `gym.make()` and to the command-line
tools; there are no short names.

| Environment | Gymnasium ID | Description |
| --- | --- | --- |
| **Ant** | `superdex_gym/Ant-v0` | Quadrupedal locomotion task benchmark |
| **CartPole** | `superdex_gym/CartPole-v0` | Classic pole balancing task benchmark |
| **HalfCheetah** | `superdex_gym/HalfCheetah-v0` | Planar running task benchmark |

Base environments are `MochiEnv` subclasses in `*_env.py` modules. Each one is
registered explicitly as an `EnvSpec` in `superdex.lab.gym.registration`. This catalog
reflects a source checkout; follow [setup](./setup.md) for the install route.

To print the exact environments available in your install, run this command from the
`superdex_lab` project root:

```bash
uv run python apps/envs/run_sample.py --help
```

Python adds the script's own directory to `sys.path`, so the script's sibling imports
resolve regardless of the directory from which you invoke it.

A **config variant** is an alternative configuration of an existing environment
class: another registered `EnvSpec` that reuses the same class with a different `cfg`.
Each variant has its own Gymnasium ID.

| Variant Gymnasium ID |
| --- |
| `superdex_gym/AntFullObservation-v0` |
| `superdex_gym/AntNoContact-v0` |
| `superdex_gym/AntRotationVector-v0` |
| `superdex_gym/CartPoleActuateOnPole-v0` |
| `superdex_gym/HalfCheetahFullObservation-v0` |

See [Config variants are additional specs](./environments.mdx#config-variants-are-additional-specs)
for how variants are declared.


:::note Environment prerequisites
The included CartPole, Ant and HalfCheetah environments load their scenes and
meshes from `assets/benchmarks/` from the Project Superdex source code on Github.

All environments require `superdex-robotics`, which should be installed as a dependency of `superdex-lab`.  See [Verifying the Install](./setup.md#verifying-the-install).
:::
