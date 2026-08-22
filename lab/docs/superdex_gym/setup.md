---
title: Installation and Setup
---

# Installation and Setup

SuperDex Gym (`superdex-lab`) is a Python package that depends on the SuperDex Physics
engine (`superdex-physics`) for simulation, on SuperDex Robotics
(`superdex-robotics`), which every environment imports at module load, and on Ray/RLlib
for training.

## Installing

SuperDex Gym is installed as part of Project SuperDex. Follow the
[Project SuperDex build instructions](https://github.com/facebookresearch/project_superdex#building-from-source)
for the toolchain and the install itself: `uv sync --extra core` builds physics,
robotics and Gym together, and the README's extras table covers the other
combinations.

:::note FP64 native simulation
Build the `double` extra and set `SUPERDEX_PRECISION=double` before importing
SuperDex to run the native Physics and Robotics simulation in FP64. SuperDex Lab
has no separate FP64 package: its shipped Gym environments retain `float32`
observation and action spaces. This is FP64 native simulation under the existing
float32 Gym interface, not an end-to-end float64 Lab API.
:::

## Verifying the Install

From the `project_superdex` root:

```bash
uv run python -c "import superdex.physics; print(superdex.physics.__file__)"
uv run python -c "import superdex.robotics; print('ok')"
uv run python -c "import superdex.lab.gym; print('ok')"
uv run python -c "from superdex.physics.paths import get_assets_root; print(get_assets_root())"
```

The last command prints the `assets/` directory of your checkout.

Then run an environment:

```bash
cd superdex_lab/apps/envs
uv run python run_sample.py cart_pole --action_sampler random --num_episodes 1
```

It prints a profiling table when it finishes. On a headless host, Polyscope can use
its EGL backend when a compatible EGL device and driver are available; without one,
renderer initialization can fail. See [Run Example Environments](./running_examples.md) for
the rest of what `run_sample.py` can do.

## Asset Discovery and Environment Variables

Environments resolve their scenes and meshes through
`superdex.physics.paths.get_assets_root()`. It first uses `SUPERDEX_ASSETS_PATH` when
that points at a readable directory. Otherwise, it walks upward from the installed
`superdex.physics` module, not from the current working directory, looking for an
`assets/` directory in the same source workspace. An editable source-workspace install
therefore resolves automatically. A wheel installed outside the checkout does not:
when using it with a separate clone, set
`SUPERDEX_ASSETS_PATH=<path-to-project_superdex>/assets`.

## Dependencies for apps/

`uv sync --extra core` installs `superdex-lab` and the rest of the workspace lock,
which includes `polyscope`, `imageio`, `imageio-ffmpeg`, `pillow` and `psutil`. The
scripts under `apps/` need more than that:

```bash
# apps/envs benchmarks
uv pip install "tqdm>=4.67.1"

# apps/rllib training and video
uv pip install torch==2.7.1 --extra-index-url https://download.pytorch.org/whl/cpu
uv pip install "ray[rllib]==2.49.0" "moviepy" "pillow>=10.1" "tensorboard"
```

These land outside the lock, so `uv sync` removes them again. Once the training stack
is installed, use `uv run --no-sync` for every command - a plain `uv run` puts the
locked `gymnasium` and `pillow` back and breaks `moviepy`.

The scripts under `apps/rllib/` import their siblings by bare name, so run them from
their own directory, again starting from the `project_superdex` root:

```bash
cd superdex_lab/apps/rllib
uv run --no-sync python train_samples.py --help
```

## Running the Tests

```bash
uv pip install pytest
```

From the `superdex_lab` project root - the directory holding `pyproject.toml` and
`test/`:

```bash
uv run --no-sync python -m pytest test
```

Expect **51 passed, 3 skipped**. The three skips are the `test_batched_stepping_*`
cases, which need an `agent_pose` observation that none of the shipped environments
has.

## Next Steps

- [Running Example Environments](./running_examples.md)
- [Benchmarking](./benchmarking.md)
- [Creating Custom Environments](./environments.mdx)
- [Training with Ray/RLlib](./rllib.md)
