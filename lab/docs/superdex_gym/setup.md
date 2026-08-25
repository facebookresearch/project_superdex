---
title: Installation and Setup
---

# Installation and Setup

## Install SuperDex Gym

SuperDex Gym (`superdex-lab`) is installed as part of Project SuperDex. Its package
dependencies include SuperDex Physics (`superdex-physics`) for simulation and SuperDex
Robotics (`superdex-robotics`), which every environment imports at module load.
Ray/RLlib is an optional training-stack dependency for `apps/rllib` and is installed
separately.

Before installing, follow the
[Project SuperDex build instructions](https://github.com/facebookresearch/project_superdex#building-from-source)
to set up the required toolchain. Use the `project_superdex` root as your working
directory.

Run the core install:

```bash
uv sync --extra core
```

This command builds physics, robotics and Gym together. The README's extras table
covers the other combinations.

:::note FP64 native simulation
Build the `double` extra and set `SUPERDEX_PRECISION=double` before importing
SuperDex to run the native Physics and Robotics simulation in FP64. SuperDex Lab
has no separate FP64 package: its shipped Gym environments retain `float32`
observation and action spaces. This is FP64 native simulation under the existing
float32 Gym interface, not an end-to-end float64 Lab API.
:::

## Verifying the Install

Complete the core install first. Use the `project_superdex` root as your working
directory, then run:

```bash
uv run python -c "import superdex.physics; print(superdex.physics.__file__)"
uv run python -c "import superdex.robotics; print('ok')"
uv run python -c "import superdex.lab.gym; print('ok')"
uv run python -c "from superdex.physics.paths import get_assets_root; print(get_assets_root())"
```

The last command prints the `assets/` directory of your checkout.

## Run an Environment

Complete the install verification first. From the `project_superdex` root, run this
headless smoke test:

On macOS, Linux, or PowerShell:

```bash
uv run python -c "
from superdex.lab.gym.envs.benchmarks.cartpole_env import CartPoleEnv, CartPoleEnvCfg

env = CartPoleEnv(CartPoleEnvCfg(render_mode=None))
try:
    env.reset()
    env.step(env.action_space.sample())
finally:
    env.close()
print('Smoke test passed')
"
```

With Windows Command Prompt:

```bat
uv run python -c "from superdex.lab.gym.envs.benchmarks.cartpole_env import CartPoleEnv, CartPoleEnvCfg; env = CartPoleEnv(CartPoleEnvCfg(render_mode=None)); exec('try:\n    env.reset()\n    env.step(env.action_space.sample())\nfinally:\n    env.close()'); print('Smoke test passed')"
```

See [Run Example Environments](./running_examples.md) for the interactive and
video-capable scripts.

## Understand Asset Discovery

Environments resolve their scenes and meshes through
`superdex.physics.paths.get_assets_root()` in this order:

1. **Precedence:** It uses `SUPERDEX_ASSETS_PATH` first when that variable points at a
   readable directory.
2. **Fallback:** Otherwise, it walks upward from the installed `superdex.physics`
   module, not from the current working directory, looking for an `assets/` directory
   in the same source workspace.
3. **Editable install:** An editable source-workspace install resolves the assets
   automatically through that fallback.
4. **Wheel install:** A wheel installed outside the checkout does not resolve a
   separate clone automatically.
5. **Corrective action:** When using a wheel with a separate clone, set
   `SUPERDEX_ASSETS_PATH=<path-to-project_superdex>/assets`.

## Dependencies for apps/

Complete `uv sync --extra core` first. It installs `superdex-lab` and the rest of the
workspace lock, including `polyscope`, `imageio`, `imageio-ffmpeg`, `pillow` and
`psutil`.

The scripts under `apps/` have additional requirements. The `apps/envs` benchmarks
require `tqdm>=4.67.1`. The `apps/rllib` training and video scripts require
`torch==2.7.1`, `ray[rllib]==2.49.0`, `moviepy`, `pillow>=10.1` and `tensorboard`.
Use the `project_superdex` root as your working directory, then run:

```bash
# apps/envs benchmarks
uv pip install "tqdm>=4.67.1"

# apps/rllib training and video
uv pip install torch==2.7.1 --extra-index-url https://download.pytorch.org/whl/cpu
uv pip install "ray[rllib]==2.49.0" "moviepy" "pillow>=10.1" "tensorboard"
```

These dependencies are outside the lock. If you rerun `uv sync --extra core`,
reinstall them afterward. Run RLlib training with `uv run --no-project` to avoid Ray's
working-directory check; other app scripts use plain `uv run`.

The scripts under `apps/rllib/` import their siblings by bare name. Start in the
`project_superdex` root; the first command changes the working directory to
`superdex_lab/apps/rllib`.

```bash
cd superdex_lab/apps/rllib
uv run --no-project python train_samples.py --help
```

## Running the Tests

Complete the install first. The tests require `pytest`. Use the `superdex_lab` project
root, the directory holding `pyproject.toml` and `test/`, as your working directory.

Install the test requirement:

```bash
uv pip install pytest
```

Then run the test suite from the same directory:

```bash
uv run python -m pytest test
```

Expect **51 passed, 3 skipped**. The three skips are the `test_batched_stepping_*`
cases, which need an `agent_pose` observation that none of the shipped environments
has.

## Next Steps

- [Running Example Environments](./running_examples.md)
- [Benchmarking](./benchmarking.md)
- [Authoring a Custom Environment](./environments.mdx)
- [Training with Ray/RLlib](./rllib.md)
