---
title: Running Example Environments
---

# Running Example Environments

SuperDex Gym provides example scripts for running its environments, along with sample environments to try.

## Run Scripts

These scripts show how to run an environment and are located under `superdex_lab/apps/envs/`:

- `run_sample.py`: simulates a single sample environment.
- `run_with_gymnasium_vectorization.py`: runs multiple environment instances in parallel with vectorization.

After [setup](./setup.md), each script can be run with:

```bash
uv run python [path-to-script] [args]
```
Or, from the subdirectory:

```bash
cd superdex_lab/apps/envs
uv run python [script] [args]
```

The rest of this document assumes your working directory is `superdex_lab/apps/envs`.

## 1. `run_sample.py`

**Run one sample environment with a selected action-sampling strategy.**

Runs a sample SuperDex Gym environment through the command line, configurable with different action sampling strategies.

Polyscope is required for both interactive viewing and offscreen recording. When it is available, the script opens an interactive window by default or records video offscreen when `--video` or `--video_path` is set. Otherwise, the script runs without rendering or recording.

The essential logic in the script for running an environment is:

```python
episode = 0
episode_return = 0.0
env.reset()
while episode < num_episodes:
    action = action_sampler(env)
    _, reward, terminated, truncated, info = env.step(action)
    episode_return += reward
    if terminated or truncated:

        # (Handle episode results here)

        env.reset()
        episode += 1
        episode_return = 0.0
```

### Usage Examples

Here are some usage examples before we walk through the available options:

```bash
# Run CartPole with random actions for 5 episodes
uv run python run_sample.py cart_pole --action_sampler random --num_episodes 5

# Run Ant with sweep actions and record video
uv run python run_sample.py ant --action_sampler sweep --video

# Record at a custom resolution
uv run python run_sample.py ant --action_sampler random --video --video_size 1920x1080

# Run a config variant
uv run python run_sample.py half_cheetah_full_observation --action_sampler sweep

# Start the environment in a paused state
uv run python run_sample.py half_cheetah --start_paused
```

### Available Sample Environments

`uv run python run_sample.py --help` lists the environments discovered in your current install. The repository currently provides these options:

- **Benchmarks**: `ant`, `ant_full_observation`, `ant_no_contact`,
  `ant_rotation_vector`, `cart_pole`, `cart_pole_actuate_on_pole`, `half_cheetah`,
  `half_cheetah_full_observation`

These are environment CLI names that can be passed to the script.

Names with a suffix (e.g., `_no_contact`) are alternative configurations of the same
environment. `--help` lists base environments and registered config variants present in the current build; it omits test-only variants and `train` or `benchmark` recipe files (see [Environment file naming](./rllib.md#environment-file-naming)).

### Command-Line Options

The environment name is a required argument. The remaining options are:

| Option | Default | Notes |
| --- | --- | --- |
| `--action_sampler {zero,random,sweep}` | `sweep` | Action sampling strategy: `zero`, `random`, or `sweep`. |
| `--num_episodes NUM` | `10` | Number of episodes to simulate. |
| `--video` | off | Enables video recording and switches rendering offscreen. |
| `--video_size WIDTHxHEIGHT` | none | Sets the recorded frame resolution (e.g., `1280x720`) when `--video` or `--video_path` is used; otherwise, sets the interactive window size. |
| `--video_path DIR` | none (`apps/envs/output/` when `--video` is passed) | Enables video recording and saves the videos to an output directory. The default directory is resolved relative to the script, not to your working directory. |
| `--start_paused` | off | Starts the interactive simulation paused. |
| `--profile` | off | Toggles `cProfile` around the single run. The environment's own profiler is enabled unconditionally and always prints a summary at the end. |


### Action Samplers

These are the options for `--action_sampler`:

- `zero`: Zero action vector
- `random`: Random actions from the action space
- `sweep`: Sinusoidal sweeping actions that cycle through dimensions (default)


## 2. `run_with_gymnasium_vectorization.py`

**Run multiple environments in parallel through a vectorized API.**

Run the script with:

```bash
uv run python run_with_gymnasium_vectorization.py
```

This script shows how to run multiple SuperDex Gym environments in parallel using
the `HybridVectorEnv` wrapper, which combines asynchronous and synchronous
vectorization. See [Training with Large Batches](./batching.md) for the full
contract.

Like `run_sample.py`, it samples actions and steps the environment in a loop:

```python
for step in range(num_steps):
    action = env.action_space.sample()
    _, rewards, terminateds, truncateds, infos = env.step(action)
    returns += rewards
```

However, `env` is now a `HybridVectorEnv`. Observations, rewards, and termination
flags are batched arrays, while `infos` is a dictionary of batched values rather than
a list of per-environment dictionaries.

Notice also that the script passes a list of seeds to `env.reset()` to seed each environment explicitly; a scalar seed can lead to duplicates across the vectorization layers.


### Key Features

- Runs multiple environment instances in parallel
- Demonstrates hybrid vectorization (async + sync)
- Automatic environment reset handling
- Shows vectorized API interaction patterns

For timing and throughput numbers, use [`benchmark.py`](./benchmarking.md) instead; this
script has no timing, FPS or profiler code.

### Configuration

This is the default configuration:

- `CartPoleEnv`, constructed with `cfg={"render_mode": None}`
- 9 environments
- 3 environments per worker (3 async workers total)
- 200 control steps

Modify the `__main__` block to change:

- Environment type (`cls`)
- Environment configuration (`cfg`)
- Number of environments (`num_environments`)
- Environments per worker (`num_environments_per_worker`)
- Control steps (`num_steps`, the number of `env.step()` calls per environment).
  Each control step can advance multiple physics simulation steps.

## General Usage Notes

### Video Recording

- Videos are written to `apps/envs/output/` by default.
- The frame rate is set to the environment's control frequency.
- One MP4 video is written per completed episode, numbered from `video_001.mp4`.

:::caution Video encoding can fail silently
The core setup installs `imageio[ffmpeg]`. If the FFmpeg plugin is missing, `run_sample.py` may still exit successfully with no viewable MP4 video. Run `uv pip install "imageio[ffmpeg]"` to repair only the plugin.
:::


### Environment Configuration

The CLI controls action sampling, episode count, rendering and recording options,
paused startup, and `cProfile`. Other environment settings, such as worker threads,
simulation and control frequencies, and episode length, come from the environment's
config class or variant JSON. To change them, pick a variant, edit the call site, or
construct the environment yourself:

```python
from superdex.lab.gym.envs.benchmarks.cartpole_env import CartPoleEnv, CartPoleEnvCfg

env = CartPoleEnv(CartPoleEnvCfg(control_frequency=25, simulation_frequency=50))
```

See [Creating Custom Environments](./environments.mdx) for the full configuration
surface.

## Related Tools and Modules

Beyond the scripts above for running an environment, these tools and support modules are available under `superdex_lab/apps/`:

| Tool | Purpose |
| --- | --- |
| `envs/benchmark.py` | Measures initialization time, memory use, and simulation throughput. See [benchmarking](./benchmarking.md). |
| `envs/utils/dataset.py` | LeRobot dataset helpers. Requires `lerobot` and `torch`, which need to be installed separately. |
| `rllib/train_samples.py` | PPO/SAC training. See [Training with Ray/RLlib](./rllib.md). |
| `rllib/run_inference.py` | Roll out a trained checkpoint |
| `rllib/visualize_training_history.py` | Stitch checkpoint videos into a labeled montage. See [Visualizing Training History](./visualize_training_history.md). |
| `rllib/utils.py` | Environment registration helpers shared by the RLlib scripts |
