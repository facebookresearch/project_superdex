---
title: Training with Ray/RLlib
---

# Training with Ray/RLlib

SuperDex Gym provides a Ray/RLlib integration for RL training on SuperDex Physics
using Proximal Policy Optimization (PPO) or Soft Actor-Critic (SAC), and for running
inference with trained checkpoints.

:::caution
**Algorithm support:** PPO is the recommended workflow. SAC is available
experimentally, but its bundled settings and stopping thresholds have not been tuned
or validated for the included environments. When SAC is selected, the training script
ignores any per-environment settings under the recipe's `ppo` key and uses the same
shared SAC algorithm configuration for every environment.
:::

## Quickstart

### Dependency requirements

After `uv sync --extra core`, install the supported training dependencies from the
`project_superdex` root:

```bash
uv pip install torch==2.7.1 --extra-index-url https://download.pytorch.org/whl/cpu
uv pip install "ray[rllib]==2.49.0" "moviepy" "pillow>=10.1" "tensorboard"
```

Run training commands below with `uv run --no-project`. Inference and visualization
use plain `uv run`. See
[Dependencies for apps](./setup.md#dependencies-for-apps).

### Working directory

All paths in this guide are relative to the `superdex_lab` project root. Change to the
RLlib application directory before running the training and inference commands:

```bash
cd superdex_lab/apps/rllib
```

`train_samples.py` and `run_inference.py` use bare sibling imports such as
`from callbacks import ...` and `from utils import ...`. Running either script by path
from the project root also works because Python adds the script's directory to
`sys.path`. Package imports and `python -m` execution are unsupported because the
scripts provide no import fallback. `visualize_training_history.py` has no sibling
imports and runs from any directory.

### Minimum training run

Run one CartPole PPO iteration with one environment runner:

```bash
uv run --no-project python train_samples.py \
  --pattern "cart_pole" --num_env_runners 1 --max_iterations 1
```

A successful run creates a Tune trial named `cart_pole_<trial_id>` under
`~/ray_results/PPO/`, writes `params.json` and training logs in that trial directory,
and writes a final `checkpoint_*` directory. The console reports training progress
and exits after the first iteration. The final checkpoint is always written even when
the configured checkpoint frequency has not elapsed.

### Minimum inference run

Pass the absolute path to the generated `checkpoint_*` directory, not its parent trial
directory:

```bash
uv run python run_inference.py /path/to/checkpoint
```

The checkpoint must be a valid RLlib checkpoint with policy weights under
`learner_group/learner/rl_module/<DEFAULT_MODULE_ID>/`. Its parent trial directory
must contain `params.json`; pointing the command at the trial directory or another
level fails. The script uses Ray's `DEFAULT_MODULE_ID` constant instead of a literal
module name.

A successful default inference run opens an interactive Polyscope window, runs 10
episodes, prints each episode's return and completion reason, and reports the completed
episode count. It does not initialize Ray. Add `--video` to record MP4 files instead
of using the default `"human"` render mode.

## Script locations and runtime behavior

| Script | Location | Purpose |
| --- | --- | --- |
| `train_samples.py` | `apps/rllib/train_samples.py` | Train one or more environments |
| `run_inference.py` | `apps/rllib/run_inference.py` | Roll out a trained checkpoint |
| `visualize_training_history.py` | `apps/rllib/visualize_training_history.py` | Stitch checkpoint videos into a montage |

`train_samples.py` calls a bare `ray.init()`. It has no `--ray-address` option. To
attach it to an existing cluster, set the `RAY_ADDRESS` environment variable; Ray
reads that variable directly. `run_inference.py` does not initialize Ray.

### Training a custom environment

Environment discovery determines what `train_samples.py` trains. A separate custom
training script must register its environments with Ray Tune first:

```python
from utils import register_envs

register_envs()
```

Register a custom environment with Gymnasium before calling `register_envs()`.
`register_envs()` first calls `register_all_envs()` to populate the Gymnasium registry,
then registers every Gymnasium ID currently in that registry with Tune.

The wrapper handles the two environment families differently because RLlib does not
pass `env_config` to a Gymnasium environment constructor correctly by itself. For
`superdex_gym/*` IDs, including every environment selected by `train_samples.py`, the
wrapper merges the RLlib `env_config` over the `cfg` defaults on the Gymnasium spec.
It passes the merged values as one `cfg=` mapping, preserving variant defaults unless
a matching key overrides them. For every other Gymnasium ID, the wrapper expands the
config dictionary into constructor keyword arguments.

## Scripts Overview

### 1. `train_samples.py`

**Train multiple sample environments simultaneously.**

This script trains one Ray Tune experiment per selected environment, so several
tasks can be trained side by side with a shared configuration.

**Which environments are trainable.** The script does not hardcode a list. It walks
the discovered environments, skips test-only variants, and keeps those that ship a
`<env_module>[_<variant>].train.json` recipe next to their module. **Config variants
are trainable too**, each from its own recipe - a variant does *not* inherit the base
environment's recipe, and a base environment is not trainable unless it has a recipe
of its own. The included trainable set is:

| CLI name | Gymnasium ID | Recipe |
| --- | --- | --- |
| `ant_no_contact` | `superdex_gym/AntNoContact-v0` | `benchmarks/ant_env_no_contact.train.json` |
| `cart_pole` | `superdex_gym/CartPole-v0` | `benchmarks/cartpole_env.train.json` |
| `half_cheetah` | `superdex_gym/HalfCheetah-v0` | `benchmarks/halfcheetah_env.train.json` |

The registered base `ant` environment is not trainable because no recipe is named
after its module. The `cart_pole` and `half_cheetah` entries are trainable base
environments with base-module recipes; only `ant_no_contact` is a variant.

All three recipes use assets included in the checkout.

**Features:**

- Simultaneous training of multiple environments
- Algorithm selection: PPO (recommended) or SAC (experimental)
- Pattern-based environment selection
- Per-environment PPO hyperparameters from the recipe files
- Centralized experiment management

**Usage Examples:**

```bash
# Train all trainable environments with PPO (default).
uv run --no-project python train_samples.py

# Train specific environments using patterns.
uv run --no-project python train_samples.py --pattern "cart_pole"
uv run --no-project python train_samples.py --pattern "*cheetah*"

# Train with a custom configuration.
uv run --no-project python train_samples.py --num_env_runners 64 --checkpoint_freq 5

# Experimental SAC run on CartPole, limited to one training iteration.
# This checks the training path, not learning or convergence.
uv run --no-project python train_samples.py --algorithm SAC --pattern "cart_pole" --max_iterations 1

# Train selected environments with video recording enabled (off by default)
uv run --no-project python train_samples.py --pattern "ant*" --video_on_checkpoint --output_path ./benchmark_results

# High-throughput training for benchmarking
uv run --no-project python train_samples.py --num_env_runners 128 --checkpoint_freq 20
```

:::caution `--pattern` matches CLI short names
Patterns use `fnmatch` against the snake_case short names. Use `"cart_pole"` /
`"half_cheetah"`, or a glob such as `"cart*"` / `"ant*"`. The names `"cartpole"` and
`"halfcheetah"` match nothing. The name `"ant"` also matches nothing because the
trainable ant entry is `ant_no_contact` and `fnmatch` is not a prefix match. With no
matches, the script prints `No samples to train, exitting...` and exits with a
non-zero status.
:::

**Command-line Options:**

| Option | Default | Notes |
| --- | --- | --- |
| `--algorithm`, `-a` | `PPO` | `PPO` (recommended) or `SAC` (experimental). SAC uses shared defaults and ignores per-environment `ppo` overrides. |
| `--num_env_runners`, `-n` | `32` | Parallel experience collectors. Capped to `max(1, num_cpus - num_learners)`, with a printed `WARNING:` line, when `num_env_runners + num_learners >= num_cpus`. |
| `--checkpoint_freq`, `-cf` | `10` | Checkpoint every N training iterations. A final checkpoint is always written at the end of training. |
| `--pattern`, `-p` | `*` | Selects which environments to train |
| `--num_learners`, `-nl` | `1` | Parallel policy-update processes. Unlike `--num_env_runners`, a value at or above the CPU count **raises** rather than being capped. |
| `--output_path`, `-o` | `~/ray_results/` | Root for Tune results |
| `--video_on_checkpoint`, `-vid` | off | Encoding adds per-checkpoint overhead, so pass `--video_on_checkpoint` to enable. It is a `BooleanOptionalAction`, so `--no-video_on_checkpoint` is also accepted. When the renderer is unavailable it downgrades to disabled, emitting a warning and continuing rather than failing. |
| `--profile` | off | Enables the environment profiler and `dump_timings_to_info`. Applies to **every** environment — profiling lives in the `MochiEnv` base class. |
| `--max_iterations`, `-mi` | `0` | `0` means no iteration limit; training stops on the recipe's reward / env-steps criteria |

### 2. `run_inference.py`

**Run inference using trained RLlib policy checkpoints.**

**Usage Examples:**

```bash
# Basic inference with a trained checkpoint.
uv run python run_inference.py /path/to/checkpoint

# Use the exploration forward pass.
uv run python run_inference.py /path/to/checkpoint --explore_during_inference

# Record videos of inference episodes.
uv run python run_inference.py /path/to/checkpoint --video

# Custom number of episodes and video path.
uv run python run_inference.py /path/to/checkpoint --num_episodes 20 --video_path ./inference_videos
```

**Command-line Options:**

| Option | Default | Notes |
| --- | --- | --- |
| `checkpoint_path` | — | **Required.** Path to the checkpoint directory. |
| `--num_episodes` | `10` | |
| `--explore_during_inference` | off | Samples stochastically from the exploration distribution instead of taking the greedy action. |
| `--video` | off | Renders offscreen and records MP4 files |
| `--video_path` | the checkpoint directory | Implies `--video` |

Without `--video`, inference **opens an interactive Polyscope window** — the render
mode defaults to `"human"`, not `None`.

:::note How actions are selected
The action distribution class is taken from the restored module via
`get_inference_action_dist_cls()` / `get_exploration_action_dist_cls()`, so PPO
gets a diagonal Gaussian, SAC a squashed Gaussian, and discrete action spaces a
categorical. Without `--explore_during_inference` the distribution is collapsed
with `to_deterministic()` before sampling, yielding the greedy action; with the
flag, the action is drawn stochastically.
:::

Videos are written as `inference_000.mp4`, `inference_001.mp4`, … at a **hardcoded
30 fps** — unlike `run_sample.py`, which uses the environment's control frequency.

**Checkpoint requirements:**

- A valid RLlib checkpoint directory.
- `params.json` in the checkpoint's **parent** (trial) directory — not inside the
  checkpoint directory itself. Pointing at the wrong level is the most common
  failure here.
- Policy weights under `learner_group/learner/rl_module/<DEFAULT_MODULE_ID>/`. The
  script uses Ray's `DEFAULT_MODULE_ID` constant rather than a literal name.

## Training recipes: `*.train.json`

Per-environment training settings are **data files**, not code. Any environment that
ships `<env_module>[_<variant>].train.json` next to its module is trainable; one that
does not is skipped. This is how you make your own environment trainable by
`train_samples.py`.

### Schema

```json
{
  "description": "Training recipe for the HalfCheetah benchmark env. PPO is the recommended workflow. SAC uses the experimental shared baseline.",
  "normalize_observations": true,
  "stop_criteria": {
    "episode_return_mean": 9800,
    "num_env_steps_sampled_lifetime": 50000000
  },
  "ppo": {
    "env_runners": {
      "batch_mode": "truncate_episodes"
    },
    "training": {
      "clip_param": 0.2,
      "gamma": 0.99,
      "grad_clip": 0.5,
      "kl_coeff": 1.0,
      "lambda_": 0.95,
      "lr": 0.0003,
      "minibatch_size": 4096,
      "num_epochs": 32,
      "train_batch_size": 65536,
      "vf_loss_coeff": 0.5
    }
  }
}
```

A recipe holds training settings only. The top-level observation normalization is
algorithm-independent, while the per-environment algorithm overrides are PPO-only.
The bundled stop criteria were selected for PPO and have not been validated for SAC.
The environment configuration RLlib receives comes from the entry's *config variant*
JSON, never from the recipe. `train_samples.py` injects `profile` and
`dump_timings_to_info` into it from `--profile`.

| Key | Meaning |
| --- | --- |
| `description` | Human-readable only. The training script never reads it. |
| `normalize_observations` | When `true`, installs an RLlib `MeanStdFilter` environment-to-module connector for each environment runner. This is algorithm-independent and applies to both PPO and SAC. |
| `stop_criteria` | Two keys are recognised: `episode_return_mean` maps to RLlib's `env_runners/episode_return_mean`, and `num_env_steps_sampled_lifetime` maps to the metric of the same name. `--max_iterations` adds a `training_iteration` stop on top. |
| `ppo` | PPO overrides layered onto `default_ppo_config`. `ppo["env_runners"]` and `ppo["training"]` are splatted into the corresponding RLlib config calls. The only other accepted key is `train_batch_size_per_runner`, which is not an RLlib setting: the script multiplies it by `--num_env_runners` and writes the product to `train_batch_size`. |

:::warning Recipe scope and validation
A recipe is scoped to exactly one entry. A variant does not inherit its base module's
recipe, and a base environment does not pick up a variant's recipe. Name the target
recipe `<module>.train.json` or `<module>_<variant>.train.json`.

Under both PPO and SAC, an unrecognised top-level key or `stop_criteria` key raises
`ValueError` naming the offending key and the supported set. An `env_config` section
also raises under both algorithms; environment configuration belongs in a config
variant (see [Environment File Naming](#environment-file-naming)).

Under `--algorithm PPO`, the `ppo` block is applied and validated. An unrecognised key
inside it raises `ValueError`. Setting both `ppo.training.train_batch_size` and
`ppo.train_batch_size_per_runner` also raises `ValueError` because both configure
`train_batch_size`.

Under `--algorithm SAC`, the entire `ppo` block is ignored and not validated.
`default_sac_config` supplies every environment's settings without per-environment
overrides. Consequently, a misspelled key or both batch-size settings inside `ppo`
produce a silent no-op under SAC, although each is an error under PPO.
:::

## Environment File Naming

Each file next to an env module is one of the following, and the filename is what
decides which:

| Filename | Meaning |
| --- | --- |
| `<name>_env.py` | Env module. Its `MochiEnv` subclass becomes a Gymnasium ID (`AntEnv` &rarr; `superdex_gym/Ant-v0`, short name `ant`) |
| `<module>_<variant>.json` | Gym config variant &mdash; **the only place env configuration may live**. Registered as its own Gymnasium ID and short name (`ant_env_no_contact.json` &rarr; `superdex_gym/AntNoContact-v0`, short name `ant_no_contact`) |
| `<module>.<kind>.json` | Usage recipe for the base env, where `<kind>` is `train` or `benchmark`. A `train` recipe **may not configure the environment** |
| `<module>_<variant>.<kind>.json` | Usage recipe for that variant. A variant deliberately does *not* inherit the base env's recipe |

`<variant>` must be a snake_case token, with three rules on its segments:

- **No segment may be `env`.** Every env module ends in the `_env` segment, so this is what
  keeps a longer sibling module's files (`foo_env_extra_env.json`) from reading as a
  variant of the shorter one (`foo_env.py`).
- **No segment may be `train` or `benchmark`.** Reserved so that `foo_env_train.json` (a
  variant) cannot be confused with `foo_env.train.json` (a recipe).
- **A `test` segment marks the variant test-only.** It is discovered and smoke-tested, but
  never registered with Gymnasium and never listed by a CLI. Use this for degenerate
  configurations (no gravity, no damping) that are worth crash-checking but are not
  shippable tasks.

Because training recipes may not configure the environment, every configuration that gets
trained is also a named, runnable, smoke-tested environment. Training fails loudly if a
`train` recipe contains an `env_config` section.

Unlike `train` recipes, `benchmark` recipes carry an `env_cfg` measurement baseline.
`load_env_config` reads those recipes by name; the training script and the worker-sweep
script `apps/envs/benchmark.py` do not read them. None of the included environments
ships a `benchmark` recipe.

## General Usage Notes

### Training Configuration

**PPO** (`default_ppo_config`):

| Setting | Value |
| --- | --- |
| Network | 2-layer fully connected, `[64, 64]`, `tanh` activation |
| Value function | Shares layers with the policy (`vf_share_layers: True`) |
| Learning rate | `0.0003`, fixed — there is no schedule |
| `train_batch_size` | `16384` (`32 × 512`) |
| `minibatch_size` | `4096` |
| `num_epochs` | `15` |
| `lambda_` | `0.95` |
| `vf_loss_coeff` | `0.01` |
| `rollout_fragment_length` | `512` |
| Callbacks | `LogRewardAndInfoCallbacks` |
| Evaluation | Requests one complete episode every iteration, parallel to training, `explore=False` |

**Experimental SAC baseline** (`default_sac_config`):

These shared settings are provided to exercise the RLlib integration. They have not
been tuned or validated for the included environments, and the recipe's `ppo`
overrides are not applied. The recipe stopping thresholds are likewise not evidence
that SAC will reach them.

| Setting | Value |
| --- | --- |
| Model | RLlib's default SAC RLModule; no model overrides are applied |
| Actor / critic / alpha learning rates | `3e-5` / `3e-4` / `1e-4` |
| `initial_alpha` | `0.1`, with `target_entropy="auto"` |
| `train_batch_size` | `256` |
| `tau` | `0.005`, `target_network_update_freq` `1` |
| `n_step` | `1` |
| Replay buffer | `EpisodeReplayBuffer`, capacity `1e5`, batch `256 × 1` |
| Learning starts after | `1024` sampled steps |
| `rollout_fragment_length` | `1` |
| Callbacks | RLlib's stock `DefaultCallbacks` |
| Evaluation | Every iteration, parallel to training, `explore=False` |

Both run on CPU — `num_gpus_per_learner=0`.

### Distributed Training

- **Environment runners**: parallel processes collecting experience.
- **Learners**: parallel processes updating policy parameters.
- **Evaluation**: a dedicated evaluation runner, in parallel with training.
  PPO evaluation remains subject to Ray's `evaluation_sample_timeout_s` (120 seconds
  by default), and its reported returns use the configured five-episode smoothing
  window.

`num_env_runners` is capped to the available CPU count minus the learner count, with a
printed `WARNING:` line, once `num_env_runners + num_learners` reaches the CPU count;
`num_learners` at or above the CPU count raises instead. An example such as
`--num_env_runners 128` on a smaller machine is therefore reduced, and says so.

### Checkpointing and Monitoring

- Checkpoints every `--checkpoint_freq` iterations, plus one at the end of training.
- TensorBoard logging for training metrics.
- Optional per-checkpoint video generation.

### Performance Optimization

- **CPU usage**: worker counts are capped against the detected CPU count.
- **Memory**: batch sizes are set per algorithm in the tables above.

## Output Structure

### Training outputs

Results are rooted at `--output_path` (default `~/ray_results/`). Ray Tune creates
an algorithm directory, then one trial directory per environment named
`<short_name>_<trial_id>`:

```
<output_path>/
|-- PPO/
|   |-- ant_no_contact_a1b2c3d4/
|   |-- cart_pole_e5f6a7b8/
|   |-- half_cheetah_c9d0e1f2/
|       |-- checkpoint_000000/
|       |   |-- learner_group/
|       |   |-- video_000.mp4          # if --video_on_checkpoint
|       |   |-- video_labels.json
|       |-- params.json                # read by run_inference.py
|       |-- progress.csv
|       |-- events.out.tfevents.*
|-- SAC/
    |-- ...
```

Note that `params.json` sits in the **trial** directory, one level above each
`checkpoint_*`, and that the per-checkpoint videos live **inside** the checkpoint
directory.

### Inference outputs

- Console output with episode returns and completion reasons
- MP4 files if recording is enabled
- Completed episode count

### Debug options

- Ray dashboard at `http://localhost:8265` for cluster monitoring.
- TensorBoard: `uv run tensorboard --logdir ~/ray_results` (or your
  `--output_path`). Nothing installs the `tensorboard` command — `ray[rllib]`
  brings `tensorboardx`, which writes the event files but does not read them — so
  `uv pip install tensorboard` first.
- Tune verbosity is fixed at `V1_EXPERIMENT`; there is no flag to change it.

## Callbacks Directory

`apps/rllib/callbacks/` contains two callbacks. They are wired in different ways:
`LogRewardAndInfoCallbacks` is an RLlib `DefaultCallbacks` subclass attached to the
algorithm config, while `CheckpointVideoGeneratorCallback` is a Ray **Tune**
callback attached to the tuner.

### `LogRewardAndInfoCallbacks`

**Detailed episode statistics logging for PPO only.** This RLlib `DefaultCallbacks`
subclass is attached to the PPO algorithm config and hooks `on_episode_end` only. SAC
uses RLlib's stock `DefaultCallbacks`, so the metrics below do not appear in a SAC run.

At the end of each episode, it emits four metric families. Each reports a mean or
standard deviation **over the steps of that episode**:

| Metric | Meaning |
| --- | --- |
| `info_<key>_mean` | Mean of `info[<key>]` across the episode's steps |
| `info_<key>_stdev` | Standard deviation of the same |
| `reward_mean` | Mean **per-step** reward. Not the episode return. |
| `reward_stdev` | Standard deviation of the per-step reward |

Because `MochiEnv` surfaces every reward term as `info["reward_<term>"]`, a term
named `upright_reward` arrives as `info_reward_upright_reward_mean`. String-valued
`info` entries — `terminated_reason`, for instance — are skipped, since they cannot
be averaged.

Enabling `--profile` also switches on `dump_timings_to_info`, which adds a
`timings/*` family to `info` and therefore a corresponding set of
`info_timings/*_mean` series. That is a lot of extra metrics; turn it off once you
have the numbers you wanted.

### `CheckpointVideoGeneratorCallback`

**Video generation at training checkpoints.** A Ray **Tune** callback, attached to
the tuner rather than to the algorithm config.

```python
CheckpointVideoGeneratorCallback(
    count=1,                    # sequential rollouts, one video each
    fmt="mp4",
    fps=30,
    pattern=None,               # fnmatch pattern(s) over trial IDs; None means all
    log_to_tensorboard=True,
    tb_log_every=1,
)
```

`count` and `tb_log_every` must both be `>= 1`. The TensorBoard counter is
zero-based per trial, so the **first** checkpoint — the pre-training baseline — is
always logged, then every `tb_log_every`-th one after it.

**Output**, written into each `checkpoint_*/` directory:

| File | Contents |
| --- | --- |
| `video_000.mp4`, `video_001.mp4`, … | One per requested rollout (`count`) |
| `video_labels.json` | `{"training_iteration": int, "num_env_steps_sampled_lifetime": int}`, consumed by [`visualize_training_history.py`](./visualize_training_history.md) |

The TensorBoard summary is written to the **trial** directory rather than the
checkpoint directory, so the clips share a run with the scalar metrics. The tags are
`rollout/video_<index>`.

The callback runs the requested rollouts sequentially through one environment. That
environment is still created through `AsyncVectorEnv`, but purely for process
isolation: the renderer does not support multiple contexts per process, and the Ray
Tune driver cannot initialize it in-process. Increasing `count` therefore increases
checkpoint video-generation time approximately linearly; it does not add concurrent
render workers.

To keep checkpointing responsive, this TensorBoard preview is capped at 300 frames
and a 480-pixel longest side while preserving approximately the same playback
duration. The `video_*.mp4` in the checkpoint remains full-resolution and contains
every rendered frame; use that file for behavior review and archival.

:::caution `moviepy` is required for the TensorBoard videos
TensorBoardX encodes video summaries through `moviepy`. When it is missing, TensorBoardX
logs `moviepy is not installed.` at ERROR level once per checkpoint and writes no video
payload, but nothing is raised: the MP4 files still land in the checkpoint directory and
training continues. Nothing appears in TensorBoard, and that logged error is the only
warning you get. `moviepy` is not a declared dependency of
`superdex-lab` either; install it alongside Ray — see
[Installation and Setup](./setup.md).
:::

:::note Rendering happens in spawned subprocesses
The renderer does not support more than one GL context per process, and the Ray Tune
driver is multi-threaded. The callback therefore runs its rollouts in a Gymnasium
`AsyncVectorEnv` with the `"spawn"` start method — forking the driver and then
initializing OpenGL/EGL in the child segfaults. This also means the rollout
environments are constructed fresh, with `render_mode` forced to `"rgb_array"`.
:::

Only the sidecar-writing and TensorBoard-logging helpers are best-effort. Ray runs
`on_checkpoint` before the checkpoint is registered, so those two helpers log and
swallow failures instead of aborting training. The rest of `on_checkpoint` has no
`try`/`except`: failures while loading the module, rendering the rollout, or writing
the animations propagate. If either metric is unavailable, the sidecar is skipped
instead of being written with placeholder zeros; `visualize_training_history.py` can
then fall back to timestamp matching.

**Usage:** both callbacks are wired up by `train_samples.py`:

- `LogRewardAndInfoCallbacks` for PPO runs (SAC uses RLlib's `DefaultCallbacks`)
- `CheckpointVideoGeneratorCallback` when `--video_on_checkpoint` is passed (it is off
  by default) **and** the renderer is available
