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

## Where the scripts live and how to run them

All paths are relative to the `superdex_lab` project root:

| Script | Location | Purpose |
| --- | --- | --- |
| `train_samples.py` | `apps/rllib/train_samples.py` | Train one or more environments |
| `run_inference.py` | `apps/rllib/run_inference.py` | Roll out a trained checkpoint |
| `visualize_training_history.py` | `apps/rllib/visualize_training_history.py` | Stitch checkpoint videos into a montage |
| `callbacks/` | `apps/rllib/` | Supporting package |

`train_samples.py` and `run_inference.py` import their
siblings by bare name (`from callbacks import ...`, `from utils import ...`).
Python puts the *script's own directory* on `sys.path`, so
invoking them by path from the project root works; `cd`-ing into `apps/rllib/` first is
the convention used throughout these docs. Importing them as a package or running
`python -m` is what would need a fallback, and neither has one.
`visualize_training_history.py` has no sibling imports and runs from any directory.

```bash
cd superdex_lab/apps/rllib
```

Use `uv run --no-sync` under `apps/rllib/`, as RLlib has different dependency
versioning. See [Installation and Setup](./setup.md#dependencies-for-apps).

### Ray version

Ray is **not** a declared dependency of `superdex-lab` — `pyproject.toml` lists only
`gymnasium`, `h5py`, `numpy` and the two SuperDex packages, so nothing under
`apps/rllib/` runs after a plain install. Install it yourself, at
**`ray[rllib]==2.49.0`**; [Installation and Setup](./setup.md) gives the full command.
RLlib's API churns
hard between releases — the new API stack (`RLModule`, `EnvRunner`, `Learner`) that
these scripts target is not source-compatible with older Ray. Note that installing
`ray[rllib]` may move your `gymnasium` version.

`train_samples.py` calls a bare `ray.init()`; `run_inference.py` does not initialise Ray
at all. There
is no `--ray-address` flag; to attach `train_samples.py` to an existing cluster, set the
`RAY_ADDRESS` environment variable, which Ray reads itself.

### Training a custom environment

`train_samples.py` trains what environment discovery finds. To train an environment
of your own from your own script, register it with Ray Tune first:

```python
from utils import register_envs

register_envs()
```

`register_envs()` calls `register_all_envs()` to populate the Gymnasium registry,
then registers each id with Tune. It fans out whatever is in the Gymnasium registry
at the moment it is called, so register your own environment with Gymnasium first.
The indirection matters: RLlib does not pass its `env_config` to a Gymnasium
environment's constructor correctly on its own, and the two
families of environment need different handling. For a `superdex_gym/*` id — every
environment `train_samples.py` trains — the wrapper **merges** the RLlib `env_config`
over the `cfg` defaults registered on the Gymnasium spec and passes the result as a
single `cfg=` mapping, so a variant's defaults survive and are overridden key by key.
For any other Gymnasium id the wrapper splats the config dict into constructor keyword
arguments instead.

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
of its own. In an open-source build the trainable set is:

| CLI name | Gymnasium ID | Recipe |
| --- | --- | --- |
| `ant_no_contact` | `superdex_gym/AntNoContact-v0` | `benchmarks/ant_env_no_contact.train.json` |
| `cart_pole` | `superdex_gym/CartPole-v0` | `benchmarks/cartpole_env.train.json` |
| `half_cheetah` | `superdex_gym/HalfCheetah-v0` | `benchmarks/halfcheetah_env.train.json` |

Note that the base `ant` environment is registered but not
trainable: no recipe is named after its module. (`cart_pole` and `half_cheetah` *are*
base environments, trained from base-module recipes; only `ant_no_contact` is a variant.)

All three recipes train from a public checkout — the assets they need ship with it.

**Features:**

- Simultaneous training of multiple environments
- Algorithm selection: PPO (recommended) or SAC (experimental)
- Pattern-based environment selection
- Per-environment PPO hyperparameters from the recipe files
- Centralized experiment management

**Usage Examples:**

```bash
# Train all trainable environments with PPO (default).
uv run --no-sync python train_samples.py

# Train specific environments using patterns.
uv run --no-sync python train_samples.py --pattern "cart_pole"
uv run --no-sync python train_samples.py --pattern "*cheetah*"

# Train with a custom configuration.
uv run --no-sync python train_samples.py --num_env_runners 64 --checkpoint_freq 5

# Experimental SAC run on CartPole, limited to one training iteration.
# This checks the training path, not learning or convergence.
uv run --no-sync python train_samples.py --algorithm SAC --pattern "cart_pole" --max_iterations 1

# Train selected environments with video recording enabled (off by default)
uv run --no-sync python train_samples.py --pattern "ant*" --video_on_checkpoint --output_path ./benchmark_results

# High-throughput training for benchmarking
uv run --no-sync python train_samples.py --num_env_runners 128 --checkpoint_freq 20
```

:::caution `--pattern` matches CLI short names
Patterns are matched with `fnmatch` against the snake_case short names. `"cartpole"`
and `"halfcheetah"` match nothing, and neither does `"ant"` — the trainable ant entry
is `ant_no_contact`, and `fnmatch` is not a prefix match. The script then prints
`No samples to train, exitting...` and exits with a non-zero status rather than
failing loudly. Use `"cart_pole"` / `"half_cheetah"`, or a glob such as `"cart*"` /
`"ant*"`.
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
uv run --no-sync python run_inference.py /path/to/checkpoint

# Use the exploration forward pass.
uv run --no-sync python run_inference.py /path/to/checkpoint --explore_during_inference

# Record videos of inference episodes.
uv run --no-sync python run_inference.py /path/to/checkpoint --video

# Custom number of episodes and video path.
uv run --no-sync python run_inference.py /path/to/checkpoint --num_episodes 20 --video_path ./inference_videos
```

**Command-line Options:**

| Option | Default | Notes |
| --- | --- | --- |
| `checkpoint_path` | — | **Required positional.** The checkpoint directory. |
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
algorithm-independent, but the per-environment algorithm overrides are PPO-only.
The bundled stop criteria were selected for PPO and have not been validated for SAC.
Unknown keys are **rejected, not ignored** — at
the top level and inside `stop_criteria`, an unrecognised key raises `ValueError`
naming the offending key and the supported set, on every run. The checks *inside* the
`ppo` block are different: they run **only under `--algorithm PPO`**, because the whole
`ppo` section is skipped under SAC. An unrecognised key inside `ppo` therefore raises
under PPO and trains silently under SAC. An `env_config` section is a hard error in its
own right, under either algorithm (see [Environment File
Naming](#environment-file-naming)).

The environment configuration RLlib receives comes from the entry's *config variant*
JSON, never from the recipe. `train_samples.py` injects `profile` and
`dump_timings_to_info` into it from `--profile`.

| Key | Meaning |
| --- | --- |
| `description` | Human-readable only. The training script never reads it. |
| `normalize_observations` | When `true`, installs an RLlib `MeanStdFilter` environment-to-module connector for each environment runner. This is algorithm-independent and applies to both PPO and SAC. |
| `stop_criteria` | Two keys are recognised: `episode_return_mean` maps to RLlib's `env_runners/episode_return_mean`, and `num_env_steps_sampled_lifetime` maps to the metric of the same name. Any other key raises `ValueError`. `--max_iterations` adds a `training_iteration` stop on top. |
| `ppo` | Overrides layered onto `default_ppo_config`. `ppo["env_runners"]` and `ppo["training"]` are splatted into the corresponding RLlib config calls; the only other accepted key is `train_batch_size_per_runner`, which is not an RLlib setting — the script multiplies it by `--num_env_runners` and writes the product to `train_batch_size`. Setting both `ppo.training.train_batch_size` and `ppo.train_batch_size_per_runner` raises `ValueError`, because they configure the same value; under SAC the `ppo` block is not read at all, so the clash is not detected. |

:::warning Two properties of the recipe mechanism
- **A recipe is scoped to exactly one entry.** A variant does not inherit its base
  module's recipe, and a base env does not pick up a variant's. Point the recipe at
  the entry you mean by naming it `<module>.train.json` or
  `<module>_<variant>.train.json`.
- **The `ppo` block is ignored under SAC.** `--algorithm SAC` uses
  `default_sac_config` for every environment with no per-environment overrides at
  all. Because the block is never read, neither is it validated: a misspelled key or
  a doubled batch size inside `ppo` is an error under PPO and a silent no-op under
  SAC.
:::

## Environment File Naming

Each file next to an env module is one of the following, and the filename is what
decides which:

| Filename | Meaning |
| --- | --- |
| `<name>_env.py` | Env module. Its `MochiEnv` subclass becomes a Gymnasium id (`AntEnv` &rarr; `superdex_gym/Ant-v0`, short name `ant`) |
| `<module>_<variant>.json` | Gym config variant &mdash; **the only place env configuration may live**. Registered as its own id and short name (`ant_env_no_contact.json` &rarr; `superdex_gym/AntNoContact-v0`, short name `ant_no_contact`) |
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

(`benchmark` recipes are the exception: they carry an `env_cfg` measurement baseline,
read by name through `load_env_config` rather than by the training script. No
open-source environment ships one; the worker-sweep script `apps/envs/benchmark.py`
does not read these recipes either.)

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
- Summary statistics across inference episodes

### Debug options

- Ray dashboard at `http://localhost:8265` for cluster monitoring.
- TensorBoard: `uv run --no-sync tensorboard --logdir ~/ray_results` (or your
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

**Detailed episode statistics logging.** An RLlib `DefaultCallbacks` subclass,
attached to the algorithm config. It hooks `on_episode_end` only.

At the end of each episode it emits four families of metric, each the mean or
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

**Attached for PPO only.** SAC uses RLlib's stock `DefaultCallbacks`, so none of the
above appears in a SAC run.

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

The sidecar and the TensorBoard logging are both best-effort: Ray runs
`on_checkpoint` before the checkpoint is registered, so failures **in those two
helpers** are logged and swallowed rather than aborting training. That does not
extend to the rest of the callback — `on_checkpoint` itself has no `try`/`except`, so
a failure loading the module, rendering the rollout or writing the animations
propagates. If either metric is unavailable the sidecar
is skipped entirely rather than written with placeholder zeros, which is what lets
`visualize_training_history.py` fall back to timestamp matching.

**Usage:** both callbacks are wired up by `train_samples.py`:

- `LogRewardAndInfoCallbacks` for PPO runs (SAC uses RLlib's `DefaultCallbacks`)
- `CheckpointVideoGeneratorCallback` when `--video_on_checkpoint` is passed (it is off
  by default) **and** the renderer is available
