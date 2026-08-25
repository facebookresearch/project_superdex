---
title: Visualizing Training History
---

# Visualizing Training History

`apps/rllib/visualize_training_history.py` combines per-checkpoint rollout videos
from one Ray Tune trial into a labeled montage that shows how the policy changed
during training.

## Prerequisites

Install the core and training dependencies with `uv sync --extra core` and the
training-stack commands in [Installation and Setup](./setup.md).

Checkpoint videos are off by default. Pass `--video_on_checkpoint` to
`train_samples.py` to write `video_000.mp4` and `video_labels.json` in each
`checkpoint_*/` directory. The renderer must also be available. For example:

```bash
cd superdex_lab/apps/rllib
uv run --no-project python train_samples.py \
  --pattern "cart_pole" --checkpoint_freq 2 --video_on_checkpoint
```

If the selected checkpoints do not contain videos, the visualization command can
generate them with `--generate_missing`.

See
[CheckpointVideoGeneratorCallback](./rllib.md#checkpointvideogeneratorcallback)
for the training callback details.

## Inputs

Provide the Ray Tune trial directory as `trial_dir`, not an individual checkpoint.
It contains the `checkpoint_*` directories and normally `params.json`. Each selected
checkpoint must contain the file named by `--video_name`, which defaults to
`video_000.mp4`, unless you use `--generate_missing`; generating videos also requires
the trial's `params.json`.

Labels use `video_labels.json` when present. Otherwise, the tool matches the
checkpoint write time against `result.json`, or against `progress.csv` when
`result.json` is absent. One of those metadata files is required only when a
selected checkpoint lacks a readable label sidecar.

## Run the visualization

Use `superdex_lab/apps/rllib` as the working directory:

```bash
cd superdex_lab/apps/rllib
uv run python visualize_training_history.py \
  ~/ray_results/PPO/cart_pole_a1b2c3d4 --num_videos 8
```

The command samples up to eight checkpoints with the default `linear` schedule and
always includes the final checkpoint.

## Output

By default, the montage is written to
`<trial_dir>/training_history.mp4`. Use `--output` to choose another path; the
tool creates missing parent directories. Each clip is labeled
`iter=<training_iteration>, timestep=<num_env_steps>`, for example
`iter=250, timestep=1.02e+06`.

![Four frames from a training-history montage, labeled iter=10, 250, 500 and 750, showing a robot hand's grasp on a cube improving as training progresses](/img/superdex_gym/training_history_montage.png)

## Options

### Input and selection

| Option | Default | Description |
| --- | --- | --- |
| `trial_dir` | required | Ray Tune trial directory containing `checkpoint_<number>` directories. |
| `--num_videos` | required | Number of checkpoints to sample. Clamped to the number available. |
| `--schedule` | `linear` | `linear` spaces checkpoints evenly; `exponential` concentrates samples on early training. Both include the final checkpoint. |
| `--base` | `2.0` | Curvature for `exponential`; must be greater than `1`. Larger values put more samples early. Ignored for `linear`. |
| `--video_name` | `video_000.mp4` | Rollout video filename to read from each selected checkpoint. |
| `--output` | `<trial_dir>/training_history.mp4` | Output video path. |
| `--fps` | source frame rate | Output frame rate; each clip is resampled when set. Without it, all source videos must have the same frame rate. |

### Appearance

| Option | Default | Description |
| --- | --- | --- |
| `--speed` | `1.0` | Playback speed multiplier applied before other transforms. |
| `--text_height_pct` | `5.0` | Label font size as a percentage of frame height. |
| `--margin_pct` | `3.0` | Label inset from frame edges as a percentage of frame height. |
| `--text_location` | `bottom-left` | `bottom-left`, `bottom-right`, `top-left`, `top-right`, or `center`. |
| `--font_color` | white | Any color name or hex string accepted by PIL, such as `'#ff0000'`. |
| `--font_path` | PIL's built-in font | TrueType font file to use instead. The built-in scalable font requires Pillow 10.1 or newer. |
| `--head_seconds` | `1.0` | Seconds to hold each clip's first frame before playback. |
| `--tail_seconds` | `1.0` | Seconds to hold each clip's last frame after playback. |
| `--transition_seconds` | `0.5` | Crossfade duration between clips. It must not exceed `--head_seconds` or `--tail_seconds`. |

For example, create a 10-clip montage weighted toward early training, at half
speed, with a larger red label in the top-left:

```bash
uv run python visualize_training_history.py \
  ~/ray_results/PPO/cart_pole_a1b2c3d4 \
  --num_videos 10 \
  --schedule exponential --base 3.0 \
  --speed 0.5 \
  --text_height_pct 8 --text_location top-left --font_color '#ff0000' \
  --output ./cart_pole_history.mp4
```

## Missing Videos and Label Fallback

By default, a selected checkpoint without `video_000.mp4`, or the filename set by
`--video_name`, stops with:

```text
FileNotFoundError: Selected checkpoint is missing video_000.mp4: <checkpoint>
```

Use `--generate_missing` to run `run_inference.py` for each missing video. The
inference subprocess writes `inference_000.mp4`, which the tool renames to the
requested `--video_name`. This fallback requires the training dependencies and a
working renderer.

| Option | Default | Description |
| --- | --- | --- |
| `--generate_missing` | off | Run inference instead of failing when a selected checkpoint has no video. |
| `--inference_episodes` | `1` | Number of episodes to roll out; the first episode's clip is used. Only applies with `--generate_missing`. |
| `--explore_during_inference` | off | Use the exploration policy while generating missing videos. Only applies with `--generate_missing`. |

When `video_labels.json` is absent or unreadable, label resolution falls back to
timestamp matching against the trial metadata. The maximum accepted difference is
300 seconds. If neither `result.json` nor `progress.csv` exists, or the closest
result is more than 300 seconds away, the tool stops instead of assigning an
unreliable label.

All selected videos must have the same frame dimensions. Use `--fps` to normalize
different source frame rates before the clips are joined.
