---
title: Visualizing Training History
---

# Visualizing Training History

`apps/rllib/visualize_training_history.py` stitches the per-checkpoint rollout
videos from a Ray Tune trial into a single labelled montage, so you can watch a
policy improve from its first checkpoint to its last in one clip.

After `uv sync --extra core` and the training-stack installs from
[Installation and Setup](./setup.md):

```bash
cd superdex_lab/apps/rllib
uv run --no-sync python visualize_training_history.py ~/ray_results/PPO/cart_pole_a1b2c3d4 --num_videos 8
```

By default this writes `<trial_dir>/training_history.mp4`.

![Four frames from a training-history montage, labelled iter=10, 250, 500 and 750, showing a robot hand's grasp on a cube improving as training progresses](/img/superdex_gym/training_history_montage.png)

The montage above shows rendered frames from four checkpoints during policy training on
an in-hand manipulation task.

## Prerequisites

The tool reads and writes video with `imageio`'s FFmpeg plugin and draws its labels
with Pillow. Both arrive from `uv.lock` with `uv sync --extra core`, but `moviepy`
— installed alongside Ray — caps Pillow below 12, so keep `pillow>=10.1` in the
training-stack install listed in [Installation and Setup](./setup.md), which
carries the pinned versions. Pillow 10.1 is the floor because the label uses
`ImageFont.load_default(size=...)`.

The input is the per-checkpoint videos written by `CheckpointVideoGeneratorCallback`
during training — see [Training with Ray/RLlib](./rllib.md#checkpointvideogeneratorcallback).
Checkpoint videos are **off by default**; pass `--video_on_checkpoint` to
`train_samples.py` and each `checkpoint_*/` directory gets a `video_000.mp4` and a
`video_labels.json`. The renderer has to be available too — if it is not,
`train_samples.py` warns and drops video generation.

## Input and output

| Flag | Default | Meaning |
| --- | --- | --- |
| `trial_dir` | — | **Required positional.** The Ray Tune trial directory holding the `checkpoint_<n>` subdirectories and the training metadata (`result.json` or `progress.csv`). |
| `--video_name` | `video_000.mp4` | Name of the rollout video to read from each checkpoint directory |
| `--output` | `<trial_dir>/training_history.mp4` | Output path |
| `--fps` | source frame rate | Output frame rate; resamples every clip |

Note that `trial_dir` is the **trial** directory — the one containing
`params.json`, `progress.csv` and the `checkpoint_*` directories — not an individual
checkpoint.

## Choosing which checkpoints to show

| Flag | Default | Meaning |
| --- | --- | --- |
| `--num_videos` | — | **Required.** Number of checkpoints to sample. Clamped to the number available. |
| `--schedule` | `linear` | `linear` spaces samples evenly; `exponential` concentrates them on early training |
| `--base` | `2.0` | Curvature of the exponential schedule. Must be `> 1`; larger values front-load more heavily. Ignored for `linear`. |

Both schedules always include the final checkpoint. `exponential` is usually the
better choice for a long run, because most of the visible change happens early.

Duplicate indices are resolved by bumping forward to the next unused checkpoint, and
every remaining sample has an index slot reserved before rounding, so you always get
the number of clips you asked for. When the schedule cannot distinguish them, the
front of an `exponential` montage degenerates to consecutive checkpoints.

## Backfilling missing videos

If you trained without `--video_on_checkpoint`, which is the default, or the renderer
was unavailable at training time, the checkpoints have no videos. By default a missing
video is a hard error:

```
FileNotFoundError: Selected checkpoint is missing video_000.mp4: <checkpoint>
```

`--generate_missing` instead shells out to `run_inference.py` for that checkpoint and
renames the resulting `inference_000.mp4` into place.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--generate_missing` | off | Generate a video by running inference instead of failing |
| `--inference_episodes` | `1` | Episodes to roll out; the first episode's clip is used |
| `--explore_during_inference` | off | Use the exploration forward pass when generating |

Backfilling runs inference in a subprocess, which keeps torch, Ray and the physics
engine out of the compositing tool's own process — but they still have to be
installed, since the subprocess is `run_inference.py`. It also needs a working
renderer; if inference produces nothing, the tool reports that the renderer may be
unavailable.

## Appearance

| Flag | Default | Meaning |
| --- | --- | --- |
| `--speed` | `1.0` | Playback speed multiplier, applied before other transforms |
| `--text_height_pct` | `5.0` | Label font size, as a percentage of frame height |
| `--margin_pct` | `3.0` | Label inset from the frame edges, as a percentage of frame height |
| `--text_location` | `bottom-left` | One of `bottom-left`, `bottom-right`, `top-left`, `top-right`, `center` |
| `--font_color` | white | Any name or hex string PIL accepts, e.g. `'#ff0000'` |
| `--font_path` | PIL's built-in font | A TrueType font file |
| `--head_seconds` | `1.0` | Hold each clip's first frame before it plays |
| `--tail_seconds` | `1.0` | Hold each clip's last frame after it plays |
| `--transition_seconds` | `0.5` | Crossfade between consecutive clips |

:::note
`--transition_seconds` must not exceed `--head_seconds` or `--tail_seconds`, so the
blend only ever covers frozen frames and never cuts into the action.
:::

## How clips are labelled

Each clip is captioned `iter=<training_iteration>, timestep=<num_env_steps>`, with
the timestep in two-significant-digit scientific notation — e.g.
`iter=250, timestep=1.02e+06`. The numbers come from one of two sources, in order:

1. **`video_labels.json`**, the sidecar `CheckpointVideoGeneratorCallback` writes
   next to each video. This is authoritative when present, and the two files
   deliberately share the same filename constant.
2. **A timestamp match** against the trial's `result.json` (or `progress.csv`),
   using the checkpoint's write time. The write time is taken from
   `rllib_checkpoint.json` if present, then `algorithm_state.pkl`, then the
   directory's own mtime.

The fallback has a **300-second tolerance**. If the nearest training result is
further away than that, the tool refuses to guess:

```
Cannot reliably label <checkpoint>: the nearest training result is <N>s from the
checkpoint's write time (tolerance 300s) and no video_labels.json sidecar is
present. Regenerate the checkpoint videos to write a sidecar.
```

Backfilled videos are the common way to hit this — writing a video into a checkpoint
directory bumps its mtime, decoupling it from the original training timestamp. The
fix is to regenerate the checkpoint videos so the sidecar exists.

## Worked example

Train CartPole, checkpointing often so there is something to watch. The
`--video_on_checkpoint` flag is required: without it no checkpoint contains a video
and the montage command below fails with the `FileNotFoundError` shown above. The
renderer must be available as well.

```bash
uv run --no-sync python train_samples.py --pattern "cart_pole" --checkpoint_freq 2 --video_on_checkpoint
```

Then build a 10-clip montage weighted toward early training, at half speed, with a
larger red label in the top-left:

```bash
uv run --no-sync python visualize_training_history.py \
  ~/ray_results/PPO/cart_pole_a1b2c3d4 \
  --num_videos 10 \
  --schedule exponential --base 3.0 \
  --speed 0.5 \
  --text_height_pct 8 --text_location top-left --font_color '#ff0000' \
  --output ./cart_pole_history.mp4
```
