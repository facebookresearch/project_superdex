# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-strict

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import imageio
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Clip:
    frames: np.ndarray
    fps: float


@dataclass(frozen=True)
class _ResultRow:
    iteration: int
    timestep: int
    timestamp: float


def _validate_clip(clip: Clip) -> None:
    if clip.fps <= 0:
        raise ValueError(f"Clip fps must be positive, got {clip.fps}")
    if clip.frames.ndim != 4 or clip.frames.shape[-1] != 3:
        raise ValueError(
            f"Clip frames must have shape (T, H, W, 3), got {clip.frames.shape}"
        )
    if len(clip.frames) == 0:
        raise ValueError("Clip must contain at least one frame")
    if clip.frames.dtype != np.uint8:
        raise ValueError(f"Clip frames must use uint8, got {clip.frames.dtype}")


def load_clip(video_path: str) -> Clip:
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video does not exist: {path}")

    reader = imageio.get_reader(str(path), "ffmpeg")
    try:
        metadata = cast(dict[str, Any], reader.get_meta_data())
        fps = float(metadata["fps"])
        frames = [np.asarray(frame, dtype=np.uint8)[..., :3] for frame in reader]
    finally:
        reader.close()

    if not frames:
        raise ValueError(f"Video contains no frames: {path}")
    clip = Clip(np.stack(frames), fps)
    _validate_clip(clip)
    return clip


def _sample_indices(last: int, count: int, schedule: str, base: float) -> list[int]:
    # Validate before the single-sample early return below, so an invalid schedule or
    # base is rejected regardless of how many videos were requested.
    if schedule not in ("linear", "exponential"):
        raise ValueError(f"Unknown checkpoint schedule: {schedule}")
    if schedule == "exponential" and (not np.isfinite(base) or base <= 1):
        raise ValueError(
            f"Exponential base must be finite and greater than 1, got {base}"
        )

    # `last` is the maximum index; there are `last + 1` checkpoints available.
    # A single-video request returns the final (latest) checkpoint rather than the
    # pre-training baseline, since that is what a one-clip montage is expected to show.
    if count <= 1 or last == 0:
        return [last]
    count = min(count, last + 1)
    values: np.ndarray
    if schedule == "linear":
        values = np.linspace(0, last, count)
    else:
        # Normalize the exponent over [0, 1] so the curve always spans the full
        # [0, last] range regardless of `count`. log1p/expm1 preserve precision when
        # `base` is very close to 1, where direct subtraction would cancel. Divide by
        # `delta` before scaling by `last`: the product can overflow to infinity for a
        # very large finite base, whereas the ratio always stays within [0, 1].
        delta = base - 1.0
        fractions = np.linspace(0.0, 1.0, count)
        values = last * (np.expm1(np.log1p(delta) * fractions) / delta)

    # Round to distinct increasing indices while reserving enough integer slots for
    # every remaining sample. This preserves the requested cardinality even when
    # adjacent curve values round to the same checkpoint.
    indices: list[int] = []
    for index, value in enumerate(values):
        remaining = count - index - 1
        lower = indices[-1] + 1 if indices else 0
        upper = last - remaining
        candidate = int(round(float(value)))
        indices.append(min(max(candidate, lower), upper))
    return indices


def _generate_missing_video(
    checkpoint: Path,
    video_name: str,
    num_episodes: int,
    explore_during_inference: bool,
) -> None:
    """Backfill a missing per-checkpoint video by running inference on the checkpoint.

    Invokes ``run_inference.py`` in a subprocess (keeping its heavy torch/ray/mochi
    imports out of this compositing tool), which renders ``inference_000.mp4`` into the
    checkpoint directory; that file is then renamed to ``video_name``.
    """
    script = Path(__file__).resolve().parent / "run_inference.py"
    command = [
        sys.executable,
        str(script),
        str(checkpoint),
        "--num_episodes",
        str(num_episodes),
        "--video",
    ]
    if explore_during_inference:
        command.append("--explore_during_inference")
    logger.info(f"Generating missing {video_name} via inference: {checkpoint}")
    subprocess.run(command, check=True)

    # run_inference names its output ``inference_{episode:03d}.mp4``; rename the first
    # episode's clip to the video file this tool expects.
    produced = checkpoint / "inference_000.mp4"
    target = checkpoint / video_name
    if produced == target:
        return
    if not produced.is_file():
        raise FileNotFoundError(
            f"Inference did not produce {produced} (the renderer may be unavailable); "
            f"cannot backfill {target}."
        )
    produced.replace(target)


_CHECKPOINT_PATTERN = re.compile(r"checkpoint_(\d+)")


def _list_checkpoints(trial_dir: Path) -> list[Path]:
    """Return the trial's ``checkpoint_<number>`` directories in training order.

    Only names with a numeric suffix are accepted: a stray directory such as
    ``checkpoint_backup`` would otherwise be sampled as the newest checkpoint, since
    the schedule always includes the final index.

    Sorting is by the parsed integer rather than lexicographically, which only works
    while the suffix is zero-padded (e.g. checkpoint_000010). Unpadded names would
    order as checkpoint_1, checkpoint_10, checkpoint_2, silently mis-sampling the
    montage.
    """
    checkpoints = []
    for path in trial_dir.glob("checkpoint_*"):
        if not path.is_dir():
            continue
        match = _CHECKPOINT_PATTERN.fullmatch(path.name)
        if match is None:
            logger.warning(f"Ignoring directory with no checkpoint number: {path}")
            continue
        checkpoints.append((int(match.group(1)), path))
    return [path for _, path in sorted(checkpoints)]


def select_checkpoints(
    trial_dir: Path,
    num_videos: int,
    schedule: str,
    base: float,
    video_name: str = "video_000.mp4",
    *,
    generate_missing: bool = False,
    inference_episodes: int = 1,
    explore_during_inference: bool = False,
) -> list[Path]:
    if num_videos <= 0:
        raise ValueError(f"num_videos must be positive, got {num_videos}")
    if generate_missing and inference_episodes <= 0:
        raise ValueError(
            f"inference_episodes must be positive, got {inference_episodes}"
        )
    checkpoints = _list_checkpoints(trial_dir)
    if not checkpoints:
        raise ValueError(f"No checkpoint_<number> directories found in {trial_dir}")

    selected = [
        checkpoints[index]
        for index in _sample_indices(
            len(checkpoints) - 1, min(num_videos, len(checkpoints)), schedule, base
        )
    ]
    requested = min(num_videos, len(checkpoints))
    if len(selected) < requested:
        logger.warning(
            f"Requested {requested} checkpoints but only {len(selected)} distinct "
            "indices remain after rounding; returning fewer videos. Reduce "
            "--num_videos or widen the checkpoint range to avoid this."
        )
    for checkpoint in selected:
        video_path = checkpoint / video_name
        if not video_path.is_file():
            if generate_missing:
                _generate_missing_video(
                    checkpoint,
                    video_name,
                    inference_episodes,
                    explore_during_inference,
                )
            else:
                raise FileNotFoundError(
                    f"Selected checkpoint is missing {video_name}: {checkpoint}"
                )
        if not video_path.is_file():
            raise FileNotFoundError(
                f"Video still missing after inference for checkpoint: {checkpoint}"
            )
    return selected


def _parse_row(raw: dict[str, Any], source: Path, row_number: int) -> _ResultRow:
    required = (
        "training_iteration",
        "num_env_steps_sampled_lifetime",
        "timestamp",
    )
    missing = [name for name in required if raw.get(name) in (None, "")]
    if missing:
        raise ValueError(f"Missing {', '.join(missing)} in {source} row {row_number}")
    try:
        return _ResultRow(
            iteration=int(raw[required[0]]),
            timestep=int(raw[required[1]]),
            timestamp=float(raw[required[2]]),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid numeric metadata in {source} row {row_number}"
        ) from error


def _load_result_rows(trial_dir: Path) -> list[_ResultRow]:
    result_path = trial_dir / "result.json"
    if result_path.is_file():
        rows = []
        with result_path.open() as handle:
            for row_number, line in enumerate(handle, start=1):
                if line.strip():
                    raw = cast(dict[str, Any], json.loads(line))
                    rows.append(_parse_row(raw, result_path, row_number))
        return rows

    progress_path = trial_dir / "progress.csv"
    if not progress_path.is_file():
        raise FileNotFoundError(
            f"Neither result.json nor progress.csv exists in {trial_dir}"
        )
    with progress_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            _parse_row(cast(dict[str, Any], raw), progress_path, row_number)
            for row_number, raw in enumerate(reader, start=2)
        ]


def _checkpoint_mtime(checkpoint: Path) -> float:
    """Return the checkpoint's creation time from a stable inner file.

    The directory mtime is unreliable: writing a generated video into the checkpoint
    directory (``--generate_missing``) bumps it to generation time, which would
    mislabel the checkpoint. RLlib writes these files once at checkpoint creation and
    never touches them again, so their mtime is the true checkpoint-write time.
    """
    for name in ("rllib_checkpoint.json", "algorithm_state.pkl"):
        marker = checkpoint / name
        if marker.is_file():
            return marker.stat().st_mtime
    return checkpoint.stat().st_mtime  # fallback for non-RLlib layouts


# Sidecar file written by CheckpointVideoGeneratorCallback next to each checkpoint's
# videos. Keep in sync with `VIDEO_LABEL_SIDECAR` in
# callbacks/checkpoint_video_generator.py.
_LABEL_SIDECAR = "video_labels.json"

# Largest gap (seconds) tolerated between a checkpoint's write time and the nearest
# training-result timestamp when falling back to mtime-based labeling. Beyond this the
# match is untrustworthy (e.g. a copied or restored trial with rewritten mtimes), so we
# fail loudly instead of attaching a wrong label.
_MTIME_MATCH_TOLERANCE_SEC = 300.0


def _read_label_sidecar(checkpoint: Path) -> tuple[int, int] | None:
    """Return ``(iteration, timestep)`` from the checkpoint's label sidecar, or None.

    The sidecar is written at video-generation time from ``trial.last_result``, so it
    is authoritative. Returns None when the sidecar is absent (older checkpoints) or
    unreadable, letting the caller fall back to mtime matching.
    """
    sidecar = checkpoint / _LABEL_SIDECAR
    if not sidecar.is_file():
        return None
    try:
        with sidecar.open() as handle:
            data = cast(dict[str, Any], json.load(handle))
        return (
            int(data["training_iteration"]),
            int(data["num_env_steps_sampled_lifetime"]),
        )
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning(f"Ignoring unreadable label sidecar: {sidecar}")
        return None


def resolve_labels(
    trial_dir: Path, checkpoint_paths: Iterable[Path]
) -> list[tuple[int, int]]:
    labels: list[tuple[int, int]] = []
    # Only load the (potentially large) training-result rows if some checkpoint lacks a
    # sidecar and needs the mtime fallback.
    rows: list[_ResultRow] | None = None
    for checkpoint in checkpoint_paths:
        sidecar_label = _read_label_sidecar(checkpoint)
        if sidecar_label is not None:
            labels.append(sidecar_label)
            continue

        if rows is None:
            rows = _load_result_rows(trial_dir)
            if not rows:
                raise ValueError(f"Training metadata is empty in {trial_dir}")

        # Fall back to matching the checkpoint's write time against result timestamps.
        # Hoist the mtime out of the `min` key so `stat()` is called once per checkpoint
        # rather than once per metadata row.
        mtime = _checkpoint_mtime(checkpoint)
        nearest = min(rows, key=lambda row: abs(row.timestamp - mtime))
        # Reject the match when the nearest timestamp is implausibly far away: a copied
        # or restored trial can rewrite mtimes and silently mislabel the video.
        distance = abs(nearest.timestamp - mtime)
        if distance > _MTIME_MATCH_TOLERANCE_SEC:
            raise ValueError(
                f"Cannot reliably label {checkpoint}: the nearest training result is "
                f"{distance:.0f}s from the checkpoint's write time (tolerance "
                f"{_MTIME_MATCH_TOLERANCE_SEC:.0f}s) and no {_LABEL_SIDECAR} sidecar is "
                "present. Regenerate the checkpoint videos to write a sidecar."
            )
        labels.append((nearest.iteration, nearest.timestep))
    return labels


def extend_clip(clip: Clip, head_seconds: float, tail_seconds: float) -> Clip:
    _validate_clip(clip)
    if head_seconds < 0 or tail_seconds < 0:
        raise ValueError("Freeze padding durations cannot be negative")
    head_count = round(head_seconds * clip.fps)
    tail_count = round(tail_seconds * clip.fps)
    frames = np.concatenate(
        (
            np.repeat(clip.frames[:1], head_count, axis=0),
            clip.frames,
            np.repeat(clip.frames[-1:], tail_count, axis=0),
        )
    )
    return Clip(frames, clip.fps)


def _text_position(
    location: str,
    frame_width: int,
    frame_height: int,
    text_width: int,
    text_height: int,
    margin: int,
) -> tuple[int, int]:
    positions = {
        "bottom-left": (margin, frame_height - text_height - margin),
        "bottom-right": (
            frame_width - text_width - margin,
            frame_height - text_height - margin,
        ),
        "top-left": (margin, margin),
        "top-right": (frame_width - text_width - margin, margin),
        "center": ((frame_width - text_width) // 2, (frame_height - text_height) // 2),
    }
    if location not in positions:
        raise ValueError(f"Unknown text location: {location}")
    return positions[location]


def add_text(
    clip: Clip,
    text: str,
    *,
    text_height_pct: float,
    margin_pct: float,
    location: str = "bottom-left",
    color: tuple[int, int, int] = (255, 255, 255),
    font_path: str | None = None,
) -> Clip:
    _validate_clip(clip)
    if text_height_pct <= 0 or margin_pct < 0:
        raise ValueError("Text height must be positive and margin cannot be negative")
    frame_height, frame_width = clip.frames.shape[1:3]
    font_size = max(1, round(text_height_pct / 100 * frame_height))
    margin = round(margin_pct / 100 * frame_height)
    font = (
        ImageFont.truetype(font_path, font_size)
        if font_path is not None
        else ImageFont.load_default(size=font_size)
    )
    bounds = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox(
        (0, 0), text, font=font, anchor="lt"
    )
    position = _text_position(
        location,
        frame_width,
        frame_height,
        bounds[2] - bounds[0],
        bounds[3] - bounds[1],
        margin,
    )
    stroke_width = max(1, round(font_size * 0.08))
    rendered = []
    for frame in clip.frames:
        image = Image.fromarray(frame)
        ImageDraw.Draw(image).text(
            position,
            text,
            fill=color,
            font=font,
            anchor="lt",
            stroke_width=stroke_width,
            stroke_fill=(0, 0, 0),
        )
        rendered.append(np.asarray(image, dtype=np.uint8))
    return Clip(np.stack(rendered), clip.fps)


def crossfade(clips: list[Clip], transition_seconds: float) -> Clip:
    if not clips:
        raise ValueError("At least one clip is required")
    if transition_seconds < 0:
        raise ValueError("Transition duration cannot be negative")
    for clip in clips:
        _validate_clip(clip)
    reference = clips[0]
    for clip in clips[1:]:
        if clip.frames.shape[1:] != reference.frames.shape[1:]:
            raise ValueError("All clips must have the same frame dimensions")
        if not np.isclose(clip.fps, reference.fps):
            raise ValueError("All clips must have the same fps")

    overlap = round(transition_seconds * reference.fps)
    result = reference.frames
    for clip in clips[1:]:
        if overlap > min(len(result), len(clip.frames)):
            raise ValueError("Transition is longer than an adjacent clip")
        if overlap == 0:
            result = np.concatenate((result, clip.frames))
            continue
        # Exclude the linspace endpoints (0 and 1) so both clips contribute to every
        # blended frame. With the endpoints included, overlap == 1 yields alpha == [0.0],
        # making the single blended frame entirely the previous clip and dropping the
        # next clip's first frame (a one-frame stutter at very short transitions).
        alpha = np.linspace(0.0, 1.0, overlap + 2, dtype=np.float32)[1:-1][
            :, None, None, None
        ]
        blended = np.rint(
            result[-overlap:].astype(np.float32) * (1 - alpha)
            + clip.frames[:overlap].astype(np.float32) * alpha
        ).astype(np.uint8)
        result = np.concatenate((result[:-overlap], blended, clip.frames[overlap:]))
    return Clip(result, reference.fps)


def _validate_transition(
    fps: float, transition_seconds: float, head_seconds: float, tail_seconds: float
) -> None:
    """Reject a crossfade longer than the freeze padding it blends over.

    Compared in frame units using the same rounding rule as ``extend_clip`` and
    ``crossfade``, so the error matches the actual failure mode instead of comparing
    raw seconds.
    """
    transition_frames = round(transition_seconds * fps)
    min_padding_frames = min(round(head_seconds * fps), round(tail_seconds * fps))
    if transition_frames > min_padding_frames:
        raise ValueError(
            f"transition_seconds ({transition_frames} frames) must not exceed either "
            f"freeze padding duration ({min_padding_frames} frames) at {fps:g} fps."
        )


def change_speed(clip: Clip, speed: float) -> Clip:
    _validate_clip(clip)
    if speed <= 0:
        raise ValueError(f"Video speed must be positive, got {speed}")
    frame_count = max(1, round(len(clip.frames) / speed))
    indices = np.linspace(0, len(clip.frames) - 1, frame_count).round().astype(int)
    return Clip(clip.frames[indices], clip.fps)


def _resample_clip(clip: Clip, fps: float) -> Clip:
    if fps <= 0:
        raise ValueError(f"Output fps must be positive, got {fps}")
    frame_count = max(1, round(len(clip.frames) * fps / clip.fps))
    indices = np.linspace(0, len(clip.frames) - 1, frame_count).round().astype(int)
    return Clip(clip.frames[indices], fps)


def _parse_color(value: str) -> tuple[int, int, int]:
    try:
        return cast(tuple[int, int, int], ImageColor.getrgb(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid font color: {value}") from error


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"Must be a positive integer, got {value}")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"Must be a positive number, got {value}")
    return number


def _non_negative_float(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError(f"Must not be negative, got {value}")
    return number


class _RealDefaultsHelpFormatter(argparse.HelpFormatter):
    """Appends the default value to each option's help text.

    Unlike ``ArgumentDefaultsHelpFormatter``, required options and options defaulting
    to ``None``/``False``/``SUPPRESS`` are left alone: their help text already
    describes the fallback behaviour, and "(default: None)" is more misleading than
    informative.
    """

    def _get_help_string(self, action: argparse.Action) -> str:
        help_text = action.help or ""
        if (
            action.required
            or action.default is None
            or action.default is False
            or action.default is argparse.SUPPRESS
        ):
            return help_text
        return f"{help_text} (default: {action.default})"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stitch checkpoint rollout videos into a labelled training history."
        ),
        formatter_class=_RealDefaultsHelpFormatter,
    )
    parser.add_argument(
        "trial_dir",
        type=Path,
        help=(
            "Ray Tune trial directory holding the checkpoint_<number> subdirectories "
            "and the training metadata (result.json or progress.csv)."
        ),
    )
    parser.add_argument(
        "--num_videos",
        type=_positive_int,
        required=True,
        help=(
            "Number of checkpoints to sample into the montage. Clamped to the number "
            "of available checkpoints."
        ),
    )
    parser.add_argument(
        "--schedule",
        choices=("linear", "exponential"),
        default="linear",
        help=(
            "Checkpoint sampling schedule: 'linear' spaces samples evenly, "
            "'exponential' concentrates them on early training. Both always include "
            "the final checkpoint."
        ),
    )
    parser.add_argument(
        "--base",
        type=float,
        default=2.0,
        help=(
            "Curvature of the exponential schedule; must be greater than 1, with "
            "larger values front-loading more samples. Ignored for --schedule linear."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output video path. Defaults to <trial_dir>/training_history.mp4.",
    )
    parser.add_argument(
        "--video_name",
        default="video_000.mp4",
        help="Name of the rollout video file to read from each checkpoint directory.",
    )
    parser.add_argument(
        "--generate_missing",
        action="store_true",
        help=(
            "If a selected checkpoint has no video, generate it by running inference "
            "(run_inference.py) on that checkpoint instead of failing."
        ),
    )
    parser.add_argument(
        "--inference_episodes",
        type=_positive_int,
        default=1,
        help=(
            "Episodes to roll out when generating a missing video; the first episode's "
            "clip is used. Only applies with --generate_missing."
        ),
    )
    parser.add_argument(
        "--explore_during_inference",
        action="store_true",
        help="Use the exploration policy when generating missing videos.",
    )
    parser.add_argument(
        "--speed",
        type=_positive_float,
        default=1.0,
        help="Playback speed multiplier applied before other video transforms",
    )
    parser.add_argument(
        "--fps",
        type=_positive_float,
        help=(
            "Frame rate [frames/s] of the output video, resampling each clip. Defaults "
            "to the source videos' frame rate."
        ),
    )
    parser.add_argument(
        "--text_height_pct",
        type=_positive_float,
        default=5.0,
        help="Label font size, as a percentage of the frame height.",
    )
    parser.add_argument(
        "--margin_pct",
        type=_non_negative_float,
        default=3.0,
        help="Label inset from the frame edges, as a percentage of the frame height.",
    )
    parser.add_argument(
        "--text_location",
        choices=("bottom-left", "bottom-right", "top-left", "top-right", "center"),
        default="bottom-left",
        help="Where to draw the iteration/timestep label within the frame.",
    )
    parser.add_argument(
        "--font_color",
        type=_parse_color,
        default=(255, 255, 255),
        help="Label color, as any name or hex string accepted by PIL (e.g. '#ff0000').",
    )
    parser.add_argument(
        "--font_path",
        help=(
            "TrueType font file for the label. Defaults to PIL's built-in font, which "
            "requires Pillow 10.1+ to be scaled."
        ),
    )
    parser.add_argument(
        "--head_seconds",
        type=_non_negative_float,
        default=1.0,
        help="Duration [s] to hold each clip's first frame before it plays.",
    )
    parser.add_argument(
        "--tail_seconds",
        type=_non_negative_float,
        default=1.0,
        help="Duration [s] to hold each clip's last frame after it plays.",
    )
    parser.add_argument(
        "--transition_seconds",
        type=_non_negative_float,
        default=0.5,
        help=(
            "Crossfade duration [s] between consecutive clips. Must not exceed "
            "--head_seconds or --tail_seconds, so the blend only covers frozen frames."
        ),
    )
    return parser


def _write_clip(clip: Clip, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output), format="ffmpeg", fps=clip.fps)
    try:
        for frame in clip.frames:
            writer.append_data(frame)
    finally:
        writer.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _build_parser().parse_args()
    trial_dir = cast(Path, args.trial_dir).expanduser().resolve()

    logger.info(f"Scanning checkpoints in {trial_dir}")
    paths = select_checkpoints(
        trial_dir,
        args.num_videos,
        args.schedule,
        args.base,
        args.video_name,
        generate_missing=args.generate_missing,
        inference_episodes=args.inference_episodes,
        explore_during_inference=args.explore_during_inference,
    )
    logger.info(f"Selected {len(paths)} checkpoints using {args.schedule} sampling")
    for index, path in enumerate(paths, start=1):
        logger.info(f"Selected checkpoint {index}/{len(paths)}: {path}")

    logger.info("Resolving iteration and timestep labels from training metadata")
    labels = resolve_labels(trial_dir, paths)
    for path, (iteration, timestep) in zip(paths, labels):
        logger.info(f"Resolved {path.name}: #iter={iteration}, #step={timestep}")

    clips = []
    for index, path in enumerate(paths, start=1):
        video_path = path / args.video_name
        logger.info(f"Loading video {index}/{len(paths)}: {video_path}")
        clip = load_clip(str(video_path))
        logger.info(
            f"Loaded {len(clip.frames)} frames at {clip.fps:g} fps "
            f"with resolution {clip.frames.shape[2]}x{clip.frames.shape[1]}"
        )
        clip = change_speed(clip, args.speed)
        logger.info(
            f"Changed video {index}/{len(paths)} to {args.speed:g}x speed; "
            f"new frame count is {len(clip.frames)}"
        )
        if args.fps is not None:
            clip = _resample_clip(clip, args.fps)
            logger.info(
                f"Resampled video {index}/{len(paths)} to "
                f"{len(clip.frames)} frames at {clip.fps:g} fps"
            )
        # Validate as soon as the effective fps is known, before decoding the rest.
        # A single clip needs no validation: `crossfade` performs no transition then.
        if index == 1 and len(paths) > 1:
            _validate_transition(
                clip.fps, args.transition_seconds, args.head_seconds, args.tail_seconds
            )
        clip = extend_clip(clip, args.head_seconds, args.tail_seconds)
        logger.info(
            f"Added freeze padding to video {index}/{len(paths)}; "
            f"new frame count is {len(clip.frames)}"
        )
        iteration, timestep = labels[index - 1]
        label = f"iter={iteration}, timestep={timestep:.2e}"
        logger.info(f"Rendering label on video {index}/{len(paths)}: {label}")
        clips.append(
            add_text(
                clip,
                label,
                text_height_pct=args.text_height_pct,
                margin_pct=args.margin_pct,
                location=args.text_location,
                color=args.font_color,
                font_path=args.font_path,
            )
        )

    logger.info(
        f"Crossfading {len(clips)} clips with "
        f"{args.transition_seconds:g}-second transitions"
    )
    final = crossfade(clips, args.transition_seconds)
    output = (
        cast(Path | None, args.output) or trial_dir / "training_history.mp4"
    ).expanduser()
    logger.info(f"Encoding {len(final.frames)} frames at {final.fps:g} fps to {output}")
    _write_clip(final, output)
    logger.info(f"Finished writing training history video: {output}")


if __name__ == "__main__":
    main()
