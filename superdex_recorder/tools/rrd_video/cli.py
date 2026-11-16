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

"""rrd_video: 1x real-time videos of superdex_recorder recordings, rendered by Rerun."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from rrd_video import encode, viewer
from rrd_video.auto_camera import parse_up, Pick, pick_camera
from rrd_video.camera import (
    CameraError,
    RegistryEntry,
    ResolvedCamera,
    save_entry,
    task_key,
)
from rrd_video.camera_choice import CameraChooser, flag_camera, pick_json
from rrd_video.encode import EncodeError
from rrd_video.geometry import read_geometry
from rrd_video.recording import Recording, RecordingError
from rrd_video.storage import Storage, StorageError
from rrd_video.timing import (
    DEFAULT_FPS,
    make_schedule,
    MAX_FPS,
    MIN_FPS,
    Schedule,
    window_frames,
)
from rrd_video.viewer import RenderError

logger: logging.Logger = logging.getLogger(__name__)

MANIFEST_SCHEMA = "superdex.rrd_video/1"


@dataclass(frozen=True)
class Input:
    location: str
    path: Path
    remote: bool = False

    @property
    def stem(self) -> str:
        return Path(self.location.rsplit("/", 1)[-1]).stem

    def out_dir(self, out: str | None) -> str:
        """--out, else the directory holding a local input."""
        if out:
            return out
        if self.remote:
            raise RecordingError(
                f"Pass --out for {self.location}: outputs never go into a remote input's folder"
            )
        return str(Path(self.location).resolve().parent)


@dataclass
class _Context:
    args: argparse.Namespace
    work: Path
    storages: Sequence[Storage] = field(default_factory=tuple)

    def storage(self, location: str) -> Storage | None:
        """The storage for a URI such as 'scheme://bucket/key'; None for a local path."""
        if "://" not in location:
            return None
        for storage in self.storages:
            if location.startswith(storage.scheme):
                return storage
        raise StorageError(
            f"Cannot read or write {location}: copy it to a local path first"
        )

    def chooser(self) -> CameraChooser:
        args = self.args
        return CameraChooser(
            args.camera,
            args.registry,
            args.framing,
            args.up,
            args.exclude,
            args.size[0] / args.size[1],
        )


@dataclass
class _Staged:
    item: Input
    stage: Path
    names: dict[str, str]  # Output files in `stage`, the manifest last.
    summary: dict[str, Any]


def main() -> None:
    sys.exit(run(sys.argv[1:]))


def run(argv: list[str], storages: Sequence[Storage] = ()) -> int:
    """Run one command; 0 on success, 1 on an error the message explains.

    `storages` adds remote locations (e.g. an object store) to the local paths every
    command takes.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    # Render workers start after this, so Rerun's Rust logging there stays quiet.
    os.environ.setdefault("RUST_LOG", "error")
    args = _parser().parse_args(argv)
    try:
        with tempfile.TemporaryDirectory(prefix="rrd_video-") as temp:
            result = args.run(_Context(args, Path(temp), storages))
    except (
        CameraError,
        EncodeError,
        RecordingError,
        RenderError,
        StorageError,
        OSError,
    ) as e:
        logger.error("rrd_video: %s", e)
        return 1
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rrd_video", description=__doc__)
    commands = parser.add_subparsers(required=True)
    render = commands.add_parser("render", help="Render recordings to videos")
    _recordings(render)
    _outputs(render)
    _camera_source(render)
    _auto_camera(render)
    render.add_argument(
        "--fps",
        type=_fps_type,
        default=float(DEFAULT_FPS),
        help=f"Video frame rate, {MIN_FPS} to {MAX_FPS}, or 'native' for the recorded rate (default {DEFAULT_FPS})",
    )
    render.add_argument("--mp4", action="store_true", help="Also write an H.264 .mp4")
    render.set_defaults(run=_render)
    camera = commands.add_parser(
        "camera", help="Choose, inspect and save cameras"
    ).add_subparsers(required=True)
    pick = camera.add_parser(
        "pick", help="Automatic camera with stills of the top candidates"
    )
    _recordings(pick)
    _outputs(pick)
    _auto_camera(pick)
    pick.set_defaults(run=_pick)
    show = camera.add_parser(
        "show", help="The camera a render would use, its source and the task key"
    )
    _recordings(show)
    _camera_source(show)
    _auto_camera(show)
    show.set_defaults(run=_show)
    _save_parser(camera)
    return parser


def _save_parser(camera: Any) -> None:
    save = camera.add_parser(
        "save", help="Save a camera for a task in a registry directory"
    )
    save.add_argument(
        "--registry", required=True, help="Directory of saved task cameras"
    )
    task = save.add_mutually_exclusive_group(required=True)
    task.add_argument("--task", help="Task key, as `camera show` prints it")
    task.add_argument("--recording", help="A recording of the task, to read its key")
    save.add_argument(
        "--from",
        dest="source",
        required=True,
        help="Camera JSON file or 'eye=x,y,z target=x,y,z [up=x,y,z]'",
    )
    save.add_argument("--note", default="")
    save.set_defaults(run=_save)


def _recordings(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("recordings", nargs="+", help="Recordings (.rrd)")
    parser.add_argument(
        "--start", type=_finite, help="Seconds after the first recorded frame"
    )
    parser.add_argument(
        "--end", type=_finite, help="Seconds after the first recorded frame"
    )


def _outputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--out", help="Output directory (default: beside each local input)"
    )
    parser.add_argument(
        "--workers",
        type=_positive,
        help="Parallel viewers (default: one per three CPUs, memory permitting)",
    )


def _camera_source(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--camera",
        help="A camera JSON file or 'eye=x,y,z target=x,y,z [up=x,y,z]'; 'auto' always picks, ignoring --registry",
    )
    parser.add_argument(
        "--registry", help="Directory of saved task cameras to use (default: none)"
    )


def _auto_camera(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--framing",
        choices=("subject", "wide"),
        default="subject",
        help="What the automatic camera frames: the task's subject (default) or every moving robot link too",
    )
    parser.add_argument(
        "--up",
        type=_up_type,
        help="World up for automatic cameras: x, y, z, -x, -y or -z",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Entity pattern to leave out, e.g. '/world/cameras/**'",
    )
    parser.add_argument(
        "--size",
        type=_size_type,
        default=(1280, 720),
        help="Video size WIDTHxHEIGHT (default 1280x720)",
    )


def _size_type(text: str) -> tuple[int, int]:
    width, _, height = text.lower().partition("x")
    if (
        not (width.isdigit() and height.isdigit())
        or int(width) < 2
        or int(height) < 2
        or int(width) % 2
        or int(height) % 2
    ):
        raise argparse.ArgumentTypeError(
            f"expected two positive even numbers like 1280x720, got {text!r}"
        )
    return int(width), int(height)


def _finite(text: str) -> float:
    try:
        value = float(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"expected a number, got {text!r}") from e
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"expected a finite number, got {text!r}")
    return value


def _positive(text: str) -> int:
    if not text.isdigit() or int(text) < 1:
        raise argparse.ArgumentTypeError(
            f"expected a positive whole number, got {text!r}"
        )
    return int(text)


def _fps_type(text: str) -> float | None:
    """A frame rate, or None for 'native'; make_schedule checks its range."""
    return None if text == "native" else _finite(text)


def _up_type(text: str) -> np.ndarray:
    try:
        return parse_up(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def _fetch(context: _Context, locations: list[str], writes: bool = True) -> list[Input]:
    """Local copies of the inputs, sorted by location; refuses two that would write the same
    outputs and output directories that cannot be written."""
    inputs = [
        Input(loc, Path(loc), context.storage(loc) is not None) for loc in locations
    ]
    inputs.sort(key=lambda i: i.location if i.remote else str(i.path.resolve()))
    targets = [(i.out_dir(context.args.out), i.stem) for i in inputs] if writes else []
    for target in targets:
        if targets.count(target) > 1:
            raise RecordingError(f"Two inputs would write {target[0]}/{target[1]}.*")
    for directory in {Path(d) for d, _ in targets if context.storage(d) is None}:
        existing = next(p for p in [directory, *directory.parents] if p.exists())
        if not existing.is_dir() or not os.access(existing, os.W_OK):
            raise RecordingError(f"Cannot write into {directory}")
    for index, item in enumerate(inputs):
        storage = context.storage(item.location)
        if storage is not None:
            name = item.location.rsplit("/", 1)[-1]
            path = storage.download(
                item.location, context.work / "inputs" / str(index) / name
            )
            inputs[index] = Input(item.location, path, remote=True)
        elif not item.path.is_file():
            raise RecordingError(f"No such recording: {item.location}")
    return inputs


def _render(context: _Context) -> dict[str, Any]:
    """Render every input, then publish them all: a failure before publishing writes nothing."""
    args, work = context.args, context.work
    inputs = _fetch(context, args.recordings)
    for item in inputs:  # Refuse a bad input before rendering any.
        clock = Recording(item.path).clock
        make_schedule(clock, args.start, args.end, args.fps)
        window_frames(clock, args.start, args.end)
    tools = _Tools(
        viewer.find_toolchain(work / "viewer"), encode.find_ffmpeg(work / "viewer")
    )
    chooser = context.chooser()
    staged = [
        _render_one(args, tools, chooser, item, work / "out" / str(index))
        for index, item in enumerate(inputs)
    ]
    return {"videos": _publish_all(context, staged)}


@dataclass(frozen=True)
class _Tools:
    viewer: viewer.Toolchain
    ffmpeg: Path

    def versions(self) -> dict[str, str]:
        return {
            "rerun": viewer.rerun_version(self.viewer),
            "vulkan_driver": viewer.driver_name(self.viewer),
            "ffmpeg": encode.ffmpeg_version(self.ffmpeg),
        }


def _render_one(
    args: argparse.Namespace,
    tools: _Tools,
    chooser: CameraChooser,
    item: Input,
    stage: Path,
) -> _Staged:
    """The video, its stills and its manifest, in `stage`."""
    recording = Recording(item.path)
    schedule = make_schedule(recording.clock, args.start, args.end, args.fps)
    window = window_frames(recording.clock, args.start, args.end)
    camera, checks = chooser.choose(item.location, recording, window)
    logger.info(
        "%s: %d frames at %s fps, camera from %s",
        item.location,
        len(schedule.frames),
        schedule.fps,
        camera.source,
    )
    stage.mkdir(parents=True)
    timing = _render_frames(
        args, tools.viewer, item, recording, schedule, camera, stage
    )
    names = _encode(args, tools.ffmpeg, item, schedule, stage, timing)
    manifest = _manifest(args, item, recording, schedule, camera, names)
    manifest.update(timing=timing, checks=checks, toolchain=tools.versions())
    (stage / names["manifest"]).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    for warning in checks["warnings"]:
        logger.warning("%s: %s", item.location, warning)
    summary = {
        "input": item.location,
        "camera_source": camera.source,
        "task_key": camera.task_key,
        "warnings": checks["warnings"],
    }
    return _Staged(item, stage, names, summary)


def _render_frames(
    args: argparse.Namespace,
    toolchain: viewer.Toolchain,
    item: Input,
    recording: Recording,
    schedule: Schedule,
    camera: ResolvedCamera,
    stage: Path,
) -> dict[str, Any]:
    """Each distinct recorded frame once, then one numbered file per video frame."""
    width, height = args.size
    rendered, frames = stage / "rendered", stage / "frames"
    rendered.mkdir()
    frames.mkdir()
    blueprint = stage / "view.rbl"
    viewer.write_blueprint(
        blueprint,
        recording.application_id,
        camera.pose,
        viewer.contents_for(args.exclude),
    )
    jobs = [
        viewer.RenderJob(f, blueprint, rendered / f"{f}.png", width, height)
        for f in sorted(set(schedule.frames))
    ]
    workers = min(
        len(jobs), args.workers or viewer.default_workers(item.path.stat().st_size)
    )
    started = time.monotonic()
    viewer.render_jobs(item.path, toolchain, jobs, workers)
    elapsed = max(time.monotonic() - started, 1e-6)
    for index, frame in enumerate(schedule.frames):
        _link_or_copy(rendered / f"{frame}.png", frames / f"{index:06d}.png")
    logger.info(
        "Rendered %d frames in %.1f s (%.1f frames/s, %d viewers)",
        len(jobs),
        elapsed,
        len(jobs) / elapsed,
        workers,
    )
    return {
        "rendered_frames": len(jobs),
        "render_s": round(elapsed, 2),
        "frames_per_s": round(len(jobs) / elapsed, 2),
        "viewers": workers,
    }


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def _encode(
    args: argparse.Namespace,
    ffmpeg: Path,
    item: Input,
    schedule: Schedule,
    stage: Path,
    timing: dict[str, Any],
) -> dict[str, str]:
    """The WebM (and MP4), decoded and checked, plus first and last frames; the manifest
    name comes last."""
    names = {
        "webm": f"{item.stem}.webm",
        "first": f"{item.stem}.first.png",
        "last": f"{item.stem}.last.png",
    }
    pattern, frames, threads = (
        str(stage / "frames" / "%06d.png"),
        len(schedule.frames),
        min(16, timing["viewers"]),
    )
    started = time.monotonic()
    encode.encode_webm(ffmpeg, pattern, schedule.fps, stage / names["webm"], threads)
    encode.check(
        encode.probe(ffmpeg, stage / names["webm"]), frames, "vp9", schedule.fps
    )
    if args.mp4:
        names["mp4"] = f"{item.stem}.mp4"
        encode.encode_mp4(ffmpeg, pattern, schedule.fps, stage / names["mp4"], threads)
        encode.check(
            encode.probe(ffmpeg, stage / names["mp4"]), frames, "h264", schedule.fps
        )
    timing["encode_s"] = round(time.monotonic() - started, 2)
    logger.info("Encoded and checked %s in %.1f s", item.stem, timing["encode_s"])
    shutil.copyfile(stage / "frames" / "000000.png", stage / names["first"])
    shutil.copyfile(stage / "frames" / f"{frames - 1:06d}.png", stage / names["last"])
    return {**names, "manifest": f"{item.stem}.video.json"}


def _manifest(
    args: argparse.Namespace,
    item: Input,
    recording: Recording,
    schedule: Schedule,
    camera: ResolvedCamera,
    names: dict[str, str],
) -> dict[str, Any]:
    up = None if args.up is None else [float(x) for x in args.up]
    return {
        "schema": MANIFEST_SCHEMA,
        "input": _input_section(item, recording),
        "clock": _clock_section(recording, schedule),
        "video": _video_section(args, recording, schedule, names),
        "camera": {
            "source": camera.source,
            "task_key": camera.task_key,
            **camera.pose.to_json(),
            "details": camera.details,
        },
        "options": {
            "fps": "native" if args.fps is None else args.fps,
            "framing": args.framing,
            "exclude": args.exclude,
            "up": up,
            "camera": args.camera,
        },
    }


def _input_section(item: Input, recording: Recording) -> dict[str, Any]:
    with item.path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    return {
        "location": item.location,
        "sha256": digest,
        "bytes": item.path.stat().st_size,
        "application_id": recording.application_id,
        "recording_id": recording.recording_id,
        "properties": recording.properties,
    }


def _clock_section(recording: Recording, schedule: Schedule) -> dict[str, Any]:
    clock = recording.clock
    return {
        "recorded_frames": len(clock.frames),
        "native_hz": round(schedule.native_hz, 6),
        "uniform_step": schedule.uniform,
        "irregular_steps": schedule.irregular_steps,
        "first_sim_time_s": int(clock.times_ns[0]) / 1e9,
        "last_sim_time_s": int(clock.times_ns[-1]) / 1e9,
    }


def _video_section(
    args: argparse.Namespace,
    recording: Recording,
    schedule: Schedule,
    names: dict[str, str],
) -> dict[str, Any]:
    first_ns = int(recording.clock.times_ns[0])
    return {
        "files": names,
        "fps": f"{schedule.fps.numerator}/{schedule.fps.denominator}",
        "frames": len(schedule.frames),
        "duration_s": round(schedule.duration_s, 6),
        "subsample": schedule.subsample,
        "start_s": (schedule.start_ns - first_ns) / 1e9,
        "end_s": (schedule.end_ns - first_ns) / 1e9,
        "width": args.size[0],
        "height": args.size[1],
        "real_time_factor": 1.0,
    }


def _publish_all(context: _Context, staged: list[_Staged]) -> list[dict[str, Any]]:
    """Copy every staged output to its directory, manifests last.

    Local outputs are first copied next to their targets under temporary names, then renamed
    into place, so a failed copy leaves nothing behind. Remote outputs are uploaded in order.
    """
    plan = [
        (s, name, s.item.out_dir(context.args.out))
        for s in staged
        for name in s.names.values()
    ]
    plan.sort(key=lambda p: p[1].endswith(".video.json"))
    local = [
        (s.stage / name, Path(directory) / name)
        for s, name, directory in plan
        if context.storage(directory) is None
    ]
    for _, target in local:
        if target.is_dir():
            raise RecordingError(f"Cannot write {target}: it is a directory")
        target.parent.mkdir(parents=True, exist_ok=True)
    partials = []
    try:
        for source, target in local:
            partial = target.with_name(f".{target.name}.partial")
            partials.append(partial)
            shutil.copyfile(source, partial)
    except OSError:
        for partial in partials:
            partial.unlink(missing_ok=True)
        raise
    for partial, (_, target) in zip(partials, local):
        os.replace(partial, target)
    published: dict[int, list[str]] = {}
    for s, name, directory in plan:
        storage = context.storage(directory)
        location = (
            directory.rstrip("/") + "/" + name
            if storage
            else str(Path(directory) / name)
        )
        if storage is not None:
            storage.upload(s.stage / name, location)
        published.setdefault(id(s), []).append(location)
    return [{**s.summary, "outputs": published[id(s)]} for s in staged]


def _pick(context: _Context) -> dict[str, Any]:
    """Automatic cameras with candidate stills; publishes only when every input worked."""
    args, work = context.args, context.work
    toolchain = viewer.find_toolchain(work / "viewer")
    staged = []
    for index, item in enumerate(_fetch(context, args.recordings)):
        recording = Recording(item.path)
        window = window_frames(recording.clock, args.start, args.end)
        geometry = read_geometry(recording).without(args.exclude)
        pick = pick_camera(
            recording,
            geometry,
            window,
            args.size[0] / args.size[1],
            args.framing,
            args.up,
        )
        stage = work / "out" / str(index)
        stage.mkdir(parents=True)
        names = _candidate_stills(args, toolchain, item, recording, pick, stage)
        names["camera"] = f"{item.stem}.camera.json"
        document = {
            "schema": "superdex.rrd_video.pick/1",
            "task_key": task_key(recording),
            "camera": pick.best.pose.to_json(),
            "pick": pick_json(pick, item.location),
        }
        (stage / names["camera"]).write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n"
        )
        staged.append(
            _Staged(
                item,
                stage,
                names,
                {"input": item.location, "task_key": document["task_key"]},
            )
        )
    return {"picks": _publish_all(context, staged)}


def _candidate_stills(
    args: argparse.Namespace,
    toolchain: viewer.Toolchain,
    item: Input,
    recording: Recording,
    pick: Pick,
    stage: Path,
) -> dict[str, str]:
    """One Rerun still per top candidate, at the middle keyframe."""
    frame = pick.keyframes[len(pick.keyframes) // 2]
    jobs, names = [], {}
    for rank, candidate in enumerate(pick.alternatives(), start=1):
        blueprint = stage / f"candidate_{rank}.rbl"
        viewer.write_blueprint(
            blueprint,
            recording.application_id,
            candidate.pose,
            viewer.contents_for(args.exclude),
        )
        names[f"candidate_{rank}"] = f"{item.stem}.camera_{rank}.png"
        jobs.append(
            viewer.RenderJob(
                frame, blueprint, stage / names[f"candidate_{rank}"], *args.size
            )
        )
    viewer.render_jobs(item.path, toolchain, jobs, args.workers or len(jobs))
    return names


def _show(context: _Context) -> dict[str, Any]:
    args = context.args
    chooser = context.chooser()
    shown = []
    for item in _fetch(context, args.recordings, writes=False):
        recording = Recording(item.path)
        window = window_frames(recording.clock, args.start, args.end)
        camera, _ = chooser.choose(item.location, recording, window)
        shown.append(
            {
                "input": item.location,
                "task_key": camera.task_key,
                "source": camera.source,
                "camera": camera.pose.to_json(),
            }
        )
    return {"cameras": shown}


def _save(context: _Context) -> dict[str, Any]:
    args = context.args
    key = args.task or task_key(
        Recording(_fetch(context, [args.recording], writes=False)[0].path)
    )
    pose = flag_camera(args.source)
    if pose is None:
        raise CameraError("--from must name a camera, not 'auto'")
    origin = (
        f"from {Path(args.source).name}" if Path(args.source).is_file() else "manual"
    )
    path = save_entry(Path(args.registry), RegistryEntry(key, pose, origin, args.note))
    return {"saved": str(path), "task_key": key}
