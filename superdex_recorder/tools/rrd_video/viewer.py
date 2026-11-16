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

"""Headless Rerun renders, one recorded frame per viewer launch.

The viewer uses the system's graphics driver. RRD_VIDEO_LAVAPIPE points it at a Mesa
lavapipe install instead: CPU Vulkan, for machines without a GPU or a display.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from importlib import resources
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import rerun.blueprint as rrb  # @manual
from PIL import Image
from rrd_video.camera import CameraPose
from rrd_video.executables import ensure_executable
from rrd_video.recording import FRAME, Recording

RERUN_VERSION_PREFIX = "rerun-cli 0.33."
BACKGROUND: tuple[int, int, int] = (235, 237, 241)
# Rerun draws a view tab of 24 logical pixels above the 3D view; captures are 2x HiDPI.
TAB_HEIGHT = 24
HIDPI = 2
RENDER_TIMEOUT_S = 120.0
RENDER_ATTEMPTS = 3
WORLD_CONTENTS = ["/world/**"]
VIEWER_THREADS = 2
VIEWER_MEMORY_BYTES = 1_500_000_000  # 1.5 GB, a headless viewer's peak on large scenes
# A worker holds the recording in memory, about this many times its file size.
WORKER_MEMORY_PER_FILE_BYTE = 6


class RenderError(RuntimeError):
    pass


@dataclass(frozen=True)
class Lavapipe:
    """A Mesa lavapipe install: its driver library directory and ICD file."""

    lib_dir: Path
    icd: Path


@dataclass(frozen=True)
class Toolchain:
    rerun_cli: Path
    work_dir: Path
    lavapipe: Lavapipe | None = None  # None for the system graphics driver.


def find_toolchain(work_dir: Path) -> Toolchain:
    """The Rerun 0.33 CLI of the rerun-sdk package (RRD_VIDEO_RERUN_CLI overrides it), and
    lavapipe when RRD_VIDEO_LAVAPIPE names an install."""
    cli = Path(os.environ.get("RRD_VIDEO_RERUN_CLI") or _bundled_rerun_cli())
    if not cli.is_file():
        raise RenderError(f"No Rerun CLI at {cli}")
    work_dir.mkdir(parents=True, exist_ok=True)
    lavapipe_dir = os.environ.get("RRD_VIDEO_LAVAPIPE")
    lavapipe = _find_lavapipe(Path(lavapipe_dir).resolve()) if lavapipe_dir else None
    toolchain = Toolchain(
        ensure_executable(cli.resolve(), work_dir), work_dir, lavapipe
    )
    version = rerun_version(toolchain)
    if not version.startswith(RERUN_VERSION_PREFIX):
        raise RenderError(f"Expected {RERUN_VERSION_PREFIX}x, found {version!r}")
    (work_dir / "config").mkdir(exist_ok=True)
    _run_cli(toolchain, ["analytics", "disable"])
    return toolchain


def _find_lavapipe(root: Path) -> Lavapipe:
    """The driver library (lib/ or lib/<triplet>/) and ICD file of a lavapipe install."""
    libraries = sorted(root.glob("lib*/libvulkan_lvp.so")) + sorted(
        root.glob("lib*/*/libvulkan_lvp.so")
    )
    icds = sorted(root.glob("share/vulkan/icd.d/lvp_icd*.json"))
    if not libraries or not icds:
        raise RenderError(
            f"RRD_VIDEO_LAVAPIPE={root} holds no lib/**/libvulkan_lvp.so and share/vulkan/icd.d/lvp_icd*.json"
        )
    return Lavapipe(libraries[0].parent, icds[0])


def _bundled_rerun_cli() -> Path:
    package = resources.files("rerun_cli")
    name = "rerun.exe" if package.joinpath("rerun.exe").is_file() else "rerun"
    with resources.as_file(package.joinpath(name)) as path:
        return Path(path)


def _run_cli(toolchain: Toolchain, arguments: list[str]) -> str:
    try:
        result = subprocess.run(
            [str(toolchain.rerun_cli), *arguments],
            capture_output=True,
            text=True,
            timeout=60,
            env=viewer_env(toolchain),
            check=True,
        )
    except (subprocess.SubprocessError, OSError) as e:
        raise RenderError(f"rerun {' '.join(arguments)} failed: {e}") from e
    return result.stdout


def rerun_version(toolchain: Toolchain) -> str:
    lines = _run_cli(toolchain, ["--version"]).splitlines()
    return lines[0].strip() if lines else ""


def driver_name(toolchain: Toolchain) -> str:
    if toolchain.lavapipe is None:
        return "system"
    return f"lavapipe (Vulkan {json.loads(toolchain.lavapipe.icd.read_text())['ICD']['api_version']})"


def viewer_env(toolchain: Toolchain) -> dict[str, str]:
    """The viewer's environment: a private config with analytics off, and lavapipe if set."""
    env = dict(os.environ)
    env.update(
        {
            "XDG_CONFIG_HOME": str(toolchain.work_dir / "config"),
            "XDG_DATA_HOME": str(toolchain.work_dir / "data"),
            "MESA_SHADER_CACHE_DIR": str(toolchain.work_dir / "mesa_cache"),
            "LP_NUM_THREADS": str(VIEWER_THREADS),
            "RERUN_APP_ONLY": "true",
            "RUST_LOG": "error",
        }
    )
    lavapipe = toolchain.lavapipe
    if lavapipe is None:
        return env
    library_path = [str(lavapipe.lib_dir), env.get("LD_LIBRARY_PATH", "")]
    env.update(
        {
            "LD_LIBRARY_PATH": ":".join(p for p in library_path if p),
            "VK_ICD_FILENAMES": str(lavapipe.icd),
            "VK_DRIVER_FILES": str(lavapipe.icd),
            "VK_LOADER_LAYERS_DISABLE": "~implicit~",
            "VK_LOADER_DISABLE_DYNAMIC_LIBRARY_UNLOADING": "1",
            "WGPU_BACKEND": "vulkan",
        }
    )
    for name in ("DISPLAY", "WAYLAND_DISPLAY"):
        env.pop(name, None)
    return env


def contents_for(excludes: list[str]) -> list[str]:
    return WORLD_CONTENTS + [f"- {pattern}" for pattern in excludes]


def write_blueprint(
    path: Path,
    application_id: str,
    camera: CameraPose,
    contents: list[str] | None = None,
) -> None:
    """One 3D view at `camera`, with the time cursor following the newest frame."""
    view = rrb.Spatial3DView(
        origin="/",
        contents=contents or WORLD_CONTENTS,
        name="rrd_video",
        background=list(BACKGROUND),
        line_grid=False,
        eye_controls=rrb.EyeControls3D(
            position=list(camera.eye),
            look_target=list(camera.target),
            eye_up=list(camera.up),
        ),
    )
    blueprint = rrb.Blueprint(
        view,
        rrb.TimePanel(timeline=FRAME, play_state="Following", state="hidden"),
        rrb.BlueprintPanel(state="hidden"),
        rrb.SelectionPanel(state="hidden"),
        rrb.TopPanel(state="hidden"),
        auto_views=False,
        auto_layout=False,
    )
    blueprint.save(application_id, path)


@dataclass(frozen=True)
class RenderJob:
    frame: int
    blueprint: Path
    output: Path
    width: int
    height: int


def render_capture(
    toolchain: Toolchain,
    recording: Path,
    blueprint: Path,
    output: Path,
    width: int,
    height: int,
) -> None:
    """One headless screenshot of the whole viewer window."""
    renderer = ["--renderer", "vulkan"] if toolchain.lavapipe else []
    command = [
        str(toolchain.rerun_cli),
        "--headless",
        *renderer,
        "--threads", str(VIEWER_THREADS),
        "--bind", "127.0.0.1",
        "--port", "0",
        "--window-size", f"{width}x{height + TAB_HEIGHT}",
        "--memory-limit", "4GB",
        "--screenshot-to", str(output),
        str(recording),
        str(blueprint),
    ]  # fmt: skip
    try:
        result = subprocess.run(
            command,
            env=viewer_env(toolchain),
            capture_output=True,
            text=True,
            timeout=RENDER_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as e:
        raise RenderError(
            f"Rerun did not finish within {RENDER_TIMEOUT_S:.0f} s"
        ) from e
    if result.returncode != 0 or not _is_png(output):
        raise RenderError(
            f"Rerun exited {result.returncode}: {_cause(result.stderr, toolchain)}"
        )


def _cause(stderr: str, toolchain: Toolchain) -> str:
    """The line of Rerun's output that says why it failed."""
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    panics = [
        line for line in lines if "panicked at" in line or line.startswith("Error")
    ]
    cause = (panics or lines or ["no output"])[-1]
    if "render context" in cause and toolchain.lavapipe is None:
        cause += (
            " (no usable graphics driver: install lavapipe or set RRD_VIDEO_LAVAPIPE)"
        )
    return cause


def crop_view(capture: Path, width: int, height: int) -> Image.Image:
    """The 3D view of a capture, without its tab, downsampled from HiDPI."""
    try:
        image = Image.open(capture).convert("RGB")
    except OSError as e:
        raise RenderError(f"Unreadable capture: {e}") from e
    expected = (width * HIDPI, (height + TAB_HEIGHT) * HIDPI)
    if image.size != expected:
        raise RenderError(f"Capture is {image.size}, expected {expected}")
    view = image.crop((0, TAB_HEIGHT * HIDPI, expected[0], expected[1]))
    view = view.resize((width, height), Image.Resampling.LANCZOS)
    if _is_blank(view):
        raise RenderError(
            "The view shows nothing: the camera misses the scene, or the capture ran before it loaded"
        )
    return view


def _is_blank(image: Image.Image) -> bool:
    pixels = np.asarray(image, dtype=np.int16)
    return bool(np.all(np.abs(pixels - np.array(BACKGROUND, dtype=np.int16)) <= 3))


def _is_png(path: Path) -> bool:
    return path.is_file() and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


_worker: dict[str, Any] = {}


def _init_worker(recording_path: Path, toolchain: Toolchain) -> None:
    _worker["recording"] = Recording(recording_path)
    _worker["toolchain"] = toolchain
    _worker["dir"] = Path(tempfile.mkdtemp(dir=toolchain.work_dir))


def _run_job(job: RenderJob) -> int:
    recording: Recording = _worker["recording"]
    scratch: Path = _worker["dir"]
    cut, capture = scratch / "frame.rrd", scratch / "capture.png"
    recording.write_at(job.frame, cut)
    error: RenderError | None = None
    for _ in range(RENDER_ATTEMPTS):
        capture.unlink(missing_ok=True)
        try:
            render_capture(
                _worker["toolchain"], cut, job.blueprint, capture, job.width, job.height
            )
            crop_view(capture, job.width, job.height).save(job.output)
            return job.frame
        except RenderError as e:
            error = e
    raise RenderError(f"Frame {job.frame} failed {RENDER_ATTEMPTS} times: {error}")


def render_jobs(
    recording_path: Path, toolchain: Toolchain, jobs: list[RenderJob], workers: int
) -> None:
    """Render every job, each worker reading the recording once; stop at the first failure."""
    workers = max(1, min(workers, len(jobs)))
    pool = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
        initializer=_init_worker,
        initargs=(recording_path, toolchain),
    )
    try:
        done, _ = wait(
            [pool.submit(_run_job, job) for job in jobs], return_when=FIRST_EXCEPTION
        )
        for future in done:
            future.result()  # Raises the first failure.
    except BrokenProcessPool as e:
        raise RenderError(f"A render worker died (out of memory?): {e}") from e
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def default_workers(recording_bytes: int) -> int:
    """One viewer per three CPUs, within 60% of the memory this process may use."""
    cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count() or 1
    )
    by_cpu = max(1, cpus // (VIEWER_THREADS + 1))
    per_worker = VIEWER_MEMORY_BYTES + WORKER_MEMORY_PER_FILE_BYTE * recording_bytes
    by_memory = max(1, int(0.6 * _memory_limit() / per_worker))
    return min(by_cpu, by_memory)


def _memory_limit() -> int:
    """RRD_VIDEO_MEMORY_BYTES, else the cgroup limit, else the available memory."""
    override = os.environ.get("RRD_VIDEO_MEMORY_BYTES", "")
    if override.isdigit():
        return int(override)
    cgroup = Path("/sys/fs/cgroup/memory.max")
    if cgroup.is_file() and cgroup.read_text().strip().isdigit():
        return int(cgroup.read_text())
    meminfo = Path("/proc/meminfo")
    for line in meminfo.read_text().splitlines() if meminfo.is_file() else []:
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 8_000_000_000
