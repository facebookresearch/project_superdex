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

"""Browser-playable video from rendered frames, checked after encoding."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from importlib import resources
from pathlib import Path

import imageio_ffmpeg
from rrd_video.executables import ensure_executable

ENCODE_TIMEOUT_S = 3600.0
# Renders are sRGB: convert and tag as BT.709, limited range, as browsers expect.
_COLOR = [
    "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,setparams=range=tv:colorspace=bt709:color_primaries=bt709:color_trc=bt709",
    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
]  # fmt: skip


class EncodeError(RuntimeError):
    pass


@dataclass(frozen=True)
class Encoded:
    path: Path
    frames: int
    codec: str
    color: str  # e.g. "tv, bt709, progressive"
    fps: float


def find_ffmpeg(work_dir: Path) -> Path:
    """The static ffmpeg of the imageio-ffmpeg package (RRD_VIDEO_FFMPEG overrides it)."""
    override = os.environ.get("RRD_VIDEO_FFMPEG")
    return ensure_executable(
        Path(override) if override else _bundled_ffmpeg(), work_dir
    )


def _bundled_ffmpeg() -> Path:
    # imageio_ffmpeg.get_ffmpeg_exe() prefers any `ffmpeg` on PATH when the packaged binary
    # is not executable, and a system ffmpeg may lack libvpx-vp9.
    binaries = resources.files("imageio_ffmpeg.binaries")
    names = sorted(p.name for p in binaries.iterdir() if p.name.startswith("ffmpeg-"))
    if not names:
        return Path(imageio_ffmpeg.get_ffmpeg_exe())
    with resources.as_file(binaries.joinpath(names[-1])) as path:
        return Path(path)


def ffmpeg_version(ffmpeg: Path) -> str:
    return _run(ffmpeg, ["-version"], 60).splitlines()[0]


def _rate(fps: Fraction) -> str:
    return f"{fps.numerator}/{fps.denominator}"


def encode_webm(
    ffmpeg: Path, frames_pattern: str, fps: Fraction, output: Path, threads: int
) -> None:
    """VP9 yuv420p, BT.709 limited range: the format every browser plays."""
    _run(
        ffmpeg,
        [
            "-y", "-framerate", _rate(fps), "-i", frames_pattern, *_COLOR,
            "-c:v", "libvpx-vp9", "-crf", "28", "-b:v", "0", "-row-mt", "1",
            "-deadline", "good", "-cpu-used", "4", "-tile-columns", "2", "-threads", str(threads),
            "-r", _rate(fps), str(output),
        ],
        ENCODE_TIMEOUT_S,
    )  # fmt: skip


def encode_mp4(
    ffmpeg: Path, frames_pattern: str, fps: Fraction, output: Path, threads: int
) -> None:
    _run(
        ffmpeg,
        [
            "-y", "-framerate", _rate(fps), "-i", frames_pattern, *_COLOR,
            "-c:v", "libx264", "-crf", "18", "-preset", "slow", "-threads", str(threads),
            "-movflags", "+faststart", "-r", _rate(fps), str(output),
        ],
        ENCODE_TIMEOUT_S,
    )  # fmt: skip


def _run(ffmpeg: Path, arguments: list[str], timeout: float) -> str:
    """stdout of a quiet ffmpeg run; stderr goes into the error."""
    command = [str(ffmpeg), "-hide_banner", "-loglevel", "error", *arguments]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, OSError) as e:
        raise EncodeError(f"ffmpeg failed: {e}") from e
    if result.returncode != 0:
        raise EncodeError(f"ffmpeg failed: {_last_line(result.stderr)}")
    return result.stdout


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "no output"


def probe(ffmpeg: Path, path: Path) -> Encoded:
    """Decode the whole file: frame count, codec, color and fps as a player sees them."""
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=ENCODE_TIMEOUT_S,
            stdin=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, OSError) as e:
        raise EncodeError(f"Cannot decode {path}: {e}") from e
    if result.returncode != 0:
        raise EncodeError(f"Cannot decode {path}: {_last_line(result.stderr)}")
    stream = re.search(
        r"Stream #0:0[^:\n]*: Video: (\w+).*?, \w+\(([^)]*)\).*?, ([\d.]+) fps",
        result.stderr,
    )
    frames = re.findall(r"frame=\s*(\d+)", result.stderr)
    if stream is None or not frames:
        raise EncodeError(f"Cannot read the video stream of {path}")
    return Encoded(
        path, int(frames[-1]), stream.group(1), stream.group(2), float(stream.group(3))
    )


def check(encoded: Encoded, frames: int, codec: str, fps: Fraction) -> None:
    """Frame count, codec, frame rate, and BT.709 limited range as a player sees them."""
    problems = []
    if encoded.frames != frames:
        problems.append(f"{encoded.frames} frames, expected {frames}")
    if encoded.codec != codec:
        problems.append(f"codec {encoded.codec}, expected {codec}")
    if abs(encoded.fps - float(fps)) > 0.01 * float(fps):
        problems.append(f"{encoded.fps} fps, expected {float(fps):.6g}")
    tags = [t.strip() for t in encoded.color.split(",")]
    if "tv" not in tags or "bt709" not in tags:
        problems.append(f"color {encoded.color!r}, expected tv and bt709")
    if problems:
        raise EncodeError(f"{encoded.path}: " + "; ".join(problems))
