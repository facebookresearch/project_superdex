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

"""Which recorded frame each video frame shows, at 1x real time.

Video frame i shows the recorded frame nearest to sim_time t0 + i / fps, the earlier one on
a tie: below the recorded rate some frames are skipped, above it frames repeat evenly. At
the native rate, a constant-step recording shows every recorded frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from rrd_video.recording import Clock, RecordingError

DEFAULT_FPS = 60
MIN_FPS = 1
MAX_FPS = 240
# A recording without a constant step plays at the largest of these within its median rate.
VARIABLE_RATE_FPS = (60, 50, 30, 25, 20, 10)
# A recorded rate within 0.1% of an integer plays at that integer (59.9988 Hz at 60 fps).
_INTEGER_SNAP = 1e-3
# A step is regular when it is a whole number of median steps, give or take this
# tolerance: 1 ms (sim_time rounded to milliseconds), at most a quarter step. Longer
# regular steps are pauses or dropped frames, held in the video. A recording with more than
# 1% irregular steps is variable-rate.
_STEP_TOLERANCE_NS = 1_000_000
_IRREGULAR_SHARE = 0.01
# Video times this close to halfway between two recorded frames count as a tie; they absorb
# sim_time rounded to whole nanoseconds.
_TIE_NS = 1_000


@dataclass(frozen=True)
class Schedule:
    frames: list[int]
    fps: Fraction
    native_hz: float
    uniform: bool
    irregular_steps: int
    # Recorded frames per video frame when that is a whole number, else 0.
    subsample: int
    start_ns: int
    end_ns: int

    @property
    def duration_s(self) -> float:
        return len(self.frames) / float(self.fps)


def native_period_ns(clock: Clock) -> int:
    """The recorded step: the mean of the regular steps, each a whole number of median
    steps, so that steps rounded to milliseconds (16, 17, 17 ms) give 60 Hz."""
    steps = np.diff(clock.times_ns).astype(np.float64)
    steps = steps[steps > 0]
    if len(steps) == 0:
        raise RecordingError(
            "A video needs at least two recorded frames at distinct times"
        )
    median = float(np.median(steps))
    multiples = np.maximum(1, np.round(steps / median))
    regular = np.abs(steps - multiples * median) <= _tolerance_ns(median)
    return int(round(steps[regular].sum() / multiples[regular].sum()))


def _tolerance_ns(period_ns: float) -> float:
    return min(0.25 * period_ns, max(0.01 * period_ns, _STEP_TOLERANCE_NS))


def irregular_steps(clock: Clock, period_ns: int) -> int:
    """Steps that are not a whole number of recorded steps."""
    steps = np.diff(clock.times_ns).astype(np.float64)
    multiples = np.maximum(1, np.round(steps / period_ns))
    return int(np.sum(np.abs(steps - multiples * period_ns) > _tolerance_ns(period_ns)))


def playback_rate(hz: float) -> Fraction:
    """A recorded rate as a rational frame rate, snapped to an integer when it is one in
    practice."""
    nearest = round(hz)
    if nearest > 0 and abs(hz - nearest) <= _INTEGER_SNAP * hz:
        return Fraction(nearest)
    return Fraction(hz).limit_denominator(1001)


def make_schedule(
    clock: Clock,
    start_s: float | None = None,
    end_s: float | None = None,
    fps: float | None = None,
) -> Schedule:
    """Video frames for [start_s, end_s] seconds after the first recorded frame, at `fps`.

    `fps=None` plays at the recorded rate: every k-th frame above MAX_FPS, and the largest
    of VARIABLE_RATE_FPS within the median rate for a recording without a constant step.
    """
    period = native_period_ns(clock)
    native_hz = 1e9 / period
    irregular = irregular_steps(clock, period)
    uniform = irregular <= _IRREGULAR_SHARE * (len(clock.frames) - 1)
    rate = _rate(native_hz, uniform, fps)
    ratio = native_hz / float(rate)
    whole = abs(ratio - round(ratio)) <= _INTEGER_SNAP * ratio and round(ratio) >= 1
    start, end = window_ns(clock, start_s, end_s)
    frames = _frames_on_grid(clock, start, end, rate)
    return Schedule(
        frames,
        rate,
        native_hz,
        uniform,
        irregular,
        round(ratio) if uniform and whole else 0,
        start,
        end,
    )


def _rate(native_hz: float, uniform: bool, fps: float | None) -> Fraction:
    if fps is not None:
        if not MIN_FPS <= fps <= MAX_FPS:
            raise RecordingError(f"--fps {fps} must be between {MIN_FPS} and {MAX_FPS}")
        return Fraction(fps).limit_denominator(1001)
    if not uniform:
        return Fraction(
            next(
                (f for f in VARIABLE_RATE_FPS if f <= native_hz), VARIABLE_RATE_FPS[-1]
            )
        )
    subsample = max(1, math.ceil(native_hz / MAX_FPS - _INTEGER_SNAP))
    return playback_rate(native_hz / subsample)


def window_ns(
    clock: Clock, start_s: float | None, end_s: float | None
) -> tuple[int, int]:
    """The sim_time window [start_s, end_s] seconds after the first recorded frame."""
    first, last = int(clock.times_ns[0]), int(clock.times_ns[-1])
    start = first + _seconds_ns(start_s, 0)
    end = min(last, first + _seconds_ns(end_s, last - first))
    if not first <= start <= end:
        raise RecordingError(
            f"The window {start_s}..{end_s} s lies outside the recording (0..{(last - first) / 1e9:.6g} s)"
        )
    return start, end


def window_frames(
    clock: Clock, start_s: float | None = None, end_s: float | None = None
) -> list[int]:
    """The recorded frames inside the window."""
    start, end = window_ns(clock, start_s, end_s)
    inside = (clock.times_ns >= start) & (clock.times_ns <= end)
    if not inside.any():
        first = int(clock.times_ns[0])
        raise RecordingError(
            f"No recorded frame between {(start - first) / 1e9:.6g} and {(end - first) / 1e9:.6g} s"
        )
    return [int(f) for f in clock.frames[inside]]


def _seconds_ns(seconds: float | None, default: int) -> int:
    return default if seconds is None else round(seconds * 1e9)


def _frames_on_grid(clock: Clock, start: int, end: int, rate: Fraction) -> list[int]:
    """The recorded frame nearest to each video time, the earlier one on a tie."""
    step_num, step_den = 10**9 * rate.denominator, rate.numerator  # ns per video frame
    count = (end - start + _TIE_NS) * step_den // step_num + 1
    grid = start + np.arange(count, dtype=np.int64) * step_num // step_den
    times = clock.times_ns
    after = np.clip(np.searchsorted(times, grid, side="left"), 1, len(times) - 1)
    before = after - 1
    earlier = (grid - times[before]) <= (times[after] - grid) + _TIE_NS
    index = np.where(earlier, before, after)
    return [int(clock.frames[i]) for i in index]
