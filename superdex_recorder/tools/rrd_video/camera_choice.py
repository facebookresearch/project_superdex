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

"""Which camera each video uses, and where its subject leaves the view."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from rrd_video.auto_camera import (
    find_subject,
    keyframes_of,
    Pick,
    pick_camera,
    Subject,
    subject_in_view,
)
from rrd_video.camera import (
    CameraError,
    CameraPose,
    load_registry,
    ResolvedCamera,
    task_key,
)
from rrd_video.geometry import read_geometry, SceneGeometry
from rrd_video.recording import Recording

IN_VIEW_WARNING = 0.98
IN_VIEW_SAMPLES = 600
# A task's shared automatic camera is used only while it keeps this share of the subject.
SHARED_IN_VIEW_MIN = 0.9


class CameraChooser:
    """Cameras from `--camera`, else a saved camera in `--registry`, else an automatic pick.

    `--camera auto` always picks. Recordings of one named task share the first recording's
    pick, unless it loses this recording's subject; keys made of actor names, or unknown, are
    never shared.
    """

    def __init__(
        self,
        camera: str | None,
        registry: str | None,
        framing: str,
        up: np.ndarray | None,
        excludes: list[str],
        aspect: float,
    ) -> None:
        self.flag = flag_camera(camera)
        self.force_auto = camera == "auto"
        use_registry = registry and self.flag is None and not self.force_auto
        self.registry = load_registry(Path(registry)) if use_registry else {}
        self.framing = framing
        self.up = up
        self.excludes = excludes
        self.aspect = aspect
        self.picks: dict[str, tuple[Pick, str]] = {}

    def choose(
        self, location: str, recording: Recording, frames: list[int]
    ) -> tuple[ResolvedCamera, dict[str, Any]]:
        """The camera for the recorded `frames`, and where its subject leaves the view."""
        key = task_key(recording)
        geometry = read_geometry(recording).without(self.excludes)
        entry = self.registry.get(key)
        fixed = (
            ResolvedCamera(self.flag, "flag", key)
            if self.flag is not None
            else ResolvedCamera(entry.pose, f"registry ({entry.origin})", key)
            if entry is not None
            else None
        )
        if fixed is not None:
            return fixed, self._checks(recording, geometry, frames, fixed.pose, None)
        shared = key in self.picks and not self.force_auto
        if shared:
            pick, picked_from = self.picks[key]
            checks = self._checks(recording, geometry, frames, pick.best.pose, None)
            if (checks["subject_in_view_min"] or 0.0) >= SHARED_IN_VIEW_MIN:
                return ResolvedCamera(
                    pick.best.pose, "auto", key, pick_json(pick, picked_from)
                ), _with_up_warning(checks, pick)
        pick = pick_camera(
            recording, geometry, frames, self.aspect, self.framing, self.up
        )
        if not shared and not key.startswith("actors=") and key != "unknown":
            self.picks[key] = (pick, location)
        source = (
            "auto (own pick: the task's shared camera lost this subject)"
            if shared
            else "auto"
        )
        checks = self._checks(recording, geometry, frames, pick.best.pose, pick.subject)
        return ResolvedCamera(
            pick.best.pose, source, key, pick_json(pick, location)
        ), _with_up_warning(checks, pick)

    def _checks(
        self,
        recording: Recording,
        geometry: SceneGeometry,
        frames: list[int],
        pose: CameraPose,
        subject: Subject | None,
    ) -> dict[str, Any]:
        try:
            if subject is None:
                subject = find_subject(
                    recording, geometry, keyframes_of(frames, 12), self.framing, self.up
                )
        except CameraError as e:
            return {
                "subject_in_view_min": None,
                "frames_checked": 0,
                "warnings": [f"No in-view check: {e}"],
            }
        return view_checks(recording, geometry, subject, pose, frames, self.aspect)


def _with_up_warning(checks: dict[str, Any], pick: Pick) -> dict[str, Any]:
    if pick.subject.up_source != "assumed +Z":
        return checks
    warning = "Up axis assumed +Z (no gravity or flat static body recorded): pass --up or --camera if the view looks rolled"
    return {**checks, "warnings": [*checks["warnings"], warning]}


def flag_camera(spec: str | None) -> CameraPose | None:
    """The pose `--camera` names: a JSON file or 'eye=... target=... [up=...]'; None for auto."""
    if spec is None or spec == "auto":
        return None
    path = Path(spec)
    if path.suffix == ".json" and not path.is_file():
        raise CameraError(f"No camera file {spec}")
    if not path.is_file():
        return CameraPose.parse(spec)
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise CameraError(f"{spec} is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise CameraError(f"{spec} must hold a JSON object")
    return CameraPose.from_json(data.get("camera", data))


def pick_json(pick: Pick, picked_from: str) -> dict[str, Any]:
    subject = pick.subject
    return {
        "picked_from": picked_from,
        "up": [float(x) for x in subject.up],
        "up_source": subject.up_source,
        "robots": subject.robots,
        "robot_confidence": subject.robot_confidence,
        "subject": [{"entity": p, "role": subject.roles[p]} for p in subject.paths],
        "keyframes": pick.keyframes,
        "best": pick.best.to_json(),
        "alternatives": [c.to_json() for c in pick.alternatives()[1:]],
    }


def view_checks(
    recording: Recording,
    geometry: SceneGeometry,
    subject: Subject,
    pose: CameraPose,
    frames: list[int],
    aspect: float,
) -> dict[str, Any]:
    """Where the subject leaves the view, checked on up to IN_VIEW_SAMPLES recorded frames."""
    count = len(frames)
    checked = sorted(
        set(
            np.linspace(0, count - 1, min(count, IN_VIEW_SAMPLES))
            .round()
            .astype(int)
            .tolist()
        )
    )
    shares = subject_in_view(
        geometry, subject, pose, [frames[i] for i in checked], aspect
    )
    warnings = []
    outside = [i for i, share in zip(checked, shares) if share < IN_VIEW_WARNING]
    if outside:
        clock = recording.clock
        seconds = (
            clock.times_ns[np.searchsorted(clock.frames, frames)] - clock.times_ns[0]
        ) / 1e9
        spans = ", ".join(
            f"{seconds[a]:.1f}-{seconds[b]:.1f} s"
            for a, b in _spans(outside, count / len(checked))
        )
        warnings.append(f"Subject partly outside the view at {spans}")
    return {
        "subject_in_view_min": round(min(shares), 4),
        "frames_checked": len(checked),
        "warnings": warnings,
    }


def _spans(indices: list[int], step: float) -> list[tuple[int, int]]:
    """Runs of indices no more than `step` apart, as (first, last); at most eight."""
    spans: list[list[int]] = []
    for index in indices:
        if spans and index - spans[-1][1] <= step + 1:
            spans[-1][1] = index
        else:
            spans.append([index, index])
    return [(a, b) for a, b in spans[:8]]
