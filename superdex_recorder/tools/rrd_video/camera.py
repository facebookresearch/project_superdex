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

"""Camera poses, task keys and the per-task camera registry."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rrd_video.recording import Recording, static_text

Vector = tuple[float, float, float]
REGISTRY_SCHEMA = "superdex.rrd_video.camera/1"
# Properties that name a task, most specific first.
_TASK_PROPERTIES = ("scene.task.filename", "task", "env", "scene")
_ACTORS_PREFIX = "/data/physics/actors/"


class CameraError(ValueError):
    pass


@dataclass(frozen=True)
class CameraPose:
    eye: Vector
    target: Vector
    up: Vector = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        view = [t - e for e, t in zip(self.eye, self.target)]
        cross = [
            view[1] * self.up[2] - view[2] * self.up[1],
            view[2] * self.up[0] - view[0] * self.up[2],
            view[0] * self.up[1] - view[1] * self.up[0],
        ]
        if not all(math.isfinite(v) for v in (*self.eye, *self.target, *self.up)):
            raise CameraError(f"Camera values must be finite: {self}")
        if math.hypot(*view) < 1e-9:
            raise CameraError("The camera's eye and target are the same point")
        if math.hypot(*cross) < 1e-6 * math.hypot(*view) * math.hypot(*self.up):
            raise CameraError(
                "The camera's up direction is parallel to its view direction"
            )

    def to_json(self) -> dict[str, list[float]]:
        return {k: [float(x) for x in v] for k, v in asdict(self).items()}

    @staticmethod
    def from_json(data: Any) -> CameraPose:
        try:
            return CameraPose(
                vector(data["eye"]),
                vector(data["target"]),
                vector(data.get("up", (0, 0, 1))),
            )
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            raise CameraError(f"Invalid camera pose {data}: {e}") from e

    @staticmethod
    def parse(spec: str) -> CameraPose:
        """`eye=x,y,z target=x,y,z [up=x,y,z]`."""
        parts = spec.split()
        fields = dict(part.split("=", 1) for part in parts if "=" in part)
        allowed = {"eye", "target", "up"}
        if (
            len(fields) != len(parts)
            or not {"eye", "target"} <= fields.keys() <= allowed
        ):
            raise CameraError(
                f"Expected 'eye=x,y,z target=x,y,z [up=x,y,z]', got {spec!r}"
            )
        return CameraPose.from_json({k: v.split(",") for k, v in fields.items()})


def vector(values: Any) -> Vector:
    """Three numbers (or numeric strings) as a vector."""
    if isinstance(values, str):
        raise TypeError(f"expected three numbers, got {values!r}")
    x, y, z = (float(v) for v in values)
    return (x, y, z)


@dataclass
class ResolvedCamera:
    pose: CameraPose
    source: str
    task_key: str
    details: dict[str, Any] = field(default_factory=dict)


def task_key(recording: Recording) -> str:
    """A readable key shared by every recording of one task.

    The first task-naming episode property wins, normalized to a path relative to the
    assets root or a bare file name. Without one, the key is the sorted actor names.
    """
    for name in _TASK_PROPERTIES:
        value = recording.properties.get(name)
        if value:
            return f"{name}={_normalize(value)}"
    names = actor_names(recording)
    return "actors=" + "+".join(names) if names else "unknown"


def _normalize(value: str) -> str:
    """Paths relative to the assets root; other absolute paths keep only their file name."""
    value = value.strip().replace("\\", "/")
    if "/assets/" in value:
        return value.split("/assets/", 1)[1]
    if value.startswith("/") or re.match(r"^[A-Za-z]:/", value):
        return value.rsplit("/", 1)[-1]
    return value


def actor_names(recording: Recording) -> list[str]:
    """Sorted spawn (or runtime) names of the recorded physics actors."""
    names = set()
    for chunk in recording.statics:
        path = chunk.entity_path
        if not path.startswith(_ACTORS_PREFIX) or "/" in path[len(_ACTORS_PREFIX) :]:
            continue
        name = static_text([chunk], path, "superdex.physics.spawn_name") or static_text(
            [chunk], path, "superdex.physics.runtime_name"
        )
        if name:
            names.add(re.sub(r"[^A-Za-z0-9_.\-]+", "_", name))
    return sorted(names)


def registry_file_name(key: str) -> str:
    """'task=tasks/x/lift.superdex_task' -> 'lift.superdex_task.<hash>.json'."""
    value = key.split("=", 1)[-1].rstrip("/").rsplit("/", 1)[-1]
    slug = re.sub(r"[^A-Za-z0-9_.\-]+", "_", value).strip("_") or "camera"
    return f"{slug[:60]}.{hashlib.sha256(key.encode()).hexdigest()[:8]}.json"


@dataclass(frozen=True)
class RegistryEntry:
    """A camera saved for one task key."""

    key: str
    pose: CameraPose
    origin: str
    note: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": REGISTRY_SCHEMA,
            "key": self.key,
            "camera": self.pose.to_json(),
            "origin": self.origin,
            "note": self.note,
        }

    @staticmethod
    def from_json(data: Any) -> RegistryEntry:
        if not isinstance(data, dict) or data.get("schema") != REGISTRY_SCHEMA:
            raise CameraError(f"not a {REGISTRY_SCHEMA} object")
        if not isinstance(data.get("key"), str) or not data["key"]:
            raise CameraError("'key' must be a task key")
        return RegistryEntry(
            data["key"],
            CameraPose.from_json(data["camera"]),
            str(data.get("origin", "")),
            str(data.get("note", "")),
        )


def load_registry(directory: Path) -> dict[str, RegistryEntry]:
    """Every camera saved in `directory`, by task key."""
    if not directory.is_dir():
        raise CameraError(f"No camera registry directory {directory}")
    entries: dict[str, RegistryEntry] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            entry = RegistryEntry.from_json(json.loads(path.read_text()))
        except (json.JSONDecodeError, KeyError, CameraError) as e:
            raise CameraError(
                f"Camera registry file {path.name} is invalid: {e}"
            ) from e
        if entry.key in entries:
            raise CameraError(
                f"Camera registry has two cameras for {entry.key!r} ({path.name})"
            )
        entries[entry.key] = entry
    return entries


def save_entry(directory: Path, entry: RegistryEntry) -> Path:
    """Write `entry` to its own file in `directory`, replacing an earlier camera for its key."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / registry_file_name(entry.key)
    path.write_text(json.dumps(entry.to_json(), indent=2, sort_keys=True) + "\n")
    return path
