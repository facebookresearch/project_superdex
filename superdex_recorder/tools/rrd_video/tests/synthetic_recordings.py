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

"""Small recordings in the superdex_recorder layout, written with the Rerun SDK for tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr  # @manual

BOX_TRIANGLES = np.array(
    [
        [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
        [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
    ],
    dtype=np.uint32,
)  # fmt: skip


def box_vertices(half: tuple[float, float, float]) -> np.ndarray:
    return np.array(
        [
            [x, y, z]
            for x in (-half[0], half[0])
            for y in (-half[1], half[1])
            for z in (-half[2], half[2])
        ],
        dtype=np.float32,
    )


@dataclass
class Body:
    """A box actor: static, or with a position per frame (None where nothing is logged)."""

    name: str
    half: tuple[float, float, float]
    color: tuple[int, int, int]
    positions: list[tuple[float, float, float] | None]
    static: bool = False
    kind: str = "Rigid"


@dataclass
class Scene:
    bodies: list[Body]
    hz: float = 60.0
    first_time_s: float = 0.0
    properties: dict[str, str] = field(default_factory=dict)
    frames_entity: bool = True


def write(path: Path, scene: Scene, extra: Sequence[tuple[str, Any]] = ()) -> Path:
    """Write `scene` with the entity layout superdex_recorder uses; `extra` (entity,
    archetype) pairs are logged at the last frame."""
    rec = rr.RecordingStream("superdex", recording_id=path.stem)
    rec.save(str(path))
    _log_statics(rec, scene)
    frames = max(len(b.positions) for b in scene.bodies)
    for frame in range(frames):
        rec.set_time("frame", sequence=frame)
        rec.set_time("sim_time", duration=scene.first_time_s + frame / scene.hz)
        if scene.frames_entity:
            rec.log(
                "/data/physics/frames",
                rr.AnyValues(**{"superdex.physics.step_dt_sec": 1 / scene.hz}),
            )
        for body in scene.bodies:
            position = body.positions[frame] if frame < len(body.positions) else None
            if position is not None:
                rec.log(
                    f"/world/{body.name}",
                    rr.Transform3D(translation=position, quaternion=[0, 0, 0, 1]),
                )
    for entity, archetype in extra:
        rec.log(entity, archetype)
    rec.disconnect()
    return path


def _log_statics(rec: rr.RecordingStream, scene: Scene) -> None:
    """The episode schema and properties, and each body's actor record and mesh."""
    rec.log(
        "/episode",
        rr.AnyValues(**{"superdex.episode.schema_name": "superdex.episode"}),
        static=True,
    )
    for key, value in scene.properties.items():
        rec.log(
            f"/episode/properties/{key}",
            rr.AnyValues(
                **{
                    "superdex.episode.property_key": [key],
                    "superdex.episode.property_value": [value],
                }
            ),
            static=True,
        )
    for body in scene.bodies:
        rec.log(
            f"/data/physics/actors/{body.name}",
            rr.AnyValues(
                **{
                    "superdex.physics.actor_id": body.name,
                    "superdex.physics.runtime_name": body.name,
                    "superdex.physics.actor_type": body.kind,
                    "superdex.physics.is_static": 1.0 if body.static else 0.0,
                }
            ),
            static=True,
        )
        rec.log(
            f"/world/{body.name}",
            rr.Mesh3D(
                vertex_positions=box_vertices(body.half),
                triangle_indices=BOX_TRIANGLES,
                albedo_factor=[*body.color, 255],
            ),
            static=True,
        )


def sliding_box_scene(frames: int = 30, up: str = "z") -> Scene:
    """A box sliding along +x on a floor, with a tall wall on the -y side."""

    def oriented(v: tuple[float, float, float]) -> tuple[float, float, float]:
        x, y, z = v
        return (x, z, -y) if up == "y" else (x, y, z)

    def half(v: tuple[float, float, float]) -> tuple[float, float, float]:
        x, y, z = v
        return (x, z, y) if up == "y" else (x, y, z)

    floor = Body(
        "floor",
        half((1.0, 1.0, 0.01)),
        (120, 160, 120),
        [oriented((0, 0, -0.01))],
        static=True,
    )
    wall = Body(
        "wall",
        half((0.5, 0.02, 0.4)),
        (90, 90, 160),
        [oriented((0, -0.3, 0.4))],
        static=True,
    )
    path = [oriented((-0.3 + 0.6 * i / (frames - 1), 0, 0.05)) for i in range(frames)]
    cube = Body("cube", (0.05, 0.05, 0.05), (220, 100, 80), path)
    return Scene([floor, wall, cube])
