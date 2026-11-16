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

"""World-space geometry of a recording's meshes at recorded frames.

Used only to choose and check the camera. The video itself is a Rerun render.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from rrd_video.recording import Recording

WORLD = "/world"
_TRANSLATION = "Transform3D:translation"
_QUATERNION = "Transform3D:quaternion"
_VERTICES = "Mesh3D:vertex_positions"
_TRIANGLES = "Mesh3D:triangle_indices"


Transform = tuple[np.ndarray, np.ndarray]  # rotation, translation


@dataclass
class _Track:
    """Values of one component over the frame timeline, ascending by frame."""

    frames: np.ndarray
    values: list[Any]

    def at(self, frame: int) -> Any | None:
        index = int(np.searchsorted(self.frames, frame, side="right")) - 1
        return None if index < 0 else self.values[index]


@dataclass
class _Entity:
    static_transform: Transform | None = None
    transforms: _Track | None = None
    static_vertices: np.ndarray | None = None
    vertices: _Track | None = None
    triangles: np.ndarray | None = None
    samples: tuple[np.ndarray, np.ndarray] | None = None  # triangle ids, barycentrics
    # Temporal rows as read, chunk by chunk: (frames, values).
    transform_parts: list[tuple[np.ndarray, list[Transform]]] = field(
        default_factory=list
    )
    vertex_parts: list[tuple[np.ndarray, list[np.ndarray]]] = field(
        default_factory=list
    )


@dataclass
class SceneGeometry:
    """Meshes under /world with their transforms, read from a recording."""

    entities: dict[str, _Entity] = field(default_factory=dict)
    # Entity path patterns ('/world/x' or '/world/x/**') whose meshes are left out.
    excluded: tuple[str, ...] = ()

    @property
    def mesh_paths(self) -> list[str]:
        return sorted(
            p
            for p, e in self.entities.items()
            if (e.static_vertices is not None or e.vertices is not None)
            and not _matches(p, self.excluded)
        )

    def without(self, patterns: list[str]) -> SceneGeometry:
        return SceneGeometry(self.entities, self.excluded + tuple(patterns))

    def world_transform(self, path: str, frame: int) -> Transform:
        """Rotation and translation of `path` in world at `frame`."""
        rotation, translation = np.eye(3), np.zeros(3)
        parts = path.strip("/").split("/")
        for depth in range(1, len(parts) + 1):
            entity = self.entities.get("/" + "/".join(parts[:depth]))
            local = _local_transform(entity, frame) if entity else None
            if local is not None:
                translation = rotation @ local[1] + translation
                rotation = rotation @ local[0]
        return rotation, translation

    def local_vertices(self, path: str, frame: int) -> np.ndarray | None:
        entity = self.entities[path]
        if entity.static_vertices is not None:
            return entity.static_vertices
        return entity.vertices.at(frame) if entity.vertices else None

    def world_vertices(self, path: str, frame: int) -> np.ndarray | None:
        local = self.local_vertices(path, frame)
        if local is None or len(local) == 0:
            return None
        rotation, translation = self.world_transform(path, frame)
        return local @ rotation.T + translation

    def world_box(self, path: str, frame: int) -> tuple[np.ndarray, np.ndarray] | None:
        vertices = self.world_vertices(path, frame)
        if vertices is None:
            return None
        return vertices.min(axis=0), vertices.max(axis=0)

    def surface_area(self, path: str) -> float:
        entity = self.entities[path]
        vertices = self.local_vertices(path, _first_frame(entity))
        if vertices is None or entity.triangles is None or len(entity.triangles) == 0:
            return 0.0
        return float(_triangle_areas(vertices, entity.triangles).sum())

    def sample_points(self, path: str, frame: int, count: int) -> np.ndarray | None:
        """World positions of `count` fixed surface samples; None without triangles."""
        entity = self.entities[path]
        vertices = self.world_vertices(path, frame)
        if vertices is None or entity.triangles is None or len(entity.triangles) == 0:
            return None
        if entity.samples is None or len(entity.samples[0]) != count:
            entity.samples = _surface_samples(path, vertices, entity.triangles, count)
        triangle_ids, weights = entity.samples
        corners = vertices[entity.triangles[triangle_ids]]
        return np.einsum("nk,nkd->nd", weights, corners)


def read_geometry(recording: Recording) -> SceneGeometry:
    geometry = SceneGeometry()
    for chunk in recording.statics:
        if _in_world(chunk.entity_path):
            _add_static(geometry, chunk.entity_path, chunk.to_record_batch())
    for temporal in recording.temporal:
        if _in_world(temporal.entity_path):
            _add_temporal(
                geometry, temporal.entity_path, temporal.batch, temporal.frames
            )
    for entity in geometry.entities.values():
        entity.transforms = _track(entity.transform_parts)
        entity.vertices = _track(entity.vertex_parts)
        if entity.triangles is None:
            entity.triangles = _soup_triangles(entity)
    return geometry


def _track(parts: list[tuple[np.ndarray, list[Any]]]) -> _Track | None:
    """One track from the rows of every chunk, sorted by frame."""
    parts = [(frames, values) for frames, values in parts if values]
    if not parts:
        return None
    frames = np.concatenate([f for f, _ in parts]).astype(np.int64)
    values = [v for _, part in parts for v in part]
    order = np.argsort(frames, kind="stable")
    return _Track(frames[order], [values[i] for i in order])


def _soup_triangles(entity: _Entity) -> np.ndarray | None:
    """A mesh without indices is a triangle list: every three vertices form a triangle."""
    vertices = entity.static_vertices
    if vertices is None and entity.vertices is not None:
        vertices = entity.vertices.values[0]
    if vertices is None or len(vertices) == 0 or len(vertices) % 3:
        return None
    return np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)


def _in_world(path: str) -> bool:
    return path == WORLD or path.startswith(WORLD + "/")


def _add_static(geometry: SceneGeometry, path: str, batch: pa.RecordBatch) -> None:
    entity = geometry.entities.setdefault(path, _Entity())
    names = batch.schema.names
    if _TRANSLATION in names or _QUATERNION in names:
        entity.static_transform = _transform_row(batch, 0)
    if _VERTICES in names and batch.column(_VERTICES)[0].is_valid:
        entity.static_vertices = _points(batch.column(_VERTICES)[0])
    if _TRIANGLES in names and batch.column(_TRIANGLES)[0].is_valid:
        entity.triangles = _triangles(batch.column(_TRIANGLES)[0])


def _add_temporal(
    geometry: SceneGeometry, path: str, batch: pa.RecordBatch, frames: np.ndarray
) -> None:
    entity = geometry.entities.setdefault(path, _Entity())
    names = batch.schema.names
    transform_columns = [n for n in (_TRANSLATION, _QUATERNION) if n in names]
    if transform_columns:
        valid = _valid_any(batch, transform_columns)
        rotations, translations = _transform_rows(batch, np.flatnonzero(valid))
        entity.transform_parts.append(
            (frames[valid], list(zip(rotations, translations)))
        )
    if _VERTICES in names:
        column = batch.column(_VERTICES)
        valid = np.asarray(column.is_valid().to_numpy(zero_copy_only=False))
        rows = [_points(column[int(i)]) for i in np.flatnonzero(valid)]
        entity.vertex_parts.append((frames[valid], rows))
    if _TRIANGLES in names and entity.triangles is None:
        column = batch.column(_TRIANGLES)
        for i in range(len(column)):
            if column[i].is_valid:
                entity.triangles = _triangles(column[i])
                break


def _matches(path: str, patterns: tuple[str, ...]) -> bool:
    for pattern in patterns:
        prefix = pattern.removesuffix("/**")
        if path == prefix or (
            pattern.endswith("/**") and path.startswith(prefix + "/")
        ):
            return True
    return False


def _valid_any(batch: pa.RecordBatch, names: list[str]) -> np.ndarray:
    """Rows where any of the named columns holds a value."""
    valid = pc.is_valid(batch.column(names[0]))
    for name in names[1:]:
        valid = pc.or_(valid, pc.is_valid(batch.column(name)))
    return np.asarray(valid.to_numpy(zero_copy_only=False))


def _points(value: Any) -> np.ndarray:
    flat = value.values.flatten().to_numpy(zero_copy_only=False)
    return np.asarray(flat, dtype=np.float64).reshape(-1, 3)


def _triangles(value: Any) -> np.ndarray:
    flat = value.values.flatten().to_numpy(zero_copy_only=False)
    return np.asarray(flat, dtype=np.int64).reshape(-1, 3)


def _transform_row(batch: pa.RecordBatch, row: int) -> Transform:
    translation, rotation = np.zeros(3), np.eye(3)
    names = batch.schema.names
    if _TRANSLATION in names and batch.column(_TRANSLATION)[row].is_valid:
        values = _points(batch.column(_TRANSLATION)[row])
        if len(values):
            translation = values[0]
    if _QUATERNION in names and batch.column(_QUATERNION)[row].is_valid:
        flat = (
            batch.column(_QUATERNION)[row]
            .values.flatten()
            .to_numpy(zero_copy_only=False)
        )
        if len(flat) >= 4:
            rotation = quaternion_matrices(
                np.asarray(flat[:4], dtype=np.float64)[None]
            )[0]
    return rotation, translation


def _transform_rows(
    batch: pa.RecordBatch, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rotations and translations of the given rows, vectorized when every row holds one value."""
    translations = _row_vectors(batch, _TRANSLATION, rows, 3)
    quaternions = _row_vectors(batch, _QUATERNION, rows, 4)
    if translations is None or quaternions is None:
        pairs = [_transform_row(batch, int(i)) for i in rows]
        return (
            np.array([p[0] for p in pairs]).reshape(-1, 3, 3),
            np.array([p[1] for p in pairs]).reshape(-1, 3),
        )
    return quaternion_matrices(quaternions), translations


def _row_vectors(
    batch: pa.RecordBatch, name: str, rows: np.ndarray, width: int
) -> np.ndarray | None:
    if name not in batch.schema.names:
        identity = np.zeros(width)
        if width == 4:
            identity[3] = 1.0
        return np.tile(identity, (len(rows), 1))
    column = batch.column(name).take(pa.array(rows, type=pa.int64()))
    if column.null_count:
        return None
    lengths = np.asarray(column.value_lengths().to_numpy(zero_copy_only=False))
    if np.any(lengths != 1):
        return None
    flat = column.flatten().flatten().to_numpy(zero_copy_only=False)
    return np.asarray(flat, dtype=np.float64).reshape(-1, width)


def quaternion_matrices(quaternions: np.ndarray) -> np.ndarray:
    """Rotation matrices of xyzw quaternions; identity where one is degenerate."""
    x, y, z, w = (quaternions[:, i] for i in range(4))
    norm = x * x + y * y + z * z + w * w
    good = np.isfinite(norm) & (norm > 0)
    s = np.where(good, 2.0 / np.where(good, norm, 1.0), 0.0)
    matrices = np.empty((len(quaternions), 3, 3))
    matrices[:, 0, 0] = 1 - s * (y * y + z * z)
    matrices[:, 0, 1] = s * (x * y - z * w)
    matrices[:, 0, 2] = s * (x * z + y * w)
    matrices[:, 1, 0] = s * (x * y + z * w)
    matrices[:, 1, 1] = 1 - s * (x * x + z * z)
    matrices[:, 1, 2] = s * (y * z - x * w)
    matrices[:, 2, 0] = s * (x * z - y * w)
    matrices[:, 2, 1] = s * (y * z + x * w)
    matrices[:, 2, 2] = 1 - s * (x * x + y * y)
    return matrices


def _local_transform(entity: _Entity, frame: int) -> Transform | None:
    # Static data takes precedence over temporal data, as in Rerun.
    if entity.static_transform is not None:
        return entity.static_transform
    return entity.transforms.at(frame) if entity.transforms else None


def _first_frame(entity: _Entity) -> int:
    return int(entity.vertices.frames[0]) if entity.vertices else 0


def _triangle_areas(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def _surface_samples(
    path: str, vertices: np.ndarray, triangles: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted triangle ids and barycentric weights, seeded by the entity path."""
    seed = int.from_bytes(hashlib.sha256(path.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    areas = _triangle_areas(vertices, triangles)
    total = areas.sum()
    probabilities = areas / total if total > 0 else None
    triangle_ids = rng.choice(len(triangles), size=count, p=probabilities)
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    weights = np.stack([1 - u - v, u, v], axis=1)
    return triangle_ids, weights
