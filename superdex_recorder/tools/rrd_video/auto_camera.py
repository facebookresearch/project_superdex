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

"""Chooses a camera from what a recording shows: the task's subject, seen unoccluded.

The subject is what moves and is not the robot, the robot links near it, and the fixtures
it touches.
Candidate directions on a fixed grid are scored on subject visibility (a point-splat
z-buffer over the recorded surfaces), on how much of the subject's motion crosses the
image, and on mild priors. Everything is deterministic.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from rrd_video.camera import CameraError, CameraPose, vector
from rrd_video.geometry import SceneGeometry
from rrd_video.recording import FRAMES_ENTITY, Recording, static_text

VERTICAL_FOV_DEG = 55.0  # Rerun 0.33's fixed 3D view field of view.
AZIMUTHS_DEG: tuple[float, ...] = tuple(range(0, 360, 15))
ELEVATIONS_DEG: tuple[float, ...] = (15.0, 30.0, 45.0, 60.0)
FILL = 0.75  # Subject points stay inside this fraction of the half field of view.
GRAVITY = "superdex.physics.gravity_xyz"
# Recorder entities that are not scene geometry.
EXCLUDED_ENTITIES = ("/world/shadows", "/world/cameras")
_ZBUFFER_WIDTH = 192
_POINT_BUDGET = 240_000
_WEIGHTS = {"visibility": 0.6, "legibility": 0.25, "prior": 0.15}
_ROLE_WEIGHTS = {"object": 2.0, "marker": 1.0, "robot": 1.0, "fixture": 0.5}


@dataclass(frozen=True)
class Actor:
    entity: str
    name: str
    kind: str
    robot: bool
    controlled: bool


@dataclass
class Subject:
    paths: list[str]
    roles: dict[str, str]
    robots: list[str]
    up: np.ndarray
    up_source: str
    robot_confidence: str
    # Keyframes at which each robot link is framed: while it is within link_radius of an object.
    link_keyframes: dict[str, np.ndarray] = field(default_factory=dict)
    link_radius: float = 0.0


@dataclass
class Candidate:
    azimuth: float
    elevation: float
    pose: CameraPose
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def score(self) -> float:
        return sum(_WEIGHTS[k] * self.scores[k] for k in _WEIGHTS)

    def to_json(self) -> dict[str, Any]:
        return {
            "azimuth_deg": self.azimuth,
            "elevation_deg": self.elevation,
            "score": round(self.score, 4),
            **{k: round(v, 4) for k, v in self.scores.items()},
            "camera": self.pose.to_json(),
        }


def read_actors(recording: Recording) -> list[Actor]:
    """Physics actors, with robots joined by runtime name to the robotics recording."""
    statics = recording.statics
    robot_names = {
        static_text(statics, c.entity_path, "superdex.robotics.robot_id")
        for c in statics
        if c.entity_path.startswith("/data/robotics/robots/")
    } - {None}
    controlled = {
        t.entity_path
        for t in recording.temporal
        if "superdex.physics.controller_target_positions" in t.batch.schema.names
    }
    actors = []
    for chunk in statics:
        path = chunk.entity_path
        prefix = "/data/physics/actors/"
        if not path.startswith(prefix) or "/" in path[len(prefix) :]:
            continue
        kind = static_text([chunk], path, "superdex.physics.actor_type")
        if kind is None:
            continue
        name = static_text([chunk], path, "superdex.physics.runtime_name") or ""
        actors.append(
            Actor(
                "/world/" + path[len(prefix) :],
                name,
                kind.lower(),
                name in robot_names,
                path in controlled,
            )
        )
    return sorted(actors, key=lambda a: a.entity)


def _owner(actors: list[Actor], path: str) -> Actor | None:
    matches = [a for a in actors if path == a.entity or path.startswith(a.entity + "/")]
    return max(matches, key=lambda a: len(a.entity)) if matches else None


def find_subject(
    recording: Recording,
    geometry: SceneGeometry,
    frames: list[int],
    framing: str = "subject",
    up: np.ndarray | None = None,
) -> Subject:
    """What the task is about; `up` overrides the inferred up axis."""
    actors = read_actors(recording)
    paths = [p for p in geometry.mesh_paths if not _excluded(p)]
    corners = {p: _corners_over(geometry, p, frames) for p in paths}
    corners = {p: c for p, c in corners.items() if c is not None}
    if not corners:
        raise CameraError(
            "The recording has no meshes under /world to frame; pass --camera"
        )
    scale = _scene_scale(corners)
    moving = {p for p, c in corners.items() if _motion(c) > max(0.002, 0.005 * scale)}
    robots, confidence = _robots(actors, geometry, frames)
    roles = _roles(actors, robots, corners, moving)
    objects = [p for p, r in roles.items() if r in ("object", "marker")]
    links = _near_links(roles, corners, objects, framing)
    selected = objects + links or sorted(moving) or sorted(corners)
    if up is None:
        up, up_source = up_axis(recording, corners, moving)
    else:
        up_source = "--up"
    fixtures = _fixtures(roles, corners, objects, selected, up)
    for path in fixtures:
        roles[path] = "fixture"
    return Subject(
        sorted(selected + fixtures),
        {p: roles.get(p, "context") for p in selected + fixtures},
        sorted(a.entity for a in robots),
        up,
        up_source,
        confidence,
        *(({}, 0.0) if framing == "wide" else _link_keyframes(links, objects, corners)),
    )


def _fixtures(
    roles: dict[str, str],
    corners: dict[str, np.ndarray],
    objects: list[str],
    selected: list[str],
    up: np.ndarray,
) -> list[str]:
    """Static bodies the objects touch, such as rods or pegs, but not tables or floors."""
    if not objects:
        return []
    extent = float(
        np.ptp(
            np.concatenate([corners[p].reshape(-1, 3) for p in selected]), axis=0
        ).max()
    )
    radius = max(0.005, 0.25 * extent)
    fixtures = []
    for path, role in sorted(roles.items()):
        if role != "context":
            continue
        box = corners[path][0]
        if np.ptp(box, axis=0).max() > 10 * extent or _is_flat_support(box, up):
            continue
        if min(_gap(corners[path], corners[o]) for o in objects) <= radius:
            fixtures.append(path)
    return fixtures


def _is_flat_support(box: np.ndarray, up: np.ndarray) -> bool:
    extent = np.ptp(box, axis=0)
    axis = int(np.argmax(np.abs(up)))
    return bool(extent[axis] < 0.25 * np.delete(extent, axis).min())


def _robots(
    actors: list[Actor], geometry: SceneGeometry, frames: list[int]
) -> tuple[list[Actor], str]:
    joined = [a for a in actors if a.robot]
    if joined:
        return joined, "robotics recording"
    controlled = [a for a in actors if a.controlled]
    if controlled:
        return controlled, "pose-controlled articulations"
    # No controllers either: an articulation whose root link never moves is taken as a robot.
    guessed = []
    for actor in actors:
        if actor.kind != "articulated":
            continue
        links = [
            p for p in geometry.mesh_paths if p.startswith(actor.entity + "/links/")
        ]
        depth = min((p.count("/") for p in links), default=0)
        roots = [p for p in links if p.count("/") == depth] or [actor.entity]
        origins = np.array(
            [[geometry.world_transform(r, f)[1] for r in roots] for f in frames]
        )
        if np.ptp(origins, axis=0).max() < 1e-4:
            guessed.append(actor)
    return guessed, "low (fixed-root articulations)" if guessed else "none"


def _roles(
    actors: list[Actor],
    robots: list[Actor],
    corners: dict[str, np.ndarray],
    moving: set[str],
) -> dict[str, str]:
    robot_entities = {a.entity for a in robots}
    roles = {}
    for path in corners:
        actor = _owner(actors, path)
        if actor is not None and actor.entity in robot_entities:
            roles[path] = "robot"
        elif path not in moving:
            roles[path] = "context"
        else:
            roles[path] = "object" if actor is not None else "marker"
    return roles


def _excluded(path: str) -> bool:
    return any(path == e or path.startswith(e + "/") for e in EXCLUDED_ENTITIES)


def _near_links(
    roles: dict[str, str],
    corners: dict[str, np.ndarray],
    objects: list[str],
    framing: str,
) -> list[str]:
    links = [p for p, r in roles.items() if r == "robot"]
    if framing == "wide":
        return [p for p in links if _motion(corners[p]) > 0]
    if not objects:
        return []
    extents = [np.ptp(corners[p].reshape(-1, 3), axis=0).max() for p in objects]
    radius = float(np.clip(0.5 * np.median(extents), 0.005, 0.05))
    near = []
    for link in links:
        gaps = [_gap(corners[link], corners[o]) for o in objects]
        if min(gaps) <= radius:
            near.append(link)
    return near


def _link_keyframes(
    links: list[str], objects: list[str], corners: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], float]:
    """Keyframes at which each link is within reach of an object, and that reach.

    A gripper that starts far above a small block would otherwise shrink the block to a
    few pixels; the video still shows it arrive.
    """
    if not objects:
        return {}, 0.0
    radius = link_radius([corners[p][0] for p in objects])
    keyframes = {}
    for link in links:
        gaps = np.min([_gaps(corners[link], corners[o]) for o in objects], axis=0)
        keyframes[link] = gaps <= radius
    return keyframes, radius


def link_radius(object_boxes: list[np.ndarray]) -> float:
    """How near an object a robot link must be to be framed: about one object size."""
    extents = [np.ptp(box, axis=0).max() for box in object_boxes]
    return float(np.clip(np.median(extents), 0.03, 0.15))


def _box_gap(
    low_a: np.ndarray, high_a: np.ndarray, low_b: np.ndarray, high_b: np.ndarray
) -> np.ndarray:
    """Distance between axis-aligned boxes, along the last axis."""
    separation = np.maximum(0, np.maximum(low_a - high_b, low_b - high_a))
    return np.sqrt(np.sum(separation * separation, axis=-1))


def _gaps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance between two entities' boxes at each keyframe."""
    return _box_gap(a.min(axis=1), a.max(axis=1), b.min(axis=1), b.max(axis=1))


def _gap(a: np.ndarray, b: np.ndarray) -> float:
    """Smallest distance between two entities' boxes over shared keyframes."""
    return float(_gaps(a, b).min())


def _corners_over(
    geometry: SceneGeometry, path: str, frames: list[int]
) -> np.ndarray | None:
    """World box corners per keyframe: (frames, 8, 3)."""
    boxes = [geometry.world_box(path, f) for f in frames]
    if any(b is None for b in boxes):
        return None
    return np.stack([box_corners(low, high) for low, high in boxes])


def _near(box: np.ndarray, others: list[np.ndarray], radius: float) -> bool:
    """Whether a (low, high) box lies within `radius` of any of `others`."""
    return any(
        float(_box_gap(box[0], box[1], other[0], other[1])) <= radius
        for other in others
    )


def box_corners(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [x, y, z]
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (low[2], high[2])
        ]
    )


def _motion(corners: np.ndarray) -> float:
    return float(np.linalg.norm(corners - corners[0], axis=2).max())


def _scene_scale(corners: dict[str, np.ndarray]) -> float:
    if not corners:
        return 1.0
    sizes = np.array([np.ptp(c[0], axis=0).max() for c in corners.values()])
    keep = [c for c, s in zip(corners.values(), sizes) if s <= 20 * np.median(sizes)]
    points = np.concatenate([c[0] for c in keep])
    return float(np.linalg.norm(np.ptp(points, axis=0)))


def up_axis(
    recording: Recording, corners: dict[str, np.ndarray], moving: set[str]
) -> tuple[np.ndarray, str]:
    """World up: against recorded gravity, else the normal of flat static bodies, else +Z."""
    gravity = _gravity(recording)
    if gravity is not None and _norm(gravity) > 0:
        return _snap(-gravity), "gravity"
    votes = np.zeros(3)
    above = [corners[p][0].mean(axis=0) for p in moving]
    for path, box in corners.items():
        if path in moving:
            continue
        extent = np.ptp(box[0], axis=0)
        thin = int(np.argmin(extent))
        others = np.delete(extent, thin)
        if extent[thin] < 0.25 * others.min() and above:
            sign = (
                1.0
                if np.mean([c[thin] for c in above]) >= box[0][:, thin].mean()
                else -1.0
            )
            votes[thin] += sign * others.prod()
    if np.abs(votes).max() > 0 and np.abs(votes).max() >= 0.6 * np.abs(votes).sum():
        axis = int(np.argmax(np.abs(votes)))
        up = np.zeros(3)
        up[axis] = math.copysign(1.0, votes[axis])
        return up, "flat static bodies"
    return np.array([0.0, 0.0, 1.0]), "assumed +Z"


def _gravity(recording: Recording) -> np.ndarray | None:
    """The scene's gravity, logged on the frame clock (static, or per frame)."""
    batches = [
        c.to_record_batch() for c in recording.statics if c.entity_path == FRAMES_ENTITY
    ]
    batches += [t.batch for t in recording.temporal if t.entity_path == FRAMES_ENTITY]
    for batch in batches:
        if GRAVITY not in batch.schema.names:
            continue
        for value in batch.column(GRAVITY).to_pylist():
            if value is not None:
                return np.asarray(value, dtype=np.float64).reshape(-1)[:3]
    return None


def parse_up(text: str) -> np.ndarray:
    """'z', '+y' or '-x' as a unit axis."""
    sign = -1.0 if text.startswith("-") else 1.0
    axis = "xyz".find(text.lstrip("+-").lower())
    if axis < 0 or len(text.lstrip("+-")) != 1:
        raise ValueError(f"--up must be one of x, y, z, -x, -y, -z; got {text!r}")
    up = np.zeros(3)
    up[axis] = sign
    return up


def _snap(direction: np.ndarray) -> np.ndarray:
    axis = int(np.argmax(np.abs(direction)))
    up = np.zeros(3)
    up[axis] = math.copysign(1.0, direction[axis])
    return up


def _basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = int(np.argmax(np.abs(up)))
    first = np.eye(3)[(axis + 1) % 3]
    second = np.cross(up, first)
    return first, second / _norm(second)


def direction(up: np.ndarray, azimuth: float, elevation: float) -> np.ndarray:
    """Unit vector from the target toward the camera."""
    first, second = _basis(up)
    a, e = math.radians(azimuth), math.radians(elevation)
    return math.cos(e) * (math.cos(a) * first + math.sin(a) * second) + math.sin(e) * up


def camera_axes(offset: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Rows right, up, forward for a camera looking along -offset."""
    forward = -offset / _norm(offset)
    right = np.cross(forward, up)
    if _norm(right) < 1e-9:
        raise CameraError("The camera's up direction is parallel to its view direction")
    right /= _norm(right)
    return np.stack([right, np.cross(right, forward), forward])


# Scoring runs on threads: these avoid BLAS, whose concurrent calls gave wrong results.
def _project(points: np.ndarray, origin: np.ndarray, axes: np.ndarray) -> np.ndarray:
    return np.einsum("ij,kj->ik", points - origin, axes)


def _dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(a * b))


def _norm(a: np.ndarray) -> float:
    return math.sqrt(_dot(a, a))


def fit_distance(
    points: np.ndarray, target: np.ndarray, axes: np.ndarray, aspect: float
) -> float:
    """Smallest distance that keeps every point within FILL of the field of view."""
    tan_v = math.tan(math.radians(VERTICAL_FOV_DEG / 2)) * FILL
    tan_h = tan_v * aspect
    local = _project(points, target, axes)
    needed = np.maximum(
        np.abs(local[:, 0]) / tan_h - local[:, 2],
        np.abs(local[:, 1]) / tan_v - local[:, 2],
    )
    radius = float(np.linalg.norm(local, axis=1).max())
    return max(float(needed.max()), 1.2 * radius, 1e-3)


@dataclass
class _Samples:
    """Fixed surface samples of every visible entity at each keyframe."""

    points: list[np.ndarray]  # per keyframe, (N, 3)
    owners: np.ndarray  # (N,) entity index
    paths: list[str]


def _sample(geometry: SceneGeometry, subject: Subject, frames: list[int]) -> _Samples:
    paths = [p for p in geometry.mesh_paths if not _excluded(p)]
    subject_points = [geometry.world_vertices(p, frames[0]) for p in subject.paths]
    stacked = np.concatenate([v for v in subject_points if v is not None])
    spacing = max(float(np.ptp(stacked, axis=0).max()) / (_ZBUFFER_WIDTH * 0.6), 1e-4)
    counts = {
        p: int(np.clip(geometry.surface_area(p) / spacing**2, 30, 30_000))
        for p in paths
    }
    total = sum(counts.values())
    if total > _POINT_BUDGET:
        scale = _POINT_BUDGET / total
        counts = {p: max(30, int(c * scale)) for p, c in counts.items()}
    kept = [
        p for p in paths if geometry.sample_points(p, frames[0], counts[p]) is not None
    ]
    owners = np.concatenate([np.full(counts[p], i) for i, p in enumerate(kept)])
    points = [
        np.concatenate([geometry.sample_points(p, f, counts[p]) for p in kept])
        for f in frames
    ]
    return _Samples(points, owners, kept)


def _visibility(
    samples: _Samples,
    subject: Subject,
    eye: np.ndarray,
    axes: np.ndarray,
    aspect: float,
) -> float:
    width, height = _ZBUFFER_WIDTH, max(1, round(_ZBUFFER_WIDTH / aspect))
    tan_v = math.tan(math.radians(VERTICAL_FOV_DEG / 2))
    index = {p: i for i, p in enumerate(samples.paths)}
    subject_ids = np.array([index[p] for p in subject.paths if p in index])
    roles = [subject.roles[p] for p in subject.paths if p in index]
    scores = []
    for points in samples.points:
        local = _project(points, eye, axes)
        depth = local[:, 2]
        front = depth > 1e-4
        u = local[front, 0] / (depth[front] * tan_v * aspect)
        v = local[front, 1] / (depth[front] * tan_v)
        inside = (np.abs(u) < 1) & (np.abs(v) < 1)
        x = ((u[inside] + 1) * 0.5 * width).astype(np.int64)
        y = ((1 - v[inside]) * 0.5 * height).astype(np.int64)
        pixel = y * width + x
        owner = samples.owners[front][inside]
        order = np.lexsort((depth[front][inside], pixel))
        first = np.unique(pixel[order], return_index=True)[1]
        nearest = owner[order][first]
        shown = np.bincount(nearest, minlength=len(samples.paths))[subject_ids]
        alone = np.array([len(np.unique(pixel[owner == i])) for i in subject_ids])
        ratio = np.where(alone > 0, shown / np.maximum(alone, 1), 0.0)
        scores.append(_by_role(ratio, roles))
    return float(np.mean(scores))


def _by_role(ratio: np.ndarray, roles: list[str]) -> float:
    """Mean visible share per role, weighted by role: two objects outweigh thirty links."""
    total = weight = 0.0
    for role in sorted(set(roles)):
        mask = np.array([r == role for r in roles])
        total += _ROLE_WEIGHTS.get(role, 1.0) * float(ratio[mask].mean())
        weight += _ROLE_WEIGHTS.get(role, 1.0)
    return total / weight


def _legibility(subject_tracks: np.ndarray, forward: np.ndarray) -> float:
    """Share of subject motion across the image rather than along the view."""
    steps = np.diff(subject_tracks, axis=0).reshape(-1, 3)
    total = np.linalg.norm(steps, axis=1).sum()
    if total < 1e-9:
        return 1.0
    along = np.einsum("ij,j->i", steps, forward)
    across = np.linalg.norm(steps - np.outer(along, forward), axis=1).sum()
    return float(across / total)


def _prior(
    offset: np.ndarray,
    up: np.ndarray,
    elevation: float,
    target: np.ndarray,
    bases: list[np.ndarray],
) -> float:
    elevation_score = math.exp(-(((elevation - 35.0) / 20.0) ** 2))
    if not bases:
        return 0.5 * elevation_score + 0.25
    toward_robot = np.mean(bases, axis=0) - target
    toward_robot -= up * _dot(toward_robot, up)
    horizontal = offset - up * _dot(offset, up)
    if _norm(toward_robot) < 1e-6 or _norm(horizontal) < 1e-6:
        return 0.5 * elevation_score + 0.25
    facing = 0.5 * (
        1 - _dot(toward_robot, horizontal) / (_norm(toward_robot) * _norm(horizontal))
    )
    return 0.5 * elevation_score + 0.5 * facing


@dataclass
class Pick:
    best: Candidate
    candidates: list[Candidate]
    subject: Subject
    keyframes: list[int]

    def alternatives(
        self, count: int = 4, min_angle_deg: float = 40.0
    ) -> list[Candidate]:
        chosen: list[Candidate] = []
        for candidate in self.candidates:
            offset = _offset(candidate)
            if all(_angle(offset, _offset(c)) >= min_angle_deg for c in chosen):
                chosen.append(candidate)
            if len(chosen) == count:
                break
        return chosen


def _offset(candidate: Candidate) -> np.ndarray:
    return np.subtract(candidate.pose.eye, candidate.pose.target)


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    cos = _dot(a, b) / (_norm(a) * _norm(b))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def pick_camera(
    recording: Recording,
    geometry: SceneGeometry,
    frames: list[int],
    aspect: float,
    framing: str = "subject",
    up: np.ndarray | None = None,
    keyframe_count: int = 12,
) -> Pick:
    """The best camera for the recorded frames `frames`, with every scored candidate."""
    keyframes = keyframes_of(frames, keyframe_count)
    subject = find_subject(recording, geometry, keyframes, framing, up)
    samples = _sample(geometry, subject, keyframes)
    core = [p for p in subject.paths if subject.roles[p] != "fixture"]
    tracks = _tracks(samples, core)
    pooled = np.concatenate(
        [
            _framed_points(samples, subject, core),
            _fixture_points(samples, subject, tracks),
        ]
    )
    pooled = _within_scene(pooled, samples.points[0])
    target = 0.5 * (pooled.min(axis=0) + pooled.max(axis=0))
    bases = [geometry.world_transform(r, keyframes[0])[1] for r in subject.robots]
    visibility_frames = _Samples(
        samples.points[1::3] or samples.points, samples.owners, samples.paths
    )

    def score(azimuth: float, elevation: float) -> Candidate:
        offset = direction(subject.up, azimuth, elevation)
        axes = camera_axes(offset, subject.up)
        distance = fit_distance(pooled, target, axes, aspect)
        eye = target + distance * offset
        pose = CameraPose(vector(eye), vector(target), vector(subject.up))
        candidate = Candidate(azimuth, elevation, pose)
        candidate.scores = {
            "visibility": _visibility(visibility_frames, subject, eye, axes, aspect),
            "legibility": _legibility(tracks, axes[2]),
            "prior": _prior(offset, subject.up, elevation, target, bases),
        }
        return candidate

    grid = [(a, e) for e in ELEVATIONS_DEG for a in AZIMUTHS_DEG]
    with ThreadPoolExecutor(max_workers=16) as pool:
        candidates = list(pool.map(lambda ae: score(*ae), grid))
    candidates.sort(key=lambda c: (-round(c.score, 9), c.elevation, c.azimuth))
    return Pick(candidates[0], candidates, subject, keyframes)


def _framed_points(samples: _Samples, subject: Subject, paths: list[str]) -> np.ndarray:
    """Subject points over all keyframes, robot links only while they are near objects."""
    framed = []
    for k, points in enumerate(samples.points):
        kept = [
            p
            for p in paths
            if p not in subject.link_keyframes or subject.link_keyframes[p][k]
        ]
        framed.append(
            points[
                np.isin(
                    samples.owners,
                    [samples.paths.index(p) for p in kept if p in samples.paths],
                )
            ]
        )
    return np.concatenate(framed)


def _tracks(samples: _Samples, paths: list[str]) -> np.ndarray:
    """Sample positions of `paths` per keyframe: (keyframes, points, 3)."""
    mask = np.isin(
        samples.owners, [samples.paths.index(p) for p in paths if p in samples.paths]
    )
    return np.stack([points[mask] for points in samples.points])


def _fixture_points(
    samples: _Samples, subject: Subject, tracks: np.ndarray
) -> np.ndarray:
    """Fixture samples within one subject extent of the subject: the part of a rod or peg
    near the action frames the shot, not its whole length."""
    fixtures = [p for p in subject.paths if subject.roles[p] == "fixture"]
    if not fixtures:
        return np.zeros((0, 3))
    points = tracks.reshape(-1, 3)
    low, high = points.min(axis=0), points.max(axis=0)
    margin = float(np.max(high - low))
    near = _tracks(samples, fixtures).reshape(-1, 3)
    return np.clip(near, low - margin, high + margin)


def _within_scene(points: np.ndarray, initial: np.ndarray) -> np.ndarray:
    """Points within a quarter of the scene's size of where everything started.

    An object that falls off the table into empty space would otherwise pull the camera
    metres away; the video still shows it leave, and the in-view check reports it.
    """
    low, high = initial.min(axis=0), initial.max(axis=0)
    margin = 0.25 * float(np.max(high - low))
    inside = np.all((points >= low - margin) & (points <= high + margin), axis=1)
    return points[inside] if inside.any() else points


def keyframes_of(frames: list[int], count: int) -> list[int]:
    if len(frames) <= count:
        return list(frames)
    positions = np.linspace(0, len(frames) - 1, count).round().astype(int)
    return [frames[i] for i in positions]


def subject_in_view(
    geometry: SceneGeometry,
    subject: Subject,
    pose: CameraPose,
    frames: list[int],
    aspect: float,
) -> list[float]:
    """Per frame, the share of subject box corners inside the view.

    Fixtures are context, and robot links count only while they are near an object, as
    in the framing.
    """
    eye, target, up = (np.asarray(v) for v in (pose.eye, pose.target, pose.up))
    axes = camera_axes(eye - target, up)
    tan_v = math.tan(math.radians(VERTICAL_FOV_DEG / 2))
    checked = [p for p in subject.paths if subject.roles[p] != "fixture"]
    objects = [p for p in checked if subject.roles[p] in ("object", "marker")]
    shares = []
    for frame in frames:
        found = {p: geometry.world_box(p, frame) for p in checked}
        found = {p: b for p, b in found.items() if b}
        near = [np.stack(found[o]) for o in objects if o in found]
        boxes = [
            b
            for p, b in found.items()
            if p not in subject.link_keyframes
            or _near(np.stack(b), near, subject.link_radius)
        ]
        if not boxes:
            shares.append(1.0)
            continue
        corners = np.concatenate([box_corners(lo, hi) for lo, hi in boxes])
        local = _project(corners, eye, axes)
        depth = np.maximum(local[:, 2], 1e-9)
        inside = (
            (np.abs(local[:, 0] / depth) <= tan_v * aspect)
            & (np.abs(local[:, 1] / depth) <= tan_v)
            & (local[:, 2] > 0)
        )
        shares.append(float(inside.mean()))
    return shares
