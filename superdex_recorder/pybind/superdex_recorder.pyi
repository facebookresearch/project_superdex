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

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

import numpy as np
import numpy.typing as npt

class RecorderConfig:
    def __init__(self, output_directory: str | Path) -> None: ...
    output_directory: Path

class EpisodeMetadata:
    def __init__(self, properties: dict[str, str] = ...) -> None: ...

class Episode:
    @property
    def episode_id(self) -> str: ...
    @property
    def output_path(self) -> Path: ...
    @property
    def is_active(self) -> bool: ...
    def commit(self) -> None: ...
    def abort(self, reason: str) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]: ...

class Recorder:
    def __init__(self, config: RecorderConfig) -> None: ...
    def begin_episode(self, metadata: EpisodeMetadata = ...) -> Episode: ...

class PhysicsRecorderConfig:
    def __init__(self) -> None: ...
    record_contact_forces: bool
    contact_force_visualization_scale_meters_per_newton: float
    spawn_names: list[tuple[Any, str]]

class PhysicsRecorder:
    def __init__(
        self,
        episode: Episode,
        scene: Any,
        config: PhysicsRecorderConfig = ...,
    ) -> None: ...
    def record(self, frame: int, simulation_time_sec: float) -> None: ...

class RoboticsRecorder:
    def __init__(self, episode: Episode, robot: Any) -> None: ...
    def record(self, frame: int, simulation_time_sec: float) -> None: ...

class RobotShadow:
    def __init__(
        self,
        episode: Episode,
        robot: Any,
        id: str,
        subtree_root_link_index: int | None = ...,
        color_rgba: tuple[int, int, int, int] = ...,
    ) -> None: ...
    def record(
        self,
        frame: int,
        simulation_time_sec: float,
        world_from_anchor: Any,
        joint_positions: npt.NDArray[np.floating[Any]],
    ) -> None: ...

class SkeletonShadow:
    def __init__(
        self,
        episode: Episode,
        robot: Any,
        id: str,
        keypoint_names: Sequence[str],
        parent_indices: Sequence[int],
        label: str = ...,
        color_rgba: tuple[int, int, int, int] = ...,
        keypoint_radius_meters: float = ...,
    ) -> None: ...
    # world_keypoints_xyz holds every keypoint's world X, Y and Z, in keypoint order, as a flat
    # array of 3N values or an (N, 3) array.
    def record(
        self,
        frame: int,
        simulation_time_sec: float,
        world_keypoints_xyz: npt.NDArray[np.floating[Any]],
    ) -> None: ...
    def record_untracked(self, frame: int, simulation_time_sec: float) -> None: ...

# Native episode reader, which the parity tests compare with Rerun's own reader.
class _EpisodeComponent:
    entity_path: str
    component: str
    is_static: bool

class _ScalarRows:
    frames: list[int]
    simulation_times_ns: list[int]
    # Row i holds values[offsets[i] : offsets[i + 1]].
    offsets: list[int]
    values: list[float]

class _EpisodeReader:
    def __init__(self, path: str | Path) -> None: ...
    def components(self) -> list[_EpisodeComponent]: ...
    def read_static_texts(self, entity_path: str, component: str) -> list[str]: ...
    def read_static_scalars(self, entity_path: str, component: str) -> list[float]: ...
    def read_scalars(self, entity_path: str, component: str) -> _ScalarRows: ...

# Native core of superdex.recorder.EpisodeRestorer.
class _RecordedActor:
    path: str
    handle: Any
    is_articulated: bool
    is_static: bool

def _bind_recorded_actors(
    scene: Any, recording: _EpisodeReader
) -> list[_RecordedActor]: ...

class _EpisodeRestorer:
    def __init__(self, scene: Any, recording: _EpisodeReader) -> None: ...
    @property
    def frames(self) -> list[int]: ...
    def actors(self) -> list[_RecordedActor]: ...
    def restore_frame(self, frame_index: int) -> None: ...

def _hash_file(path: str) -> str: ...
