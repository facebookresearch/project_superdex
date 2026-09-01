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

"""
State management dataclasses for the UnrealCV viewer.

These dataclasses follow the same patterns as the Polyscope viewer's state management.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from superdex.physics import Actor, ActorHandle, Scene, SceneHandle
from superdex.physics.utils.coordinate_systems import (
    CoordinateSystem,
    CoordinateTransform,
)
from superdex.physics.viewer.unrealcv.unrealcv_viewer_cfg import (
    CaptureFormat,
    CaptureMode,
)
from superdex.physics.viewer.utils.aabb import AABB

if TYPE_CHECKING:
    from .updaters.unrealcv_updater import UnrealCVUpdater

########################################################################################


@dataclasses.dataclass
class UnrealCVActorState:
    """Structure holding the state of an actor in the UnrealCV scene."""

    handle: ActorHandle
    """Unique handle of the actor."""

    instance: Actor
    """Pointer to the mochi actor."""

    ue_actor_name: str
    """The name of the corresponding actor in Unreal Engine."""

    updater: "UnrealCVUpdater | None" = None
    """Associated updater, if any."""


@dataclasses.dataclass
class UnrealCVCameraState:
    """Structure holding the camera state for UnrealCV."""

    camera_id: int = 0
    """The camera ID in UnrealCV (typically 0)."""

    use_follow_camera: bool = False
    """If True, the camera will follow the scene actors."""

    frame_camera_on_next_update: bool = False
    """If True, the camera will frame the scene on the next frame."""

    frame_camera_direction: npt.NDArray[np.floating] | None = None
    """Direction of the camera when framing the scene."""

    frame_fly_to: bool = False
    """If True, animate the camera to the target position."""

    automatic_distance: bool = False
    """If True, automatically compute the distance to frame the scene."""

    smoothing: float = 0.4
    """Smoothing factor for camera movement."""

    smoothed_target_position: npt.NDArray[np.floating] | None = None
    """Smoothed target position for the follow camera."""

    smoothed_position: npt.NDArray[np.floating] | None = None
    """Smoothed camera position (look_from). Updated by follow camera."""

    cached_rotation_ue: npt.NDArray[np.floating] | None = None
    """Last-set UE rotation (pitch, yaw, roll). Avoids GET during render."""

    distance: float = 500.0
    """Current view distance in UE units (centimeters)."""

    smoothed_distance: float = 500.0
    """Smoothed view distance."""


@dataclasses.dataclass
class UnrealCVSceneState:
    """Structure holding the state of the scene in UnrealCV."""

    instance: Scene | None = None
    """Pointer to the mochi scene."""

    handle: SceneHandle | None = None
    """Handle to the mochi scene."""

    actors: dict[ActorHandle, UnrealCVActorState] = dataclasses.field(
        default_factory=dict
    )
    """Actors present in the scene, keyed by unique actor handle."""

    aabb: AABB = dataclasses.field(default_factory=AABB.empty)
    """Bounding box of the scene."""

    selected_actor: ActorHandle | None = None
    """Handle of the currently selected actor (if any)."""

    excluded_actors: set[str] = dataclasses.field(default_factory=set)
    """Actor names that should be excluded from rendering."""


@dataclasses.dataclass
class UnrealCVViewerState:
    """
    Complete state structure for the UnrealCV viewer.

    This state object is passed to different components to allow them to access
    and modify the state of the viewer.
    """

    # Coordinate system (must be specified - no defaults)
    coordinate_system: CoordinateSystem
    """Coordinate system of the source (mochi) simulation."""

    coordinate_transform: CoordinateTransform
    """Transform for converting from mochi to Unreal Engine coordinates."""

    # Connection state
    connected: bool = False
    """Whether the viewer is connected to the UnrealCV server."""

    # General state
    paused: bool = False
    """Whether the simulation is paused."""

    step_once: bool = False
    """If True, the simulation must step once and then pause."""

    requires_update: bool = False
    """If True, the viewer must update the scene actors."""

    offscreen: bool = False
    """Whether rendering is offscreen (RGB_ARRAY mode)."""

    close_requested: bool = False
    """If True, the user requested to close the viewer."""

    # Unit conversion
    meters_to_cm: float = 100.0
    """Scale factor for converting from meters to centimeters."""

    # Scene state
    scene: UnrealCVSceneState = dataclasses.field(default_factory=UnrealCVSceneState)
    """State of the scene."""

    # Camera state
    camera: UnrealCVCameraState = dataclasses.field(default_factory=UnrealCVCameraState)
    """State of the camera."""

    # Use center of mass transform
    use_com_transform: bool = True
    """If True, use CoM transform instead of root transform for rigid actors."""

    # Capture format
    capture_format: CaptureFormat = CaptureFormat.NPY
    """Image capture format (npy or png) for camera responses."""

    # Capture mode
    capture_mode: CaptureMode = CaptureMode.PIPELINED
    """Capture timing mode (pipelined or synchronized)."""
