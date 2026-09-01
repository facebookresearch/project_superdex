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
Configuration dataclass for the MochiRenderer viewer.
"""

from __future__ import annotations

from superdex.physics.utils.configclasses import configclass

########################################################################################


@configclass
class CameraCfg:
    """Configuration for a single named camera.

    Camera pose can be specified in two ways (checked in this order):
      1. ``position`` + ``rotation`` — full 6-DOF transform (preferred).
      2. ``look_from`` + ``look_at`` — classic look-at (fallback).
    If neither pair is set the server default is used.
    """

    width: int = 640
    """Width of the camera image in pixels."""

    height: int = 480
    """Height of the camera image in pixels."""

    position: tuple[float, float, float] | None = None
    """Camera position (x, y, z). Used with ``rotation``."""

    rotation: tuple[float, float, float, float] | None = None
    """Camera orientation as an XYZW quaternion. Used with ``position``."""

    look_from: tuple[float, float, float] | None = None
    """Camera eye position (x, y, z). Fallback when ``position``/``rotation``
    are not set. None means use server default."""

    look_at: tuple[float, float, float] | None = None
    """Camera target position (x, y, z). Fallback when ``position``/``rotation``
    are not set. None means use server default."""

    horizontal_fov_deg: float | None = None
    """Horizontal field of view in degrees. If set, converted to vertical
    FOV and sent to the renderer server. None means use server default."""


########################################################################################


@configclass
class MochiRendererViewerCfg:
    """Configuration options for the MochiRenderer viewer."""

    host: str = "localhost"
    """Hostname of the mochi_renderer server."""

    port: int = 9000
    """Port for the mochi_renderer TCP connection."""

    size: tuple[int, int] = (640, 480)
    """Width and height of captured images."""

    offscreen: bool = False
    """Whether rendering should occur offscreen. If True, render() returns
    captured images as numpy arrays."""

    camera_name: str = "default"
    """Name of the camera to use for image capture."""

    cameras: dict[str, CameraCfg] | None = None
    """Optional multi-camera configuration. Keys are camera names, values are
    per-camera settings. When set, named cameras are created at startup."""

    connection_timeout: float = 5.0
    """Timeout in seconds for the initial connection to the server."""

    retry_attempts: int = 3
    """Number of connection retry attempts before giving up."""

    environment_gltf: str | None = None
    """Path to a static environment glTF file (table, room, etc.)."""

    actor_gltfs: dict[str, str] | None = None
    """Mapping from mochi actor name to glTF file path.
    Actors with a glTF mapping will be rendered using the glTF mesh
    instead of the physics mesh. For articulated actors, map each
    link name to its glTF file."""

    environment_ibl: str | None = None
    """Path to an HDR/EXR environment map for image-based lighting.
    When set, the IBL is loaded on connection and provides realistic
    indirect lighting and reflections."""

    skybox_visible: bool = False
    """Whether the IBL skybox is visible in the background. When False
    (default), the IBL still provides indirect lighting but the
    background remains the solid-color skybox."""
