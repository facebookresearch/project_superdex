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
Configuration dataclass for the UnrealCV viewer.
"""

from __future__ import annotations

from enum import Enum

from superdex.physics.utils.configclasses import configclass
from superdex.physics.utils.coordinate_systems import CoordinateSystem

########################################################################################


class CaptureFormat(Enum):
    """Image capture format for UnrealCV camera responses."""

    NPY = "npy"
    """NumPy binary format. Returns raw BGRA uint8 arrays over the wire.
    Fastest for in-process consumption but larger payloads."""

    PNG = "png"
    """PNG compressed format. Smaller payloads over the wire but requires
    decode on the client side."""


class CaptureMode(Enum):
    """Capture timing mode for UnrealCV camera responses."""

    PIPELINED = "pipelined"
    """1-frame-lag pipeline: start an async capture at the end of each frame
    and poll the *previous* frame's result at the start of the next frame.
    Lower per-frame latency at the cost of returning images that are one
    simulation step behind."""

    SYNCHRONIZED = "synchronized"
    """Same-frame capture: start an async capture, then poll it within the
    same render() call.  The returned images correspond to the current
    simulation state, but the frame takes longer because the GPU capture
    must complete before render() returns."""


########################################################################################


@configclass
class UnrealCVViewerCfg:
    """Configuration options for the UnrealCV viewer."""

    host: str = "localhost"
    """Hostname of the Unreal Engine instance running UnrealCV."""

    port: int = 9000
    """Port for UnrealCV TCP connection."""

    size: tuple[int, int] | None = (1280, 720)
    """Width and height of captured images. If None, uses the UE viewport size."""

    offscreen: bool = False
    """Whether rendering should occur offscreen (RGB_ARRAY mode). If True, render()
    will return captured images instead of displaying in UE."""

    coordinate_system: CoordinateSystem | str | None = None
    """Coordinate system used by the mochi simulation. If None, the default coordinate
    system (Polyscope convention: right-handed, Y-up, -Z-forward) will be used. The
    viewer will automatically convert to Unreal Engine's coordinate system."""

    start_paused: bool = False
    """Whether to start with the simulation paused."""

    actor_name_mapping: dict[str, str] | None = None
    """Optional mapping from mochi actor names to Unreal Engine actor names. If provided,
    actor names will be translated when sending commands to UE. This is useful when
    the UE level uses different naming conventions than the mochi scene."""

    articulated_actor_mapping: dict[str, dict] | None = None
    """Optional mapping from mochi articulated actors to UE skeletal meshes.
    Use this for articulated actors (robots) where transforms from get_articulated_pose()
    should be used to drive bone transforms.
    Example: {
        "fer_allegrohandv5_right": {
            "ue_actor": "BP_Robot_Franka_C_1",
            "link_to_bone": {
                "base": "base",
                "fer_link0": "fer_link0",
                "fer_link1": "fer_link1",
            }
        }
    }
    If link_to_bone is not provided, mochi link names are used directly as bone names."""

    connection_timeout: float = 5.0
    """Timeout in seconds for the initial connection to the UnrealCV server."""

    retry_attempts: int = 3
    """Number of connection retry attempts before giving up."""

    camera_id: int = 0
    """The camera ID to use for rendering and image capture in UnrealCV."""

    meters_to_cm: float = 100.0
    """Scale factor for converting from mochi units (meters) to UE units (centimeters)."""

    capture_format: CaptureFormat = CaptureFormat.NPY
    """Image capture format for camera responses. NPY returns raw numpy arrays
    (faster for in-process use, larger payloads). PNG returns compressed images
    (smaller payloads, requires decode)."""

    capture_mode: CaptureMode = CaptureMode.PIPELINED
    """Capture timing mode. PIPELINED starts an async capture at the end of
    each frame and polls the previous frame's result at the start of the next
    frame (lower latency, images lag by one step). SYNCHRONIZED starts and
    polls the capture within the same render() call (no lag, but higher
    per-frame cost)."""

    camera_mapping: dict[str, dict] | None = None
    """Optional mapping from observation keys (e.g. ``"image.depth_camera_0_rgb"``) to camera
    configuration dicts containing ``ue_name``, ``image_width``, and ``image_height``.
    When provided, the mapping is resolved to camera indices immediately after
    connecting to the UnrealCV server, and camera capture sizes are configured
    per-camera based on each camera's dimensions.

    Example::

        camera_mapping = {
            "image.depth_camera_0_rgb": {
                "ue_name": "FusionCamera0",
                "image_width": 640,
                "image_height": 480,
            },
            "image.wrist": {
                "ue_name": "WristFusionCamera",
                "image_width": 320,
                "image_height": 240,
            },
        }
    """
