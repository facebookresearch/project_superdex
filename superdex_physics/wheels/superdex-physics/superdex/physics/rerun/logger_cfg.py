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

from pathlib import Path

from superdex.physics.utils.configclasses import configclass
from superdex.physics.utils.coordinate_systems import CoordinateSystem

########################################################################################


@configclass
class RerunLoggerCfg:
    """Configuration for the RerunLogger."""

    application_id: str = "mochi"
    """Application ID for the rerun recording."""

    recording_id: str | None = None
    """Optional recording ID for this session."""

    recording_name: str | None = None
    """Optional human-readable recording name shown in the Rerun Viewer's
    recording list. Distinct from ``application_id`` (which groups recordings):
    set ``application_id`` to the task and ``recording_name`` to the specific
    experiment/run so sibling recordings are distinguishable."""

    spawn: bool = False
    """Spawn a local rerun viewer on initialization."""

    connect: bool = True
    """Connect to a running rerun viewer (default: localhost:9876)."""

    connect_addr: str | None = None
    """gRPC address to connect to (e.g., '192.168.1.100:9876')."""

    save_path: Path | str | None = None
    """Path to save .rrd recording file."""

    coordinate_system: CoordinateSystem | str | None = None
    """Source coordinate system for the scene data. If None, the default coordinate
    system (Polyscope convention: right-handed, Y-up, -Z-forward) will be used. You can
    specify a custom coordinate system using CoordinateSystem class or use named presets
    like "unity", "unreal", etc."""

    log_meshes: bool = True
    """Whether to log actor meshes."""

    log_transforms: bool = True
    """Whether to log actor transforms."""

    log_debug_draw: bool = True
    """Whether to log debug draw primitives."""

    use_timeline: bool = True
    """Whether to use rerun's timeline for frame sequencing."""

    mesh_assets_dir: str | None = None
    """Optional directory containing visual mesh files (DAE/GLB) for textured
    rendering. When set, the logger loads visual meshes from this directory
    instead of using Mochi's physics collision meshes. The directory should
    contain franka_description/ and dg_description/ subdirectories with
    URDF-source visual meshes."""
