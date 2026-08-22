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

from superdex.physics.utils.configclasses import configclass
from superdex.physics.utils.coordinate_systems import CoordinateSystem

########################################################################################


@configclass
class ViewerCfg:
    """Options for the Polyscope renderer."""

    backend: str = ""
    """Backend to use for rendering. If empty, the default backend will be used."""
    size: tuple[int, int] | None = None
    """Width and height of the renderer window. If None, the default size will be used."""
    offscreen: bool = False
    """Whether the rendering should occur offscreen."""
    start_paused: bool = False
    """Whether the rendering should start paused. If True, the user will need to press a
    button in the UI to start the simulation."""
    coordinate_system: CoordinateSystem | str | None = None
    """Coordinate system to use for visualization. If None, the default coordinate system
    (Polyscope convention: right-handed, Y-up, -Z-forward) will be used. You can specify
    a custom coordinate system using CoordinateSystem class or use named presets like
    "unity", "unreal", etc."""
