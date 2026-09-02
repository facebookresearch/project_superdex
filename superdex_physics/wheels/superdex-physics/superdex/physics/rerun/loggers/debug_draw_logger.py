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

import numpy as np
import rerun as rr
from superdex.physics import DebugDraw, Scene
from superdex.physics.rerun.loggers.base import Logger
from superdex.physics.utils.coordinate_systems import CoordinateTransform
from superdex.physics.utils.decorators import override_from
from superdex.physics.utils.transformations import apply_linear_map

########################################################################################


class DebugDrawLogger(Logger):
    """
    Logger for debug draw visualization to rerun.

    This class handles logging debug draw primitives (lines and spheres) from
    a Mochi scene to rerun.
    """

    ####################################################################################
    # Members
    ####################################################################################

    _debug_draw: DebugDraw | None

    ####################################################################################
    # Constructor
    ####################################################################################

    def __init__(
        self,
        scene: Scene,
        coordinate_transform: CoordinateTransform,
        entity_path_prefix: str = "world/debug_draw",
    ):
        """Initialize the debug draw logger.

        Args:
            scene: The Mochi scene containing the debug draw data.
            coordinate_transform: Converter for coordinate system transformations.
            entity_path_prefix: Prefix for the entity path in the rerun hierarchy.
        """
        super().__init__(
            entity_path=entity_path_prefix,
            coordinate_transform=coordinate_transform,
        )
        self._debug_draw = scene.get_debug_draw()

    ####################################################################################
    # Logging
    ####################################################################################

    @override_from(Logger)
    def log(self, static: bool = False) -> None:
        """Log the debug draw primitives to rerun.

        Args:
            static: If True, log as static (timeless) data.
        """
        if self._debug_draw is None:
            return
        debug_draw = self._debug_draw

        if not debug_draw.is_enabled():
            return

        # Gather the latest debug draw data from the scene.
        data = debug_draw.gather_data()
        num_line_vertices = len(data.line_vertices.positions)
        num_spheres = len(data.spheres.positions)

        # Retrieve the change of basis matrix.
        source_to_target = self._coordinate_transform.source_to_target

        # Handle line rendering.
        if num_line_vertices > 0:
            line_vertices = np.asarray(data.line_vertices.positions)
            line_colors = np.asarray(data.line_vertices.colors)
            line_colors = line_colors[:, 0:3]  # Keep as uint8 (0-255)

            # Convert to the target coordinate system.
            line_vertices_target = apply_linear_map(
                source_to_target[:3, :3], line_vertices
            )

            # Reshape for LineStrips3D - pairs of vertices form line segments.
            # Each pair [start, end] is a separate line strip.
            line_strips = line_vertices_target.reshape(-1, 2, 3)
            strip_colors = line_colors[::2]  # Use start vertex color for each line

            rr.log(
                f"{self._entity_path}/lines",
                rr.LineStrips3D(
                    strips=line_strips,
                    colors=strip_colors,
                ),
                static=static,
            )
        else:
            # Clear lines if none present.
            rr.log(f"{self._entity_path}/lines", rr.Clear(recursive=False))

        # Handle sphere rendering.
        if num_spheres > 0:
            sphere_centers = np.asarray(data.spheres.positions)
            sphere_radii = np.asarray(data.spheres.radii)
            sphere_colors = np.asarray(data.spheres.colors)
            sphere_colors = sphere_colors[:, 0:3]  # Keep as uint8 (0-255)

            # Convert to the target coordinate system.
            sphere_centers_target = apply_linear_map(
                source_to_target[:3, :3], sphere_centers
            )

            rr.log(
                f"{self._entity_path}/spheres",
                rr.Points3D(
                    positions=sphere_centers_target,
                    radii=sphere_radii,
                    colors=sphere_colors,
                ),
                static=static,
            )
        else:
            # Clear spheres if none present.
            rr.log(f"{self._entity_path}/spheres", rr.Clear(recursive=False))

    @override_from(Logger)
    def clear(self) -> None:
        """Clear/remove this entity from rerun."""
        rr.log(self._entity_path, rr.Clear(recursive=True))
        self._debug_draw = None
