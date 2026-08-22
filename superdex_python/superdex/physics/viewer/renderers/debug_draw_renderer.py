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
from superdex.physics import DebugDraw, Scene
from superdex.physics.utils.coordinate_systems import CoordinateTransform
from superdex.physics.utils.decorators import override_from
from superdex.physics.utils.transformations import apply_linear_map
from superdex.physics.viewer.backend import polyscope as ps
from superdex.physics.viewer.renderers.renderer import Renderer

########################################################################################


class DebugDrawRenderer(Renderer):
    """
    Renderer for debug draw visualization.
    """

    ####################################################################################
    # Members
    ####################################################################################

    _debug_draw: DebugDraw | None
    _lines_struct: ps.CurveNetwork | None
    _lines_radius: float
    _spheres_struct: ps.PointCloud | None
    _spheres_scale: float

    ####################################################################################
    # Constants
    ####################################################################################

    _DEFAULT_LINE_RADIUS = 0.00005  # 0.5mm
    """Default line radius for rendered lines. Adjusted based on the scale of the
    Allegro-in-Hand scene."""
    _DEFAULT_RADIUS_SCALE = 1
    """Default scaling factor for sphere radii."""

    ####################################################################################
    # Constructor
    ####################################################################################

    def __init__(self, coordinate_transform: CoordinateTransform) -> None:
        """
        Initialize the debug draw renderer with default settings.
        """

        # Initialize base class with a default name
        super().__init__("DebugDraw", coordinate_transform)

        # Initialize with no debug draw instance.
        self._debug_draw = None
        self._lines_struct = None
        self._spheres_struct = None

        # Set default rendering properties.
        self._lines_radius = DebugDrawRenderer._DEFAULT_LINE_RADIUS
        self._spheres_scale = DebugDrawRenderer._DEFAULT_RADIUS_SCALE

    ####################################################################################
    # Functions handling the render structure creation and update
    ####################################################################################

    def reset(self, scene: Scene | None) -> None:
        """
        Reset the renderer with a (possibly) new scene and update the visualization.
        If no scene is provided, the debug draw renderer will be disabled.
        """

        # Update the debug draw instance.
        # If no scene is provided, disable debug draw and remove associated renderers.
        # Otherwise, update the debug draw instance and create renderers if necessary.
        if scene is None:
            self.remove()
        else:
            self._debug_draw = scene.get_debug_draw()
            self.update()

    @override_from(Renderer)
    def update(self) -> None:
        """
        Update the visualization by gathering debug draw data and refreshing the
        rendered elements.
        """

        # Early exit if there is no bound debug draw instance.
        debug_draw = self._debug_draw
        if debug_draw is None:
            return

        # Early exit if debug drawing is disabled.
        if not debug_draw.is_enabled():
            if self._spheres_struct is not None:
                self._spheres_struct.set_enabled(False)
            if self._lines_struct is not None:
                self._lines_struct.set_enabled(False)
            return

        # Gather the latest debug draw data from the scene.
        data = debug_draw.gather_data()
        num_line_vertices = len(data.line_vertices.positions)
        num_spheres = len(data.spheres.positions)

        # Retrieve the change of basis matrix.
        source_to_ps = self._coordinate_transform.source_to_target

        # Handle line rendering.
        if num_line_vertices > 0:
            line_indices = np.arange(num_line_vertices).reshape(-1, 2)
            line_vertices = np.asarray(data.line_vertices.positions)
            line_colors = np.asarray(data.line_vertices.colors)
            line_colors = line_colors[:, 0:3] / 255.0

            # Convert to the target coordinate system.
            line_vertices_ps = apply_linear_map(source_to_ps[:3, :3], line_vertices)

            # Register the curve network with polyscope.
            lines_struct = self._lines_struct
            if lines_struct is None or lines_struct.n_nodes() != num_line_vertices:
                lines_struct = ps.register_curve_network(
                    "[Debug Draw] Lines",
                    line_vertices_ps,
                    line_indices,
                    radius=self._lines_radius,
                    material="flat",
                )
                self._lines_struct = lines_struct
            else:
                lines_struct.update_node_positions(line_vertices_ps)
                lines_struct.set_radius(self._lines_radius)
            lines_struct.add_color_quantity("Colors", line_colors, enabled=True)
            lines_struct.set_enabled(True)

        # No lines to display, hide existing line structure if it exists.
        elif self._lines_struct is not None:
            self._lines_struct.set_enabled(False)

        # Handle sphere rendering.
        if num_spheres > 0:
            # Extract sphere data from debug draw.
            # Apply global scaling factor to all sphere radii.
            sphere_centers = np.asarray(data.spheres.positions)
            sphere_radii = np.asarray(data.spheres.radii)
            sphere_radii = sphere_radii * self._spheres_scale
            sphere_colors = np.asarray(data.spheres.colors)
            sphere_colors = sphere_colors[:, 0:3] / 255.0

            # Convert to the target coordinate system.
            sphere_centers_ps = apply_linear_map(source_to_ps[:3, :3], sphere_centers)

            # Register spheres as a point cloud with quad rendering.
            # Set per-sphere radius and color data.
            # Use the radius data to control point sizes, disable auto-scaling.
            spheres_struct = self._spheres_struct
            if spheres_struct is None or spheres_struct.n_points() != num_spheres:
                spheres_struct = ps.register_point_cloud(
                    "[Debug Draw] Spheres",
                    sphere_centers_ps,
                    point_render_mode="quad",
                    material="flat",
                )
                self._spheres_struct = spheres_struct
            else:
                spheres_struct.update_point_positions(sphere_centers_ps)
            spheres_struct.add_scalar_quantity("Radius", sphere_radii)
            spheres_struct.add_color_quantity("Colors", sphere_colors, enabled=True)
            spheres_struct.set_point_radius_quantity("Radius", autoscale=False)
            spheres_struct.set_enabled(True)

        # No spheres to display, hide existing sphere structure if it exists.
        elif self._spheres_struct is not None:
            self._spheres_struct.set_enabled(False)

    @override_from(Renderer)
    def remove(self) -> None:
        """
        Remove all rendered structures and clean up resources.
        """

        if self._spheres_struct is not None:
            self._spheres_struct.remove()
            self._spheres_struct = None
        if self._lines_struct is not None:
            self._lines_struct.remove()
            self._lines_struct = None
        self._debug_draw = None

    ####################################################################################
    # Rendering properties
    ####################################################################################

    def get_sphere_radius_scale(self) -> float:
        """
        Get the current sphere radius scaling factor.
        """
        return self._spheres_scale

    def set_sphere_radius_scale(self, value: float) -> None:
        """
        Set the sphere radius scaling factor. Note that this operation triggers a full
        update.
        """
        self._spheres_scale = value
        self.update()  # No way to update radii, perform full update.

    def get_line_radius(self) -> float:
        """
        Get the current line radius for rendered lines.
        """
        return self._lines_radius

    def set_line_radius(self, value: float) -> None:
        """
        Set the line radius and update the existing line structure if available.
        """
        self._lines_radius = value
        if self._lines_struct is not None:
            self._lines_struct.set_radius(value)
