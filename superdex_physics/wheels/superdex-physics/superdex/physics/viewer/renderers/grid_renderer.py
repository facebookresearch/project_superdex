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

from typing import Literal

import numpy as np
import numpy.typing as npt
from superdex.physics.utils.coordinate_systems import CoordinateTransform
from superdex.physics.utils.decorators import override_from
from superdex.physics.utils.transformations import make_transform
from superdex.physics.viewer.backend import polyscope as ps
from superdex.physics.viewer.renderers.renderer import Renderer

########################################################################################


Axes = Literal["xy", "xz", "yz"]
"""Possible axes for the grid plane."""

Styles = Literal["grid", "checker"]
"""Possible styles for the grid."""


class GridRenderer(Renderer):
    """
    A renderer for creating and managing a customizable grid in a 3D scene using
    Polyscope. This class provides functionality to create, update, and manipulate a
    grid plane that can be positioned and oriented in 3D space. The grid can be
    customized in terms of size, line spacing, orientation, and colors. Infinite grids
    are also supported.
    """

    ####################################################################################
    # Members
    ####################################################################################

    # Private members.
    _render_struct: ps.SurfaceMesh | None
    _center: npt.NDArray[float]
    _size: float
    _period: float
    _axes: Axes
    _color_1: npt.NDArray[float]
    _color_2: npt.NDArray[float]
    _style: Styles
    _transform_dirty: bool
    _param_dirty: bool

    ####################################################################################
    # Constants
    ####################################################################################

    _VERTICES = np.array(
        [[-0.5, -0.5, 0], [-0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, -0.5, 0]]
    )
    """Vertex positions for the quad primitive used to represent the grid."""
    _PARAM = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    """Parametric coordinates (uvs) for the quad primitive used to represent the grid."""
    _FACES = np.array([[0, 1, 2, 3]])
    """Face indices for the quad primitive used to represent the grid."""
    _PROJECTION_AXES = {
        "xy": np.array([0, 0, 1]),
        "xz": np.array([0, 1, 0]),
        "yz": np.array([1, 0, 0]),
    }
    """Projection axes per axes."""
    _ROTATIONS = {
        "xy": np.array([0, 0, 0]),
        "xz": np.array([np.pi / 2, 0, 0]),
        "yz": np.array([0, np.pi / 2, 0]),
    }
    """Rotation vectors per axes."""
    _INFINITE_SIZE = 1e3
    """A large number used to represent infinite size. This is used to ensure that the
    grid is always visible in the scene."""

    ####################################################################################
    # Constructor
    ####################################################################################

    def __init__(
        self,
        name: str,
        coordinate_transform: CoordinateTransform,
        size: float,
        center: npt.NDArray[float],
        axes: Axes,
        period: float,
        style: Styles,
        color_1: npt.NDArray[float],
        color_2: npt.NDArray[float],
        double_sided: bool,
    ):
        """Initializes the grid renderer."""

        super().__init__(name, coordinate_transform)

        # Initialize rendering structure.
        # Generate a surface mesh consisting of a unit quad of length 1 centered at the
        # origin. We will rescale it through the transform on update.
        # We flip the faces if the coordinate system is left-handed to ensure the
        # appropriate winding order.
        flip_faces = coordinate_transform.encodes_reflection
        faces = self._FACES[:, ::-1] if flip_faces else self._FACES
        self._render_struct = None
        self._center = np.zeros(3, dtype=np.float32)
        self._size = size
        self._period = period
        self._axes = axes
        self._color_1 = np.zeros(3, dtype=np.float32)
        self._color_2 = np.zeros(3, dtype=np.float32)
        self._style = style
        self._transform_dirty = True
        self._param_dirty = True
        self._render_struct = ps.register_surface_mesh(
            self._name,
            self._VERTICES,
            faces,
            material="flat",
        )

        # Initialize render settings.
        self.set_center(center)
        self.set_size(size)
        self.set_period(period)
        self.set_axes(axes)
        self.set_color_1(color_1)
        self.set_color_2(color_2)
        self.set_style(style)
        self.set_double_sided(double_sided)

        # Force update the grid - This will initialize the transform and the
        # parameterized quantity displaying the grid.
        self.update()

    ####################################################################################
    # Render structure update and destruction
    ####################################################################################

    @override_from(Renderer)
    def update(self) -> None:
        """Updates the grid, updating its transform and parameterization to reflect the
        current settings of the grid."""

        assert self._render_struct is not None, "Render structure not initialized."

        # Retrieve the change of basis matrix.
        source_to_ps = self._coordinate_transform.source_to_target
        ps_to_source = self._coordinate_transform.target_to_source

        # Update transform matrix.
        is_infinite_grid = self._size == np.inf
        if self._transform_dirty or is_infinite_grid:
            center, size = self._center, self._size
            # If the grid is infinite, we use a large number to ensure it is always
            # visible in the scene, and compute the center on the fly to give the
            # illusion of an infinite grid.
            if is_infinite_grid:
                size = self._INFINITE_SIZE
                proj = self._PROJECTION_AXES[self._axes]
                camera_params = ps.get_view_camera_parameters()
                camera_pos_ps = camera_params.get_position()
                camera_pos = ps_to_source[:3, :3] @ camera_pos_ps
                camera_pos -= proj * np.dot(camera_pos, proj)
                center = (center % size) + size * np.round(camera_pos / size)

            # Compute the transform matrix.
            rotvec = self._ROTATIONS[self._axes]
            transform = make_transform(center, rotvec, size)
            transform_ps = source_to_ps @ transform  # Vertices in src coordinates.
            self.get_render_structure().set_transform(transform_ps)
            self._transform_dirty = False

        # Update (or create) parameterization.
        if self._param_dirty:
            size = self._size
            if size == np.inf:
                size = self._INFINITE_SIZE

            self.get_render_structure().add_parameterization_quantity(
                "UV",
                self._PARAM,
                enabled=True,
                viz_style=self._style,
                coords_type="unit",
                grid_colors=(self._color_1, self._color_2),
                checker_colors=(self._color_1, self._color_2),
                checker_size=self._period / size,
            )
            self._param_dirty = False

    @override_from(Renderer)
    def remove(self) -> None:
        """Removes the grid renderer from the scene and cleans up resources."""
        if self._render_struct is not None:
            self._render_struct.remove()
            self._render_struct = None

    ####################################################################################
    # Grid properties accessors
    ####################################################################################

    def get_name(self) -> str:
        """Returns the name of the grid renderer."""
        return self._name

    def get_render_structure(self) -> ps.SurfaceMesh:
        """Returns the underlying Polyscope surface mesh used for rendering."""
        assert self._render_struct is not None
        return self._render_struct

    def get_center(self) -> npt.NDArray[float]:
        """Returns the center position of the grid."""
        return self._center

    def set_center(self, center: npt.NDArray[float]) -> None:
        """Sets the center position of the grid."""
        center = np.asarray(center, dtype=np.float32)
        if center.shape != (3,):
            raise ValueError("Invalid center. Expected 3-element array.")
        self._center[:] = center
        self._transform_dirty = True

    def get_size(self) -> float:
        """Returns the size (side length) of the grid."""
        return self._size

    def set_size(self, size: float) -> None:
        """Sets the size (side length) of the grid."""
        size = float(size)
        if size <= 0:
            raise ValueError("Size must be positive.")
        self._size = size
        self._transform_dirty = True
        self._param_dirty = True

    def get_period(self) -> float:
        """Returns the grid line period (spacing between grid lines)."""
        return self._period

    def set_period(self, period: float) -> None:
        """Sets the grid line period (spacing between grid lines)."""
        if period <= 0:
            raise ValueError("Period must be positive.")
        self._period = period
        self._param_dirty = True

    def get_axes(self) -> Axes:
        """Returns the orientation axes of the grid plane."""
        return self._axes

    def set_axes(self, axes: Axes) -> None:
        """Sets the orientation axes of the grid plane. Possible values are "xy", "xz"
        and "yz"."""
        if axes not in self._PROJECTION_AXES:
            raise ValueError(f"Invalid axes: {axes}.")
        self._axes = axes
        self._transform_dirty = True

    def get_color_1(self) -> npt.NDArray[float]:
        """Returns the RGB values of the first color. If the style is "grid", this
        corresponds to the grid lines."""
        return self._color_1

    def set_color_1(self, color: npt.ArrayLike) -> None:
        """Sets the RGB values of the first color. If the style is "grid", this
        corresponds to the grid lines."""
        color = np.asarray(color, dtype=np.float32)
        if color.shape != (3,):
            raise ValueError("Invalid color. Expected 3-element array.")
        self._color_1[:] = color
        self._param_dirty = True

    def get_color_2(self) -> npt.NDArray[float]:
        """Returns the RGB values of the second color. If the style is "grid", this
        corresponds to the grid cells."""
        return self._color_2

    def set_color_2(self, color: npt.ArrayLike) -> None:
        """Sets the RGB values of the second color. If the style is "grid", this
        corresponds to the grid cells."""
        color = np.asarray(color, dtype=np.float32)
        if color.shape != (3,):
            raise ValueError("Invalid color. Expected 3-element array.")
        self._color_2[:] = color
        self._param_dirty = True

    def get_style(self) -> Styles:
        """Returns the style of the grid. Possible values are "grid" and "checker"."""
        return self._style

    def set_style(self, style: Styles) -> None:
        """Sets the style of the grid. Possible values are "grid" and "checker"."""
        if style not in ("grid", "checker"):
            raise ValueError(f"Invalid style: {style}.")
        self._style = style
        self._param_dirty = True

    def is_double_sided(self) -> bool:
        """Returns whether the grid is double-sided."""
        return self.get_render_structure().get_back_face_policy() != "cull"

    def set_double_sided(self, double_sided: bool) -> None:
        """Sets whether the grid is double-sided."""
        policy = "identical" if double_sided else "cull"
        self.get_render_structure().set_back_face_policy(policy)
