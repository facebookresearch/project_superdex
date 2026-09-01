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
import numpy.typing as npt
from superdex.physics.utils.coordinate_systems import CoordinateTransform
from superdex.physics.utils.decorators import override_from
from superdex.physics.utils.transformations import apply_linear_map
from superdex.physics.viewer.backend import polyscope as ps
from superdex.physics.viewer.renderers.renderer import Renderer

########################################################################################


class PointCloudRenderer(Renderer):
    """
    Renderer for displaying 3D point clouds using Polyscope. Provides functionality
    to manage and render point cloud geometries, including point coordinates, radii,
    and colors. The PointCloudRenderer supports both global and per-point radius and
    color specifications. The class follows the same patterns as other renderers in
    the SuperDex Physics viewer, including stack-based property management for easy overrides.
    """

    ####################################################################################
    # Members
    ####################################################################################

    # Point cloud geometry.
    _coordinates: npt.NDArray[float]

    # Render settings (stack-based for easy overrides).
    _color: list[npt.NDArray[float]]
    _transparency: list[float]

    # Backing Polyscope render structure.
    _render_struct: ps.PointCloud | None
    _coordinates_dirty: bool

    # Per-point radii (optional).
    _radii: npt.NDArray[float] | None
    _radii_dirty: bool

    # Per-point colors (optional).
    _colors: npt.NDArray[float] | None
    _colors_dirty: bool

    ####################################################################################
    # Constants
    ####################################################################################

    _TAG: str = "[PointCloud]"
    """Tag used to identify PointCloudRenderer-related render structures."""

    _DEFAULT_RADIUS: float = 0.01
    """Default radius for all points."""

    ####################################################################################
    # Constructor
    ####################################################################################

    def __init__(
        self,
        name: str,
        coordinates: npt.NDArray[float] | npt.ArrayLike,
        coordinate_transform: CoordinateTransform,
        radius: float | None = None,
        radii: npt.NDArray[float] | npt.ArrayLike | None = None,
        color: npt.NDArray[float] | npt.ArrayLike | None = None,
        colors: npt.NDArray[float] | npt.ArrayLike | None = None,
        transparency: float | None = None,
    ) -> None:
        """Initialize the PointCloudRenderer with given parameters.

        Args:
            name: Name identifier for this renderer.
            coordinates: Nx3 array of point coordinates in global space.
            coordinate_transform: Converter for coordinate system transformations.
            radius: Global radius for all points. If not provided, a default value
                will be used. Ignored if per-point radii are specified.
            radii: Optional N-element array of per-point radii. If provided, overrides
                the global radius.
            color: RGB color for all points as a 3-element array. If not provided,
                a default color will be used. Ignored if per-point colors are specified.
            colors: Optional Nx3 array of per-point RGB colors. If provided, overrides
                the global color.
            transparency: Transparency level (0.0 = opaque, 1.0 = fully transparent).
        """

        # Initialize base class.
        super().__init__(name, coordinate_transform)

        # Initialize rendering structure.
        self._render_struct = None
        self._radii = None
        self._colors = None
        self._color = []
        self._transparency = []
        self.set_points(coordinates, radii, colors)
        self._create_render_structure()
        self.update()

        # Apply provided settings or defaults.
        if color is not None:
            self.set_color(color)
        if transparency is not None:
            self.set_transparency(transparency)

        # Set global radius (only used if per-point radii is None).
        if radius is None:
            radius = self._DEFAULT_RADIUS
        self.set_radius(radius)

    ####################################################################################
    # General Management Methods
    ####################################################################################

    def _create_render_structure(self) -> None:
        """Creates the backing Polyscope render structure. Called internally during
        initialization and whenever the point cloud topology changes."""

        tagged_name = f"{self._TAG} {self._name}"

        # Convert coordinates to target coordinate system.
        source_to_ps = self._coordinate_transform.source_to_target
        coordinates_ps = apply_linear_map(source_to_ps[:3, :3], self._coordinates)

        # Generate point cloud structure.
        self._render_struct = ps.register_point_cloud(
            name=tagged_name,
            points=coordinates_ps,
        )

    @override_from(Renderer)
    def update(self) -> None:
        """Update the renderer's visualization if any properties have changed."""

        # Retrieve the change of basis matrix.
        source_to_ps = self._coordinate_transform.source_to_target

        # Update coordinates if dirty.
        if self._coordinates_dirty and self.get_render_structure().is_enabled():
            if self.get_render_structure().n_points() != self._coordinates.shape[0]:
                self._create_render_structure()
            else:
                coordinates_ps = apply_linear_map(
                    source_to_ps[:3, :3], self._coordinates
                )
                self.get_render_structure().update_point_positions(coordinates_ps)
            self._coordinates_dirty = False

        # Update per-point radii if dirty.
        if self._radii_dirty and self.get_render_structure().is_enabled():
            if self._radii is not None:
                self.get_render_structure().add_scalar_quantity(
                    "Radii", self._radii, enabled=False
                )
                self.get_render_structure().set_point_radius_quantity(
                    "Radii", autoscale=False
                )
            else:
                # Clear per-point radii if set to None.
                self.get_render_structure().remove_quantity(
                    "Radii", error_if_absent=False
                )
                self.get_render_structure().clear_point_radius_quantity()
            self._radii_dirty = False

        # Update per-point colors if dirty.
        if self._colors_dirty and self.get_render_structure().is_enabled():
            if self._colors is not None:
                self.get_render_structure().add_color_quantity(
                    "Colors", self._colors, enabled=True
                )
            else:
                self.get_render_structure().remove_quantity(
                    "Colors", error_if_absent=False
                )
            self._colors_dirty = False

    @override_from(Renderer)
    def remove(self) -> None:
        """Remove the renderer and clean up resources."""
        if self._render_struct is not None:
            self._render_struct.remove()
            self._render_struct = None

    def is_dirty(self) -> bool:
        """Returns True if the render structure is marked as dirty."""
        return self._coordinates_dirty or self._radii_dirty or self._colors_dirty

    ####################################################################################
    # Geometry Methods
    ####################################################################################

    def set_points(
        self,
        coordinates: npt.NDArray[float] | npt.ArrayLike,
        radii: npt.NDArray[float] | npt.ArrayLike | None = None,
        colors: npt.NDArray[float] | npt.ArrayLike | None = None,
    ) -> None:
        """Set the points of the point cloud. The coordinates are expected to be in
        global space. The coordinate transform will be applied to convert to the
        rendering coordinate system.

        Args:
            coordinates: Nx3 array of point coordinates in global space.
            radii: Optional N-element array of per-point radii. If None, the global
                radius will be used for all points.
            colors: Optional Nx3 array of per-point RGB colors. If None, the global
                color will be used for all points.
        """

        # Validate input data and generate copies to avoid modifying originals.
        coordinates = np.array(coordinates, dtype=np.float32)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("Invalid coordinates. Expected Nx3 array.")

        # Validate per-point radii if provided.
        if radii is not None:
            radii = np.asarray(radii, dtype=np.float32)
            if radii.ndim != 1:
                raise ValueError("Invalid radii. Expected 1D array.")
            if radii.shape[0] != coordinates.shape[0]:
                raise ValueError(
                    "Invalid radii. Expected same number of elements as points."
                )
            self._radii = radii.copy()
        else:
            self._radii = None

        # Validate per-point colors if provided.
        if colors is not None:
            colors = np.asarray(colors, dtype=np.float32)
            if colors.ndim != 2 or colors.shape[1] != 3:
                raise ValueError("Invalid colors. Expected Nx3 array.")
            if colors.shape[0] != coordinates.shape[0]:
                raise ValueError(
                    "Invalid colors. Expected same number of elements as points."
                )
            self._colors = colors.copy()
        else:
            self._colors = None

        # Update members.
        self._coordinates = coordinates
        self._coordinates_dirty = True
        self._radii_dirty = True
        self._colors_dirty = True

    def get_coordinates(self) -> npt.NDArray[float]:
        """Returns the coordinates of the point cloud."""
        return self._coordinates

    def get_num_points(self) -> int:
        """Returns the number of points in the point cloud."""
        return self._coordinates.shape[0]

    ####################################################################################
    # Per-Point Appearance Properties
    ####################################################################################

    def get_point_radii(self) -> npt.NDArray[float] | None:
        """Returns the per-point radii array, or None if using global radius."""
        return self._radii

    def set_point_radii(self, radii: npt.NDArray[float] | npt.ArrayLike | None) -> None:
        """Sets per-point radii for the point cloud.

        Args:
            radii: N-element array of per-point radii, or None to clear per-point
                radii and use the global radius.
        """
        if radii is not None:
            radii = np.asarray(radii, dtype=np.float32)
            if radii.ndim != 1:
                raise ValueError("Invalid radii. Expected 1D array.")
            if radii.shape[0] != self._coordinates.shape[0]:
                raise ValueError(
                    "Invalid radii. Expected same number of elements as points. If you "
                    "want to change the number of points, call set_points() instead."
                )
            self._radii = radii.copy()
        else:
            self._radii = None
        self._radii_dirty = True

    def get_point_colors(self) -> npt.NDArray[float] | None:
        """Returns the per-point colors array, or None if using global color."""
        return self._colors

    def set_point_colors(
        self, colors: npt.NDArray[float] | npt.ArrayLike | None
    ) -> None:
        """Sets per-point colors for the point cloud.

        Args:
            colors: Nx3 array of per-point RGB colors with values in [0, 1],
                or None to clear per-point colors and use the global color.
        """
        if colors is not None:
            colors = np.asarray(colors, dtype=np.float32)
            if colors.ndim != 2 or colors.shape[1] != 3:
                raise ValueError("Invalid colors. Expected Nx3 array.")
            if colors.shape[0] != self._coordinates.shape[0]:
                raise ValueError(
                    "Invalid colors. Expected same number of elements as points. If "
                    "you want to change the number of points, call set_points() "
                    "instead."
                )
            self._colors = colors.copy()
        else:
            self._colors = None
        self._colors_dirty = True

    ####################################################################################
    # Appearance Properties
    ####################################################################################

    def is_enabled(self) -> bool:
        """Returns whether the point cloud rendering is enabled."""
        return self.get_render_structure().is_enabled()

    def set_enabled(self, enabled: bool) -> None:
        """Enables or disables the point cloud rendering."""
        enabled = bool(enabled)
        self.get_render_structure().set_enabled(enabled)

    def get_render_structure(self) -> ps.PointCloud:
        """Returns the underlying Polyscope point cloud structure."""
        assert self._render_struct is not None
        return self._render_struct

    def get_radius(self) -> float:
        """Returns the current global point radius."""
        return self.get_render_structure().get_radius()

    def set_radius(self, radius: float) -> None:
        """Sets a new global radius for all points. This radius is used when per-point
        radii are not specified.

        Args:
            radius: Global radius for all points.
        """
        radius = float(radius)
        self.get_render_structure().set_radius(radius, relative=False)

    def get_color(self) -> npt.NDArray[float]:
        """Returns the current global point color."""
        return np.asarray(self.get_render_structure().get_color(), dtype=np.float32)

    def set_color(self, color: npt.ArrayLike) -> None:
        """Sets a new color for all points.

        Args:
            color: RGB color as a 3-element array with values in [0, 1].
        """
        color = np.asarray(color, dtype=np.float32)
        if color.shape != (3,):
            raise ValueError("Invalid color. Expected 3-element array.")
        self.get_render_structure().set_color(color)

    def push_color(self, color: npt.ArrayLike) -> None:
        """Pushes a new color onto the stack.

        Args:
            color: RGB color as a 3-element array with values in [0, 1].
        """
        color = np.asarray(color, dtype=np.float32)
        if color.shape != (3,):
            raise ValueError("Invalid color. Expected 3-element array.")
        self._color.append(self.get_color())
        self.get_render_structure().set_color(color)

    def pop_color(self) -> None:
        """Pops the current color from the stack."""
        self.get_render_structure().set_color(self._color[-1])
        self._color.pop()

    def get_transparency(self) -> float:
        """Returns the current transparency level."""
        return self.get_render_structure().get_transparency()

    def set_transparency(self, transparency: float) -> None:
        """Sets a new transparency level.

        Args:
            transparency: Transparency level (0.0 = opaque, 1.0 = fully transparent).
        """
        transparency = float(transparency)
        self.get_render_structure().set_transparency(transparency)

    def push_transparency(self, transparency: float) -> None:
        """Pushes a new transparency level onto the stack.

        Args:
            transparency: Transparency level (0.0 = opaque, 1.0 = fully transparent).
        """
        transparency = float(transparency)
        self._transparency.append(self.get_transparency())
        self.get_render_structure().set_transparency(transparency)

    def pop_transparency(self) -> None:
        """Pops the current transparency level from the stack."""
        self.get_render_structure().set_transparency(self._transparency[-1])
        self._transparency.pop()
