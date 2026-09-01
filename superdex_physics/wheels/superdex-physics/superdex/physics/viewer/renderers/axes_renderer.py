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

import warnings

import numpy as np
import numpy.typing as npt
from superdex.physics.utils.coordinate_systems import CoordinateTransform
from superdex.physics.utils.decorators import override_from
from superdex.physics.utils.transformations import (
    apply_linear_map,
    is_pure_rotation,
    orthonormalize_transform,
)
from superdex.physics.viewer.backend import polyscope as ps
from superdex.physics.viewer.renderers.renderer import Renderer
from superdex.physics.viewer.ui import styling

########################################################################################


class AxesRenderer(Renderer):
    """Helper renderer handling the renderer of axes of a frame using Polyscop. The
    frame is defined by a homogeneous transformation matrix, whose first three columns
    define the axes of the frame, and the last column defines the origin of the frame."""

    ####################################################################################
    # Members
    ####################################################################################

    _render_struct: ps.PointCloud | None
    _radius: float
    _length: float
    _xaxis_color: npt.NDArray[float]
    _yaxis_color: npt.NDArray[float]
    _zaxis_color: npt.NDArray[float]
    _transform: npt.NDArray[float]
    _transform_dirty: bool
    _struct_dirty: bool

    ####################################################################################
    # Constructor
    ####################################################################################

    def __init__(
        self,
        name: str,
        coordinate_transform: CoordinateTransform,
        transform: npt.NDArray[float] | npt.ArrayLike | None = None,
        radius: float = 0.008,
        length: float = 0.016,
        color: npt.NDArray[float] | npt.ArrayLike | tuple[float, float, float] = (
            0.75,
            0.75,
            0.75,
        ),
        xaxis_color: npt.NDArray[float] | npt.ArrayLike = styling.AXES_COLORS[0],
        yaxis_color: npt.NDArray[float] | npt.ArrayLike = styling.AXES_COLORS[1],
        zaxis_color: npt.NDArray[float] | npt.ArrayLike = styling.AXES_COLORS[2],
    ):
        """
        Initialize the AxesRenderer with the given parameters.
        """

        # Initialize base class
        super().__init__(name, coordinate_transform)

        # Fall back to default values if not provided.
        if transform is None:
            transform = np.eye(4, dtype=np.float32)

        # Initialize rendering structure.
        self._radius = radius
        self._length = length
        self._color = np.zeros(3, dtype=np.float32)
        self._xaxis_color = np.zeros(3, dtype=np.float32)
        self._yaxis_color = np.zeros(3, dtype=np.float32)
        self._zaxis_color = np.zeros(3, dtype=np.float32)
        self._transform = np.eye(4, dtype=np.float32)
        self._render_struct = ps.register_point_cloud(
            f"[Axes] {name}", points=np.zeros((1, 3))
        )

        # Initialize render settings.
        self.set_transform(np.asarray(transform))
        self.set_radius(radius)
        self.set_length(length)
        self.set_color(color)
        self.set_xaxis_color(xaxis_color)
        self.set_yaxis_color(yaxis_color)
        self.set_zaxis_color(zaxis_color)
        self.update()

    ####################################################################################
    # General Management Methods
    ####################################################################################

    @override_from(Renderer)
    def update(self) -> None:
        """Updates the rendering of the axes if any properties have changed."""

        render_struct = self.get_structure()

        # Retrieve the change of basis matrices.
        source_to_ps = self._coordinate_transform.source_to_target
        ps_to_source = self._coordinate_transform.target_to_source

        # Create (or update) the vector quantities displaying the axes.
        if self._struct_dirty:
            x, y, z = apply_linear_map(source_to_ps[:3, :3], np.eye(3))
            render_struct.add_vector_quantity(
                "X",
                x[None, ...],
                length=self._length,
                radius=self._radius / 8,
                color=self._xaxis_color,
                enabled=True,
            )
            render_struct.add_vector_quantity(
                "Y",
                y[None, ...],
                length=self._length,
                radius=self._radius / 8,
                color=self._yaxis_color,
                enabled=True,
            )
            render_struct.add_vector_quantity(
                "Z",
                z[None, ...],
                length=self._length,
                radius=self._radius / 8,
                color=self._zaxis_color,
                enabled=True,
            )
            self._struct_dirty = False

        # Update the transform if it has changed.
        if self._transform_dirty:
            # Build conjugate transform to account for the change in coordinate system.
            transform_ps = source_to_ps @ self._transform @ ps_to_source
            render_struct.set_transform(transform_ps)
            self._transform_dirty = False

    @override_from(Renderer)
    def remove(self) -> None:
        """
        Remove the axes renderer from the scene.
        """
        if self._render_struct is not None:
            self._render_struct.remove()
            self._render_struct = None
            self._struct_dirty = False
            self._transform_dirty = False

    ####################################################################################
    # Basic Renderer Properties
    ####################################################################################

    def get_structure(self) -> ps.PointCloud:
        """Get the underlying Polyscope structure of the axes renderer."""
        assert self._render_struct is not None, "Render structure not initialized."
        return self._render_struct

    def is_enabled(self) -> bool:
        """Get the enabled state of the axes renderer."""
        return self.get_structure().is_enabled()

    def set_enabled(self, enabled: bool) -> None:
        """Set the enabled state of the axes renderer."""
        self.get_structure().set_enabled(enabled)

    def get_transform(self) -> npt.NDArray[float]:
        """Get the transformation matrix defining the frame of reference."""
        return self._transform

    def set_transform(self, transform: npt.NDArray[float] | npt.ArrayLike) -> None:
        """Set the transformation matrix defining the frame of reference."""
        transform = np.asarray(transform, dtype=np.float32)
        if not transform.shape == (4, 4):
            raise ValueError("Invalid transform. Expected 4x4 array.")
        if not is_pure_rotation(transform):
            warnings.warn(
                "The given transform does not represent a pure rotation. The resulting "
                "axes orientations may be wrong.",
                stacklevel=2,
            )
            transform = orthonormalize_transform(transform)
        self._transform[:] = transform
        self._transform_dirty = True

    def get_radius(self) -> float:
        """Get the radius of the axes lines."""
        return self._radius

    def set_radius(self, radius: float) -> None:
        """Set the radius of the axes lines."""
        radius = float(radius)
        self._radius = radius
        self.get_structure().set_radius(radius, relative=False)
        self._struct_dirty = True

    def get_length(self) -> float:
        """Get the length of the axes lines."""
        return self._length

    def set_length(self, length: float) -> None:
        """Set the length of the axes lines."""
        length = float(length)
        self._length = length
        self._struct_dirty = True

    def get_color(self) -> npt.NDArray[float]:
        """Get the color of the sphere denoting the origin of the axes."""
        return np.asarray(self.get_structure().get_color(), dtype=np.float32)

    def set_color(self, color: npt.NDArray[float] | npt.ArrayLike) -> None:
        """Set the color of the sphere denoting the origin of the axes."""
        color = np.asarray(color, dtype=np.float32)
        if color.shape != (3,):
            raise ValueError("Invalid color. Expected 3-element array.")
        self.get_structure().set_color(color)

    def get_xaxis_color(self) -> npt.NDArray[float]:
        """Get the color of the x-axis."""
        return self._xaxis_color

    def set_xaxis_color(self, color: npt.NDArray[float] | npt.ArrayLike) -> None:
        """Set the color of the x-axis."""
        color = np.asarray(color, dtype=np.float32)
        if color.shape != (3,):
            raise ValueError("Invalid color. Expected 3-element array.")
        self._xaxis_color = color
        self._struct_dirty = True

    def get_yaxis_color(self) -> npt.NDArray[float]:
        """Get the color of the y-axis."""
        return self._yaxis_color

    def set_yaxis_color(self, color: npt.NDArray[float] | npt.ArrayLike) -> None:
        """Set the color of the y-axis."""
        color = np.asarray(color, dtype=np.float32)
        if color.shape != (3,):
            raise ValueError("Invalid color. Expected 3-element array.")
        self._yaxis_color = color
        self._struct_dirty = True

    def get_zaxis_color(self) -> npt.NDArray[float]:
        """Get the color of the z-axis."""
        return self._zaxis_color

    def set_zaxis_color(self, color: npt.NDArray[float] | npt.ArrayLike) -> None:
        """Set the color of the z-axis."""
        color = np.asarray(color, dtype=np.float32)
        if color.shape != (3,):
            raise ValueError("Invalid color. Expected 3-element array.")
        self._zaxis_color = color
        self._struct_dirty = True
