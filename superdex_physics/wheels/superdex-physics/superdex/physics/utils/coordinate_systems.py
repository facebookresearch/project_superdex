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
This module provides classes and utilities for defining, converting, and working with
different coordinate systems used by various 3D rendering engines and tools. Different
engines use different conventions for which direction is "right", "up", and "forward",
and this module allows transparent conversion between these conventions.
"""

from __future__ import annotations

import enum

import numpy as np
import numpy.typing as npt

########################################################################################


class Axis(enum.Enum):
    """Enumeration of coordinate axes and their directions.

    Represents the six possible axis directions in 3D space: positive and negative
    X, Y, and Z. Used to define which physical directions correspond to "right",
    "up", and "forward" in a coordinate system.
    """

    POS_X = "+X"
    """Positive X axis direction."""
    NEG_X = "-X"
    """Negative X axis direction."""
    POS_Y = "+Y"
    """Positive Y axis direction."""
    NEG_Y = "-Y"
    """Negative Y axis direction."""
    POS_Z = "+Z"
    """Positive Z axis direction."""
    NEG_Z = "-Z"
    """Negative Z axis direction."""

    @staticmethod
    def from_str(axis_name: str) -> Axis:
        """Convert a string representation of an axis to an Axis enum value.

        The string is case-insensitive and should be in the format "+X", "-X", "+Y",
        "-Y", "+Z", or "-Z". Raises a ValueError if the string doesn't match any
        valid axis direction.

        Args:
            axis_name: String representation of an axis (e.g., "+X", "-y", "+Z").

        Returns:
            The corresponding Axis enum value.

        Raises:
            ValueError: If the string does not match any valid axis direction.
        """
        normalized_axis_name = axis_name.upper()
        try:
            return Axis(normalized_axis_name)
        except ValueError:
            raise ValueError(f"Unknown axis direction: {axis_name}")

    def to_vector(self) -> npt.NDArray[np.float32]:
        """Convert the axis to a 3D unit vector.

        Returns a normalized 3D vector pointing in the direction specified by this axis.
        For example, POS_X returns [1, 0, 0] and NEG_Y returns [0, -1, 0].

        Returns:
            A 3D numpy array representing the unit vector for this axis direction.
        """
        match self:
            case Axis.POS_X:
                return np.array([1.0, 0.0, 0.0], dtype=np.float32)
            case Axis.NEG_X:
                return np.array([-1.0, 0.0, 0.0], dtype=np.float32)
            case Axis.POS_Y:
                return np.array([0.0, 1.0, 0.0], dtype=np.float32)
            case Axis.NEG_Y:
                return np.array([0.0, -1.0, 0.0], dtype=np.float32)
            case Axis.POS_Z:
                return np.array([0.0, 0.0, 1.0], dtype=np.float32)
            case Axis.NEG_Z:
                return np.array([0.0, 0.0, -1.0], dtype=np.float32)
            case _:
                raise ValueError(f"Unknown axis direction: {self.value}")


class Handedness(enum.Enum):
    """Represents the handedness of a coordinate system.

    The handedness of a coordinate system determines the orientation convention used
    for rotations and cross products. Different rendering engines use different
    conventions.
    """

    LEFT = "left"
    """Left-handed coordinate system (e.g., Unity, Unreal)."""
    RIGHT = "right"
    """Right-handed coordinate system (e.g., OpenGL, Blender)."""


class CoordinateSystem:
    """Defines a coordinate system for visualization.

    A coordinate system specifies which directions correspond to "right", "up",
    and "forward" in the visualization. This is important for rendering as different
    engines and tools use different conventions.

    The coordinate system is defined by three orthogonal axes. For example:
    - Unity uses: right=+X, up=+Y, forward=+Z (left-handed)
    - Unreal uses: right=+Y, up=+Z, forward=+X (right-handed)
    - Polyscope uses: right=+X, up=+Y, forward=-Z (right-handed)

    The class automatically validates that the three axes are mutually orthogonal
    and computes the handedness of the system.
    """

    _right: Axis
    _up: Axis
    _forward: Axis
    _axes_matrix: npt.NDArray[np.float32]
    _handedness: Handedness

    def __init__(
        self,
        right: Axis | str,
        up: Axis | str,
        forward: Axis | str,
    ):
        """Initialize the coordinate system.

        Args:
            right: The axis direction for the right direction.
            up: The axis direction for the up direction.
            forward: The axis direction for the forward direction.
        """

        # Convert strings to Axis enum values.
        if isinstance(right, str):
            right = Axis.from_str(right)
        if isinstance(up, str):
            up = Axis.from_str(up)
        if isinstance(forward, str):
            forward = Axis.from_str(forward)

        # Extract the basis vectors.
        right_vec = right.to_vector()
        up_vec = up.to_vector()
        forward_vec = forward.to_vector()

        # Build the matrix specifying the semantic axes of the coordinate system.
        axes_matrix = np.column_stack([right_vec, forward_vec, up_vec])
        det_axes_matrix = np.linalg.det(axes_matrix)

        # Check that vectors are orthogonal.
        if np.isclose(det_axes_matrix, 0.0):
            raise ValueError(
                f"The specified axes (right={right.value}, up={up.value}, forward="
                f"{forward.value}) are not orthogonal"
            )

        # Try infer the handedness of the coordinate system from this.
        handedness = Handedness.LEFT if det_axes_matrix < 0.0 else Handedness.RIGHT

        # Store the coordinate system.
        self._right = right
        self._up = up
        self._forward = forward
        self._axes_matrix = axes_matrix
        self._handedness = handedness

    @property
    def right(self) -> Axis:
        """The axis direction for the right direction."""
        return self._right

    @property
    def up(self) -> Axis:
        """The axis direction for the up direction."""
        return self._up

    @property
    def forward(self) -> Axis:
        """The axis direction for the forward direction."""
        return self._forward

    @property
    def axes_matrix(self) -> npt.NDArray[np.float32]:
        """Returns the 3x3 matrix for the coordinate system. The axes matrix consists of
        three column vectors representing the right, forward and up directions."""
        return self._axes_matrix

    @property
    def handedness(self) -> Handedness:
        """Determines the handedness of the coordinate system."""
        return self._handedness

    @property
    def is_left_handed(self) -> bool:
        """Determines if the coordinate system is left-handed."""
        return self._handedness == Handedness.LEFT

    @property
    def is_right_handed(self) -> bool:
        """Determines if the coordinate system is right-handed."""
        return self._handedness == Handedness.RIGHT

    def __repr__(self) -> str:
        """Returns a string representation of the coordinate system."""
        return (
            f"{self._handedness.value.capitalize()}-handed, "
            f"{self._up.value}-up, "
            f"{self._forward.value}-forward"
        )


class CoordinateTransform:
    """Provides transformation matrices for converting between coordinate systems.

    This class computes and provides the transformation matrices needed to convert
    geometric data (points, vectors, rotation matrices, transformation matrices) from
    one coordinate system convention to another. This is essential when working with
    content authored in a specific coordinate system (e.g., Unity) but rendered in a
    different one (e.g., Polyscope).

    The transformation matrices are computed using a change-of-basis matrix derived
    from the source and target coordinate systems' axes vectors. The origin of the
    coordinate system is preserved, but the axes are rotated to match the target
    coordinate system.

    Mathematical Foundation:
        Let 𝐒 and 𝐓 be matrices whose columns encode the right, up, and forward
        directions as unit vectors, each expressed in their respective native coordinate
        systems 𝕊 and 𝕋. Let 𝐮 ∈ 𝕊 and 𝐯 ∈ 𝕋 be coordinates representing the same
        geometric point, defined in their respective coordinate systems.

        When projecting both points onto their semantic basis vectors (right, forward,
        up), the resulting coordinates must be identical:

                                        𝐒ᵀ𝐮 = 𝐓ᵀ𝐯

        That is, "how far right, foward, and up" is invariant — it describes the same
        point in space regardless of the underlying coordinate system representation.

        Solving for 𝐯 yields:

                                        𝐯 = 𝐓𝐒ᵀ𝐮 = 𝐂𝐮

        where 𝐂 = 𝐓𝐒ᵀ is the change-of-basis matrix.
    """

    _source_system: CoordinateSystem
    _target_system: CoordinateSystem
    _source_to_target: npt.NDArray[np.float32]

    def __init__(
        self,
        source_system: CoordinateSystem,
        target_system: CoordinateSystem,
    ):
        """Initialize the coordinate transform.

        Computes the conversion matrix between the source and target coordinate
        systems using a change-of-basis transformation.

        Args:
            source_system: The source coordinate system to convert from.
            target_system: The target coordinate system to convert to.
        """

        source_axes = source_system.axes_matrix
        target_axes = target_system.axes_matrix
        self._source_system = source_system
        self._target_system = target_system
        self._source_to_target = np.eye(4, dtype=np.float32)
        self._source_to_target[:3, :3] = target_axes @ source_axes.T

    @property
    def source_system(self) -> CoordinateSystem:
        """The source coordinate system to convert from."""
        return self._source_system

    @property
    def target_system(self) -> CoordinateSystem:
        """The target coordinate system to convert to."""
        return self._target_system

    @property
    def source_to_target(self) -> npt.NDArray[np.float32]:
        """Returns the 4x4 matrix doing the basis change from source to target."""
        return self._source_to_target

    @property
    def target_to_source(self) -> npt.NDArray[np.float32]:
        """Returns the 4x4 matrix doing the basis change from target to source."""
        return self._source_to_target.T

    @property
    def encodes_reflection(self) -> bool:
        """Determines if the coordinate transform encodes a reflection."""
        return np.linalg.det(self._source_to_target[:3, :3]) < 0.0

    def direction_to_target(
        self, direction: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Transform a direction vector from source to target coordinate system.

        Unlike positions, directions are not affected by translation, only by
        the rotation/reflection component of the transformation.

        Args:
            direction: A 3D direction vector in source coordinates.

        Returns:
            The direction vector transformed to target coordinates.
        """
        return self._source_to_target[:3, :3] @ direction

    def position_to_target(
        self,
        position: npt.NDArray[np.floating],
        scale: float = 1.0,
    ) -> npt.NDArray[np.floating]:
        """Transform a position from source to target coordinate system.

        Applies the coordinate system transformation and an optional scale factor
        (e.g., for meters to centimeters conversion).

        Args:
            position: A 3D position in source coordinates.
            scale: Optional scale factor to apply (default 1.0).

        Returns:
            The position transformed to target coordinates.
        """
        return self._source_to_target[:3, :3] @ position * scale

    def rotation_to_target(
        self, quaternion: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Transform a rotation quaternion from source to target coordinate system.

        This method correctly handles the conversion of rotations between coordinate
        systems, including the handedness flip when converting between left-handed
        and right-handed systems.

        The algorithm matches the C++ coordinate-space conversion:

        1. If the transform encodes a reflection (handedness flip), negate the
           quaternion's imaginary components (x, y, z).
        2. Apply the direction transformation to the axis (x, y, z components).
        3. Keep the scalar component (w) unchanged.

        Args:
            quaternion: A rotation quaternion in [x, y, z, w] format (scipy convention).

        Returns:
            The quaternion transformed to target coordinates in [x, y, z, w] format.
        """
        x, y, z, w = quaternion

        # If handedness flips, negate the quaternion axis
        if self.encodes_reflection:
            axis = np.array([-x, -y, -z])
        else:
            axis = np.array([x, y, z])

        # Transform the axis using the direction transformation
        axis_out = self.direction_to_target(axis)

        return np.array([axis_out[0], axis_out[1], axis_out[2], w])


########################################################################################
# Preset Coordinate Systems
########################################################################################

COORDINATE_SYSTEMS = {
    "mochi": CoordinateSystem(right="+x", up="+y", forward="-z"),
    "unity": CoordinateSystem(right="+x", up="+y", forward="+z"),
    "unreal": CoordinateSystem(right="+y", up="+z", forward="+x"),
    "blender": CoordinateSystem(right="+x", up="+z", forward="+y"),
    "polyscope": CoordinateSystem(right="+x", up="+y", forward="-z"),
    "ros": CoordinateSystem(right="-y", up="+z", forward="+x"),
}
"""Preset coordinate systems for common 3D engines and tools.

This dictionary provides ready-to-use coordinate system definitions used in popular
3D rendering engines and modeling tools. These coordinate systems are defined in terms
of the semantic axes (right, up, and forward) derived from the canonical basis vectors
(X, Y, and Z)."""

DEFAULT_COORDINATE_SYSTEM = COORDINATE_SYSTEMS["polyscope"]
"""The default coordinate system used by the SuperDex Physics viewer.

This is set to the legacy OpenGL view convention (right-handed, Y-up, -Z-forward) as it
matches the internal coordinate system of Polyscope, the rendering backend used by the
viewer."""
