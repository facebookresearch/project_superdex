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

import unittest

import numpy as np
from superdex.physics.utils.coordinate_systems import (
    Axis,
    CoordinateSystem,
    CoordinateTransform,
    Handedness,
)

########################################################################################


class TestAxis(unittest.TestCase):
    """Test class for Axis enum."""

    def test_axis_values(self):
        """Test that all axis enum values are correctly defined."""
        assert Axis.POS_X.value == "+X"
        assert Axis.NEG_X.value == "-X"
        assert Axis.POS_Y.value == "+Y"
        assert Axis.NEG_Y.value == "-Y"
        assert Axis.POS_Z.value == "+Z"
        assert Axis.NEG_Z.value == "-Z"

    def test_from_str_case_insensitive(self):
        """Test that from_str correctly converts strings to Axis enum values regardless
        of case."""
        assert Axis.from_str("+x") == Axis.POS_X
        assert Axis.from_str("+X") == Axis.POS_X
        assert Axis.from_str("-x") == Axis.NEG_X
        assert Axis.from_str("-X") == Axis.NEG_X
        assert Axis.from_str("+y") == Axis.POS_Y
        assert Axis.from_str("+Y") == Axis.POS_Y
        assert Axis.from_str("-y") == Axis.NEG_Y
        assert Axis.from_str("-Y") == Axis.NEG_Y
        assert Axis.from_str("+z") == Axis.POS_Z
        assert Axis.from_str("+Z") == Axis.POS_Z
        assert Axis.from_str("-z") == Axis.NEG_Z
        assert Axis.from_str("-Z") == Axis.NEG_Z

    def test_from_str_invalid_raises_error(self):
        """Test that from_str raises a ValueError for invalid axis strings."""
        with self.assertRaises(ValueError) as context:
            Axis.from_str("invalid")
        assert "Unknown axis direction: invalid" in str(context.exception)

        with self.assertRaises(ValueError):
            Axis.from_str("X")
        with self.assertRaises(ValueError):
            Axis.from_str("+W")
        with self.assertRaises(ValueError):
            Axis.from_str("")

    def test_to_vector(self):
        """Test that the to_vector method returns the correct unit vector for each axis
        direction."""
        # Test that POS_X converts to the correct unit vector.
        vector = Axis.POS_X.to_vector()
        expected = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert np.allclose(vector, expected)
        assert vector.dtype == np.float32

        # Test that NEG_X converts to the correct unit vector.
        vector = Axis.NEG_X.to_vector()
        expected = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        assert np.allclose(vector, expected)
        assert vector.dtype == np.float32

        # Test that POS_Y converts to the correct unit vector.
        vector = Axis.POS_Y.to_vector()
        expected = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert np.allclose(vector, expected)
        assert vector.dtype == np.float32

        # Test that NEG_Y converts to the correct unit vector.
        vector = Axis.NEG_Y.to_vector()
        expected = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        assert np.allclose(vector, expected)
        assert vector.dtype == np.float32

        # Test that POS_Z converts to the correct unit vector.
        vector = Axis.POS_Z.to_vector()
        expected = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        assert np.allclose(vector, expected)
        assert vector.dtype == np.float32

        # Test that NEG_Z converts to the correct unit vector.
        vector = Axis.NEG_Z.to_vector()
        expected = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        assert np.allclose(vector, expected)


class TestCoordinateSystem(unittest.TestCase):
    """Test class for CoordinateSystem."""

    def test_init_with_axis_enums(self):
        """Test that CoordinateSystem can be initialized with Axis enums."""
        coord_system = CoordinateSystem(
            right=Axis.POS_X, up=Axis.POS_Y, forward=Axis.POS_Z
        )
        assert coord_system.right == Axis.POS_X
        assert coord_system.up == Axis.POS_Y
        assert coord_system.forward == Axis.POS_Z

    def test_init_with_strings(self):
        """Test that CoordinateSystem can be initialized with string representations."""
        coord_system = CoordinateSystem(right="+X", up="+Y", forward="+Z")
        assert coord_system.right == Axis.POS_X
        assert coord_system.up == Axis.POS_Y
        assert coord_system.forward == Axis.POS_Z

    def test_init_with_mixed_types(self):
        """Test that CoordinateSystem can be initialized with mixed Axis and string types."""
        coord_system = CoordinateSystem(right=Axis.POS_X, up="+Y", forward=Axis.POS_Z)
        assert coord_system.right == Axis.POS_X
        assert coord_system.up == Axis.POS_Y
        assert coord_system.forward == Axis.POS_Z

    def test_init_non_orthogonal_raises_error(self):
        """Test that initializing with non-orthogonal axes raises an error."""
        with self.assertRaises(ValueError):
            CoordinateSystem("+X", "+X", "+Z")
        with self.assertRaises(ValueError):
            CoordinateSystem("+X", "+Y", "+X")
        with self.assertRaises(ValueError):
            CoordinateSystem("+X", "+Y", "+Y")

    def test_axes_matrix(self):
        """Test that the axes matrix is correctly constructed from axis vectors."""
        coord_system = CoordinateSystem("+X", "+Z", "-Y")
        expected_axes = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
        assert np.allclose(coord_system.axes_matrix, expected_axes)

    def test_left_handed_system(self):
        """Test that a left-handed coordinate system is correctly identified."""
        coord_system = CoordinateSystem("+X", "+Y", "+Z")
        assert coord_system.handedness == Handedness.LEFT
        assert coord_system.is_left_handed
        assert not coord_system.is_right_handed

    def test_right_handed_system(self):
        """Test that a right-handed coordinate system is correctly identified."""
        coord_system = CoordinateSystem("+X", "+Y", "-Z")
        assert coord_system.handedness == Handedness.RIGHT
        assert coord_system.is_right_handed
        assert not coord_system.is_left_handed

    def test_axes_matrix_orthonormal(self):
        """Test that the axes matrix is orthonormal.

        Verifies that all axis vectors are unit length and mutually orthogonal.
        """
        coord_system = CoordinateSystem("+X", "+Z", "-Y")
        axes = coord_system.axes_matrix

        # Check that all columns are unit vectors
        for i in range(3):
            column_length = np.linalg.norm(axes[:, i])
            assert np.isclose(column_length, 1.0)

        # Check that all columns are orthogonal to each other
        for i in range(3):
            for j in range(i + 1, 3):
                dot_product = np.dot(axes[:, i], axes[:, j])
                assert np.isclose(dot_product, 0.0)


class TestCoordinateTransform(unittest.TestCase):
    """Test class for CoordinateTransform."""

    def test_init_same_system(self):
        """Test that a transform can be created between identical coordinate systems."""
        coord_system = CoordinateSystem("+X", "+Y", "+Z")
        transform = CoordinateTransform(coord_system, coord_system)
        assert transform.source_system == coord_system
        assert transform.target_system == coord_system

    def test_init_different_systems(self):
        """Test that a transform can be created between different coordinate systems."""
        source_system = CoordinateSystem("+X", "+Y", "+Z")
        target_system = CoordinateSystem("+X", "+Y", "-Z")
        transform = CoordinateTransform(source_system, target_system)
        assert transform.source_system == source_system
        assert transform.target_system == target_system

    def test_source_to_target_matrix_identity_for_same_system(self):
        """Test that the source_to_target matrix is identity when source and target systems are
        identical."""
        coord_system = CoordinateSystem("+X", "+Y", "+Z")
        transform = CoordinateTransform(coord_system, coord_system)
        expected = np.eye(4, dtype=np.float32)
        assert np.allclose(transform.source_to_target, expected)

    def test_source_to_target_matrix_shape(self):
        """Test that the source_to_target matrix has the correct shape and data type."""
        source_system = CoordinateSystem("+X", "+Y", "+Z")
        target_system = CoordinateSystem("+X", "+Y", "-Z")
        transform = CoordinateTransform(source_system, target_system)
        assert transform.source_to_target.shape == (4, 4)
        assert transform.source_to_target.dtype == np.float32

    def test_target_to_source_matrix(self):
        """Test that the target_to_source matrix is the transpose of source_to_target."""
        source_system = CoordinateSystem("+X", "+Y", "+Z")
        target_system = CoordinateSystem("+X", "+Z", "-Y")
        transform = CoordinateTransform(source_system, target_system)
        assert np.allclose(transform.target_to_source, transform.source_to_target.T)

    def test_encodes_reflection_same_handedness(self):
        """Test that transforms between systems with same handedness don't encode reflection."""
        source_system = CoordinateSystem("+X", "+Y", "+Z")
        target_system = CoordinateSystem("+Y", "+Z", "+X")
        transform = CoordinateTransform(source_system, target_system)
        assert not transform.encodes_reflection

    def test_encodes_reflection_different_handedness(self):
        """Test that transforms between systems with different handedness encode reflection."""
        source_system = CoordinateSystem("+X", "+Y", "+Z")
        target_system = CoordinateSystem("+X", "+Y", "-Z")
        transform = CoordinateTransform(source_system, target_system)
        assert transform.encodes_reflection

    def test_direction_to_target_identity(self):
        """Test that direction_to_target returns the same direction for identical systems."""
        coord_system = CoordinateSystem("+X", "+Y", "+Z")
        transform = CoordinateTransform(coord_system, coord_system)
        direction = np.array([1.0, 2.0, 3.0])
        result = transform.direction_to_target(direction)
        assert np.allclose(result, direction)

    def test_direction_to_target_transforms_correctly(self):
        """Test that direction_to_target correctly transforms a direction vector."""
        source_system = CoordinateSystem("+X", "+Y", "+Z")
        target_system = CoordinateSystem("+X", "+Z", "-Y")
        transform = CoordinateTransform(source_system, target_system)

        direction = np.array([1.0, 0.0, 0.0])
        result = transform.direction_to_target(direction)
        assert np.allclose(result, [1.0, 0.0, 0.0])

        direction = np.array([0.0, 1.0, 0.0])
        result = transform.direction_to_target(direction)
        assert np.allclose(result, [0.0, 0.0, 1.0])

        direction = np.array([0.0, 0.0, 1.0])
        result = transform.direction_to_target(direction)
        assert np.allclose(result, [0.0, -1.0, 0.0])

    def test_direction_to_target_preserves_magnitude(self):
        """Test that direction_to_target preserves the magnitude of the direction."""
        source_system = CoordinateSystem("+X", "+Y", "+Z")
        target_system = CoordinateSystem("+Y", "+Z", "+X")
        transform = CoordinateTransform(source_system, target_system)

        direction = np.array([3.0, 4.0, 0.0])
        result = transform.direction_to_target(direction)
        assert np.isclose(np.linalg.norm(result), 5.0)

    def test_position_to_target_identity(self):
        """Test that position_to_target returns the same position for identical systems."""
        coord_system = CoordinateSystem("+X", "+Y", "+Z")
        transform = CoordinateTransform(coord_system, coord_system)
        position = np.array([1.0, 2.0, 3.0])
        result = transform.position_to_target(position)
        assert np.allclose(result, position)

    def test_position_to_target_with_scale(self):
        """Test that position_to_target correctly applies scale."""
        coord_system = CoordinateSystem("+X", "+Y", "+Z")
        transform = CoordinateTransform(coord_system, coord_system)
        position = np.array([1.0, 2.0, 3.0])
        scale = 100.0
        result = transform.position_to_target(position, scale=scale)
        assert np.allclose(result, position * scale)

    def test_position_to_target_transforms_correctly(self):
        """Test that position_to_target correctly transforms a position."""
        source_system = CoordinateSystem("+X", "+Y", "+Z")
        target_system = CoordinateSystem("+X", "+Z", "-Y")
        transform = CoordinateTransform(source_system, target_system)

        position = np.array([1.0, 2.0, 3.0])
        result = transform.position_to_target(position)
        expected = np.array([1.0, -3.0, 2.0])
        assert np.allclose(result, expected)

    def test_position_to_target_with_scale_and_transform(self):
        """Test position_to_target with both coordinate transform and scale."""
        source_system = CoordinateSystem("+X", "+Y", "+Z")
        target_system = CoordinateSystem("+X", "+Z", "-Y")
        transform = CoordinateTransform(source_system, target_system)

        position = np.array([1.0, 2.0, 3.0])
        scale = 10.0
        result = transform.position_to_target(position, scale=scale)
        expected = np.array([10.0, -30.0, 20.0])
        assert np.allclose(result, expected)

    def test_rotation_to_target_identity(self):
        """Test that rotation_to_target returns identity for same system."""
        coord_system = CoordinateSystem("+X", "+Y", "+Z")
        transform = CoordinateTransform(coord_system, coord_system)

        quat = np.array([0.0, 0.0, 0.0, 1.0])
        result = transform.rotation_to_target(quat)
        assert np.allclose(result, quat)

    def test_rotation_to_target_same_handedness(self):
        """Test rotation_to_target between systems with same handedness.

        Source system (+X, +Y, +Z): X is "right", Y is "up", Z is "forward"
        Target system (+Y, +Z, +X): Y is "right", Z is "up", X is "forward"

        A quaternion [1, 0, 0, 0] represents rotation around X axis.
        In source, X is "right". In target, "right" is Y.
        So the rotation axis should transform from X to Y.
        """
        source_system = CoordinateSystem("+X", "+Y", "+Z")
        target_system = CoordinateSystem("+Y", "+Z", "+X")
        transform = CoordinateTransform(source_system, target_system)

        quat = np.array([1.0, 0.0, 0.0, 0.0])
        result = transform.rotation_to_target(quat)
        expected = np.array([0.0, 1.0, 0.0, 0.0])
        assert np.allclose(result, expected)

    def test_rotation_to_target_different_handedness(self):
        """Test rotation_to_target between systems with different handedness.

        When handedness flips, the quaternion's imaginary components are negated
        before applying the direction transformation.
        """
        source_system = CoordinateSystem("+X", "+Y", "+Z")
        target_system = CoordinateSystem("+X", "+Y", "-Z")
        transform = CoordinateTransform(source_system, target_system)

        quat = np.array([1.0, 0.0, 0.0, 0.0])
        result = transform.rotation_to_target(quat)
        expected = np.array([-1.0, 0.0, 0.0, 0.0])
        assert np.allclose(result, expected)

    def test_rotation_to_target_preserves_scalar(self):
        """Test that rotation_to_target preserves the scalar (w) component."""
        source_system = CoordinateSystem("+X", "+Y", "+Z")
        target_system = CoordinateSystem("+Y", "+Z", "+X")
        transform = CoordinateTransform(source_system, target_system)

        quat = np.array([0.1, 0.2, 0.3, 0.9])
        result = transform.rotation_to_target(quat)
        assert np.isclose(result[3], 0.9)


########################################################################################

if __name__ == "__main__":
    unittest.main()
