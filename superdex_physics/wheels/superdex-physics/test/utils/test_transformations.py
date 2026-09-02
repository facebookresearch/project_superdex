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
from superdex.physics.utils.transformations import (
    apply_affine_transform,
    is_pure_rotation,
    make_transform,
    orthonormalize_transform,
)

########################################################################################


class TestTransformations(unittest.TestCase):
    """Test class for transformation utilities."""

    def test_make_transform(self):
        # Test that make_transform correctly creates a homogeneous transformation matrix.

        # Test with identity transform
        transform = make_transform([0, 0, 0], [0, 0, 0])
        expected = np.eye(4)
        assert np.allclose(transform, expected)

        # Test with translation only
        transform = make_transform([1, 2, 3], [0, 0, 0])
        expected = np.eye(4)
        expected[:3, 3] = [1, 2, 3]
        assert np.allclose(transform, expected)

        # Test with rotation only (90 degrees around z-axis)
        transform = make_transform([0, 0, 0], [0, 0, np.pi / 2])
        expected = np.eye(4)
        expected[0, 0] = 0
        expected[0, 1] = -1
        expected[1, 0] = 1
        expected[1, 1] = 0
        assert np.allclose(transform, expected)

        # Test with scale
        transform = make_transform([0, 0, 0], [0, 0, 0], scale=2.0)
        expected = np.eye(4)
        expected[:3, :3] *= 2.0
        assert np.allclose(transform, expected)

        # Test with translation, rotation, and scale
        transform = make_transform([1, 2, 3], [0, 0, np.pi / 2], scale=2.0)
        expected = np.eye(4)
        expected[0, 0] = 0
        expected[0, 1] = -2
        expected[1, 0] = 2
        expected[1, 1] = 0
        expected[2, 2] = 2
        expected[:3, 3] = [1, 2, 3]
        assert np.allclose(transform, expected)

    def test_apply_affine_transform(self):
        # Test that apply_affine_transform correctly applies an affine transformation to points.

        # Create a simple translation transform
        transform = np.eye(4)
        transform[:3, 3] = [1, 2, 3]

        # Apply to a single point
        point = np.array([0, 0, 0])
        transformed = apply_affine_transform(transform, point)
        expected = np.array([1, 2, 3])
        assert np.allclose(transformed, expected)

        # Apply to multiple points
        points = np.array([[0, 0, 0], [1, 1, 1], [2, 3, 4]])
        transformed = apply_affine_transform(transform, points)
        expected = np.array([[1, 2, 3], [2, 3, 4], [3, 5, 7]])
        assert np.allclose(transformed, expected)

        # Create a rotation transform (90 degrees around z-axis)
        transform = np.eye(4)
        transform[0, 0] = 0
        transform[0, 1] = -1
        transform[1, 0] = 1
        transform[1, 1] = 0

        # Apply to a single point
        point = np.array([1, 0, 0])
        transformed = apply_affine_transform(transform, point)
        expected = np.array([0, 1, 0])
        assert np.allclose(transformed, expected)

        # Apply to multiple points
        points = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]])
        transformed = apply_affine_transform(transform, points)
        expected = np.array([[0, 1, 0], [-1, 0, 0], [-1, 1, 0]])
        assert np.allclose(transformed, expected)

    def test_is_pure_rotation(self):
        # Test that is_pure_rotation correctly identifies pure rotation matrices.

        # Test identity matrix (pure rotation)
        identity_transform = np.eye(4)
        assert is_pure_rotation(identity_transform)
        assert is_pure_rotation(identity_transform[:3, :3])

        # Test pure rotation matrix (90 degrees around z-axis)
        rotation_transform = np.eye(4)
        rotation_transform[0, 0] = 0
        rotation_transform[0, 1] = -1
        rotation_transform[1, 0] = 1
        rotation_transform[1, 1] = 0
        assert is_pure_rotation(rotation_transform)
        assert is_pure_rotation(rotation_transform[:3, :3])

        # Test pure rotation matrix (arbitrary rotation)
        angle = np.pi / 3
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_transform = np.eye(4)
        rotation_transform[0, 0] = cos_a
        rotation_transform[0, 1] = -sin_a
        rotation_transform[1, 0] = sin_a
        rotation_transform[1, 1] = cos_a
        assert is_pure_rotation(rotation_transform)
        assert is_pure_rotation(rotation_transform[:3, :3])

        # Test scaled matrix (not pure rotation)
        scaled_transform = np.eye(4)
        scaled_transform[:3, :3] *= 2.0
        assert not is_pure_rotation(scaled_transform)
        assert not is_pure_rotation(scaled_transform[:3, :3])

        # Test matrix with negative determinant (reflection, not pure rotation)
        reflection_transform = np.eye(4)
        reflection_transform[0, 0] = -1
        assert not is_pure_rotation(reflection_transform)
        assert not is_pure_rotation(reflection_transform[:3, :3])

        # Test non-orthogonal matrix (shear)
        shear_transform = np.eye(4)
        shear_transform[0, 1] = 0.5  # Add shear
        assert not is_pure_rotation(shear_transform)
        assert not is_pure_rotation(shear_transform[:3, :3])

        # Test matrix with scaling and rotation combined
        scaled_rotation_transform = make_transform(
            [0, 0, 0], [0, 0, np.pi / 4], scale=1.5
        )
        assert not is_pure_rotation(scaled_rotation_transform)
        assert not is_pure_rotation(scaled_rotation_transform[:3, :3])

        # Test pure rotation created with make_transform
        pure_rotation_transform = make_transform(
            [1, 2, 3], [0, 0, np.pi / 6], scale=1.0
        )
        assert is_pure_rotation(pure_rotation_transform)
        assert is_pure_rotation(pure_rotation_transform[:3, :3])

    def test_orthonormalize_transform(self):
        # Test that orthonormalize_transform computes a rotation matrix.

        # Test with identity matrix (should return identity)
        input_transform = np.eye(4)
        result_transform = orthonormalize_transform(input_transform)
        assert np.allclose(result_transform, input_transform)

        # Test with pure rotation matrix (should return the same rotation)
        input_transform = make_transform([1, 2, 3], [0, 0, np.pi / 4], scale=1.0)
        result_transform = orthonormalize_transform(input_transform)
        assert np.allclose(result_transform, input_transform)

        # Test with scaled rotation matrix
        # The closest rotation should be orthogonal and have determinant 1
        input_transform = make_transform([0, 0, 0], [0, 0, np.pi / 3], scale=2.0)
        result_transform = orthonormalize_transform(input_transform)
        assert is_pure_rotation(result_transform)

        # Test with a matrix that has shear
        input_transform = np.eye(4)
        input_transform[0, 1] = 0.3
        input_transform[1, 2] = 0.2
        result_transform = orthonormalize_transform(input_transform)
        assert is_pure_rotation(result_transform)

        # Test with a matrix that combines scaling, rotation, and shear
        input_transform = np.array(
            [
                [2.1, -0.9, 0.1, 5],
                [1.1, 1.9, 0.2, 6],
                [0.05, 0.1, 2.05, 7],
                [0, 0, 0, 1],
            ]
        )
        result_transform = orthonormalize_transform(input_transform)
        assert is_pure_rotation(result_transform)

        # Test that the function preserves the general orientation For a simple scaled
        # rotation, the closest rotation should have similar orientation
        arbitrary_rotation = np.array(
            [
                [np.cos(np.pi / 6), 0, -np.sin(np.pi / 6)],
                [0, 1, 0],
                [np.sin(np.pi / 6), 0, np.cos(np.pi / 6)],
            ]
        )
        input_transform = np.eye(4)
        input_transform[:3, :3] = 1.5 * arbitrary_rotation
        result_transform = orthonormalize_transform(input_transform)
        assert np.allclose(result_transform[:3, :3], arbitrary_rotation, atol=1e-10)

        # Test with simple reflection matrix (reflection across x-axis)
        input_transform = np.eye(4)
        input_transform[0, 0] = -1
        result_transform = orthonormalize_transform(input_transform)
        assert is_pure_rotation(result_transform)


########################################################################################

if __name__ == "__main__":
    unittest.main()
