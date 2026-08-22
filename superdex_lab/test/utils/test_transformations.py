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
from superdex.lab.gym.utils.transformations import (
    angular_distance,
    pi_minus_pi_cap,
    unwrap_angle_sequence,
)

########################################################################################


class TestTransformations(unittest.TestCase):
    """Test class for transformation utilities."""

    def test_pi_minus_pi_cap(self):
        # Test that pi_minus_pi_cap correctly wraps angles to [-pi, pi).

        # Test with scalar values
        assert np.isclose(pi_minus_pi_cap(0.0), 0.0)
        assert np.isclose(pi_minus_pi_cap(np.pi), -np.pi)
        assert np.isclose(pi_minus_pi_cap(-np.pi), -np.pi)
        assert np.isclose(pi_minus_pi_cap(3 * np.pi), -np.pi)
        assert np.isclose(pi_minus_pi_cap(-3 * np.pi), -np.pi)
        assert np.isclose(pi_minus_pi_cap(2 * np.pi), 0.0)
        assert np.isclose(pi_minus_pi_cap(-2 * np.pi), 0.0)
        assert np.isclose(pi_minus_pi_cap(np.pi / 2), np.pi / 2)
        assert np.isclose(pi_minus_pi_cap(-np.pi / 2), -np.pi / 2)

        # Test with array values
        angles = np.array(
            [0.0, np.pi, -np.pi, 3 * np.pi, -3 * np.pi, 2 * np.pi, -2 * np.pi]
        )
        expected = np.array([0.0, -np.pi, -np.pi, -np.pi, -np.pi, 0.0, 0.0])
        wrapped = pi_minus_pi_cap(angles)
        assert np.allclose(wrapped, expected)

    def test_angular_distance(self):
        # Test that angular_distance correctly computes the signed angular distance.

        # Test with scalar values
        assert np.isclose(angular_distance(0.0, 0.0), 0.0)
        assert np.isclose(angular_distance(np.pi / 2, 0.0), -np.pi / 2)
        assert np.isclose(angular_distance(0.0, np.pi / 2), np.pi / 2)
        assert np.isclose(
            angular_distance(np.pi, -np.pi), 0.0
        )  # They are the same angle
        assert np.isclose(
            angular_distance(3 * np.pi / 4, -3 * np.pi / 4), np.pi / 2
        )  # Shortest path is clockwise

        # Test with array values
        x = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        y = np.array([np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 4])
        expected = np.array([np.pi / 4, -np.pi / 4, -3 * np.pi / 4, 3 * np.pi / 4])
        distances = angular_distance(x, y)
        assert np.allclose(distances, expected)

    def test_unwrap_angle_sequence(self):
        # Test that unwrap_angle_sequence correctly unwraps a sequence of angles.

        # Test with a simple sequence
        angles = np.array([0.0, np.pi / 2, -np.pi, np.pi / 2])
        unwrapped = unwrap_angle_sequence(angles)

        # First angle stays the same, others are cumulative differences
        assert np.isclose(unwrapped[0], 0.0)
        assert np.isclose(unwrapped[1], np.pi / 2)
        assert np.isclose(unwrapped[2], np.pi)
        assert np.isclose(unwrapped[3], np.pi / 2)

        # Test with a sequence that crosses the -pi/pi boundary
        # The sequence should be continuous without jumps
        angles = np.array([np.pi - 0.15, np.pi - 0.05, -np.pi + 0.05, -np.pi + 0.15])
        unwrapped = unwrap_angle_sequence(angles)
        assert np.isclose(unwrapped[0], np.pi - 0.15)
        diffs = np.diff(unwrapped)
        expected_diffs = np.array([0.1, 0.1, 0.1])  # Continuous differences
        assert np.allclose(diffs, expected_diffs)


########################################################################################

if __name__ == "__main__":
    unittest.main()
