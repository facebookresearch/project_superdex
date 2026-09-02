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

import superdex.physics as sdp
from superdex.physics.utils.testing.testcases import (
    make_empty_scene,
    make_single_rigid_cube_scene,
    MochiContextTestCase,
)

########################################################################################


class TestMochiContextTestCase(unittest.TestCase):
    """Test cases for the test case with Mochi context."""

    def test_initialization_and_shutdown(self):
        """Test that the test case properly initializes and shuts down Mochi."""

        self.assertFalse(sdp.is_initialized())
        MochiContextTestCase.setUpClass()
        self.assertTrue(sdp.is_initialized())
        MochiContextTestCase.tearDownClass()
        self.assertFalse(sdp.is_initialized())


########################################################################################


class TestScenes(MochiContextTestCase):
    """Test cases for scene creation utilities"""

    def test_empty_scene(self):
        """Test that make_empty_scene creates a valid empty scene."""
        with make_empty_scene() as scene:
            self.assertIsNotNone(scene)
            self.assertEqual(scene.get_num_actors(), 0)

    def test_single_rigid_cube_scene(self):
        """Test that make_single_rigid_cube_scene creates a valid scene with a single
        rigid cube."""
        with make_single_rigid_cube_scene() as scene:
            self.assertIsNotNone(scene)
            self.assertEqual(scene.get_num_actors(), 1)


########################################################################################

if __name__ == "__main__":
    unittest.main()
