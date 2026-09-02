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
from unittest import TestCase

import numpy as np
from superdex.physics.utils.testing.decorators import skip_if
from superdex.physics.viewer import Viewer, VIEWER_AVAILABLE, ViewerCfg
from superdex.physics.viewer.renderers.point_cloud_renderer import PointCloudRenderer

########################################################################################


@skip_if(not VIEWER_AVAILABLE, "Requires Polyscope >= 2.5.0")
class TestPointCloudRenderer(TestCase):
    """Test suite for PointCloudRenderer functionality."""

    # Helper method to create a simple point cloud
    @staticmethod
    def _create_simple_point_cloud() -> np.ndarray:
        """Create a simple point cloud for testing."""
        return np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float32
        )

    ############################################################################
    # Initialization Tests
    ############################################################################

    def test_initialization_with_default_properties(self):
        """Test that point clouds can be initialized with default properties."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            coordinates = self._create_simple_point_cloud()
            point_cloud = viewer.add_point_cloud("test_cloud", coordinates)

            self.assertIsNotNone(point_cloud)
            self.assertIsInstance(point_cloud, PointCloudRenderer)
            self.assertEqual(point_cloud.get_num_points(), 5)
            # Default global radius should be 0.01
            self.assertAlmostEqual(point_cloud.get_radius(), 0.01)

    def test_initialization_with_custom_properties(self):
        """Test that point clouds can be initialized with custom rendering properties."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            coordinates = self._create_simple_point_cloud()
            point_cloud = viewer.add_point_cloud(
                "test_cloud",
                coordinates,
                radius=0.05,
                color=np.array([1.0, 0.0, 0.0]),
                transparency=0.5,
            )

            self.assertIsNotNone(point_cloud)
            self.assertAlmostEqual(point_cloud.get_radius(), 0.05)
            self.assertTrue(np.allclose(point_cloud.get_color(), [1.0, 0.0, 0.0]))
            self.assertAlmostEqual(point_cloud.get_transparency(), 0.5)

    def test_initialization_with_per_point_radii(self):
        """Test that point clouds can be initialized with per-point radii."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            coordinates = self._create_simple_point_cloud()
            radii = np.array([0.01, 0.02, 0.03, 0.04, 0.05], dtype=np.float32)
            point_cloud = viewer.add_point_cloud(
                "test_cloud",
                coordinates,
                radii=radii,
            )

            self.assertIsNotNone(point_cloud)
            point_radii = point_cloud.get_point_radii()
            self.assertIsNotNone(point_radii)
            self.assertTrue(np.allclose(point_radii, radii))

    def test_initialization_with_per_point_colors(self):
        """Test that point clouds can be initialized with per-point colors."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            coordinates = self._create_simple_point_cloud()
            colors = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0],
                ],
                dtype=np.float32,
            )
            point_cloud = viewer.add_point_cloud(
                "test_cloud",
                coordinates,
                colors=colors,
            )

            self.assertIsNotNone(point_cloud)
            point_colors = point_cloud.get_point_colors()
            self.assertIsNotNone(point_colors)
            self.assertTrue(np.allclose(point_colors, colors))

            # Global color can still be set (used as fallback)
            point_cloud.set_color(np.array([0.5, 0.5, 0.5]))
            self.assertTrue(np.allclose(point_cloud.get_color(), [0.5, 0.5, 0.5]))

    ############################################################################
    # Geometry Tests
    ############################################################################

    def test_get_coordinates(self):
        """Test getting point cloud coordinates."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            coordinates = self._create_simple_point_cloud()
            point_cloud = viewer.add_point_cloud("test_cloud", coordinates)

            retrieved = point_cloud.get_coordinates()
            self.assertTrue(np.allclose(retrieved, coordinates))

    def test_get_num_points(self):
        """Test getting the number of points in the cloud."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            # Test with different point counts
            for num_points in [1, 5, 10, 100]:
                coordinates = np.random.rand(num_points, 3).astype(np.float32)
                point_cloud = viewer.add_point_cloud(f"cloud_{num_points}", coordinates)
                self.assertEqual(point_cloud.get_num_points(), num_points)

    def test_set_points(self):
        """Test setting new points for the point cloud."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            coordinates = self._create_simple_point_cloud()
            point_cloud = viewer.add_point_cloud("test_cloud", coordinates)

            # Set new coordinates
            new_coordinates = np.array(
                [[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.float32
            )
            point_cloud.set_points(new_coordinates)

            self.assertEqual(point_cloud.get_num_points(), 3)
            self.assertTrue(np.allclose(point_cloud.get_coordinates(), new_coordinates))

    def test_set_points_with_radii(self):
        """Test setting points with per-point radii."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            coordinates = self._create_simple_point_cloud()
            point_cloud = viewer.add_point_cloud("test_cloud", coordinates)

            # Initially using global radius
            self.assertIsNone(point_cloud.get_point_radii())

            # Set points with per-point radii
            new_coords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
            radii = np.array([0.01, 0.02, 0.03], dtype=np.float32)
            point_cloud.set_points(new_coords, radii)

            self.assertEqual(point_cloud.get_num_points(), 3)
            self.assertIsNotNone(point_cloud.get_point_radii())
            self.assertTrue(np.allclose(point_cloud.get_point_radii(), radii))

    def test_set_points_clears_radii_when_none(self):
        """Test that set_points with radii=None clears per-point radii."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            coordinates = self._create_simple_point_cloud()
            radii = np.array([0.01, 0.02, 0.03, 0.04, 0.05], dtype=np.float32)
            point_cloud = viewer.add_point_cloud("test_cloud", coordinates, radii=radii)

            # Verify radii are set
            self.assertIsNotNone(point_cloud.get_point_radii())

            # Set new points without radii
            new_coords = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
            point_cloud.set_points(new_coords, radii=None)

            # Radii should be cleared
            self.assertIsNone(point_cloud.get_point_radii())

    def test_set_points_with_colors(self):
        """Test setting points with per-point colors and clearing them."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            coordinates = self._create_simple_point_cloud()
            colors = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0],
                ],
                dtype=np.float32,
            )
            point_cloud = viewer.add_point_cloud(
                "test_cloud", coordinates, colors=colors
            )

            # Verify colors are set
            self.assertIsNotNone(point_cloud.get_point_colors())
            self.assertTrue(np.allclose(point_cloud.get_point_colors(), colors))

            # Set new points with new colors
            new_coords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
            new_colors = np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
            )
            point_cloud.set_points(new_coords, colors=new_colors)

            self.assertEqual(point_cloud.get_num_points(), 3)
            self.assertIsNotNone(point_cloud.get_point_colors())
            self.assertTrue(np.allclose(point_cloud.get_point_colors(), new_colors))

            # Set new points without colors (should clear them)
            point_cloud.set_points(new_coords, colors=None)
            self.assertIsNone(point_cloud.get_point_colors())

    ############################################################################
    # Coordinate Validation Tests
    ############################################################################

    def test_coordinates_validation_shape(self):
        """Test that coordinates must be Nx3 arrays."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            coordinates = self._create_simple_point_cloud()
            point_cloud = viewer.add_point_cloud("test_cloud", coordinates)

            # Test 1D array (invalid)
            with self.assertRaises(ValueError) as cm:
                point_cloud.set_points(np.array([1.0, 2.0, 3.0]))
            self.assertIn("Expected Nx3", str(cm.exception))

            # Test wrong number of columns (invalid)
            with self.assertRaises(ValueError) as cm:
                point_cloud.set_points(np.array([[0, 0], [1, 1]], dtype=np.float32))
            self.assertIn("Expected Nx3", str(cm.exception))

            # Test 3D array (invalid)
            with self.assertRaises(ValueError) as cm:
                point_cloud.set_points(np.random.rand(2, 3, 4).astype(np.float32))
            self.assertIn("Expected Nx3", str(cm.exception))

    def test_radii_validation(self):
        """Test that radii must be 1D arrays matching point count."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            coordinates = self._create_simple_point_cloud()  # 5 points
            point_cloud = viewer.add_point_cloud("test_cloud", coordinates)

            # Test wrong number of radii (too few)
            with self.assertRaises(ValueError) as cm:
                point_cloud.set_points(coordinates, np.array([0.01, 0.02]))
            self.assertIn("same number of elements", str(cm.exception))

            # Test wrong number of radii (too many)
            with self.assertRaises(ValueError) as cm:
                point_cloud.set_points(
                    coordinates, np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
                )
            self.assertIn("same number of elements", str(cm.exception))

            # Test 2D radii array (invalid)
            with self.assertRaises(ValueError) as cm:
                point_cloud.set_points(
                    coordinates, np.array([[0.01], [0.02], [0.03], [0.04], [0.05]])
                )
            self.assertIn("Expected 1D", str(cm.exception))

    def test_per_point_colors_validation(self):
        """Test that per-point colors must be Nx3 arrays matching point count."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            coordinates = self._create_simple_point_cloud()  # 5 points
            point_cloud = viewer.add_point_cloud("test_cloud", coordinates)

            # Test wrong number of colors (too few) via set_points
            with self.assertRaises(ValueError) as cm:
                point_cloud.set_points(
                    coordinates,
                    colors=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                )
            self.assertIn("same number of elements", str(cm.exception))

            # Test wrong number of colors (too many) via set_points
            with self.assertRaises(ValueError) as cm:
                point_cloud.set_points(
                    coordinates,
                    colors=np.array(
                        [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [1.0, 1.0, 0.0],
                            [0.0, 1.0, 1.0],
                            [1.0, 0.0, 1.0],
                        ]
                    ),
                )
            self.assertIn("same number of elements", str(cm.exception))

            # Test wrong shape (not Nx3) via set_points
            with self.assertRaises(ValueError) as cm:
                point_cloud.set_points(
                    coordinates,
                    colors=np.array(
                        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
                    ),
                )
            self.assertIn("Expected Nx3", str(cm.exception))

            # Test wrong number of colors via set_point_colors
            with self.assertRaises(ValueError) as cm:
                point_cloud.set_point_colors(
                    np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
                )
            self.assertIn("same number of elements", str(cm.exception))

            # Test wrong shape via set_point_colors
            with self.assertRaises(ValueError) as cm:
                point_cloud.set_point_colors(
                    np.array(
                        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
                    )
                )
            self.assertIn("Expected Nx3", str(cm.exception))

    ############################################################################
    # Radius Properties Tests
    ############################################################################

    def test_global_radius_get_set(self):
        """Test getting and setting global radius."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            coordinates = self._create_simple_point_cloud()
            point_cloud = viewer.add_point_cloud("test_cloud", coordinates)

            # Test setting different radii
            test_radii = [0.001, 0.01, 0.05, 0.1, 1.0]
            for radius in test_radii:
                point_cloud.set_radius(radius)
                self.assertAlmostEqual(point_cloud.get_radius(), radius)

    def test_point_radii_with_global_radius(self):
        """Test interaction between per-point radii and global radius."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            coordinates = self._create_simple_point_cloud()
            radii = np.array([0.01, 0.02, 0.03, 0.04, 0.05], dtype=np.float32)
            point_cloud = viewer.add_point_cloud(
                "test_cloud",
                coordinates,
                radii=radii,
            )

            # Per-point radii should be set
            self.assertIsNotNone(point_cloud.get_point_radii())

            # Global radius can still be set (used as fallback)
            point_cloud.set_radius(0.1)
            self.assertAlmostEqual(point_cloud.get_radius(), 0.1)

    def test_set_point_colors(self):
        """Test getting, setting, and clearing per-point colors via set_point_colors."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            coordinates = self._create_simple_point_cloud()
            point_cloud = viewer.add_point_cloud("test_cloud", coordinates)

            # Initially no per-point colors
            self.assertIsNone(point_cloud.get_point_colors())

            # Set per-point colors
            colors = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0],
                ],
                dtype=np.float32,
            )
            point_cloud.set_point_colors(colors)
            self.assertIsNotNone(point_cloud.get_point_colors())
            self.assertTrue(np.allclose(point_cloud.get_point_colors(), colors))

            # Verify setting colors marks dirty flag
            point_cloud.update()
            self.assertFalse(point_cloud.is_dirty())
            point_cloud.set_point_colors(colors)
            self.assertTrue(point_cloud.is_dirty())

            # Clear colors
            point_cloud.set_point_colors(None)
            self.assertIsNone(point_cloud.get_point_colors())

    ############################################################################
    # Color Property Tests
    ############################################################################

    def test_color_get_set(self):
        """Test getting and setting point cloud color."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            coordinates = self._create_simple_point_cloud()
            point_cloud = viewer.add_point_cloud(
                "test_cloud",
                coordinates,
                color=np.array([1.0, 0.0, 0.0]),
            )

            # Test initial color
            self.assertTrue(np.allclose(point_cloud.get_color(), [1.0, 0.0, 0.0]))

            # Test setting different colors
            test_colors = [
                np.array([0.0, 1.0, 0.0]),  # Green
                np.array([0.0, 0.0, 1.0]),  # Blue
                np.array([0.5, 0.5, 0.5]),  # Gray
            ]
            for color in test_colors:
                point_cloud.set_color(color)
                self.assertTrue(np.allclose(point_cloud.get_color(), color))

    def test_color_push_pop(self):
        """Test push/pop stack operations for color."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            coordinates = self._create_simple_point_cloud()
            point_cloud = viewer.add_point_cloud(
                "test_cloud",
                coordinates,
                color=np.array([1.0, 0.0, 0.0]),
            )

            # Verify initial color
            self.assertTrue(np.allclose(point_cloud.get_color(), [1.0, 0.0, 0.0]))

            # Push new color
            point_cloud.push_color(np.array([0.0, 1.0, 0.0]))
            self.assertTrue(np.allclose(point_cloud.get_color(), [0.0, 1.0, 0.0]))

            # Push another color
            point_cloud.push_color(np.array([0.0, 0.0, 1.0]))
            self.assertTrue(np.allclose(point_cloud.get_color(), [0.0, 0.0, 1.0]))

            # Pop back through stack
            point_cloud.pop_color()
            self.assertTrue(np.allclose(point_cloud.get_color(), [0.0, 1.0, 0.0]))

            point_cloud.pop_color()
            self.assertTrue(np.allclose(point_cloud.get_color(), [1.0, 0.0, 0.0]))

    def test_color_validation(self):
        """Test that color must be a 3-element array."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            coordinates = self._create_simple_point_cloud()
            point_cloud = viewer.add_point_cloud("test_cloud", coordinates)

            # Test wrong size
            with self.assertRaises(ValueError) as cm:
                point_cloud.set_color(np.array([1.0, 0.0]))
            self.assertIn("Expected 3-element", str(cm.exception))

            with self.assertRaises(ValueError) as cm:
                point_cloud.set_color(np.array([1.0, 0.0, 0.0, 1.0]))
            self.assertIn("Expected 3-element", str(cm.exception))

            with self.assertRaises(ValueError) as cm:
                point_cloud.push_color(np.array([1.0, 0.0]))
            self.assertIn("Expected 3-element", str(cm.exception))

    ############################################################################
    # Transparency Property Tests
    ############################################################################

    def test_transparency_get_set(self):
        """Test getting and setting transparency."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            coordinates = self._create_simple_point_cloud()
            point_cloud = viewer.add_point_cloud(
                "test_cloud",
                coordinates,
                transparency=0.3,
            )

            # Test initial transparency
            self.assertAlmostEqual(point_cloud.get_transparency(), 0.3)

            # Test setting different transparencies
            for transparency in [0.0, 0.25, 0.5, 0.75, 1.0]:
                point_cloud.set_transparency(transparency)
                self.assertAlmostEqual(point_cloud.get_transparency(), transparency)

    def test_transparency_push_pop(self):
        """Test push/pop stack operations for transparency."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            coordinates = self._create_simple_point_cloud()
            point_cloud = viewer.add_point_cloud(
                "test_cloud",
                coordinates,
                transparency=0.2,
            )

            # Verify initial
            self.assertAlmostEqual(point_cloud.get_transparency(), 0.2)

            # Push new values
            point_cloud.push_transparency(0.5)
            self.assertAlmostEqual(point_cloud.get_transparency(), 0.5)

            point_cloud.push_transparency(0.8)
            self.assertAlmostEqual(point_cloud.get_transparency(), 0.8)

            # Pop back through stack
            point_cloud.pop_transparency()
            self.assertAlmostEqual(point_cloud.get_transparency(), 0.5)

            point_cloud.pop_transparency()
            self.assertAlmostEqual(point_cloud.get_transparency(), 0.2)

    ############################################################################
    # Enable/Disable Tests
    ############################################################################

    def test_enabled_state(self):
        """Test enabling and disabling point cloud rendering."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            coordinates = self._create_simple_point_cloud()
            point_cloud = viewer.add_point_cloud("test_cloud", coordinates)

            # Should be enabled by default
            self.assertTrue(point_cloud.is_enabled())

            # Disable
            point_cloud.set_enabled(False)
            self.assertFalse(point_cloud.is_enabled())

            # Re-enable
            point_cloud.set_enabled(True)
            self.assertTrue(point_cloud.is_enabled())

    ############################################################################
    # Dirty Flag Tests
    ############################################################################

    def test_is_dirty(self):
        """Test the dirty flag mechanism."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            coordinates = self._create_simple_point_cloud()
            point_cloud = viewer.add_point_cloud("test_cloud", coordinates)

            # After update, should not be dirty
            point_cloud.update()
            self.assertFalse(point_cloud.is_dirty())

            # After setting new points, should be dirty
            point_cloud.set_points(coordinates)
            self.assertTrue(point_cloud.is_dirty())

            # After update, should not be dirty again
            point_cloud.update()
            self.assertFalse(point_cloud.is_dirty())

    ############################################################################
    # Render Structure Tests
    ############################################################################

    def test_get_render_structure(self):
        """Test getting the underlying Polyscope render structure."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            coordinates = self._create_simple_point_cloud()
            point_cloud = viewer.add_point_cloud("test_cloud", coordinates)

            render_struct = point_cloud.get_render_structure()
            self.assertIsNotNone(render_struct)

    ############################################################################
    # Property Stack Independence Tests
    ############################################################################

    def test_property_stack_independence(self):
        """Test that color and transparency stacks are independent."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            coordinates = self._create_simple_point_cloud()
            point_cloud = viewer.add_point_cloud(
                "test_cloud",
                coordinates,
                color=np.array([1.0, 0.0, 0.0]),
                transparency=0.3,
            )

            # Push on both stacks
            point_cloud.push_color(np.array([0.0, 1.0, 0.0]))
            point_cloud.push_transparency(0.7)

            # Verify both have pushed values
            self.assertTrue(np.allclose(point_cloud.get_color(), [0.0, 1.0, 0.0]))
            self.assertAlmostEqual(point_cloud.get_transparency(), 0.7)

            # Pop color - transparency should remain unchanged
            point_cloud.pop_color()
            self.assertTrue(np.allclose(point_cloud.get_color(), [1.0, 0.0, 0.0]))
            self.assertAlmostEqual(point_cloud.get_transparency(), 0.7)

            # Pop transparency - color should remain unchanged
            point_cloud.pop_transparency()
            self.assertTrue(np.allclose(point_cloud.get_color(), [1.0, 0.0, 0.0]))
            self.assertAlmostEqual(point_cloud.get_transparency(), 0.3)

    ############################################################################
    # Data Type Conversion Tests
    ############################################################################

    def test_coordinates_data_type_conversion(self):
        """Test that coordinates are properly converted to float32."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            # Test with Python list
            coords_list = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
            point_cloud = viewer.add_point_cloud("list_cloud", coords_list)
            self.assertEqual(point_cloud.get_coordinates().dtype, np.float32)

            # Test with int array
            coords_int = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int32)
            point_cloud2 = viewer.add_point_cloud("int_cloud", coords_int)
            self.assertEqual(point_cloud2.get_coordinates().dtype, np.float32)

            # Test with float64 array
            coords_float64 = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
            point_cloud3 = viewer.add_point_cloud("float64_cloud", coords_float64)
            self.assertEqual(point_cloud3.get_coordinates().dtype, np.float32)


########################################################################################

if __name__ == "__main__":
    unittest.main()
