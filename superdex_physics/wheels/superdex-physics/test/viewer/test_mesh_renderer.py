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
from superdex.physics.viewer.renderers.mesh_renderer import MeshRenderer

########################################################################################


@skip_if(not VIEWER_AVAILABLE, "Requires Polyscope >= 2.5.0")
class TestMeshRenderer(TestCase):
    """Test suite for MeshRenderer functionality."""

    # Helper method to create a simple triangle mesh
    @staticmethod
    def _create_simple_mesh() -> tuple[np.ndarray, np.ndarray]:
        """Create a simple triangle mesh for testing."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        return vertices, faces

    ############################################################################
    # Initialization Tests
    ############################################################################

    def test_initialization_with_custom_properties(self):
        """Test that meshes can be initialized with custom rendering properties."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            vertices, faces = self._create_simple_mesh()
            mesh = viewer.add_mesh(
                "test_mesh",
                vertices,
                faces,
                front_face_color=np.array([1.0, 0.0, 0.0]),
                transparency=0.5,
            )

            self.assertIsNotNone(mesh)
            self.assertTrue(np.allclose(mesh.get_front_face_color(), [1.0, 0.0, 0.0]))
            self.assertEqual(mesh.get_transparency(), 0.5)

    ############################################################################
    # Material Tests
    ############################################################################

    def test_material_initialization(self):
        """Test initializing mesh with different materials."""

        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            vertices, faces = self._create_simple_mesh()

            # Test each material type
            for material in MeshRenderer.Material:
                mesh = viewer.add_mesh(
                    f"mesh_{material.name}",
                    vertices,
                    faces,
                    material=material,
                )
                self.assertEqual(mesh.get_material(), material)

    def test_material_get_set(self):
        """Test getting and setting material."""

        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            vertices, faces = self._create_simple_mesh()
            mesh = viewer.add_mesh("test_mesh", vertices, faces)

            # Test setting and getting each material
            for material in MeshRenderer.Material:
                mesh.set_material(material)
                self.assertEqual(mesh.get_material(), material)

    def test_material_push_pop(self):
        """Test push/pop stack operations for material."""

        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            vertices, faces = self._create_simple_mesh()
            mesh = viewer.add_mesh(
                "test_mesh",
                vertices,
                faces,
                material=MeshRenderer.Material.CLAY,
            )

            # Verify initial
            self.assertEqual(mesh.get_material(), MeshRenderer.Material.CLAY)

            # Push new values
            mesh.push_material(MeshRenderer.Material.JADE)
            self.assertEqual(mesh.get_material(), MeshRenderer.Material.JADE)

            mesh.push_material(MeshRenderer.Material.WAX)
            self.assertEqual(mesh.get_material(), MeshRenderer.Material.WAX)

            # Pop back through stack
            mesh.pop_material()
            self.assertEqual(mesh.get_material(), MeshRenderer.Material.JADE)

            mesh.pop_material()
            self.assertEqual(mesh.get_material(), MeshRenderer.Material.CLAY)

    ############################################################################
    # Smooth Shading Tests
    ############################################################################

    def test_smooth_shading_initialization(self):
        """Test initializing mesh with smooth shading enabled/disabled."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            vertices, faces = self._create_simple_mesh()

            # Test with smooth shading enabled
            mesh_smooth = viewer.add_mesh(
                "smooth_mesh",
                vertices,
                faces,
                smooth_shading=True,
            )
            self.assertTrue(mesh_smooth.get_smooth_shading())

            # Test with smooth shading disabled (flat)
            mesh_flat = viewer.add_mesh(
                "flat_mesh",
                vertices,
                faces,
                smooth_shading=False,
            )
            self.assertFalse(mesh_flat.get_smooth_shading())

    def test_smooth_shading_get_set(self):
        """Test getting and setting smooth shading."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            vertices, faces = self._create_simple_mesh()
            mesh = viewer.add_mesh("test_mesh", vertices, faces)

            # Test toggling
            mesh.set_smooth_shading(True)
            self.assertTrue(mesh.get_smooth_shading())

            mesh.set_smooth_shading(False)
            self.assertFalse(mesh.get_smooth_shading())

            mesh.set_smooth_shading(True)
            self.assertTrue(mesh.get_smooth_shading())

    def test_smooth_shading_push_pop(self):
        """Test push/pop stack operations for smooth shading."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            vertices, faces = self._create_simple_mesh()
            mesh = viewer.add_mesh(
                "test_mesh",
                vertices,
                faces,
                smooth_shading=True,
            )

            # Verify initial
            self.assertTrue(mesh.get_smooth_shading())

            # Push new values
            mesh.push_smooth_shading(False)
            self.assertFalse(mesh.get_smooth_shading())

            mesh.push_smooth_shading(True)
            self.assertTrue(mesh.get_smooth_shading())

            # Pop back through stack
            mesh.pop_smooth_shading()
            self.assertFalse(mesh.get_smooth_shading())

            mesh.pop_smooth_shading()
            self.assertTrue(mesh.get_smooth_shading())

    ############################################################################
    # Color Property Tests
    ############################################################################

    def test_front_face_color_operations(self):
        """Test front face color get/set and push/pop operations."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            vertices, faces = self._create_simple_mesh()

            # Test initialization and get/set
            mesh = viewer.add_mesh(
                "test_mesh",
                vertices,
                faces,
                front_face_color=np.array([1.0, 0.0, 0.0]),
            )
            self.assertTrue(np.allclose(mesh.get_front_face_color(), [1.0, 0.0, 0.0]))

            # Test setting different colors
            test_colors = [
                np.array([0.0, 1.0, 0.0]),  # Green
                np.array([0.0, 0.0, 1.0]),  # Blue
            ]
            for color in test_colors:
                mesh.set_front_face_color(color)
                self.assertTrue(np.allclose(mesh.get_front_face_color(), color))

            # Test push/pop
            mesh.push_front_face_color(np.array([0.5, 0.5, 0.5]))
            self.assertTrue(np.allclose(mesh.get_front_face_color(), [0.5, 0.5, 0.5]))

            mesh.pop_front_face_color()
            self.assertTrue(np.allclose(mesh.get_front_face_color(), [0.0, 0.0, 1.0]))

    def test_back_face_color_operations(self):
        """Test back face color get/set and push/pop operations."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            vertices, faces = self._create_simple_mesh()
            mesh = viewer.add_mesh(
                "test_mesh",
                vertices,
                faces,
                back_face_color=np.array([1.0, 0.0, 1.0]),
            )

            # Test get
            initial = mesh.get_back_face_color()
            self.assertTrue(np.allclose(initial, [1.0, 0.0, 1.0]))

            # Test set
            mesh.set_back_face_color(np.array([1.0, 1.0, 0.0]))
            self.assertTrue(np.allclose(mesh.get_back_face_color(), [1.0, 1.0, 0.0]))

            # Test push/pop
            mesh.push_back_face_color(np.array([0.0, 1.0, 1.0]))
            self.assertTrue(np.allclose(mesh.get_back_face_color(), [0.0, 1.0, 1.0]))

            mesh.pop_back_face_color()
            self.assertTrue(np.allclose(mesh.get_back_face_color(), [1.0, 1.0, 0.0]))

    def test_edge_color_operations(self):
        """Test edge color get/set and push/pop operations."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            vertices, faces = self._create_simple_mesh()
            mesh = viewer.add_mesh(
                "test_mesh",
                vertices,
                faces,
                edge_color=np.array([0.2, 0.2, 0.2]),
            )

            # Test set
            mesh.set_edge_color(np.array([1.0, 0.5, 0.0]))
            self.assertTrue(np.allclose(mesh.get_edge_color(), [1.0, 0.5, 0.0]))

            # Test push/pop
            mesh.push_edge_color(np.array([0.8, 0.8, 0.8]))
            self.assertTrue(np.allclose(mesh.get_edge_color(), [0.8, 0.8, 0.8]))

            mesh.pop_edge_color()
            self.assertTrue(np.allclose(mesh.get_edge_color(), [1.0, 0.5, 0.0]))

    def test_node_color_operations(self):
        """Test node color get/set and push/pop operations."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            vertices, faces = self._create_simple_mesh()
            mesh = viewer.add_mesh(
                "test_mesh",
                vertices,
                faces,
                node_color=np.array([0.1, 0.1, 0.1]),
            )

            # Test set
            mesh.set_node_color(np.array([0.5, 0.5, 1.0]))
            self.assertTrue(np.allclose(mesh.get_node_color(), [0.5, 0.5, 1.0]))

            # Test push/pop
            mesh.push_node_color(np.array([0.9, 0.9, 0.9]))
            self.assertTrue(np.allclose(mesh.get_node_color(), [0.9, 0.9, 0.9]))

            mesh.pop_node_color()
            self.assertTrue(np.allclose(mesh.get_node_color(), [0.5, 0.5, 1.0]))

    ############################################################################
    # Back Face Policy Tests
    ############################################################################

    def test_back_face_policy_operations(self):
        """Test back face policy get/set and push/pop operations."""

        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            vertices, faces = self._create_simple_mesh()
            mesh = viewer.add_mesh("test_mesh", vertices, faces)

            # Test get/set for all policies
            for policy in MeshRenderer.BackFacePolicy:
                mesh.set_back_face_policy(policy)
                self.assertEqual(mesh.get_back_face_policy(), policy)

            # Test push/pop
            mesh.set_back_face_policy(MeshRenderer.BackFacePolicy.IDENTICAL)
            initial = mesh.get_back_face_policy()

            mesh.push_back_face_policy(MeshRenderer.BackFacePolicy.CULL)
            self.assertEqual(
                mesh.get_back_face_policy(), MeshRenderer.BackFacePolicy.CULL
            )

            mesh.pop_back_face_policy()
            self.assertEqual(mesh.get_back_face_policy(), initial)

    ############################################################################
    # Scalar Property Tests (Transparency, Widths, Radii)
    ############################################################################

    def test_transparency_operations(self):
        """Test transparency get/set and push/pop operations."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            vertices, faces = self._create_simple_mesh()
            mesh = viewer.add_mesh("test_mesh", vertices, faces, transparency=0.3)

            # Test initialization
            self.assertAlmostEqual(mesh.get_transparency(), 0.3)

            # Test get/set with different values
            for transparency in [0.0, 0.5, 1.0]:
                mesh.set_transparency(transparency)
                self.assertAlmostEqual(mesh.get_transparency(), transparency)

            # Test push/pop
            mesh.set_transparency(0.3)
            mesh.push_transparency(0.7)
            self.assertAlmostEqual(mesh.get_transparency(), 0.7)

            mesh.push_transparency(0.9)
            self.assertAlmostEqual(mesh.get_transparency(), 0.9)

            mesh.pop_transparency()
            self.assertAlmostEqual(mesh.get_transparency(), 0.7)

            mesh.pop_transparency()
            self.assertAlmostEqual(mesh.get_transparency(), 0.3)

    def test_edge_dimensions_operations(self):
        """Test edge width and radius get/set and push/pop operations."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            vertices, faces = self._create_simple_mesh()
            mesh = viewer.add_mesh(
                "test_mesh", vertices, faces, edge_width=0.02, edge_radius=0.002
            )

            # Test edge width
            for width in [0.01, 0.05, 0.1]:
                mesh.set_edge_width(width)
                self.assertAlmostEqual(mesh.get_edge_width(), width)

            mesh.push_edge_width(0.08)
            self.assertAlmostEqual(mesh.get_edge_width(), 0.08)
            mesh.pop_edge_width()
            self.assertAlmostEqual(mesh.get_edge_width(), 0.1)

            # Test edge radius
            for radius in [0.001, 0.005, 0.01]:
                mesh.set_edge_radius(radius)
                self.assertAlmostEqual(mesh.get_edge_radius(), radius)

            mesh.push_edge_radius(0.006)
            self.assertAlmostEqual(mesh.get_edge_radius(), 0.006)
            mesh.pop_edge_radius()
            self.assertAlmostEqual(mesh.get_edge_radius(), 0.01)

    def test_node_radius_operations(self):
        """Test node radius get/set and push/pop operations."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            vertices, faces = self._create_simple_mesh()
            mesh = viewer.add_mesh("test_mesh", vertices, faces, node_radius=0.003)

            # Test get/set
            for radius in [0.001, 0.005, 0.01]:
                mesh.set_node_radius(radius)
                self.assertAlmostEqual(mesh.get_node_radius(), radius)

            # Test push/pop
            mesh.set_node_radius(0.003)
            mesh.push_node_radius(0.007)
            self.assertAlmostEqual(mesh.get_node_radius(), 0.007)

            mesh.pop_node_radius()
            self.assertAlmostEqual(mesh.get_node_radius(), 0.003)

    ############################################################################
    # Geometry Tests
    ############################################################################

    def test_transform_operations(self):
        """Test transform get/set operations."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            vertices, faces = self._create_simple_mesh()
            mesh = viewer.add_mesh("test_mesh", vertices, faces)

            # Test identity transform
            identity = np.eye(4, dtype=np.float32)
            mesh.set_transform(identity)
            self.assertTrue(np.allclose(mesh.get_transform(), identity))

            # Test translation
            translation = np.eye(4, dtype=np.float32)
            translation[:3, 3] = [1.0, 2.0, 3.0]
            mesh.set_transform(translation)
            self.assertTrue(np.allclose(mesh.get_transform(), translation))

    def test_local_coordinates_operations(self):
        """Test local coordinates get/set operations."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            vertices, faces = self._create_simple_mesh()
            mesh = viewer.add_mesh("test_mesh", vertices, faces)

            # Test get initial coordinates
            coords = mesh.get_local_coordinates()
            self.assertTrue(np.allclose(coords, vertices))

            # Test set new coordinates
            new_vertices = np.array([[1, 1, 1], [2, 1, 1], [1, 2, 1]], dtype=np.float32)
            mesh.set_local_coordinates(new_vertices)
            coords = mesh.get_local_coordinates()
            self.assertTrue(np.allclose(coords, new_vertices))

    ############################################################################
    # Enable/Disable Tests
    ############################################################################

    def test_render_component_visibility(self):
        """Test enabling/disabling surface, edges, nodes, and axes rendering."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            vertices, faces = self._create_simple_mesh()
            mesh = viewer.add_mesh("test_mesh", vertices, faces)

            # Test surface (enabled by default)
            self.assertTrue(mesh.is_surface_enabled())
            mesh.set_enable_surface(False)
            self.assertFalse(mesh.is_surface_enabled())
            mesh.set_enable_surface(True)
            self.assertTrue(mesh.is_surface_enabled())

            # Test edges (disabled by default)
            self.assertFalse(mesh.are_edges_enabled())
            mesh.set_enable_edges(True)
            self.assertTrue(mesh.are_edges_enabled())
            mesh.set_enable_edges(False)
            self.assertFalse(mesh.are_edges_enabled())

            # Test nodes (disabled by default)
            self.assertFalse(mesh.are_nodes_enabled())
            mesh.set_enable_nodes(True)
            self.assertTrue(mesh.are_nodes_enabled())
            mesh.set_enable_nodes(False)
            self.assertFalse(mesh.are_nodes_enabled())

            # Test axes (disabled by default)
            self.assertFalse(mesh.are_axes_enabled())
            mesh.set_enable_axes(True)
            self.assertTrue(mesh.are_axes_enabled())
            mesh.set_enable_axes(False)
            self.assertFalse(mesh.are_axes_enabled())

    ############################################################################
    # Cross-Property Interaction Tests
    ############################################################################

    def test_property_stack_independence(self):
        """Test that all property stacks are independent of each other.

        This comprehensive test ensures that push/pop operations on one property
        do not affect the values or stacks of other properties. Tests all properties
        that support stack operations.
        """

        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            vertices, faces = self._create_simple_mesh()

            # Initialize mesh with specific values for all stackable properties
            mesh = viewer.add_mesh(
                "test_mesh",
                vertices,
                faces,
                material=MeshRenderer.Material.CLAY,
                smooth_shading=True,
                transparency=0.3,
                front_face_color=np.array([1.0, 0.0, 0.0]),
                back_face_color=np.array([0.0, 1.0, 0.0]),
                back_face_policy=MeshRenderer.BackFacePolicy.IDENTICAL,
                edge_color=np.array([0.1, 0.1, 0.1]),
                edge_width=0.01,
                edge_radius=0.001,
                node_color=np.array([0.2, 0.2, 0.2]),
                node_radius=0.002,
            )

            # Push new values onto all stacks
            mesh.push_material(MeshRenderer.Material.JADE)
            mesh.push_smooth_shading(False)
            mesh.push_transparency(0.7)
            mesh.push_front_face_color(np.array([0.0, 0.0, 1.0]))
            mesh.push_back_face_color(np.array([1.0, 1.0, 0.0]))
            mesh.push_back_face_policy(MeshRenderer.BackFacePolicy.CULL)
            mesh.push_edge_color(np.array([0.9, 0.9, 0.9]))
            mesh.push_edge_width(0.05)
            mesh.push_edge_radius(0.005)
            mesh.push_node_color(np.array([0.8, 0.8, 0.8]))
            mesh.push_node_radius(0.008)

            # Verify all properties have the pushed values
            self.assertEqual(mesh.get_material(), MeshRenderer.Material.JADE)
            self.assertFalse(mesh.get_smooth_shading())
            self.assertAlmostEqual(mesh.get_transparency(), 0.7)
            self.assertTrue(np.allclose(mesh.get_front_face_color(), [0.0, 0.0, 1.0]))
            self.assertTrue(np.allclose(mesh.get_back_face_color(), [1.0, 1.0, 0.0]))
            self.assertEqual(
                mesh.get_back_face_policy(), MeshRenderer.BackFacePolicy.CULL
            )
            self.assertTrue(np.allclose(mesh.get_edge_color(), [0.9, 0.9, 0.9]))
            self.assertAlmostEqual(mesh.get_edge_width(), 0.05)
            self.assertAlmostEqual(mesh.get_edge_radius(), 0.005)
            self.assertTrue(np.allclose(mesh.get_node_color(), [0.8, 0.8, 0.8]))
            self.assertAlmostEqual(mesh.get_node_radius(), 0.008)

            # Pop transparency - all others should remain unchanged
            mesh.pop_transparency()
            self.assertAlmostEqual(mesh.get_transparency(), 0.3)
            self.assertEqual(mesh.get_material(), MeshRenderer.Material.JADE)
            self.assertFalse(mesh.get_smooth_shading())
            self.assertTrue(np.allclose(mesh.get_front_face_color(), [0.0, 0.0, 1.0]))

            # Pop color properties - others should remain unchanged
            mesh.pop_front_face_color()
            self.assertTrue(np.allclose(mesh.get_front_face_color(), [1.0, 0.0, 0.0]))
            mesh.pop_back_face_color()
            self.assertTrue(np.allclose(mesh.get_back_face_color(), [0.0, 1.0, 0.0]))
            mesh.pop_edge_color()
            self.assertTrue(np.allclose(mesh.get_edge_color(), [0.1, 0.1, 0.1]))
            mesh.pop_node_color()
            self.assertTrue(np.allclose(mesh.get_node_color(), [0.2, 0.2, 0.2]))

            # Material and smooth shading should still have pushed values
            self.assertEqual(mesh.get_material(), MeshRenderer.Material.JADE)
            self.assertFalse(mesh.get_smooth_shading())

            # Pop dimensional properties - others should remain unchanged
            mesh.pop_edge_width()
            self.assertAlmostEqual(mesh.get_edge_width(), 0.01)
            mesh.pop_edge_radius()
            self.assertAlmostEqual(mesh.get_edge_radius(), 0.001)
            mesh.pop_node_radius()
            self.assertAlmostEqual(mesh.get_node_radius(), 0.002)

            # Material and smooth shading should still have pushed values
            self.assertEqual(mesh.get_material(), MeshRenderer.Material.JADE)
            self.assertFalse(mesh.get_smooth_shading())

            # Pop policy - material and smooth shading should remain unchanged
            mesh.pop_back_face_policy()
            self.assertEqual(
                mesh.get_back_face_policy(), MeshRenderer.BackFacePolicy.IDENTICAL
            )
            self.assertEqual(mesh.get_material(), MeshRenderer.Material.JADE)
            self.assertFalse(mesh.get_smooth_shading())

            # Pop smooth shading - material should remain unchanged
            mesh.pop_smooth_shading()
            self.assertTrue(mesh.get_smooth_shading())
            self.assertEqual(mesh.get_material(), MeshRenderer.Material.JADE)

            # Finally pop material - back to initial value
            mesh.pop_material()
            self.assertEqual(mesh.get_material(), MeshRenderer.Material.CLAY)

            # Verify all properties are back to their initial values
            self.assertAlmostEqual(mesh.get_transparency(), 0.3)
            self.assertTrue(np.allclose(mesh.get_front_face_color(), [1.0, 0.0, 0.0]))
            self.assertTrue(np.allclose(mesh.get_back_face_color(), [0.0, 1.0, 0.0]))
            self.assertEqual(
                mesh.get_back_face_policy(), MeshRenderer.BackFacePolicy.IDENTICAL
            )
            self.assertTrue(np.allclose(mesh.get_edge_color(), [0.1, 0.1, 0.1]))
            self.assertAlmostEqual(mesh.get_edge_width(), 0.01)
            self.assertAlmostEqual(mesh.get_edge_radius(), 0.001)
            self.assertTrue(np.allclose(mesh.get_node_color(), [0.2, 0.2, 0.2]))
            self.assertAlmostEqual(mesh.get_node_radius(), 0.002)

    ############################################################################
    # Texture Tests
    ############################################################################

    def test_scalar_texture_operations(self):
        """Test scalar texture get/set, filter, colormap, and range operations."""

        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            vertices, faces = self._create_simple_mesh()
            uvs = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float32)
            mesh = viewer.add_mesh(
                "test_mesh",
                vertices,
                faces,
                texture_coordinates=uvs,
            )

            # Test initial state - no texture
            self.assertIsNone(mesh.get_texture())
            self.assertEqual(
                mesh.get_texture_filter(), MeshRenderer.TextureFilter.LINEAR
            )
            self.assertEqual(mesh.get_texture_colormap(), MeshRenderer.Colormap.VIRIDIS)
            self.assertIsNone(mesh.get_texture_colormap_range())

            # Test setting scalar texture (H, W)
            texture_scalar = np.random.rand(64, 64).astype(np.float32)
            mesh.set_texture(texture_scalar)
            self.assertIsNotNone(mesh.get_texture())
            self.assertEqual(mesh.get_texture().shape, (64, 64))
            self.assertTrue(np.allclose(mesh.get_texture(), texture_scalar))

            # Test updating texture with same dimensions (smart reallocation)
            texture_scalar2 = np.random.rand(64, 64).astype(np.float32)
            mesh.set_texture(texture_scalar2)
            self.assertTrue(np.allclose(mesh.get_texture(), texture_scalar2))

            # Test updating texture with different dimensions
            texture_scalar3 = np.random.rand(128, 128).astype(np.float32)
            mesh.set_texture(texture_scalar3)
            self.assertEqual(mesh.get_texture().shape, (128, 128))
            self.assertTrue(np.allclose(mesh.get_texture(), texture_scalar3))

            # Test texture filter
            mesh.set_texture_filter(MeshRenderer.TextureFilter.NEAREST)
            self.assertEqual(
                mesh.get_texture_filter(), MeshRenderer.TextureFilter.NEAREST
            )
            mesh.set_texture_filter(MeshRenderer.TextureFilter.LINEAR)
            self.assertEqual(
                mesh.get_texture_filter(), MeshRenderer.TextureFilter.LINEAR
            )

            # Test colormap
            for colormap in [
                MeshRenderer.Colormap.MAGMA,
                MeshRenderer.Colormap.INFERNO,
                MeshRenderer.Colormap.PLASMA,
            ]:
                mesh.set_texture_colormap(colormap)
                self.assertEqual(mesh.get_texture_colormap(), colormap)

            # Test colormap range
            mesh.set_texture_colormap_range((0.0, 1.0))
            self.assertEqual(mesh.get_texture_colormap_range(), (0.0, 1.0))

            mesh.set_texture_colormap_range((-1.0, 2.5))
            self.assertEqual(mesh.get_texture_colormap_range(), (-1.0, 2.5))

            mesh.set_texture_colormap_range(None)
            self.assertIsNone(mesh.get_texture_colormap_range())

            # Test removing texture
            mesh.set_texture(None)
            self.assertIsNone(mesh.get_texture())

    def test_color_texture_operations(self):
        """Test RGB color texture operations."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            vertices, faces = self._create_simple_mesh()
            uvs = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float32)
            mesh = viewer.add_mesh(
                "test_mesh",
                vertices,
                faces,
                texture_coordinates=uvs,
            )

            # Test setting RGB texture (H, W, 3)
            texture_color = np.random.rand(64, 64, 3).astype(np.float32)
            mesh.set_texture(texture_color)
            self.assertIsNotNone(mesh.get_texture())
            self.assertEqual(mesh.get_texture().shape, (64, 64, 3))
            self.assertTrue(np.allclose(mesh.get_texture(), texture_color))

            # Test updating RGB texture with same dimensions
            texture_color2 = np.random.rand(64, 64, 3).astype(np.float32)
            mesh.set_texture(texture_color2)
            self.assertTrue(np.allclose(mesh.get_texture(), texture_color2))

            # Test updating RGB texture with different dimensions
            texture_color3 = np.random.rand(128, 128, 3).astype(np.float32)
            mesh.set_texture(texture_color3)
            self.assertEqual(mesh.get_texture().shape, (128, 128, 3))
            self.assertTrue(np.allclose(mesh.get_texture(), texture_color3))

            # Test switching from RGB to scalar
            texture_scalar = np.random.rand(64, 64).astype(np.float32)
            mesh.set_texture(texture_scalar)
            self.assertEqual(mesh.get_texture().shape, (64, 64))

            # Test switching from scalar to RGB
            mesh.set_texture(texture_color)
            self.assertEqual(mesh.get_texture().shape, (64, 64, 3))

    def test_texture_validation(self):
        """Test texture validation for invalid dimensions."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            vertices, faces = self._create_simple_mesh()
            uvs = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float32)
            mesh = viewer.add_mesh(
                "test_mesh",
                vertices,
                faces,
                texture_coordinates=uvs,
            )

            # Test invalid 1D array
            with self.assertRaises(ValueError) as cm:
                mesh.set_texture(np.array([1.0, 2.0, 3.0]))
            self.assertIn(
                "Expected (H,W) for scalar or (H,W,3) for RGB", str(cm.exception)
            )

            # Test invalid 3D array with wrong third dimension
            with self.assertRaises(ValueError) as cm:
                mesh.set_texture(np.random.rand(64, 64, 4))
            self.assertIn(
                "Expected (H,W) for scalar or (H,W,3) for RGB", str(cm.exception)
            )

            # Test invalid 4D array
            with self.assertRaises(ValueError) as cm:
                mesh.set_texture(np.random.rand(64, 64, 3, 1))
            self.assertIn(
                "Expected (H,W) for scalar or (H,W,3) for RGB", str(cm.exception)
            )

    def test_texture_colormap_range_validation(self):
        """Test colormap range validation."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            vertices, faces = self._create_simple_mesh()
            uvs = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float32)
            mesh = viewer.add_mesh(
                "test_mesh",
                vertices,
                faces,
                texture_coordinates=uvs,
            )

            # Test invalid range - not 2 elements
            with self.assertRaises(ValueError) as cm:
                mesh.set_texture_colormap_range((0.0, 1.0, 2.0))
            self.assertIn("2-element tuple", str(cm.exception))

            # Test invalid range - vmin >= vmax
            with self.assertRaises(ValueError) as cm:
                mesh.set_texture_colormap_range((1.0, 1.0))
            self.assertIn("vmin < vmax", str(cm.exception))

            with self.assertRaises(ValueError) as cm:
                mesh.set_texture_colormap_range((2.0, 1.0))
            self.assertIn("vmin < vmax", str(cm.exception))

            # Test valid ranges
            mesh.set_texture_colormap_range((0.0, 1.0))
            self.assertEqual(mesh.get_texture_colormap_range(), (0.0, 1.0))

            mesh.set_texture_colormap_range((-10.0, 10.0))
            self.assertEqual(mesh.get_texture_colormap_range(), (-10.0, 10.0))

    def test_texture_update_mechanism(self):
        """Test that texture updates trigger the dirty flag and work with update()."""

        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            vertices, faces = self._create_simple_mesh()
            uvs = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float32)
            mesh = viewer.add_mesh(
                "test_mesh",
                vertices,
                faces,
                texture_coordinates=uvs,
            )

            # Set texture and update
            texture = np.random.rand(64, 64).astype(np.float32)
            mesh.set_texture(texture)
            mesh.update()
            self.assertTrue(np.allclose(mesh.get_texture(), texture))

            # Change filter and update
            mesh.set_texture_filter(MeshRenderer.TextureFilter.NEAREST)
            mesh.update()
            self.assertEqual(
                mesh.get_texture_filter(), MeshRenderer.TextureFilter.NEAREST
            )

            # Change colormap and update
            mesh.set_texture_colormap(MeshRenderer.Colormap.MAGMA)
            mesh.update()
            self.assertEqual(mesh.get_texture_colormap(), MeshRenderer.Colormap.MAGMA)

            # Change range and update
            mesh.set_texture_colormap_range((0.0, 1.0))
            mesh.update()
            self.assertEqual(mesh.get_texture_colormap_range(), (0.0, 1.0))

            # Remove texture and update
            mesh.set_texture(None)
            mesh.update()
            self.assertIsNone(mesh.get_texture())

    def test_texture_with_different_data_types(self):
        """Test texture handling with various input data types."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            vertices, faces = self._create_simple_mesh()
            uvs = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float32)
            mesh = viewer.add_mesh(
                "test_mesh",
                vertices,
                faces,
                texture_coordinates=uvs,
            )

            # Test with Python list
            texture_list = [[0.5, 0.6], [0.7, 0.8]]
            mesh.set_texture(texture_list)
            self.assertIsNotNone(mesh.get_texture())
            self.assertEqual(mesh.get_texture().dtype, np.float32)

            # Test with int array (should be converted to float32)
            texture_int = np.array([[1, 2], [3, 4]], dtype=np.int32)
            mesh.set_texture(texture_int)
            self.assertEqual(mesh.get_texture().dtype, np.float32)

            # Test with float64 (should be converted to float32)
            texture_float64 = np.random.rand(32, 32).astype(np.float64)
            mesh.set_texture(texture_float64)
            self.assertEqual(mesh.get_texture().dtype, np.float32)

    ############################################################################
    # Texture Coordinates Tests
    ############################################################################

    def test_texture_coordinates_basic_operations(self):
        """Test texture coordinate initialization, get/set, and removal operations."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            vertices, faces = self._create_simple_mesh()
            uvs = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float32)

            # Test initialization without UVs
            mesh_no_uvs = viewer.add_mesh("mesh_no_uvs", vertices, faces)
            self.assertIsNone(mesh_no_uvs.get_texture_coordinates())

            # Test initialization with UVs
            mesh_with_uvs = viewer.add_mesh(
                "mesh_with_uvs",
                vertices,
                faces,
                texture_coordinates=uvs,
            )
            self.assertIsNotNone(mesh_with_uvs.get_texture_coordinates())
            self.assertTrue(np.allclose(mesh_with_uvs.get_texture_coordinates(), uvs))

            # Test setting UVs on a mesh without them
            mesh_no_uvs.set_texture_coordinates(uvs)
            mesh_no_uvs.update()
            self.assertIsNotNone(mesh_no_uvs.get_texture_coordinates())
            self.assertTrue(np.allclose(mesh_no_uvs.get_texture_coordinates(), uvs))

            # Test updating UVs
            uvs2 = np.array([[0.25, 0.25], [0.75, 0.25], [0.5, 0.75]], dtype=np.float32)
            mesh_no_uvs.set_texture_coordinates(uvs2)
            mesh_no_uvs.update()
            self.assertTrue(np.allclose(mesh_no_uvs.get_texture_coordinates(), uvs2))

            # Test removing UVs
            mesh_with_uvs.set_texture_coordinates(None)
            mesh_with_uvs.update()
            self.assertIsNone(mesh_with_uvs.get_texture_coordinates())

    def test_texture_coordinates_validation(self):
        """Test texture coordinate validation for shape and vertex count."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            vertices, faces = self._create_simple_mesh()
            mesh = viewer.add_mesh("test_mesh", vertices, faces)

            # Test incorrect shape - wrong number of columns (should be Nx2)
            with self.assertRaises(ValueError) as cm:
                mesh.set_texture_coordinates(
                    np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
                )
            self.assertIn("Expected Nx2", str(cm.exception))

            # Test incorrect dimensions (should be 2D array)
            with self.assertRaises(ValueError) as cm:
                mesh.set_texture_coordinates(np.array([0.0, 0.0], dtype=np.float32))
            self.assertIn("Expected Nx2", str(cm.exception))

            # Test wrong vertex count - too few UVs
            with self.assertRaises(ValueError) as cm:
                mesh.set_texture_coordinates(
                    np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
                )
            self.assertIn("same number of entries as vertices", str(cm.exception))

            # Test wrong vertex count - too many UVs
            with self.assertRaises(ValueError) as cm:
                mesh.set_texture_coordinates(
                    np.array(
                        [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.5, 0.5]],
                        dtype=np.float32,
                    )
                )
            self.assertIn("same number of entries as vertices", str(cm.exception))

    def test_texture_coordinates_replace_geometry(self):
        """Test texture coordinates through replace_geometry operations."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            vertices, faces = self._create_simple_mesh()
            uvs = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float32)
            mesh = viewer.add_mesh(
                "test_mesh", vertices, faces, texture_coordinates=uvs
            )

            # Test replace_geometry with new UVs
            new_vertices = np.array([[1, 1, 1], [2, 1, 1], [1, 2, 1]], dtype=np.float32)
            new_faces = np.array([[0, 1, 2]], dtype=np.int32)
            new_uvs = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float32)
            transform = np.eye(4, dtype=np.float32)

            mesh.replace_geometry(new_vertices, new_faces, transform, new_uvs)
            self.assertTrue(np.allclose(mesh.get_local_coordinates(), new_vertices))
            self.assertIsNotNone(mesh.get_texture_coordinates())
            self.assertTrue(np.allclose(mesh.get_texture_coordinates(), new_uvs))

            # Test replace_geometry without UVs removes existing coordinates
            mesh.replace_geometry(vertices, faces, transform)
            self.assertIsNone(mesh.get_texture_coordinates())


########################################################################################

if __name__ == "__main__":
    unittest.main()
