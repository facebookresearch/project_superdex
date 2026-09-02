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
from unittest.mock import MagicMock, patch

import numpy as np
from superdex.physics.utils.coordinate_systems import (
    COORDINATE_SYSTEMS,
    CoordinateSystem,
    DEFAULT_COORDINATE_SYSTEM,
)
from superdex.physics.utils.testing.decorators import skip_if
from superdex.physics.utils.testing.testcases import (
    add_rigid_cube,
    make_empty_scene,
    make_single_rigid_cube_scene,
    MochiContextTestCase,
)
from superdex.physics.viewer import Viewer, VIEWER_AVAILABLE, ViewerCfg
from superdex.physics.viewer.viewer_state import ActorState

########################################################################################


@skip_if(not VIEWER_AVAILABLE, "Requires Polyscope >= 2.5.0")
class TestViewer(MochiContextTestCase):
    """Test suite for the Viewer class functionality."""

    def test_sequential_viewer_creation_and_destruction(self):
        # Tests that multiple viewers can be created and destroyed in the same process
        # as long as they aren't concurrent.
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg):
            pass
        with Viewer(cfg):
            pass

    def test_concurrent_viewer_creation_raises_runtime_error(self):
        # Tests that multiple viewers cannot be created concurrently.
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg):
            with self.assertRaises(RuntimeError):
                Viewer(cfg)

    def test_onscreen_rendering_produces_none(self):
        # Tests if the viewer returns None when offscreen is False.
        cfg = ViewerCfg(backend="openGL_mock", offscreen=False)
        with Viewer(cfg) as viewer:
            result = viewer.render()
            assert result is None

    def test_offscreen_rendering_produces_image(self):
        # Tests if the viewer returns a frame when offscreen is True.
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            result = viewer.render()
            assert isinstance(result, np.ndarray)

    def test_set_get_scene(self):
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            # Viewer is initialized with no scene.
            self.assertIsNone(viewer.get_scene())
            self.assertEqual(len(viewer.get_actors()), 0)
            viewer.render()

            # Set an empty scene.
            # Scene must be the same, but the number of actors must be 0.
            with make_empty_scene() as scene:
                viewer.set_scene(scene)
                self.assertEqual(viewer.get_scene(), scene)
                self.assertEqual(len(viewer.get_actors()), 0)
                viewer.render()

            # Set a single rigid cube scene.
            with make_single_rigid_cube_scene() as scene:
                viewer.set_scene(scene)
                self.assertEqual(viewer.get_scene(), scene)
                actors = viewer.get_actors()
                self.assertEqual(scene.get_num_actors(), 1)
                self.assertEqual(len(actors), 1)
                # Verify the actor is returned (not checking specific type since
                # Actor might not be directly importable in tests, just verify it
                # exists)
                self.assertIsNotNone(actors[0])
                viewer.render()
                self.assertFalse(viewer.get_scene_bounds().is_empty)

            # Transition from populated to empty and verify the old bounds are cleared.
            with (
                make_empty_scene() as scene,
                patch(
                    "superdex.physics.viewer.viewer.ps.set_bounding_box"
                ) as set_bounding_box,
            ):
                viewer.set_scene(scene)
                self.assertTrue(viewer.get_scene_bounds().is_empty)
                set_bounding_box.assert_called_once()
                call = set_bounding_box.call_args
                np.testing.assert_array_equal(call.kwargs["low"], np.zeros(3))
                np.testing.assert_array_equal(call.kwargs["high"], np.zeros(3))
                viewer.render()
                self.assertTrue(viewer.get_scene_bounds().is_empty)

            # Set no scene.
            viewer.set_scene(None)
            self.assertIsNone(viewer.get_scene())
            self.assertEqual(len(viewer.get_actors()), 0)

    def test_rebuild_registered_plane_uses_plane_renderer_fallback(self) -> None:
        viewer = object.__new__(Viewer)
        handle = MagicMock()
        handle.value = 1
        instance = MagicMock()
        instance.get_surface_mesh.return_value.is_empty.return_value = True
        old_renderer = MagicMock()
        actor_state = ActorState(handle, instance, old_renderer)
        state = MagicMock()
        state.scene.handle.value = 2
        state.scene.actors = {handle: actor_state}
        viewer._state = state
        viewer._create_glb_renderer = MagicMock(return_value=None)
        viewer._is_plane_actor = MagicMock(return_value=True)

        with (
            patch(
                "superdex.physics.viewer.viewer.render_model_registry.get",
                return_value=object(),
            ),
            patch(
                "superdex.physics.viewer.viewer.StaticPlaneRenderer"
            ) as plane_renderer,
        ):
            viewer._rebuild_render_model_actors()

        old_renderer.remove.assert_called_once_with()
        plane_renderer.assert_called_once_with(instance, state.coordinate_transform)
        self.assertIs(actor_state.renderer, plane_renderer.return_value)

    def test_actor_creation_and_removal(self):
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer, make_empty_scene() as scene:
            viewer.set_scene(scene)
            self.assertEqual(len(viewer.get_actors()), 0)

            # Add a rigid cube.
            cube_actor_1 = add_rigid_cube(scene, "Cube 1")
            viewer.render()
            self.assertEqual(len(viewer.get_actors()), 1)
            self.assertEqual(viewer.get_actors()[0], cube_actor_1)

            # Add another rigid cube.
            cube_actor_2 = add_rigid_cube(scene, "Cube 2")
            viewer.render()
            self.assertEqual(len(viewer.get_actors()), 2)
            self.assertEqual(viewer.get_actors()[0], cube_actor_1)
            self.assertEqual(viewer.get_actors()[1], cube_actor_2)

            # Remove the first cube.
            scene.destroy_actor(cube_actor_1)
            viewer.render()
            self.assertEqual(len(viewer.get_actors()), 1)
            self.assertEqual(viewer.get_actors()[0], cube_actor_2)

            # Remove the second cube.
            scene.destroy_actor(cube_actor_2)
            viewer.render()
            self.assertEqual(len(viewer.get_actors()), 0)

    def test_actors_with_repeated_names(self):
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer, make_empty_scene() as scene:
            viewer.set_scene(scene)
            self.assertEqual(len(viewer.get_actors()), 0)

            actor_1 = add_rigid_cube(scene, "Cube")
            actor_2 = add_rigid_cube(scene, "Cube")
            viewer.render()

            actors = viewer.get_actors()
            self.assertEqual(len(actors), 2)
            self.assertIn(actor_1, actors)
            self.assertIn(actor_2, actors)

    def test_excluded_actors(self):
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer, make_single_rigid_cube_scene() as scene:
            viewer.set_scene(scene)
            self.assertEqual(len(viewer.get_actors()), 1)
            viewer.set_excluded_actors(["Must", "Not", "Exclude", "Anything"])
            self.assertEqual(len(viewer.get_actors()), 1)
            viewer.set_excluded_actors(["Cube"])  # Exact match
            self.assertEqual(len(viewer.get_actors()), 0)
            viewer.set_excluded_actors(["*be"])  # Wildcard
            self.assertEqual(len(viewer.get_actors()), 0)
            viewer.set_excluded_actors(["Cu*"])  # Wildcard
            self.assertEqual(len(viewer.get_actors()), 0)
            viewer.set_excluded_actors(["cube"])  # Case sensitiveness
            self.assertEqual(len(viewer.get_actors()), 1)
            viewer.set_excluded_actors(["Cub"])  # Partial match
            self.assertEqual(len(viewer.get_actors()), 1)

    def test_initialize_with_coordinate_system(self):
        """Test initializing viewer with user-specified coordinate systems."""

        # No coordinate system specified.
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            coordinate_system = viewer.get_coordinate_system()
            self.assertEqual(coordinate_system, DEFAULT_COORDINATE_SYSTEM)

        # Coordinate system from string.
        cfg.coordinate_system = "unreal"
        with Viewer(cfg) as viewer:
            coordinate_system = viewer.get_coordinate_system()
            self.assertEqual(coordinate_system, COORDINATE_SYSTEMS["unreal"])

        # Custom coordinate system.
        cfg.coordinate_system = CoordinateSystem("-x", "+z", "-y")
        with Viewer(cfg) as viewer:
            coordinate_system = viewer.get_coordinate_system()
            self.assertEqual(coordinate_system, cfg.coordinate_system)

        # Invalid coordinate system string.
        cfg.coordinate_system = "invalid system"
        with self.assertRaises(ValueError):
            with Viewer(cfg):
                pass

    def test_viewer_camera_respects_coordinate_system(self):
        """Test that camera operations respect the configured coordinate system."""
        cfg = ViewerCfg(
            backend="openGL_mock", offscreen=True, coordinate_system="unreal"
        )
        with Viewer(cfg) as viewer:
            look_from = np.array([1.0, 2.0, 3.0])
            look_at = np.array([0.0, 0.0, 0.0])
            look_dir = (look_at - look_from) / np.sqrt(14)
            viewer.set_camera_view(look_from, look_at)
            camera_pos = viewer.get_camera_position()
            camera_look_dir = viewer.get_camera_look_dir()
            assert np.allclose(camera_pos, look_from)
            assert np.allclose(camera_look_dir, look_dir)

    def test_set_camera_view_updates_view_center(self):
        """Test that set_camera_view automatically sets the view center to look_at point."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            look_from = np.array([5.0, 5.0, 5.0])
            look_at = np.array([1.0, 2.0, 3.0])
            viewer.set_camera_view(look_from, look_at)

            # Verify that the view center is set to the look_at point
            view_center = viewer.get_camera_view_center()
            assert np.allclose(view_center, look_at)

    def test_set_get_camera_view_center(self):
        """Test that view center can be set to various positions."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            test_centers = [
                np.array([0.0, 0.0, 0.0]),
                np.array([10.0, 20.0, 30.0]),
                np.array([-5.0, -10.0, -15.0]),
                np.array([1.5, 2.5, 3.5]),
            ]

            for center in test_centers:
                viewer.set_camera_view_center(center)
                retrieved_center = viewer.get_camera_view_center()
                assert np.allclose(retrieved_center, center)

    def test_camera_view_center_respects_coordinate_system(self):
        """Test that camera view center operations respect the configured coordinate system."""
        cfg = ViewerCfg(
            backend="openGL_mock", offscreen=True, coordinate_system="unreal"
        )
        with Viewer(cfg) as viewer:
            # Set view center in world space
            world_center = np.array([1.0, 2.0, 3.0])
            viewer.set_camera_view_center(world_center)

            # Retrieve and verify it matches
            retrieved_center = viewer.get_camera_view_center()
            assert np.allclose(retrieved_center, world_center)

    def test_mesh_creation_and_removal(self):
        """Test adding and removing helper meshes."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            # Initially no meshes
            self.assertEqual(len(viewer.get_meshes()), 0)

            # Create a simple cube mesh
            vertices = np.array(
                [
                    [-1, -1, -1],
                    [1, -1, -1],
                    [1, 1, -1],
                    [-1, 1, -1],
                    [-1, -1, 1],
                    [1, -1, 1],
                    [1, 1, 1],
                    [-1, 1, 1],
                ],
                dtype=np.float32,
            )
            faces = np.array(
                [
                    [0, 1, 2],
                    [0, 2, 3],  # front
                    [4, 5, 6],
                    [4, 6, 7],  # back
                ],
                dtype=np.int32,
            )

            # Add a mesh
            mesh1 = viewer.add_mesh("cube1", vertices, faces)
            self.assertIsNotNone(mesh1)
            self.assertEqual(len(viewer.get_meshes()), 1)
            self.assertEqual(viewer.get_mesh("cube1"), mesh1)

            # Add another mesh
            mesh2 = viewer.add_mesh("cube2", vertices * 2, faces)
            self.assertEqual(len(viewer.get_meshes()), 2)
            self.assertIn(mesh1, viewer.get_meshes())
            self.assertIn(mesh2, viewer.get_meshes())

            # Remove the first mesh
            viewer.remove_mesh("cube1")
            self.assertEqual(len(viewer.get_meshes()), 1)
            self.assertIsNone(viewer.get_mesh("cube1"))
            self.assertEqual(viewer.get_mesh("cube2"), mesh2)

            # Remove the second mesh by instance
            viewer.remove_mesh(mesh2)
            self.assertEqual(len(viewer.get_meshes()), 0)

    def test_mesh_with_repeated_names(self):
        """Test that adding a mesh with the same name replaces the previous one."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            # Create a simple triangle mesh
            vertices1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
            faces = np.array([[0, 1, 2]], dtype=np.int32)

            mesh1 = viewer.add_mesh("test_mesh", vertices1, faces)
            self.assertEqual(len(viewer.get_meshes()), 1)

            # Add mesh with same name but different vertices
            vertices2 = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=np.float32)
            mesh2 = viewer.add_mesh("test_mesh", vertices2, faces)

            # Should still have only one mesh, and it should be the same instance
            self.assertEqual(len(viewer.get_meshes()), 1)
            self.assertEqual(mesh1, mesh2)

            # Verify the geometry was updated
            coords = mesh2.get_local_coordinates()
            self.assertTrue(np.allclose(coords, vertices2))

    def test_point_cloud_creation_and_removal(self):
        """Test adding and removing helper point clouds."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            coordinates = np.array(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
                dtype=np.float32,
            )

            # Initially no point clouds
            self.assertEqual(len(viewer.get_point_clouds()), 0)

            # Add a point cloud
            cloud1 = viewer.add_point_cloud("cloud1", coordinates)
            self.assertIsNotNone(cloud1)
            self.assertEqual(len(viewer.get_point_clouds()), 1)
            self.assertEqual(viewer.get_point_cloud("cloud1"), cloud1)

            # Add another point cloud
            cloud2 = viewer.add_point_cloud("cloud2", coordinates * 2)
            self.assertEqual(len(viewer.get_point_clouds()), 2)

            # Remove by name
            viewer.remove_point_cloud("cloud1")
            self.assertEqual(len(viewer.get_point_clouds()), 1)
            self.assertIsNone(viewer.get_point_cloud("cloud1"))

            # Remove by instance
            viewer.remove_point_cloud(cloud2)
            self.assertEqual(len(viewer.get_point_clouds()), 0)

            # Remove nonexistent (should not raise)
            viewer.remove_point_cloud("nonexistent")

    def test_point_cloud_update_with_same_name(self):
        """Test that adding a point cloud with the same name updates existing one."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            coordinates = np.array(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
                dtype=np.float32,
            )

            # Create with initial properties
            cloud = viewer.add_point_cloud(
                "test_cloud",
                coordinates,
                radius=0.01,
                color=np.array([1.0, 0.0, 0.0]),
            )

            # Update with new coordinates and properties
            new_coords = coordinates * 2
            cloud_updated = viewer.add_point_cloud(
                "test_cloud",
                new_coords,
                radius=0.05,
                color=np.array([0.0, 1.0, 0.0]),
            )

            # Same instance and updated properties
            self.assertEqual(cloud, cloud_updated)
            self.assertEqual(len(viewer.get_point_clouds()), 1)
            self.assertAlmostEqual(cloud.get_radius(), 0.05)
            self.assertTrue(np.allclose(cloud.get_color(), [0.0, 1.0, 0.0]))
            self.assertTrue(np.allclose(cloud.get_coordinates(), new_coords))

    def test_point_cloud_per_point_radii_and_colors_update(self):
        """Test updating point cloud with per-point radii and colors via add_point_cloud."""
        cfg = ViewerCfg(backend="openGL_mock", offscreen=True)
        with Viewer(cfg) as viewer:
            coordinates = np.array(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
                dtype=np.float32,
            )
            radii = np.array([0.01, 0.02, 0.03, 0.04, 0.05], dtype=np.float32)
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

            # Create without per-point attributes
            cloud = viewer.add_point_cloud("test_cloud", coordinates)
            self.assertIsNone(cloud.get_point_radii())
            self.assertIsNone(cloud.get_point_colors())

            # Update with per-point radii and colors
            viewer.add_point_cloud(
                "test_cloud", coordinates, radii=radii, colors=colors
            )
            self.assertTrue(np.allclose(cloud.get_point_radii(), radii))
            self.assertTrue(np.allclose(cloud.get_point_colors(), colors))

            # Update without per-point attributes (should clear them)
            viewer.add_point_cloud("test_cloud", coordinates)
            self.assertIsNone(cloud.get_point_radii())
            self.assertIsNone(cloud.get_point_colors())


########################################################################################

if __name__ == "__main__":
    unittest.main()
