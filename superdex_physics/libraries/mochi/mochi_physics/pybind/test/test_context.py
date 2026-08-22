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

import gc
import inspect
import math
import os

from test.conftest import (
    assets_dir,
    default_num_worker_threads,
    mochi,
    MochiTestBase,
    np,
    requires_internal_assets,
    small_cube_tet_mesh_connectivity,
    small_cube_tet_mesh_coordinates,
    small_cube_tri_mesh_connectivity,
    small_cube_tri_mesh_coordinates,
)


class TestContext(MochiTestBase):
    def test_num_threads(self):
        try:
            # Initialize and shutdown a few times with various numbers of threads
            num_thread_cases = [-1, 0, 1, 2, 4]
            for n in num_thread_cases:
                self.assertTrue(mochi.is_initialized())
                mochi.shutdown()
                self.assertFalse(mochi.is_initialized())
                mochi.initialize(num_worker_threads=n)
                self.assertTrue(mochi.is_initialized())
                if n == -1:
                    self.assertLessEqual(
                        0, mochi.get_num_threads()
                    )  # -1 means Mochi gets to choose
                else:
                    self.assertEqual(n, mochi.get_num_threads())

                # Single threaded mode can be applied regardless of the number of initiliazed threads
                self.assertEqual(n == 0, mochi.is_single_threaded())
                mochi.set_is_single_threaded(True)
                self.assertTrue(mochi.is_single_threaded())
                mochi.set_is_single_threaded(False)
                self.assertEqual(n == 0, mochi.is_single_threaded())
        finally:
            # Restore default number of threads
            mochi.shutdown()
            mochi.initialize(num_worker_threads=default_num_worker_threads)
            self.assertTrue(mochi.is_initialized())

    def test_enable_log_channel(self):
        # Verbose is disabled by default
        self.assertFalse(mochi.is_log_channel_enabled(mochi.LogChannel.VERBOSE))

        # Info, Warning, Error are enabled by default
        self.assertTrue(mochi.is_log_channel_enabled(mochi.LogChannel.INFO))
        self.assertTrue(mochi.is_log_channel_enabled(mochi.LogChannel.WARNING))
        self.assertTrue(mochi.is_log_channel_enabled(mochi.LogChannel.ERROR))

        try:
            # Disable Warning, verify
            mochi.enable_log_channel(mochi.LogChannel.WARNING, False)
            self.assertFalse(mochi.is_log_channel_enabled(mochi.LogChannel.WARNING))

            # Re-enable Warning
            mochi.enable_log_channel(mochi.LogChannel.WARNING, enable=True)
            self.assertTrue(mochi.is_log_channel_enabled(mochi.LogChannel.WARNING))

            # Enable Verbose, verify
            mochi.enable_log_channel(channel=mochi.LogChannel.VERBOSE, enable=True)
            self.assertTrue(mochi.is_log_channel_enabled(mochi.LogChannel.VERBOSE))
        finally:
            # Restore defaults
            mochi.enable_log_channel(mochi.LogChannel.VERBOSE, False)
            mochi.enable_log_channel(mochi.LogChannel.WARNING, True)

    def test_enable_log_channel_filters_callback(self):
        """Verify that disabled channels do not reach the log callback."""
        messages = []

        def log_callback(channel, message, file, line):
            messages.append((channel, message))

        mochi.set_log_callback(log_callback)
        try:
            # Disable Warning
            mochi.enable_log_channel(mochi.LogChannel.WARNING, False)

            # Log to Warning — should be filtered before reaching callback
            mochi.log(message="should not appear", channel=mochi.LogChannel.WARNING)
            self.assertEqual(len(messages), 0)

            # Log to Info — should reach callback
            mochi.log(message="hello")
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0][0], mochi.LogChannel.INFO)
        finally:
            mochi.enable_log_channel(mochi.LogChannel.WARNING, True)
            mochi.set_log_callback(None)

    def test_set_log_callback(self):
        try:
            expected_channel = mochi.LogChannel.INFO
            expected_message = ""
            expected_file = __file__
            expected_line = 0

            def log_callback(channel, message, file, line):
                self.assertEqual(expected_channel, channel)
                self.assertEqual(expected_message, message)
                # Compare just the filename stem to handle path and .py/.pyc differences.
                expected_stem = os.path.splitext(os.path.basename(expected_file))[0]
                actual_stem = os.path.splitext(os.path.basename(file))[0]
                self.assertEqual(expected_stem, actual_stem)
                self.assertEqual(expected_line, line)

            # Set a custom log callback
            mochi.set_log_callback(log_callback)

            # Send a few test messages
            expected_channel = mochi.LogChannel.INFO
            expected_message = "Test message\n"
            expected_line = inspect.currentframe().f_lineno + 1
            mochi.log("Test message")

            expected_channel = mochi.LogChannel.WARNING
            expected_message = "Oh my!\n"
            expected_line = inspect.currentframe().f_lineno + 1
            mochi.log(message="Oh my!", channel=mochi.LogChannel.WARNING)

        finally:
            # Restore default log callback
            mochi.set_log_callback(None)

    @requires_internal_assets
    def test_load_shape_from_file(self):
        vzeros = [0, 0, 0]
        vones = [1, 1, 1]

        # Load from file with default scale and transform
        path = os.path.join(assets_dir, "cube/cube_minimal.mochi.json")
        self.assertEqual(0, mochi.get_num_shapes())
        shape1 = mochi.load_shape_from_file(path)
        self.assertTrue(shape1.is_valid())
        self.assertEqual(mochi.Aabb(vzeros, vones), mochi.get_shape_aabb(shape1))
        self.assertEqual(1, mochi.get_num_shapes())

        # Custom scale
        shape2 = mochi.load_shape_from_file(file_path=path, bake_scale=[2, 3, 4])
        self.assertTrue(shape2.is_valid())
        self.assertNotEqual(shape1, shape2)
        self.assertEqual(mochi.Aabb(vzeros, [2, 3, 4]), mochi.get_shape_aabb(shape2))
        self.assertEqual(2, mochi.get_num_shapes())

        # Custom Transform
        shape3 = mochi.load_shape_from_file(
            file_path=path,
            bake_transform=mochi.TransformRT(
                rotation=mochi.Quaternion.rotation_x(math.pi / 2),
                translation=[1, 2, 3],
            ),
        )
        self.assertTrue(shape3.is_valid())
        self.assertNotEqual(shape1, shape3)
        for i in range(0, 3):
            self.assertAlmostEqual(
                [1, 1, 3][i], mochi.get_shape_aabb(shape3).min[i], places=6
            )
            self.assertAlmostEqual(
                [2, 2, 4][i], mochi.get_shape_aabb(shape3).max[i], places=6
            )
        self.assertEqual(3, mochi.get_num_shapes())

        # Cleanup
        mochi.release_shape(shape1)
        mochi.release_shape(shape2)
        mochi.release_shape(shape3)
        self.assertEqual(0, mochi.get_num_shapes())

    @requires_internal_assets
    def test_load_shape_from_bytes(self):
        vzeros = [0, 0, 0]
        vones = [1, 1, 1]
        path = os.path.join(assets_dir, "cube/cube_minimal.mochi.json")
        file_data = None
        with open(path, "rb") as file:
            file_data = bytearray(file.read())

        # Load from bytearray with default scale
        self.assertEqual(0, mochi.get_num_shapes())
        shape = mochi.load_shape_from_bytes(file_data)
        self.assertTrue(shape.is_valid())
        self.assertEqual(mochi.Aabb(vzeros, vones), mochi.get_shape_aabb(shape))
        self.assertEqual(1, mochi.get_num_shapes())
        mochi.release_shape(shape)
        self.assertEqual(0, mochi.get_num_shapes())

        # Load from bytearray with custom scale and transform
        self.assertEqual(0, mochi.get_num_shapes())
        shape = mochi.load_shape_from_bytes(
            file_data=file_data,
            bake_scale=[2, 3, 4],
            bake_transform=mochi.TransformRT(
                rotation=mochi.Quaternion.rotation_x(math.pi / 2),
                translation=[1, 2, 3],
            ),
        )
        self.assertTrue(shape.is_valid())
        print(str(mochi.get_shape_aabb(shape)))
        expected_aabb = mochi.Aabb([1, -2, 3], [3, 2, 6])
        actual_aabb = mochi.get_shape_aabb(shape)
        for i in range(0, 3):
            self.assertAlmostEqual(expected_aabb.min[i], actual_aabb.min[i], places=6)
            self.assertAlmostEqual(expected_aabb.max[i], actual_aabb.max[i], places=6)
        self.assertEqual(1, mochi.get_num_shapes())
        mochi.release_shape(shape)
        self.assertEqual(0, mochi.get_num_shapes())

    def test_create_tet_mesh_shape(self):
        coordinates = small_cube_tet_mesh_coordinates
        connectivity = small_cube_tet_mesh_connectivity
        expected_aabb = mochi.Aabb([-0.1, -0.1, -0.1], [0.1, 0.1, 0.1])

        def check_and_release(shape):
            self.assertTrue(shape.is_valid())
            self.assertEqual(expected_aabb, mochi.get_shape_aabb(shape))
            self.assertEqual(1, mochi.get_num_shapes())
            mochi.release_shape(shape)
            self.assertEqual(0, mochi.get_num_shapes())

        # Create mesh from numpy arrays
        self.assertEqual(0, mochi.get_num_shapes())
        shape = mochi.create_tet_mesh_shape(
            coordinates=coordinates, connectivity=connectivity
        )
        check_and_release(shape)

        # Repeat using create_mesh_shape (more generic) with MeshData.
        # Note that pybind will implicitly convert MeshData to MeshDataView.
        shape = mochi.create_mesh_shape(
            mochi.MeshData(
                nodes_per_element=4, coordinates=coordinates, connectivity=connectivity
            )
        )
        check_and_release(shape)

        # Repeat the above with expanded kwargs
        shape = mochi.create_mesh_shape(
            nodes_per_element=4, coordinates=coordinates, connectivity=connectivity
        )
        check_and_release(shape)

        # Repeat using create_model_shape (even more generic) with ModelData
        shape = mochi.create_model_shape(
            mochi.ModelData(
                mesh=mochi.MeshData(
                    nodes_per_element=4,
                    coordinates=coordinates,
                    connectivity=connectivity,
                )
            )
        )
        check_and_release(shape)

        # Repeat the above with expanded kwargs
        shape = mochi.create_model_shape(
            mesh=mochi.MeshData(
                nodes_per_element=4,
                coordinates=coordinates,
                connectivity=connectivity,
            )
        )
        check_and_release(shape)

    def test_create_tri_mesh_shape(self):
        coordinates = small_cube_tri_mesh_coordinates
        connectivity = small_cube_tri_mesh_connectivity
        expected_aabb = mochi.Aabb([-0.1, -0.1, -0.1], [0.1, 0.1, 0.1])

        def check_and_release(shape):
            self.assertTrue(shape.is_valid())
            for i in range(0, 3):
                self.assertAlmostEqual(
                    expected_aabb.min[i], mochi.get_shape_aabb(shape).min[i], places=6
                )
                self.assertAlmostEqual(
                    expected_aabb.max[i], mochi.get_shape_aabb(shape).max[i], places=6
                )
            self.assertEqual(1, mochi.get_num_shapes())
            mochi.release_shape(shape)
            self.assertEqual(0, mochi.get_num_shapes())

        # Create mesh from numpy arrays with 32-bit floats (converts to real as necessary)
        self.assertEqual(0, mochi.get_num_shapes())
        shape = mochi.create_tri_mesh_shape(
            coordinates=coordinates.astype(np.float32), connectivity=connectivity
        )  # ordinal
        check_and_release(shape)

        # Create mesh from numpy arrays with 64-bit floats (converts to real as necessary)
        shape = mochi.create_tri_mesh_shape(
            coordinates=coordinates.astype(np.float64), connectivity=connectivity
        )
        check_and_release(shape)

        # Create from Python lists
        shape = mochi.create_tri_mesh_shape(coordinates.tolist(), connectivity.tolist())
        check_and_release(shape)

        # Now use create_mesh_shape (more generic)
        shape = mochi.create_mesh_shape(
            nodes_per_element=3, coordinates=coordinates, connectivity=connectivity
        )
        check_and_release(shape)

        # Now use create_model_shape (even more generic)
        shape = mochi.create_model_shape(
            mesh=mochi.MeshData(
                nodes_per_element=3, coordinates=coordinates, connectivity=connectivity
            )
        )
        check_and_release(shape)

    def test_create_plane_shape(self):
        # Plane with +Y normal
        self.assertEqual(0, mochi.get_num_shapes())
        shape = mochi.create_plane_shape([0, 1, 0], 0.5)  # ordinal
        self.assertTrue(shape.is_valid())
        self.assertEqual(0.5, mochi.get_shape_aabb(shape).max[1])
        self.assertEqual(1, mochi.get_num_shapes())
        mochi.release_shape(shape)
        self.assertEqual(0, mochi.get_num_shapes())

        # Plane with +Z normal
        shape = mochi.create_plane_shape(normal=[0, 0, 1], distance=3)
        self.assertTrue(shape.is_valid())
        self.assertEqual(3, mochi.get_shape_aabb(shape).max[2])
        self.assertEqual(1, mochi.get_num_shapes())
        mochi.release_shape(shape)
        self.assertEqual(0, mochi.get_num_shapes())

        # Repeat with create_model_shape (more generic)
        shape = mochi.create_model_shape(
            plane=mochi.Plane(normal=[0, 0, 1], distance=3)
        )
        self.assertTrue(shape.is_valid())
        self.assertEqual(3, mochi.get_shape_aabb(shape).max[2])
        self.assertEqual(1, mochi.get_num_shapes())
        mochi.release_shape(shape)
        self.assertEqual(0, mochi.get_num_shapes())

    def test_create_sphere_shape(self):
        self.assertEqual(0, mochi.get_num_shapes())
        shape = mochi.create_sphere_shape(center=[1, 2, 3], radius=0.5)
        self.assertTrue(shape.is_valid())
        self.assertEqual(mochi.Real3(0.5, 1.5, 2.5), mochi.get_shape_aabb(shape).min)
        self.assertEqual(mochi.Real3(1.5, 2.5, 3.5), mochi.get_shape_aabb(shape).max)
        self.assertEqual(1, mochi.get_num_shapes())
        mochi.release_shape(shape)
        self.assertEqual(0, mochi.get_num_shapes())

        # Repeat with create_model_shape (more generic)
        shape = mochi.create_model_shape(
            sphere=mochi.Sphere(center=[1, 2, 3], radius=0.5)
        )
        self.assertTrue(shape.is_valid())
        self.assertEqual(mochi.Real3(0.5, 1.5, 2.5), mochi.get_shape_aabb(shape).min)
        self.assertEqual(mochi.Real3(1.5, 2.5, 3.5), mochi.get_shape_aabb(shape).max)
        self.assertEqual(1, mochi.get_num_shapes())
        mochi.release_shape(shape)
        self.assertEqual(0, mochi.get_num_shapes())

    def test_release_shape(self):
        self.assertEqual(0, mochi.get_num_shapes())
        mochi.release_shape(None)  # Safe no-op
        self.assertEqual(0, mochi.get_num_shapes())  # No change
        shape = mochi.create_sphere_shape(center=mochi.Real3(1, 2, 3), radius=0.5)
        self.assertEqual(1, mochi.get_num_shapes())
        mochi.release_shape(shape)
        self.assertEqual(0, mochi.get_num_shapes())
        mochi.release_shape(shape)  # Safe no-op (already released)
        self.assertEqual(0, mochi.get_num_shapes())

    def test_shape_handle_auto_release(self):
        self.assertEqual(0, mochi.get_num_shapes())
        scene = mochi.create_scene("my_scene")

        # Use a ShapeHandle in isolation
        shape = mochi.create_sphere_shape(center=mochi.Real3(1, 2, 3), radius=0.5)
        self.assertEqual(1, mochi.get_num_shapes())

        # When the shape object goes out of scope, Python will clean it up (and thus
        # release the ShapeHandle) automatically, but we want to see it happen right now.
        del shape
        gc.collect()
        self.assertEqual(0, mochi.get_num_shapes())

        # RigidActorParams
        params = mochi.RigidActorParams()
        params.shape = mochi.create_sphere_shape(
            center=mochi.Real3(1, 2, 3), radius=0.5
        )
        params.is_static = True
        self.assertEqual(1, mochi.get_num_shapes())
        actor = scene.create_rigid_actor(params)
        del params  # Release the params containing the handle
        gc.collect()
        self.assertEqual(0, mochi.get_num_shapes())

        # CreateRigidActor with kwargs
        actor = scene.create_rigid_actor(
            shape=mochi.create_sphere_shape(center=mochi.Real3(1, 2, 3), radius=0.5),
            is_static=True,
        )
        gc.collect()
        self.assertEqual(0, mochi.get_num_shapes())  # Temporary kwargs were cleaned up

        scene.destroy_actor(actor)
        mochi.destroy_scene(scene)

    def test_file_cache(self):
        initial_state = mochi.is_file_cache_enabled()
        mochi.enable_file_cache(enable=True)
        self.assertTrue(mochi.is_file_cache_enabled())
        mochi.enable_file_cache(False)
        self.assertFalse(mochi.is_file_cache_enabled())
        mochi.enable_file_cache(initial_state)

    def test_get_shape_mesh_tet(self):
        coordinates = small_cube_tet_mesh_coordinates
        connectivity = small_cube_tet_mesh_connectivity
        shape = mochi.create_tet_mesh_shape(
            coordinates=coordinates, connectivity=connectivity
        )
        mesh = mochi.get_shape_mesh(shape)
        self.assertEqual(4, mesh.nodes_per_element)
        self.assertEqual(8, mesh.get_num_nodes())
        self.assertEqual(5, mesh.get_num_elements())
        self.assertIsNone(mesh.skinning)

    def test_get_shape_mesh_tri(self):
        coordinates = small_cube_tri_mesh_coordinates
        connectivity = small_cube_tri_mesh_connectivity
        shape = mochi.create_tri_mesh_shape(
            coordinates=coordinates, connectivity=connectivity
        )
        mesh = mochi.get_shape_mesh(shape)
        self.assertEqual(3, mesh.nodes_per_element)
        self.assertEqual(8, mesh.get_num_nodes())
        self.assertEqual(12, mesh.get_num_elements())
        self.assertIsNone(mesh.skinning)

    def test_get_shape_mesh_invalid_handle(self):
        with self.assertRaises(mochi.Error):
            mochi.get_shape_mesh(mochi.ShapeHandle())

    def test_get_shape_mesh_no_mesh(self):
        shape = mochi.create_sphere_shape(center=[0.0, 0.0, 0.0], radius=1.0)
        mesh = mochi.get_shape_mesh(shape)
        self.assertEqual(mochi.MeshDataView(), mesh)

    def test_get_shape_surface_mesh_tet(self):
        coordinates = small_cube_tet_mesh_coordinates
        connectivity = small_cube_tet_mesh_connectivity
        shape = mochi.create_tet_mesh_shape(
            coordinates=coordinates, connectivity=connectivity
        )
        surface = mochi.get_shape_surface_mesh(shape)
        self.assertEqual(3, surface.nodes_per_element)
        self.assertGreater(surface.get_num_nodes(), 0)
        self.assertGreater(surface.get_num_elements(), 0)

    def test_get_shape_surface_mesh_invalid_handle(self):
        with self.assertRaises(mochi.Error):
            mochi.get_shape_surface_mesh(mochi.ShapeHandle())

    def test_get_shape_surface_mesh_no_surface_mesh(self):
        shape = mochi.create_sphere_shape(center=[0.0, 0.0, 0.0], radius=1.0)
        surface = mochi.get_shape_surface_mesh(shape)
        self.assertEqual(mochi.MeshDataView(), surface)

    def test_get_shape_surface_mesh_tri(self):
        coordinates = small_cube_tri_mesh_coordinates
        connectivity = small_cube_tri_mesh_connectivity
        shape = mochi.create_tri_mesh_shape(
            coordinates=coordinates, connectivity=connectivity
        )
        surface = mochi.get_shape_surface_mesh(shape)
        self.assertEqual(3, surface.nodes_per_element)
        self.assertEqual(8, surface.get_num_nodes())
        self.assertEqual(12, surface.get_num_elements())

    def test_get_shape_visual_mesh_invalid_handle(self):
        with self.assertRaises(mochi.Error):
            mochi.get_shape_visual_mesh(mochi.ShapeHandle())

    def test_get_shape_visual_mesh_no_visual_mesh(self):
        shape = mochi.create_tet_mesh_shape(
            coordinates=small_cube_tet_mesh_coordinates,
            connectivity=small_cube_tet_mesh_connectivity,
        )
        visual = mochi.get_shape_visual_mesh(shape)
        self.assertEqual(mochi.MeshDataView(), visual)

    def test_get_shape_visual_mesh_with_visual(self):
        model = mochi.ModelData()
        model.mesh = mochi.MeshData()
        model.mesh.nodes_per_element = 4
        model.mesh.coordinates = small_cube_tet_mesh_coordinates
        model.mesh.connectivity = small_cube_tet_mesh_connectivity
        model.visual_mesh = mochi.MeshData()
        model.visual_mesh.nodes_per_element = 3
        model.visual_mesh.coordinates = small_cube_tri_mesh_coordinates
        model.visual_mesh.connectivity = small_cube_tri_mesh_connectivity
        shape = mochi.create_model_shape(model)
        visual = mochi.get_shape_visual_mesh(shape)
        self.assertEqual(3, visual.nodes_per_element)
        self.assertEqual(8, visual.get_num_nodes())
        self.assertEqual(12, visual.get_num_elements())
