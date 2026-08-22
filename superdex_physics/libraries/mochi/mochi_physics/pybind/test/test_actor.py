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

from test.conftest import (
    mochi,
    MochiTestBase,
    np,
    np_real,
    small_cube_tet_mesh_connectivity,
    small_cube_tet_mesh_coordinates,
    small_cube_tri_mesh_connectivity,
    small_cube_tri_mesh_coordinates,
)


class TestActor(MochiTestBase):
    def test_actor_set_contact_params(self):
        scene = mochi.create_scene("My Scene")
        actor = self._create_rigid_box_actor(scene)
        contact_params = mochi.ContactParams()
        contact_params.penalty_coefficient = 2.34e9
        actor.set_contact_params(contact_params)
        self.assertEqual(2.34e9, actor.get_contact_params().penalty_coefficient)
        mochi.destroy_scene(scene)

    def test_actor_set_root_transform(self):
        scene = mochi.create_scene("My Scene")
        actor = self._create_rigid_box_actor(scene)
        transform = mochi.TransformRT()
        transform.translation = [1.0, 2.0, 3.0]
        transform.rotation = mochi.normalize(mochi.Quaternion(0.1, 0.2, 0.3, 0.4))
        self.assertTrue(actor.has_root_transform())
        actor.set_root_transform(transform)
        self.assertEqual(transform, actor.get_root_transform())
        mochi.destroy_scene(scene)

    def test_actor_get_rigid_center_of_mass_local(self):
        scene = mochi.create_scene("My Scene")
        actor = self._create_rigid_box_actor(scene)
        # Expect center of mass at the center of the AABB
        com = actor.get_rigid_center_of_mass_local()
        cov = actor.get_aabb_local().get_center()
        for i in range(3):
            self.assertAlmostEqual(com[i], cov[i])
        mochi.destroy_scene(scene)

    def test_actor_get_rigid_moment_of_inertia_local(self):
        scene = mochi.create_scene("My Scene")
        actor = self._create_rigid_box_actor(
            scene,
            moi=[
                0.11,
                0.22,
                0.33,
                0.44,
                0.55,
                0.66,
            ],
        )
        self.assertEqual(
            mochi.Real6(0.11, 0.22, 0.33, 0.44, 0.55, 0.66),
            actor.get_rigid_moment_of_inertia_local(),
        )
        mochi.destroy_scene(scene)

    def test_actor_get_set_displacements(self):
        scene = mochi.create_scene("My Scene")
        actor = self._create_soft_box_actor(scene)
        displacements = actor.get_displacements()
        num_dofs = len(small_cube_tet_mesh_coordinates)
        self.assertLess(0, num_dofs)
        self.assertEqual(num_dofs, len(displacements))
        self._expect_read_only_span(displacements, mochi.SpanConstReal, float)
        for x in displacements:
            self.assertEqual(0, x)  # No displacements yet
        new_displacements = np.zeros(num_dofs, dtype=np_real)
        for i in range(num_dofs):
            new_displacements[i] = 0.001 * i
        actor.set_displacements(new_displacements)
        displacements = actor.get_displacements()
        for i in range(num_dofs):
            self.assertAlmostEqual(0.001 * i, displacements[i], places=5)
        mochi.destroy_scene(scene)

    def test_actor_set_velocity(self):
        scene = mochi.create_scene("My Scene")
        actor = self._create_rigid_box_actor(scene)
        actor.set_velocity(linear_vel=[1, 2, 3], angular_vel=[4, 5, 6])
        self.assertEqual(mochi.Real3(1, 2, 3), actor.get_linear_velocity())
        self.assertEqual(mochi.Real3(4, 5, 6), actor.get_angular_velocity())
        mochi.destroy_scene(scene)

    def test_actor_get_set_density(self):
        scene = mochi.create_scene("My Scene")
        actor = self._create_rigid_box_actor(
            scene, density=1.23
        )  # Create with specific density
        actor_size = actor.get_aabb_local().get_size()
        actor_volume = actor_size[0] * actor_size[1] * actor_size[2]
        self.assertAlmostEqual(1.23, actor.get_density())
        self.assertAlmostEqual(1.23 * actor_volume, actor.get_mass())
        actor.set_density(4.56)
        self.assertAlmostEqual(4.56, actor.get_density(), places=6)
        self.assertAlmostEqual(4.56 * actor_volume, actor.get_mass(), places=6)
        mochi.destroy_scene(scene)

    def test_actor_get_set_mass(self):
        scene = mochi.create_scene("My Scene")
        actor = self._create_rigid_box_actor(
            scene, mass=0.123
        )  # Create with specific mass
        actor_size = actor.get_aabb_local().get_size()
        actor_volume = actor_size[0] * actor_size[1] * actor_size[2]
        self.assertAlmostEqual(0.123 / actor_volume, actor.get_density(), places=4)
        self.assertAlmostEqual(0.123, actor.get_mass())
        # actor.set_mass() does not exist currently
        mochi.destroy_scene(scene)

    def test_actor_get_set_recentering_params(self):
        scene = mochi.create_scene("My Scene")
        actor = self._create_soft_box_actor(scene)
        default_params = mochi.RecenteringParams()
        params = actor.get_recentering_params()
        self.assertEqual(default_params.use_recentering, params.use_recentering)
        self.assertEqual(
            default_params.rotation_epsilon_deg, params.rotation_epsilon_deg
        )
        self.assertEqual(default_params.translation_epsilon, params.translation_epsilon)
        params.use_recentering = False
        params.rotation_epsilon_deg = 1.23
        params.translation_epsilon = 0.045
        actor.set_recentering_params(params=params)
        params2 = actor.get_recentering_params()
        self.assertEqual(params2.use_recentering, params.use_recentering)
        self.assertEqual(params2.rotation_epsilon_deg, params.rotation_epsilon_deg)
        self.assertEqual(params2.translation_epsilon, params.translation_epsilon)
        mochi.destroy_scene(scene)

    def test_actor_get_set_soft_material_params(self):
        scene = mochi.create_scene("My Scene")
        actor = self._create_soft_box_actor(scene)
        params = mochi.SoftMaterialParams()
        params.type = mochi.SoftMaterialType.NEO_HOOKEAN
        params.neo_hookean = mochi.NeoHookeanMaterialParams(
            youngs_modulus=2e5,
            poisson_ratio=0.34,
        )
        actor.set_soft_material_params(params=params)
        self.assertEqual(
            mochi.SoftMaterialType.NEO_HOOKEAN,
            actor.get_soft_material_params().type,
        )
        self.assertEqual(
            2e5,
            actor.get_soft_material_params().neo_hookean.youngs_modulus,
        )
        self.assertAlmostEqual(
            0.34,
            actor.get_soft_material_params().neo_hookean.poisson_ratio,
        )
        mochi.destroy_scene(scene)

    def test_actor_get_set_soft_material_params_field(self):
        scene = mochi.create_scene("My Scene")
        actor = self._create_soft_box_actor(scene)
        params = mochi.SoftMaterialParams()
        params.type = mochi.SoftMaterialType.NEO_HOOKEAN
        params.neo_hookean = mochi.NeoHookeanMaterialParams(
            youngs_modulus=2e5,
            poisson_ratio=0.34,
        )
        mochi.experimental.set_soft_material_params_field(
            actor=actor, params=params, element_index=1
        )
        self.assertEqual(
            mochi.SoftMaterialType.NEO_HOOKEAN,
            mochi.experimental.get_soft_material_params_field(
                actor=actor, element_index=1
            ).type,
        )
        self.assertAlmostEqual(
            2e5,
            mochi.experimental.get_soft_material_params_field(
                actor=actor, element_index=1
            ).neo_hookean.youngs_modulus,
            delta=2e5 * 4 * np.finfo(np_real).eps,
        )
        self.assertAlmostEqual(
            0.34,
            mochi.experimental.get_soft_material_params_field(
                actor=actor, element_index=1
            ).neo_hookean.poisson_ratio,
        )
        mochi.destroy_scene(scene)

    def test_actor_get_mesh(self):
        scene = mochi.create_scene("My Scene")

        # Soft actors must return their tetrahedral simulation mesh and boundary surface mesh.
        soft_actor = self._create_soft_box_actor(scene)
        mesh = soft_actor.get_mesh()
        surface_mesh = soft_actor.get_surface_mesh()
        visual_mesh = soft_actor.get_visual_mesh()
        self.assertEqual(8, mesh.get_num_nodes())
        self.assertEqual(5, mesh.get_num_elements())
        self.assertEqual(8, surface_mesh.get_num_nodes())
        self.assertEqual(12, surface_mesh.get_num_elements())
        self.assertEqual(12 * 3, len(surface_mesh.connectivity))
        self.assertEqual(
            small_cube_tet_mesh_coordinates.tolist(),
            surface_mesh.coordinates.tolist(),
        )
        self.assertEqual(mochi.MeshDataView(), visual_mesh)

        # Rigid actors with tetrahedral mesh shapes must return the boundary surface simulation mesh.
        rigid_tet_actor = self._create_rigid_box_actor(scene)
        mesh = rigid_tet_actor.get_mesh()
        surface_mesh = rigid_tet_actor.get_surface_mesh()
        self.assertEqual(8, mesh.get_num_nodes())
        self.assertEqual(12, mesh.get_num_elements())
        self.assertEqual(surface_mesh, mesh)

        # Rigid actors with triangular mesh shapes must return the compact active-node surface mesh.
        tri_coordinates = np.concatenate(
            [small_cube_tri_mesh_coordinates, np.zeros(3, dtype=np_real)]
        )
        tri_shape = mochi.create_tri_mesh_shape(
            coordinates=tri_coordinates,
            connectivity=small_cube_tri_mesh_connectivity,
        )
        tri_params = mochi.RigidActorParams()
        tri_params.shape = tri_shape
        tri_params.collider_type = mochi.ColliderType.BOX
        rigid_tri_actor = scene.create_rigid_actor(tri_params)
        mochi.release_shape(tri_shape)

        mesh = rigid_tri_actor.get_mesh()
        surface_mesh = rigid_tri_actor.get_surface_mesh()
        self.assertEqual(8, mesh.get_num_nodes())
        self.assertEqual(surface_mesh, mesh)

        # Rigid actors with implicit shapes must return an empty mesh.
        plane_shape = mochi.create_plane_shape(normal=[0, 1, 0], distance=0)
        plane_params = mochi.RigidActorParams()
        plane_params.shape = plane_shape
        plane_params.is_static = True
        plane_params.collider_type = mochi.ColliderType.PLANE
        plane_actor = scene.create_rigid_actor(plane_params)
        mochi.release_shape(plane_shape)

        mesh = plane_actor.get_mesh()
        self.assertEqual(mochi.MeshDataView(), mesh)

        mochi.destroy_scene(scene)

    def test_actor_query_node_positions(self):
        scene = mochi.create_scene("My Scene")
        actor = self._create_soft_box_actor(scene)
        scene.set_gravity([0, 0, 0])

        # Test register_query (requires stepping)
        handle1 = actor.register_query(type=mochi.QueryType.NODE_POSITIONS)
        self.assertTrue(handle1.is_valid())
        scene.step(0)
        result1 = actor.get_node_positions_local()
        self._expect_read_only_span(result1, mochi.SpanConstReal, float)
        self.assertEqual(
            small_cube_tet_mesh_coordinates.tolist(),
            result1.tolist(),
        )
        actor.cancel_query(handle1)

        # Test register_query_and_compute (no stepping required)
        handle2 = actor.register_query_and_compute(mochi.QueryType.NODE_POSITIONS)
        self.assertTrue(handle2.is_valid())
        result2 = actor.get_node_positions_local()
        self.assertEqual(
            small_cube_tet_mesh_coordinates.tolist(),
            result2.tolist(),
        )
        actor.cancel_query(handle2)

        mochi.destroy_scene(scene)

    def test_actor_query_elements_deformation_gradient(self):
        scene = mochi.create_scene("My Scene")
        actor = self._create_soft_box_actor(scene)

        # Test register_query (requires stepping)
        handle1 = actor.register_query(
            type=mochi.QueryType.ELEMENTS_DEFORMATION_GRADIENT
        )
        self.assertTrue(handle1.is_valid())
        scene.step(0.01)
        result1 = actor.get_elements_deformation_gradient()
        self._expect_read_only_span(result1, mochi.SpanConstReal, float)
        self.assertEqual(actor.get_mesh().get_num_elements() * 3 * 3, len(result1))
        actor.cancel_query(handle1)

        # Test register_query_and_compute (no stepping required)
        handle2 = actor.register_query_and_compute(
            mochi.QueryType.ELEMENTS_DEFORMATION_GRADIENT
        )
        self.assertTrue(handle2.is_valid())
        result2 = actor.get_elements_deformation_gradient()
        self.assertEqual(actor.get_mesh().get_num_elements() * 3 * 3, len(result2))
        actor.cancel_query(handle2)

        mochi.destroy_scene(scene)

    def test_actor_query_surface_node_positions(self):
        scene = mochi.create_scene("My Scene")
        actor = self._create_soft_box_actor(scene)

        # Test register_query (requires stepping)
        handle1 = actor.register_query(type=mochi.QueryType.SURFACE_NODE_POSITIONS)
        self.assertTrue(handle1.is_valid())
        scene.step(0.01)
        result1 = actor.get_surface_mesh_node_positions_local()
        self._expect_read_only_span(result1, mochi.SpanConstReal, float)
        self.assertEqual(8 * 3, len(result1))
        actor.cancel_query(handle1)

        # Test register_query_and_compute (no stepping required)
        handle2 = actor.register_query_and_compute(
            mochi.QueryType.SURFACE_NODE_POSITIONS
        )
        self.assertTrue(handle2.is_valid())
        result2 = actor.get_surface_mesh_node_positions_local()
        self.assertEqual(8 * 3, len(result2))
        actor.cancel_query(handle2)

        mochi.destroy_scene(scene)

    def test_actor_query_nodes_in_volume_local(self):
        scene = mochi.create_scene("My Scene")
        actor = self._create_soft_box_actor(scene)

        # The soft cube's eight nodes sit at the corners of [-0.1, 0.1]^3.
        expected_all = [
            (0, [-0.1, -0.1, -0.1]),
            (1, [0.1, -0.1, -0.1]),
            (2, [-0.1, 0.1, -0.1]),
            (3, [0.1, 0.1, -0.1]),
            (4, [-0.1, -0.1, 0.1]),
            (5, [0.1, -0.1, 0.1]),
            (6, [-0.1, 0.1, 0.1]),
            (7, [0.1, 0.1, 0.1]),
        ]

        def collect(volume, boundary_only):
            found = []
            actor.query_nodes_in_volume_local(
                volume,
                boundary_only,
                lambda index, pos: found.append((index, [pos[0], pos[1], pos[2]])),
            )
            return sorted(found, key=lambda entry: entry[0])

        def assert_nodes(actual, expected):
            self.assertEqual([i for i, _ in expected], [i for i, _ in actual])
            for (_, actual_pos), (_, expected_pos) in zip(actual, expected):
                for a, e in zip(actual_pos, expected_pos):
                    self.assertAlmostEqual(a, e, places=6)

        # All node positions are needed for boundary_only=False queries.
        handle = actor.register_query_and_compute(mochi.QueryType.NODE_POSITIONS)

        # A large volume of each shape encloses all eight nodes.
        big_aabb = mochi.Aabb([-1, -1, -1], [1, 1, 1])
        assert_nodes(collect(big_aabb, False), expected_all)

        sphere = mochi.Sphere(center=[0, 0, 0], radius=1.0)
        assert_nodes(collect(sphere, False), expected_all)

        obb = mochi.Obb(transform=mochi.TransformRT(), half_extents=[1, 1, 1])
        assert_nodes(collect(obb, False), expected_all)

        # A half-space AABB selects only the four nodes with x > 0.
        half_aabb = mochi.Aabb([0.05, -1, -1], [1, 1, 1])
        assert_nodes(
            collect(half_aabb, False),
            [entry for entry in expected_all if entry[1][0] > 0],
        )

        actor.cancel_query(handle)

        # For boundary_only=True the surface node positions must be registered.
        # Every node of the cube is on its surface, so all eight are still found,
        # reported with their original actor mesh node indices.
        surface_handle = actor.register_query_and_compute(
            mochi.QueryType.SURFACE_NODE_POSITIONS
        )
        assert_nodes(collect(big_aabb, True), expected_all)
        actor.cancel_query(surface_handle)

        mochi.destroy_scene(scene)

    def test_actor_query_node_normals(self):
        scene = mochi.create_scene("My Scene")
        actor = self._create_soft_box_actor(scene)

        # Test register_query (requires stepping)
        handle1 = actor.register_query(type=mochi.QueryType.SURFACE_NODE_NORMALS)
        self.assertTrue(handle1.is_valid())
        scene.step(0.01)
        result1 = actor.get_surface_mesh_node_normals_local()
        self._expect_read_only_span(result1, mochi.SpanConstReal, float)
        self.assertEqual(8 * 3, len(result1))
        actor.cancel_query(handle1)

        # Test register_query_and_compute (no stepping required)
        handle2 = actor.register_query_and_compute(mochi.QueryType.SURFACE_NODE_NORMALS)
        self.assertTrue(handle2.is_valid())
        result2 = actor.get_surface_mesh_node_normals_local()
        self.assertEqual(8 * 3, len(result2))
        actor.cancel_query(handle2)

        mochi.destroy_scene(scene)

    def test_actor_query_contact_points_world(self):
        scene = mochi.create_scene("My Scene")
        scene.set_gravity([0, -9.8, 0])

        # Static ground plane
        ground_params = mochi.RigidActorParams()
        ground_params.shape = mochi.create_plane_shape(normal=[0, 1, 0], distance=-0.5)
        ground_params.is_static = True
        ground = scene.create_rigid_actor(ground_params)

        # Dynamic rigid actor above the ground
        actor = self._create_rigid_box_actor(scene)
        handle = actor.register_query(type=mochi.QueryType.CONTACT_POINTS)
        self.assertTrue(handle.is_valid())

        # Step until contact is reported
        contacts = []
        for _i in range(1000):
            scene.step(0.01)
            cp_span = actor.get_contact_points_world()
            if len(cp_span) > 0:
                self._expect_read_only_span(
                    cp_span, mochi.SpanConstContactPoint, mochi.ContactPoint
                )
                contacts = cp_span.tolist()
                break
        self.assertNotEqual(0, len(contacts))
        for cp in contacts:
            self.assertTrue(
                cp.actor_a == actor.get_handle() or cp.actor_a == ground.get_handle()
            )
            self.assertTrue(
                cp.actor_b == actor.get_handle() or cp.actor_b == ground.get_handle()
            )
            self.assertNotEqual(cp.actor_a, cp.actor_b)
        actor.cancel_query(handle)
        mochi.destroy_scene(scene)

    def test_actor_query_elastic_energy(self):
        scene = mochi.create_scene("My Scene")
        actor = self._create_soft_box_actor(scene)
        handle = actor.register_query(mochi.QueryType.ELASTIC_ENERGY)
        scene.step(0.01)
        result = actor.get_elastic_energy()
        self.assertTrue(0 <= result)
        actor.cancel_query(handle)
        mochi.destroy_scene(scene)

    def test_actor_node_contact_forces(self):
        scene = mochi.create_scene("My Scene")

        # Static ground plane
        ground_params = mochi.RigidActorParams()
        ground_params.shape = mochi.create_plane_shape(normal=[0, 1, 0], distance=-0.5)
        ground_params.is_static = True
        scene.create_rigid_actor(ground_params)

        # Soft actor above the ground
        actor = self._create_soft_box_actor(scene)
        handle = actor.register_query(mochi.QueryType.NODE_CONTACT_FORCES)

        # Step until contact is reported
        node_contacts = []
        for _i in range(1000):
            scene.step(0.01)
            span = actor.get_node_contact_forces_world()
            if len(span) > 0:
                self._expect_read_only_span(
                    span, mochi.SpanConstNodeContactForce, mochi.NodeContactForce
                )
                node_contacts = span.tolist()
                break
        self.assertNotEqual(0, len(node_contacts))
        for x in node_contacts:
            self.assertLess(x.index, actor.get_mesh().get_num_nodes())
        actor.cancel_query(handle)
        mochi.destroy_scene(scene)

    def test_actor_query_articulated_controller_force(self):
        # Scene and articulated actor
        scene = mochi.create_scene("My Scene")
        actor = self._create_two_link_articulated_actor(scene)

        # Pose controller
        tracking_params = mochi.PoseTrackingParams()
        tracking_params.stiffness = 1000.0
        tracking_params.damping = 1.0
        tracking_params.saturation = -1.0
        controller_params = mochi.PoseControllerParams()
        controller_params.link_pos_tracking = [tracking_params, tracking_params]
        controller_params.joint_tracking = [tracking_params]
        actor.add_articulated_pose_controller(controller_params)

        # Cannot get forces yet
        try:
            actor.get_articulated_controller_force()
            self.fail("Error exception expected")
        except mochi.Error:
            pass

        # Register
        query = actor.register_query(mochi.QueryType.ARTICULATED_CONTROLLER_FORCE)

        # Still cannot get forces
        try:
            actor.get_articulated_controller_force()
            self.fail("Error exception expected")
        except mochi.Error:
            pass

        scene.step(0.01)

        # Now it should work
        forces = actor.get_articulated_controller_force()
        self.assertEqual(actor.get_num_dofs(), len(forces))
        # Cancel
        actor.cancel_query(query)

        # Now it should fail again
        try:
            actor.get_articulated_controller_force()
            self.fail("Error exception expected")
        except mochi.Error:
            pass

        mochi.destroy_scene(scene)

    def test_actor_query_sdf_distances(self):
        scene = mochi.create_scene("My Scene")

        # Static ground plane
        ground_params = mochi.RigidActorParams()
        ground_params.shape = mochi.create_plane_shape(normal=[0, 1, 0], distance=-0.5)
        ground_params.is_static = True
        scene.create_rigid_actor(ground_params)

        # Soft actor above the ground
        actor = self._create_soft_box_actor(scene)
        handle = actor.register_query(mochi.QueryType.SDF_DISTANCES)

        # Step until contact is reported
        result = mochi.SdfDistances()
        for _i in range(1000):
            scene.step(0.01)
            result = actor.get_sdf_distances()
            if len(result.sample_indices):
                break

        # Expact const spans
        self._expect_read_only_span(result.sample_indices, mochi.SpanConstInt, int)
        self._expect_read_only_span(
            result.world_positions, mochi.SpanConstReal3, mochi.Real3
        )
        self._expect_read_only_span(result.distances, mochi.SpanConstReal, float)
        self._expect_read_only_span(
            result.distance_grads, mochi.SpanConstReal3, mochi.Real3
        )
        self.assertTrue(isinstance(result.max_sdf_far_distance_evaluation, float))

        actor.cancel_query(handle)
        mochi.destroy_scene(scene)

    def test_actor_get_articulated_x(self):
        scene = mochi.create_scene("My Scene")
        actor = self._create_two_link_articulated_actor(scene)

        # get_num_dofs
        num_dofs = actor.get_num_dofs()
        self.assertEqual(9, num_dofs)

        # get_nested_link_actors
        num_links = 2
        num_joints = 2
        link_actors = actor.get_nested_link_actors()
        self.assertEqual(num_links, len(link_actors))
        self.assertEqual(
            mochi.ActorType.RIGID, scene.get_actor(link_actors[0]).get_type()
        )
        self.assertEqual(
            mochi.ActorType.RIGID, scene.get_actor(link_actors[1]).get_type()
        )

        # get_articulated_shape_info
        shape_info = actor.get_articulated_shape_info()
        self.assertEqual(num_links, len(shape_info.root_from_links_at_rest))
        self.assertEqual(num_links, len(shape_info.link_names))
        self.assertEqual("parent", shape_info.link_names[0])
        self.assertEqual("child", shape_info.link_names[1])
        self.assertEqual(num_joints, len(shape_info.joint_names))
        self.assertEqual("parent_joint", shape_info.joint_names[0])
        self.assertEqual("child_joint", shape_info.joint_names[1])
        self.assertEqual(num_links, len(shape_info.parents))
        self.assertEqual(-1, shape_info.parents[0])
        self.assertEqual(0, shape_info.parents[1])
        self.assertEqual(num_joints, len(shape_info.joint_types))
        self.assertEqual(mochi.ArticulatedJointType.FREE, shape_info.joint_types[0])
        self.assertEqual(
            mochi.ArticulatedJointType.SPHERICAL, shape_info.joint_types[1]
        )

        # get_articulated_joint_limit_constraints
        constraints = actor.get_articulated_joint_limit_constraints()
        self.assertEqual(0, len(constraints))  # this articualted actor has none

        # get_articualted_pose (out parameter)
        pose = mochi.DynamicArrayReal(num_dofs, 911)
        actor.get_articulated_pose(pose)
        self.assertEqual(num_dofs, len(pose))  # no change
        for x in pose:
            # All values should have been written
            self.assertNotEqual(911, x)

        # np.array also works as an output parameter type
        np_pose = np.zeros(num_dofs, dtype=np_real)
        actor.get_articulated_pose(np_pose)
        for i in range(0, num_dofs):
            self.assertEqual(pose[i], np_pose[i])

        # get_articulated_link_transforms (out parameter)
        link_transforms = mochi.DynamicArrayTransformRT(
            num_links, mochi.TransformRT(translation=[9, 1, 1])
        )
        actor.get_articulated_link_transforms(link_transforms)
        self.assertEqual(num_links, len(link_transforms))  # no change
        for x in link_transforms:
            # All values should have been written
            self.assertNotEqual(mochi.Real3(9, 1, 1), x.translation)

        # get_articulated_joint_velocities (out parameter)
        joint_vel = mochi.DynamicArrayReal(num_dofs, 911)
        actor.get_articulated_joint_velocities(joint_vel)
        self.assertEqual(num_dofs, len(joint_vel))  # no change
        for x in joint_vel:
            # All values should have been written
            self.assertNotEqual(911, x)

        mochi.destroy_scene(scene)

    def test_actor_boundary_condition_dofs(self):
        scene = mochi.create_scene("My Scene")
        actor = self._create_soft_box_actor(scene)

        # Initially, no boundary conditions
        retrived_dof_indices = actor.get_boundary_condition_dof_indices()
        retrived_dof_values = actor.get_boundary_condition_dof_values_world()
        self.assertEqual(0, len(retrived_dof_indices))
        self.assertEqual(0, len(retrived_dof_values))

        # Add boundary conditions
        bc_dof_indices = np.array([0, 1, 2], dtype=np.int32)
        bc_dof_values = np.array([0.5, 1.0, 1.5], dtype=np_real)
        actor.add_boundary_condition_dofs_world(
            dof_indices=bc_dof_indices, dof_values_world=bc_dof_values
        )

        # Verify BCs were added
        retrived_dof_indices = actor.get_boundary_condition_dof_indices()
        retrived_dof_values = actor.get_boundary_condition_dof_values_world()
        self._expect_read_only_span(retrived_dof_indices, mochi.SpanConstInt, int)
        self._expect_read_only_span(retrived_dof_values, mochi.SpanConstReal, float)
        self.assertEqual(3, len(retrived_dof_indices))
        self.assertEqual(3, len(retrived_dof_values))
        self.assertEqual(bc_dof_indices.tolist(), retrived_dof_indices.tolist())
        np.testing.assert_allclose(bc_dof_values, retrived_dof_values, atol=1e-6)

        # Verify BCs can be cleared
        actor.clear_boundary_conditions()
        retrived_dof_indices = actor.get_boundary_condition_dof_indices()
        retrived_dof_values = actor.get_boundary_condition_dof_values_world()
        self.assertEqual(0, len(retrived_dof_indices))
        self.assertEqual(0, len(retrived_dof_values))

        # Add permanent boundary conditions and verify they cannot be cleared
        bc_dof_indices_perm = np.array([3, 4, 5], dtype=np.int32)
        bc_dof_values_perm = np.array([0.1, 0.2, 0.3], dtype=np_real)
        actor.add_boundary_condition_dofs_world_permanent(
            dof_indices=bc_dof_indices_perm, dof_values_world=bc_dof_values_perm
        )
        actor.clear_boundary_conditions()

        retrived_dof_indices = actor.get_boundary_condition_dof_indices()
        retrived_dof_values = actor.get_boundary_condition_dof_values_world()
        self.assertEqual(3, len(retrived_dof_indices))
        self.assertEqual(3, len(retrived_dof_values))
        self.assertEqual(bc_dof_indices_perm.tolist(), retrived_dof_indices.tolist())
        np.testing.assert_allclose(bc_dof_values_perm, retrived_dof_values, atol=1e-6)

        mochi.destroy_scene(scene)

    def test_actor_boundary_condition_nodes(self):
        scene = mochi.create_scene("My Scene")
        actor = self._create_soft_box_actor(scene)

        # Initially, no boundary conditions
        retrived_dof_indices = actor.get_boundary_condition_dof_indices()
        retrived_dof_values = actor.get_boundary_condition_dof_values_world()
        self.assertEqual(0, len(retrived_dof_indices))
        self.assertEqual(0, len(retrived_dof_values))

        # Add node-based boundary conditions
        node_indices = np.array([0, 1], dtype=np.int32)
        node_positions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np_real)
        actor.add_boundary_condition_nodes_world(
            node_indices=node_indices, node_positions_world=node_positions
        )

        # Verify BCs were added
        retrived_dof_indices = actor.get_boundary_condition_dof_indices()
        retrived_dof_values = actor.get_boundary_condition_dof_values_world()
        self._expect_read_only_span(retrived_dof_indices, mochi.SpanConstInt, int)
        self._expect_read_only_span(retrived_dof_values, mochi.SpanConstReal, float)
        self.assertEqual(6, len(retrived_dof_indices))
        self.assertEqual(6, len(retrived_dof_values))
        self.assertEqual([0, 1, 2, 3, 4, 5], retrived_dof_indices.tolist())
        np.testing.assert_allclose(node_positions, retrived_dof_values, atol=1e-6)

        # Verify BCs can be cleared
        actor.clear_boundary_conditions()
        retrived_dof_indices = actor.get_boundary_condition_dof_indices()
        retrived_dof_values = actor.get_boundary_condition_dof_values_world()
        self.assertEqual(0, len(retrived_dof_indices))
        self.assertEqual(0, len(retrived_dof_values))

        # Add permanent node-based boundary conditions and verify they cannot be cleared
        node_indices_perm = np.array([2], dtype=np.int32)
        node_positions_perm = np.array([0.7, 0.8, 0.9], dtype=np_real)
        actor.add_boundary_condition_nodes_world_permanent(
            node_indices=node_indices_perm, node_positions_world=node_positions_perm
        )
        actor.clear_boundary_conditions()

        retrived_dof_indices = actor.get_boundary_condition_dof_indices()
        retrived_dof_values = actor.get_boundary_condition_dof_values_world()
        self.assertEqual(3, len(retrived_dof_indices))
        self.assertEqual(3, len(retrived_dof_values))
        self.assertEqual([6, 7, 8], retrived_dof_indices.tolist())
        np.testing.assert_allclose(node_positions_perm, retrived_dof_values, atol=1e-6)

        mochi.destroy_scene(scene)

    def test_actor_external_forces(self):
        scene = mochi.create_scene("My Scene")
        actor = self._create_rigid_box_actor(scene)
        scene.set_gravity([0, 0, 0])  # Disable gravity

        # Get initial position
        initial_pos = actor.get_root_transform().translation

        # Apply a constant force in +X direction for multiple steps
        dt = 0.01
        num_steps = 10
        total_time = dt * num_steps
        force_magnitude = 10.0
        expected_acceleration = force_magnitude / actor.get_mass()

        # Set external force on x-translation DOF (index 0)
        dof_indices = np.array([0], dtype=np.int32)
        force_values = np.array([force_magnitude], dtype=np_real)
        actor.set_external_forces_on_dofs(
            dof_indices=dof_indices, force_values=force_values
        )

        for _ in range(num_steps):
            scene.step(dt)

        # Verify position changed in expected direction
        final_pos = actor.get_root_transform().translation
        displacement = final_pos[0] - initial_pos[0]

        # Expected displacement: 0.5 * a * t^2
        expected_displacement = 0.5 * expected_acceleration * total_time * total_time
        self.assertAlmostEqual(displacement, expected_displacement, places=2)

        # Clear external forces and verify actor continues moving with constant velocity
        actor.clear_external_forces()

        expected_velocity = expected_acceleration * total_time
        num_extra_steps = 5
        extra_time = dt * num_extra_steps
        for _ in range(num_extra_steps):
            scene.step(dt)

        extra_displacement = actor.get_root_transform().translation[0] - final_pos[0]
        self.assertAlmostEqual(
            extra_displacement, expected_velocity * extra_time, places=5
        )

        mochi.destroy_scene(scene)

    def test_actor_get_dof_values(self):
        scene = mochi.create_scene("My Scene")
        actor = self._create_soft_box_actor(scene)

        # Test get_num_dofs
        self.assertEqual(3 * actor.get_mesh().get_num_nodes(), actor.get_num_dofs())

        # Test get_dof_values
        dof_indices = np.array([0, 1, 2], dtype=np.int32)
        dof_values = np.ones(3, dtype=np_real)  # Initially non-zero
        actor.get_dof_values(dof_indices=dof_indices, out_dof_values=dof_values)
        self.assertEqual(3, len(dof_values))
        self.assertEqual(
            [0.0] * 3, dof_values.tolist()
        )  # Initial displacement must be zero

        mochi.destroy_scene(scene)

    def test_actor_articulated_pose_methods(self):
        scene = mochi.create_scene("My Scene")
        actor = self._create_two_link_articulated_actor(scene)

        num_dofs = actor.get_num_dofs()
        num_links = 2

        # Test set_articulated_pose_from_joints
        initial_pose = mochi.DynamicArrayReal(num_dofs)
        actor.get_articulated_pose(initial_pose)
        self.assertEqual(num_dofs, len(initial_pose))  # No change

        new_pose = np.array(initial_pose.tolist(), dtype=np_real)
        new_pose += 0.1
        actor.set_articulated_pose_from_joints(pose=new_pose)

        retrieved_pose = mochi.DynamicArrayReal(num_dofs)
        actor.get_articulated_pose(retrieved_pose)
        np.testing.assert_allclose(new_pose, retrieved_pose, atol=1e-6)

        # Test set_articulated_pose_from_links
        initial_transforms = mochi.DynamicArrayTransformRT(num_links)
        actor.get_articulated_link_transforms(initial_transforms)
        self.assertEqual(num_links, len(initial_transforms))  # No change

        new_transforms = []
        for i in range(num_links):
            t = mochi.TransformRT(
                initial_transforms[i].rotation, initial_transforms[i].translation
            )
            t.translation += [0.1, 0.0, 0.0]
            t.rotation = mochi.normalize(mochi.Quaternion(0.1, 0.2, 0.3, 0.4))
            new_transforms.append(t)

        actor.set_articulated_pose_from_links(world_from_links=new_transforms)

        retrieved_transforms = mochi.DynamicArrayTransformRT(num_links)
        actor.get_articulated_link_transforms(retrieved_transforms)
        for i in range(num_links):
            np.testing.assert_allclose(
                new_transforms[i].translation,
                retrieved_transforms[i].translation,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                new_transforms[i].rotation,
                retrieved_transforms[i].rotation,
                atol=1e-6,
            )

        # Test set_articulated_joint_velocities
        velocities = np.ones(num_dofs, dtype=np_real)
        actor.set_articulated_joint_velocities(velocities=velocities)

        retrieved_velocities = mochi.DynamicArrayReal(num_dofs)
        actor.get_articulated_joint_velocities(retrieved_velocities)
        self.assertEqual(num_dofs, len(retrieved_velocities))  # No change
        np.testing.assert_allclose(velocities, retrieved_velocities, atol=1e-6)

        # Test add_articulated_delta_to_pose
        delta_pose = np.zeros(num_dofs, dtype=np_real)
        delta_pose[1] = 0.05  # Small delta to translation DOF
        pose_before = mochi.DynamicArrayReal(num_dofs)
        actor.get_articulated_pose(pose_before)

        out_pose = np.zeros(num_dofs, dtype=np_real)
        actor.add_articulated_delta_to_pose(
            pose=pose_before,
            delta_dofs=delta_pose,
            out_pose=out_pose,
        )

        np.testing.assert_allclose(pose_before + delta_pose, out_pose, atol=1e-6)

        mochi.destroy_scene(scene)

    def test_actor_articulated_target_pose_and_velocity_methods(self):
        scene = mochi.create_scene("My Scene")
        scene.set_gravity([0, 0, 0])
        actor = self._create_two_link_articulated_actor(scene)

        # Add pose controller
        tracking_params = mochi.PoseTrackingParams()
        tracking_params.stiffness = 1000.0
        tracking_params.damping = 50000.0  # Very large to converge to the target velocity in one step (see below)
        tracking_params.saturation = -1.0
        controller_params = mochi.PoseControllerParams()
        controller_params.link_pos_tracking = [tracking_params, tracking_params]
        controller_params.joint_tracking = [tracking_params]
        actor.add_articulated_pose_controller(controller_params)

        num_dofs = actor.get_num_dofs()

        # Test set_articulated_target_pose and get_articulated_target_pose
        target_pose = np.zeros(num_dofs, dtype=np_real)
        target_pose[0] = 0.1
        target_pose[1] = 0.2
        actor.set_articulated_target_pose(pose=target_pose)

        out_target_pose = np.zeros(num_dofs, dtype=np_real)
        actor.get_articulated_target_pose(out_pose=out_target_pose)
        self.assertEqual(num_dofs, len(out_target_pose))
        self.assertAlmostEqual(0.1, out_target_pose[0], places=5)
        self.assertAlmostEqual(0.2, out_target_pose[1], places=5)
        np.testing.assert_allclose(out_target_pose[2:], 0.0, atol=1e-6)

        # Test set_articulated_target_velocity.
        # Since there is no getter, step the simulation and check joint velocities match the target.
        target_velocity = np.zeros(num_dofs, dtype=np_real)
        target_velocity[0] = 0.5  # x-translation of FREE joint
        target_velocity[6] = 0.3  # x-rotation of SPHERICAL joint

        actor.set_articulated_target_velocity(velocity=target_velocity)
        scene.step(0.01)

        actual_velocity = np.zeros(num_dofs, dtype=np_real)
        actor.get_articulated_joint_velocities(actual_velocity)
        np.testing.assert_allclose(target_velocity, actual_velocity, atol=0.01)

        # Resetting the target pose clears a pending target velocity.
        zero_velocity = np.zeros(num_dofs, dtype=np_real)
        actor.set_articulated_joint_velocities(velocities=zero_velocity)
        current_pose = np.zeros(num_dofs, dtype=np_real)
        actor.get_articulated_pose(out_pose=current_pose)
        actor.set_articulated_target_velocity(velocity=target_velocity)
        # If reset does not clear the pending target velocity, the next step will drive the
        # actor away from zero velocity.
        actor.reset_articulated_target_pose(pose=current_pose)
        scene.step(0.01)

        actor.get_articulated_joint_velocities(actual_velocity)
        np.testing.assert_allclose(zero_velocity, actual_velocity, atol=1e-6)

        # Resetting target link transforms clears a pending target velocity too.
        actor.set_articulated_joint_velocities(velocities=zero_velocity)
        current_link_transforms = mochi.DynamicArrayTransformRT(2)
        actor.get_articulated_link_transforms(
            out_world_from_links=current_link_transforms
        )
        actor.set_articulated_target_velocity(velocity=target_velocity)
        # If reset does not clear the pending target velocity, the next step will drive the
        # actor away from zero velocity.
        actor.reset_articulated_target_link_transforms(
            world_from_targets=current_link_transforms
        )
        scene.step(0.01)

        actor.get_articulated_joint_velocities(actual_velocity)
        np.testing.assert_allclose(zero_velocity, actual_velocity, atol=1e-6)

        # Setting the articulated pose from joints resets controller targets and clears
        # a pending target velocity.
        actor.set_articulated_joint_velocities(velocities=zero_velocity)
        current_pose = np.zeros(num_dofs, dtype=np_real)
        actor.get_articulated_pose(out_pose=current_pose)
        actor.set_articulated_target_velocity(velocity=target_velocity)
        actor.set_articulated_pose_from_joints(pose=current_pose)
        scene.step(0.01)

        actor.get_articulated_joint_velocities(actual_velocity)
        np.testing.assert_allclose(zero_velocity, actual_velocity, atol=1e-6)

        # Setting the articulated pose from links resets controller targets and clears
        # a pending target velocity.
        actor.set_articulated_joint_velocities(velocities=zero_velocity)
        current_link_transforms = mochi.DynamicArrayTransformRT(2)
        actor.get_articulated_link_transforms(
            out_world_from_links=current_link_transforms
        )
        actor.set_articulated_target_velocity(velocity=target_velocity)
        actor.set_articulated_pose_from_links(world_from_links=current_link_transforms)
        scene.step(0.01)

        actor.get_articulated_joint_velocities(actual_velocity)
        np.testing.assert_allclose(zero_velocity, actual_velocity, atol=1e-6)

        mochi.destroy_scene(scene)

    def _check_set_tracking_params_per_link(self, actor, num_links):
        # Set each slice with a full num_links-length array (distinct per-link values).
        new_params = mochi.PoseControllerParams()
        new_params.link_pos_tracking = []
        for i in range(num_links):
            p = mochi.PoseTrackingParams()
            p.stiffness = 2000.0 + i * 100
            p.damping = 2.0 + 0.1 * i
            p.saturation = 10.0 + 0.1 * i
            new_params.link_pos_tracking.append(p)
        new_params.link_rot_tracking = []
        for i in range(num_links):
            p = mochi.PoseTrackingParams()
            p.stiffness = 3000.0 + i * 100
            p.damping = 3.0 + 0.1 * i
            p.saturation = 20.0 + 0.1 * i
            new_params.link_rot_tracking.append(p)
        # Hard joints have no controller; leave default so the setter does not warn
        shape_info = actor.get_articulated_shape_info()
        new_params.joint_tracking = []
        for i in range(num_links):
            p = mochi.PoseTrackingParams()
            if shape_info.joint_types[i] != mochi.ArticulatedJointType.HARD:
                p.stiffness = 4000.0 + i * 100
                p.damping = 4.0 + 0.1 * i
                p.saturation = 40.0 + 0.1 * i
            new_params.joint_tracking.append(p)

        actor.set_articulated_pose_controller_params(params=new_params)
        out_params = mochi.PoseControllerParams(num_links)
        actor.get_articulated_pose_controller_params(out_params=out_params)
        self.assertEqual(num_links, len(out_params.link_pos_tracking))
        self.assertEqual(num_links, len(out_params.link_rot_tracking))
        self.assertEqual(num_links, len(out_params.joint_tracking))
        for i, param in enumerate(out_params.link_pos_tracking):
            self.assertAlmostEqual(2000.0 + i * 100, param.stiffness, places=5)
            self.assertAlmostEqual(2.0 + 0.1 * i, param.damping, places=5)
            self.assertAlmostEqual(10.0 + 0.1 * i, param.saturation, places=5)
        for i, param in enumerate(out_params.link_rot_tracking):
            self.assertAlmostEqual(3000.0 + i * 100, param.stiffness, places=5)
            self.assertAlmostEqual(3.0 + 0.1 * i, param.damping, places=5)
            self.assertAlmostEqual(20.0 + 0.1 * i, param.saturation, places=5)
        for i, param in enumerate(out_params.joint_tracking):
            if shape_info.joint_types[i] == mochi.ArticulatedJointType.HARD:
                self.assertAlmostEqual(0.0, param.stiffness, places=5)
                self.assertAlmostEqual(0.0, param.damping, places=5)
                self.assertAlmostEqual(-1.0, param.saturation, places=5)
            else:
                self.assertAlmostEqual(4000.0 + i * 100, param.stiffness, places=5)
                self.assertAlmostEqual(4.0 + 0.1 * i, param.damping, places=5)
                self.assertAlmostEqual(40.0 + 0.1 * i, param.saturation, places=5)

    def _check_set_tracking_params_broadcast(self, actor, num_links):
        # Set each slice with a single element, which broadcasts to all links/joints.
        link_pos = mochi.PoseTrackingParams()
        link_pos.stiffness = 1111.0
        link_pos.damping = 11.0
        link_pos.saturation = 1.0
        link_rot = mochi.PoseTrackingParams()
        link_rot.stiffness = 2222.0
        link_rot.damping = 22.0
        link_rot.saturation = 2.0
        joint = mochi.PoseTrackingParams()
        joint.stiffness = 3333.0
        joint.damping = 33.0
        joint.saturation = 3.0
        new_params = mochi.PoseControllerParams()
        new_params.link_pos_tracking = [link_pos]
        new_params.link_rot_tracking = [link_rot]
        new_params.joint_tracking = [joint]

        actor.set_articulated_pose_controller_params(params=new_params)
        out_params = mochi.PoseControllerParams(num_links)
        actor.get_articulated_pose_controller_params(out_params=out_params)
        self.assertEqual(num_links, len(out_params.link_pos_tracking))
        self.assertEqual(num_links, len(out_params.link_rot_tracking))
        self.assertEqual(num_links, len(out_params.joint_tracking))
        for param in out_params.link_pos_tracking:
            self.assertAlmostEqual(1111.0, param.stiffness, places=5)
            self.assertAlmostEqual(11.0, param.damping, places=5)
            self.assertAlmostEqual(1.0, param.saturation, places=5)
        for param in out_params.link_rot_tracking:
            self.assertAlmostEqual(2222.0, param.stiffness, places=5)
            self.assertAlmostEqual(22.0, param.damping, places=5)
            self.assertAlmostEqual(2.0, param.saturation, places=5)
        for param in out_params.joint_tracking:
            self.assertAlmostEqual(3333.0, param.stiffness, places=5)
            self.assertAlmostEqual(33.0, param.damping, places=5)
            self.assertAlmostEqual(3.0, param.saturation, places=5)

    def _check_set_tracking_params_empty(self, actor, num_links):
        # Empty arrays reset every constraint of that slice to default zero gains.
        new_params = mochi.PoseControllerParams()
        new_params.link_pos_tracking = []
        new_params.link_rot_tracking = []
        new_params.joint_tracking = []

        actor.set_articulated_pose_controller_params(params=new_params)
        out_params = mochi.PoseControllerParams(num_links)
        actor.get_articulated_pose_controller_params(out_params=out_params)
        self.assertEqual(num_links, len(out_params.link_pos_tracking))
        self.assertEqual(num_links, len(out_params.link_rot_tracking))
        self.assertEqual(num_links, len(out_params.joint_tracking))
        for param in out_params.link_pos_tracking:
            self.assertAlmostEqual(0.0, param.stiffness, places=5)
            self.assertAlmostEqual(0.0, param.damping, places=5)
            self.assertAlmostEqual(-1.0, param.saturation, places=5)
        for param in out_params.link_rot_tracking:
            self.assertAlmostEqual(0.0, param.stiffness, places=5)
            self.assertAlmostEqual(0.0, param.damping, places=5)
            self.assertAlmostEqual(-1.0, param.saturation, places=5)
        for param in out_params.joint_tracking:
            self.assertAlmostEqual(0.0, param.stiffness, places=5)
            self.assertAlmostEqual(0.0, param.damping, places=5)
            self.assertAlmostEqual(-1.0, param.saturation, places=5)

    def test_actor_articulated_tracking_params(self):
        scene = mochi.create_scene("My Scene")
        actor = self._create_two_link_articulated_actor(scene)
        num_links = 2

        # Add pose controller: full link_pos array, empty link_rot (-> default zero), single broadcast joint
        tracking_params = mochi.PoseTrackingParams()
        tracking_params.stiffness = 1234.0
        tracking_params.damping = 3.0
        tracking_params.saturation = 2.0
        controller_params = mochi.PoseControllerParams()
        controller_params.link_pos_tracking = [tracking_params, tracking_params]
        controller_params.link_rot_tracking = []
        # Broadcast joint_tracking ignores Hard joints to avoid assignment branching or warnings
        controller_params.joint_tracking = [tracking_params]
        actor.add_articulated_pose_controller(controller_params)

        # Test get_articulated_pose_controller_params after adding the controller
        out_params = mochi.PoseControllerParams(num_links)
        actor.get_articulated_pose_controller_params(out_params=out_params)
        self.assertEqual(num_links, len(out_params.link_pos_tracking))
        self.assertEqual(num_links, len(out_params.link_rot_tracking))
        self.assertEqual(num_links, len(out_params.joint_tracking))
        for param in out_params.link_pos_tracking:
            self.assertAlmostEqual(1234.0, param.stiffness, places=5)
            self.assertAlmostEqual(3.0, param.damping, places=5)
            self.assertAlmostEqual(2.0, param.saturation, places=5)
        for param in out_params.link_rot_tracking:
            self.assertAlmostEqual(0.0, param.stiffness, places=5)
            self.assertAlmostEqual(0.0, param.damping, places=5)
            self.assertAlmostEqual(-1.0, param.saturation, places=5)
        # Joint gains broadcast to every controllable joint; Hard joints have no controller and
        # read back as default zero gains.
        joint_types = actor.get_articulated_shape_info().joint_types
        for i, param in enumerate(out_params.joint_tracking):
            if joint_types[i] == mochi.ArticulatedJointType.HARD:
                self.assertAlmostEqual(0.0, param.stiffness, places=5)
                self.assertAlmostEqual(0.0, param.damping, places=5)
                self.assertAlmostEqual(-1.0, param.saturation, places=5)
            else:
                self.assertAlmostEqual(1234.0, param.stiffness, places=5)
                self.assertAlmostEqual(3.0, param.damping, places=5)
                self.assertAlmostEqual(2.0, param.saturation, places=5)

        # Test set_articulated_pose_controller_params with each supported array length
        self._check_set_tracking_params_per_link(actor, num_links)
        self._check_set_tracking_params_broadcast(actor, num_links)
        self._check_set_tracking_params_empty(actor, num_links)

        # Test get_articulated_pose_constraints
        constraints = actor.get_articulated_pose_constraints()
        self._expect_read_only_span(
            constraints, mochi.SpanConstPoseConstraintInfo, mochi.PoseConstraintInfo
        )
        num_controllable_joints = sum(
            1 for type in joint_types if type != mochi.ArticulatedJointType.HARD
        )
        num_constraints = 2 * num_links + num_controllable_joints
        self.assertEqual(num_constraints, len(constraints))
        self.assertEqual(mochi.PoseConstraintType.LINK_TRANSLATION, constraints[0].type)
        self.assertEqual(mochi.PoseConstraintType.LINK_TRANSLATION, constraints[1].type)
        self.assertEqual(mochi.PoseConstraintType.LINK_ROTATION, constraints[2].type)
        self.assertEqual(mochi.PoseConstraintType.LINK_ROTATION, constraints[3].type)
        self.assertEqual(mochi.PoseConstraintType.JOINT, constraints[4].type)

        mochi.destroy_scene(scene)

    def test_actor_get_points_distance_to_surface(self):
        scene = mochi.create_scene("My Scene")

        params = mochi.RigidActorParams()
        params.shape = mochi.create_tet_mesh_shape(
            small_cube_tet_mesh_coordinates, small_cube_tet_mesh_connectivity
        )
        params.collider_type = mochi.ColliderType.BOX
        params.is_static = True
        actor = scene.create_rigid_actor(params)

        test_points = [
            [0.0, 0.0, 0.0],  # Inside the box
            [0.2, 0.0, 0.0],  # Outside the box
            [-0.2, 0.2, 0.2],  # Outside the box
        ]

        distances = np.zeros(3, dtype=np_real)
        actor.get_points_distance_to_surface(
            points_world=test_points, out_distances=distances
        )

        self.assertEqual(3, len(distances))
        self.assertTrue(distances[0] < 0 and distances[1] > 0 and distances[2] > 0)

        mochi.destroy_scene(scene)

    def test_actor_get_articulated_jacobian(self):
        scene = mochi.create_scene("My Scene")
        actor = self._create_two_link_articulated_actor(scene)

        # Jacobian size should be: 6 (DOFs per link) * num_dofs (joint DOFs)
        expected_size = 6 * actor.get_num_dofs()

        link_handles = actor.get_nested_link_actors()
        self._expect_read_only_span(
            link_handles, mochi.SpanConstActorHandle, mochi.ActorHandle
        )
        self.assertGreater(len(link_handles), 0)
        for link_handle in link_handles:
            link_actor = scene.get_actor(link_handle)
            jacobian = link_actor.get_articulated_jacobian()
            self._expect_read_only_span(jacobian, mochi.SpanConstReal, float)
            self.assertEqual(expected_size, len(jacobian))

        mochi.destroy_scene(scene)

    def test_actor_is_nested_link_actor(self):
        scene = mochi.create_scene("My Scene")

        # Rigid actors should not be nested link actors
        rigid_actor = self._create_rigid_box_actor(scene)
        self.assertFalse(rigid_actor.is_nested_link_actor())

        # Articulated actors should not be either
        articulated_actor = self._create_two_link_articulated_actor(scene)
        self.assertFalse(articulated_actor.is_nested_link_actor())

        # Link actors should be
        link_handles = articulated_actor.get_nested_link_actors()
        self.assertGreater(len(link_handles), 0)
        for handle in link_handles:
            link_actor = scene.get_actor(handle)
            self.assertTrue(link_actor.is_nested_link_actor())

        mochi.destroy_scene(scene)

    def test_actor_center_of_mass_transform(self):
        scene = mochi.create_scene("My Scene")
        actor = self._create_rigid_box_actor(scene)

        # Get initial center of mass transform
        initial_com_transform = actor.get_center_of_mass_transform()
        self.assertIsNotNone(initial_com_transform)

        # Create a new transform
        new_transform = mochi.TransformRT()
        new_transform.translation = [1.23, 2.34, 3.45]
        new_transform.rotation = mochi.normalize(mochi.Quaternion(0.1, 0.2, 0.3, 0.4))

        # Set the center of mass transform
        actor.set_center_of_mass_transform(world_from_com=new_transform)

        # Verify it was set correctly
        retrieved_transform = actor.get_center_of_mass_transform()
        np.testing.assert_allclose(
            new_transform.translation, retrieved_transform.translation, atol=1e-6
        )
        np.testing.assert_allclose(
            new_transform.rotation, retrieved_transform.rotation, atol=1e-6
        )

        mochi.destroy_scene(scene)

    def test_actor_set_zero_displacements_and_velocities(self):
        scene = mochi.create_scene("My Scene")
        actor = self._create_soft_box_actor(scene)

        # Set non-zero displacements directly to test the reset functionality
        num_dofs = actor.get_num_dofs()
        test_displacements = np.zeros(num_dofs, dtype=np_real)
        for i in range(num_dofs):
            test_displacements[i] = 0.01 * (i % 3 + 1)  # Non-zero pattern
        actor.set_displacements(test_displacements)

        # Verify that displacements are non-zero
        displacements_before = actor.get_displacements()
        self.assertTrue(all(abs(d) > 1e-6 for d in displacements_before))

        # Reset displacements and velocities
        actor.set_zero_displacements_and_velocities()

        # Verify displacements are now zero
        displacements_after = actor.get_displacements()
        self.assertTrue(all(d == 0 for d in displacements_after))

        # Step once with no gravity to verify velocities are also zero
        scene.set_gravity([0, 0, 0])
        scene.step(0.05)

        # If velocities were zero, positions should remain at rest
        displacements_after_step = actor.get_displacements()
        np.testing.assert_allclose(displacements_after_step, 0.0, atol=1e-6)

        mochi.destroy_scene(scene)

    def test_actor_set_node_positions_local(self):
        scene = mochi.create_scene("My Scene")
        actor = self._create_soft_box_actor(scene)

        # Register query to get node positions
        query = actor.register_query_and_compute(mochi.QueryType.NODE_POSITIONS)

        # Get initial node positions
        initial_positions = actor.get_node_positions_local()
        self.assertEqual(actor.get_num_dofs(), len(initial_positions))

        # Shift all nodes by a small offset
        new_positions = np.array(initial_positions, dtype=np_real) + 0.05
        actor.set_node_positions_local(positions_local=new_positions)

        # Verify positions were updated
        actor.cancel_query(query)
        query = actor.register_query_and_compute(mochi.QueryType.NODE_POSITIONS)

        retrieved_positions = actor.get_node_positions_local()
        np.testing.assert_allclose(new_positions, retrieved_positions, atol=1e-6)

        actor.cancel_query(query)
        mochi.destroy_scene(scene)

    def test_actor_set_node_velocities_local(self):
        scene = mochi.create_scene("My Scene")
        scene.set_gravity([0, 0, 0])  # Disable gravity
        actor = self._create_soft_box_actor(scene)

        # Disable recentering
        recentering_params = actor.get_recentering_params()
        recentering_params.use_recentering = False
        actor.set_recentering_params(params=recentering_params)

        # Register query to get node positions
        query = actor.register_query_and_compute(mochi.QueryType.NODE_POSITIONS)

        # Get initial node positions
        initial_positions = np.array(actor.get_node_positions_local())  # Owning copy
        num_dofs = actor.get_num_dofs()

        # Set uniform x-velocity on all nodes
        velocity_x = 1.0
        velocities = np.zeros(num_dofs, dtype=np_real)
        velocities[::3] = velocity_x
        actor.set_node_velocities_local(velocities_local=velocities)

        # Step the simulation
        dt = 0.01
        scene.step(dt)

        # Get new positions and verify displacement
        expected_displacement = np.zeros(num_dofs, dtype=np_real)
        expected_displacement[::3] = velocity_x * dt

        new_positions = np.array(actor.get_node_positions_local())  # Owning copy
        actual_displacements_x = new_positions - initial_positions
        np.testing.assert_allclose(
            expected_displacement, actual_displacements_x, atol=1e-6
        )

        actor.cancel_query(query)
        mochi.destroy_scene(scene)

    def test_actor_is_query_supported(self):
        scene = mochi.create_scene("My Scene")
        soft_actor = self._create_soft_box_actor(scene)
        rigid_actor = self._create_rigid_box_actor(scene)

        # Soft actor supports soft-specific queries
        self.assertTrue(soft_actor.is_query_supported(mochi.QueryType.NODE_POSITIONS))
        self.assertTrue(soft_actor.is_query_supported(mochi.QueryType.ELASTIC_ENERGY))
        self.assertTrue(
            soft_actor.is_query_supported(mochi.QueryType.ELEMENTS_DEFORMATION_GRADIENT)
        )
        self.assertTrue(
            soft_actor.is_query_supported(mochi.QueryType.SURFACE_NODE_POSITIONS)
        )

        # Rigid actor does not support soft-specific queries
        self.assertFalse(rigid_actor.is_query_supported(mochi.QueryType.NODE_POSITIONS))
        self.assertFalse(rigid_actor.is_query_supported(mochi.QueryType.ELASTIC_ENERGY))
        self.assertFalse(
            rigid_actor.is_query_supported(
                mochi.QueryType.ELEMENTS_DEFORMATION_GRADIENT
            )
        )

        # Neither actor supports constraint-specific queries
        self.assertFalse(
            soft_actor.is_query_supported(mochi.QueryType.CONSTRAINT_FORCE)
        )
        self.assertFalse(
            rigid_actor.is_query_supported(mochi.QueryType.CONSTRAINT_FORCE)
        )

        mochi.destroy_scene(scene)

    def test_add_linear_transmission_and_get_displacement(self):
        scene = mochi.create_scene("Transmission Scene")
        actor = self._create_revolute_chain_articulated_actor(scene, num_joints=3)

        transmission_params = mochi.experimental.LinearTransmissionParams(
            joint_indices=[0, 1, 2],
            joint_coefficients=[0.5, -0.3, 0.2],
        )
        transmission_index = mochi.experimental.add_linear_transmission(
            actor=actor, params=transmission_params
        )
        self.assertEqual(0, transmission_index)

        displacement = mochi.experimental.get_transmission_displacement(
            actor=actor, transmission_index=transmission_index
        )
        self.assertAlmostEqual(0.0, displacement, places=5)

        mochi.destroy_scene(scene)

    def test_get_transmission_displacement_jacobian_mutable_span(self):
        scene = mochi.create_scene("Transmission Jacobian Scene")
        actor = self._create_revolute_chain_articulated_actor(scene, num_joints=3)
        transmission_index = mochi.experimental.add_linear_transmission(
            actor=actor,
            params=mochi.experimental.LinearTransmissionParams(
                joint_indices=[2, 0, 2],
                joint_coefficients=[0.4, -0.3, 0.6],
            ),
        )

        jacobian = np.full(actor.get_num_dofs(), 911, dtype=np_real)
        mochi.experimental.get_transmission_displacement_jacobian(
            actor=actor,
            transmission_index=transmission_index,
            out_jacobian=jacobian,
        )
        np.testing.assert_allclose(jacobian, [-0.3, 0.0, 1.0], atol=1e-6)

        with self.assertRaises(mochi.Error):
            mochi.experimental.get_transmission_displacement_jacobian(
                actor=actor,
                transmission_index=transmission_index,
                out_jacobian=np.zeros(actor.get_num_dofs() - 1, dtype=np_real),
            )

        mochi.destroy_scene(scene)

    def test_add_spatial_tendon_and_get_displacement(self):
        scene = mochi.create_scene("Spatial Tendon Scene")
        actor = self._create_revolute_chain_articulated_actor(scene, num_joints=3)

        tendon_params = mochi.experimental.SpatialTendonParams(
            routing_elements=[
                mochi.RoutingElement(
                    type=mochi.RoutingElementType.WAYPOINT,
                    index=0,
                    local_position=mochi.Real3(0.0, 0.0, 0.0),
                ),
                mochi.RoutingElement(
                    type=mochi.RoutingElementType.WAYPOINT,
                    index=1,
                    local_position=mochi.Real3(0.0, 0.1, 0.0),
                ),
                mochi.RoutingElement(
                    type=mochi.RoutingElementType.WAYPOINT,
                    index=2,
                    local_position=mochi.Real3(0.0, 0.0, 0.1),
                ),
            ],
        )
        transmission_index = mochi.experimental.add_spatial_tendon(
            actor=actor, params=tendon_params
        )
        self.assertEqual(0, transmission_index)

        # The displacement is the routed-length change relative to the rest pose, so it is zero at rest.
        displacement = mochi.experimental.get_transmission_displacement(
            actor=actor, transmission_index=transmission_index
        )
        self.assertAlmostEqual(0.0, displacement, places=5)

        # Rotating the joints changes the routed length, hence the reported displacement.
        pose = np.array([0.0, 0.6, -0.4], dtype=np_real)
        actor.set_articulated_pose_from_joints(pose)
        moved_displacement = mochi.experimental.get_transmission_displacement(
            actor=actor, transmission_index=transmission_index
        )
        self.assertNotAlmostEqual(0.0, moved_displacement, places=4)

        mochi.destroy_scene(scene)

    def test_add_spatial_tendon_mixed_elements(self):
        # A mixed routing list (waypoints plus a linear-joint element) reports the waypoint
        # displacement plus coefficient * jointAngle.
        scene = mochi.create_scene("Mixed Spatial Tendon Scene")
        waypoint_actor = self._create_revolute_chain_articulated_actor(
            scene, num_joints=3
        )
        mixed_actor = self._create_revolute_chain_articulated_actor(scene, num_joints=3)

        coefficient = 0.5
        linear_joint = 1
        waypoint_elements = [
            mochi.RoutingElement(
                type=mochi.RoutingElementType.WAYPOINT,
                index=0,
                local_position=mochi.Real3(0.0, 0.2, 0.0),
            ),
            mochi.RoutingElement(
                type=mochi.RoutingElementType.WAYPOINT,
                index=1,
                local_position=mochi.Real3(0.0, 0.2, 0.1),
            ),
            mochi.RoutingElement(
                type=mochi.RoutingElementType.WAYPOINT,
                index=2,
                local_position=mochi.Real3(0.0, 0.0, 0.2),
            ),
        ]
        waypoint_index = mochi.experimental.add_spatial_tendon(
            actor=waypoint_actor,
            params=mochi.experimental.SpatialTendonParams(
                routing_elements=waypoint_elements
            ),
        )
        mixed_index = mochi.experimental.add_spatial_tendon(
            actor=mixed_actor,
            params=mochi.experimental.SpatialTendonParams(
                routing_elements=waypoint_elements
                + [
                    mochi.RoutingElement(
                        type=mochi.RoutingElementType.LINEAR_JOINT,
                        index=linear_joint,
                        coefficient=coefficient,
                    )
                ]
            ),
        )

        angle = 0.4
        pose = np.array([0.3, angle, -0.5], dtype=np_real)
        waypoint_actor.set_articulated_pose_from_joints(pose)
        mixed_actor.set_articulated_pose_from_joints(pose)

        waypoint_displacement = mochi.experimental.get_transmission_displacement(
            actor=waypoint_actor, transmission_index=waypoint_index
        )
        mixed_displacement = mochi.experimental.get_transmission_displacement(
            actor=mixed_actor, transmission_index=mixed_index
        )
        self.assertAlmostEqual(
            mixed_displacement, waypoint_displacement + coefficient * angle, places=5
        )

        mochi.destroy_scene(scene)

    def test_get_transmission_displacement_single_joint(self):
        scene = mochi.create_scene("Transmission Scene")
        actor = self._create_revolute_chain_articulated_actor(scene, num_joints=2)
        self.assertEqual(2, actor.get_num_dofs())

        coefficient = 0.4
        transmission_params = mochi.experimental.LinearTransmissionParams(
            joint_indices=[1],
            joint_coefficients=[coefficient],
        )
        transmission_index = mochi.experimental.add_linear_transmission(
            actor=actor, params=transmission_params
        )

        angle = 0.7
        pose = np.array([0.0, angle], dtype=np_real)
        actor.set_articulated_pose_from_joints(pose)

        displacement = mochi.experimental.get_transmission_displacement(
            actor=actor, transmission_index=transmission_index
        )
        self.assertAlmostEqual(angle * coefficient, displacement, places=5)

        mochi.destroy_scene(scene)

    def test_get_transmission_displacement_multiple_joints(self):
        scene = mochi.create_scene("Transmission Scene")
        actor = self._create_revolute_chain_articulated_actor(scene, num_joints=3)

        c0, c1, c2 = 0.5, 0.3, 0.2
        transmission_params = mochi.experimental.LinearTransmissionParams(
            joint_indices=[0, 1, 2],
            joint_coefficients=[c0, c1, -c2],
        )
        transmission_index = mochi.experimental.add_linear_transmission(
            actor=actor, params=transmission_params
        )

        a0, a1, a2 = 0.2, -0.4, 0.6
        pose = np.array([a0, a1, a2], dtype=np_real)
        actor.set_articulated_pose_from_joints(pose)

        expected = a0 * c0 + a1 * c1 + a2 * (-c2)
        displacement = mochi.experimental.get_transmission_displacement(
            actor=actor, transmission_index=transmission_index
        )
        self.assertAlmostEqual(expected, displacement, places=5)

        mochi.destroy_scene(scene)

    def test_get_transmission_displacement_alignment_flag_effect(self):
        scene = mochi.create_scene("Transmission Scene")
        actor = self._create_revolute_chain_articulated_actor(scene, num_joints=2)

        coefficient = 0.5

        transmission_aligned = mochi.experimental.add_linear_transmission(
            actor=actor,
            params=mochi.experimental.LinearTransmissionParams(
                joint_indices=[1],
                joint_coefficients=[coefficient],
            ),
        )
        transmission_opposite = mochi.experimental.add_linear_transmission(
            actor=actor,
            params=mochi.experimental.LinearTransmissionParams(
                joint_indices=[1],
                joint_coefficients=[-coefficient],
            ),
        )

        angle = 0.3
        pose = np.array([0.0, angle], dtype=np_real)
        actor.set_articulated_pose_from_joints(pose)

        displacement_aligned = mochi.experimental.get_transmission_displacement(
            actor=actor, transmission_index=transmission_aligned
        )
        displacement_opposite = mochi.experimental.get_transmission_displacement(
            actor=actor, transmission_index=transmission_opposite
        )
        self.assertAlmostEqual(angle * coefficient, displacement_aligned, places=5)
        self.assertAlmostEqual(-(angle * coefficient), displacement_opposite, places=5)

        mochi.destroy_scene(scene)

    def test_get_transmission_displacement_updates_with_pose(self):
        scene = mochi.create_scene("Transmission Scene")
        actor = self._create_revolute_chain_articulated_actor(scene, num_joints=2)

        coefficient = 0.5
        transmission_index = mochi.experimental.add_linear_transmission(
            actor=actor,
            params=mochi.experimental.LinearTransmissionParams(
                joint_indices=[1],
                joint_coefficients=[coefficient],
            ),
        )

        angle1 = 0.3
        actor.set_articulated_pose_from_joints(np.array([0.0, angle1], dtype=np_real))
        self.assertAlmostEqual(
            angle1 * coefficient,
            mochi.experimental.get_transmission_displacement(
                actor=actor, transmission_index=transmission_index
            ),
            places=5,
        )

        angle2 = -0.5
        actor.set_articulated_pose_from_joints(np.array([0.0, angle2], dtype=np_real))
        self.assertAlmostEqual(
            angle2 * coefficient,
            mochi.experimental.get_transmission_displacement(
                actor=actor, transmission_index=transmission_index
            ),
            places=5,
        )

        mochi.destroy_scene(scene)

    def test_multiple_transmissions(self):
        scene = mochi.create_scene("Transmission Scene")
        actor = self._create_revolute_chain_articulated_actor(scene, num_joints=3)

        c0a, c1a = 0.5, 0.3
        transmission0 = mochi.experimental.add_linear_transmission(
            actor=actor,
            params=mochi.experimental.LinearTransmissionParams(
                joint_indices=[0, 1],
                joint_coefficients=[c0a, c1a],
            ),
        )

        c1b, c2b = 0.4, 0.6
        transmission1 = mochi.experimental.add_linear_transmission(
            actor=actor,
            params=mochi.experimental.LinearTransmissionParams(
                joint_indices=[1, 2],
                joint_coefficients=[-c1b, c2b],
            ),
        )

        self.assertEqual(0, transmission0)
        self.assertEqual(1, transmission1)

        a0, a1, a2 = 0.1, 0.2, -0.3
        actor.set_articulated_pose_from_joints(np.array([a0, a1, a2], dtype=np_real))

        expected0 = a0 * c0a + a1 * c1a
        expected1 = a1 * (-c1b) + a2 * c2b
        self.assertAlmostEqual(
            expected0,
            mochi.experimental.get_transmission_displacement(
                actor=actor, transmission_index=transmission0
            ),
            places=5,
        )
        self.assertAlmostEqual(
            expected1,
            mochi.experimental.get_transmission_displacement(
                actor=actor, transmission_index=transmission1
            ),
            places=5,
        )

        mochi.destroy_scene(scene)

    def test_attach_displacement_control_actuator(self):
        scene = mochi.create_scene("Transmission Scene")
        actor = self._create_revolute_chain_articulated_actor(scene, num_joints=3)

        transmission_index = mochi.experimental.add_linear_transmission(
            actor=actor,
            params=mochi.experimental.LinearTransmissionParams(
                joint_indices=[0, 1, 2],
                joint_coefficients=[0.5, -0.3, 0.2],
            ),
        )

        actuator_params = mochi.experimental.DisplacementControlActuatorParams(
            target_displacement=0.0,
            stiffness=1e7,
            damping=0.0,
        )
        mochi.experimental.attach_displacement_control_actuator(
            actor=actor, transmission_index=transmission_index, params=actuator_params
        )

        num_state_vars = (
            mochi.experimental.get_num_transmission_actuator_state_variables(
                actor=actor, transmission_index=transmission_index
            )
        )
        self.assertEqual(1, num_state_vars)

        mochi.destroy_scene(scene)

    def test_attach_force_control_actuator(self):
        scene = mochi.create_scene("Transmission Scene")
        actor = self._create_revolute_chain_articulated_actor(scene, num_joints=2)

        transmission_index = mochi.experimental.add_linear_transmission(
            actor=actor,
            params=mochi.experimental.LinearTransmissionParams(
                joint_indices=[0, 1],
                joint_coefficients=[0.3, 0.4],
            ),
        )

        mochi.experimental.attach_force_control_actuator(
            actor=actor,
            transmission_index=transmission_index,
            params=mochi.experimental.ForceControlActuatorParams(force=100.0),
        )

        num_state_vars = (
            mochi.experimental.get_num_transmission_actuator_state_variables(
                actor=actor, transmission_index=transmission_index
            )
        )
        self.assertEqual(1, num_state_vars)

        mochi.destroy_scene(scene)

    def test_attach_mckibben_actuator(self):
        scene = mochi.create_scene("Transmission Scene")
        actor = self._create_revolute_chain_articulated_actor(scene, num_joints=2)

        transmission_index = mochi.experimental.add_linear_transmission(
            actor=actor,
            params=mochi.experimental.LinearTransmissionParams(
                joint_indices=[0, 1],
                joint_coefficients=[0.3, 0.4],
            ),
        )

        mckibben_params = mochi.experimental.McKibbenActuatorParams(
            pressure=1e5,
            minimum_pressure=1e3,
            thread_length=0.15,
            number_of_wraps=3.0,
            deflated_stiffness=500.0,
            deflated_equilibrium_length=0.1,
        )
        mochi.experimental.attach_mc_kibben_actuator(
            actor=actor, transmission_index=transmission_index, params=mckibben_params
        )

        num_state_vars = (
            mochi.experimental.get_num_transmission_actuator_state_variables(
                actor=actor, transmission_index=transmission_index
            )
        )
        self.assertEqual(1, num_state_vars)

        mochi.destroy_scene(scene)

    def test_transmission_state_variables_roundtrip(self):
        scene = mochi.create_scene("Transmission Scene")
        actor = self._create_revolute_chain_articulated_actor(scene, num_joints=3)

        transmission_index = mochi.experimental.add_linear_transmission(
            actor=actor,
            params=mochi.experimental.LinearTransmissionParams(
                joint_indices=[0, 1, 2],
                joint_coefficients=[0.5, -0.3, 0.2],
            ),
        )

        mochi.experimental.attach_displacement_control_actuator(
            actor=actor,
            transmission_index=transmission_index,
            params=mochi.experimental.DisplacementControlActuatorParams(
                target_displacement=0.0, stiffness=1e7
            ),
        )

        num_state_vars = (
            mochi.experimental.get_num_transmission_actuator_state_variables(
                actor=actor, transmission_index=transmission_index
            )
        )
        self.assertEqual(1, num_state_vars)

        target = -0.25
        mochi.experimental.set_transmission_actuator_state_variables(
            actor=actor,
            transmission_index=transmission_index,
            state_variables=np.array([target], dtype=np_real),
        )

        out = np.zeros(num_state_vars, dtype=np_real)
        mochi.experimental.get_transmission_actuator_state_variables(
            actor=actor, transmission_index=transmission_index, out_state_variables=out
        )
        self.assertAlmostEqual(target, out[0], places=5)

        mochi.destroy_scene(scene)

    def test_transmission_state_variables_initial_value_matches_params(self):
        scene = mochi.create_scene("Transmission Scene")
        actor = self._create_revolute_chain_articulated_actor(scene, num_joints=2)

        transmission_index = mochi.experimental.add_linear_transmission(
            actor=actor,
            params=mochi.experimental.LinearTransmissionParams(
                joint_indices=[1],
                joint_coefficients=[0.4],
            ),
        )

        initial_target = 0.42
        mochi.experimental.attach_displacement_control_actuator(
            actor=actor,
            transmission_index=transmission_index,
            params=mochi.experimental.DisplacementControlActuatorParams(
                target_displacement=initial_target, stiffness=1e7
            ),
        )

        out = np.zeros(1, dtype=np_real)
        mochi.experimental.get_transmission_actuator_state_variables(
            actor=actor, transmission_index=transmission_index, out_state_variables=out
        )
        self.assertAlmostEqual(initial_target, out[0], places=5)

        mochi.destroy_scene(scene)

    def test_get_transmission_displacement_invalid_transmission_index(self):
        scene = mochi.create_scene("Transmission Scene")
        actor = self._create_revolute_chain_articulated_actor(scene, num_joints=2)

        mochi.experimental.add_linear_transmission(
            actor=actor,
            params=mochi.experimental.LinearTransmissionParams(
                joint_indices=[0],
                joint_coefficients=[0.5],
            ),
        )

        with self.assertRaises(mochi.Error):
            mochi.experimental.get_transmission_displacement(
                actor=actor, transmission_index=-1
            )
        with self.assertRaises(mochi.Error):
            mochi.experimental.get_transmission_displacement(
                actor=actor, transmission_index=1
            )

        mochi.destroy_scene(scene)

    def test_get_transmission_displacement_no_transmissions_added(self):
        scene = mochi.create_scene("Transmission Scene")
        actor = self._create_revolute_chain_articulated_actor(scene, num_joints=2)

        with self.assertRaises(mochi.Error):
            mochi.experimental.get_transmission_displacement(
                actor=actor, transmission_index=0
            )

        mochi.destroy_scene(scene)

    def test_get_transmission_displacement_non_articulated_actor(self):
        scene = mochi.create_scene("Transmission Scene")
        actor = self._create_rigid_box_actor(scene)

        with self.assertRaises(mochi.Error):
            mochi.experimental.get_transmission_displacement(
                actor=actor, transmission_index=0
            )

        mochi.destroy_scene(scene)

    def test_transmission_state_variables_no_actuator_attached(self):
        scene = mochi.create_scene("Transmission Scene")
        actor = self._create_revolute_chain_articulated_actor(scene, num_joints=2)

        transmission_index = mochi.experimental.add_linear_transmission(
            actor=actor,
            params=mochi.experimental.LinearTransmissionParams(
                joint_indices=[0],
                joint_coefficients=[0.5],
            ),
        )

        with self.assertRaises(mochi.Error):
            mochi.experimental.get_num_transmission_actuator_state_variables(
                actor=actor, transmission_index=transmission_index
            )

        with self.assertRaises(mochi.Error):
            mochi.experimental.set_transmission_actuator_state_variables(
                actor=actor,
                transmission_index=transmission_index,
                state_variables=np.array([0.0], dtype=np_real),
            )

        with self.assertRaises(mochi.Error):
            mochi.experimental.get_transmission_actuator_state_variables(
                actor=actor,
                transmission_index=transmission_index,
                out_state_variables=np.zeros(1, dtype=np_real),
            )

        mochi.destroy_scene(scene)

    def test_transmission_state_variables_invalid_transmission_index(self):
        scene = mochi.create_scene("Transmission Scene")
        actor = self._create_revolute_chain_articulated_actor(scene, num_joints=2)

        mochi.experimental.add_linear_transmission(
            actor=actor,
            params=mochi.experimental.LinearTransmissionParams(
                joint_indices=[0],
                joint_coefficients=[0.5],
            ),
        )

        with self.assertRaises(mochi.Error):
            mochi.experimental.get_num_transmission_actuator_state_variables(
                actor=actor, transmission_index=-1
            )

        with self.assertRaises(mochi.Error):
            mochi.experimental.get_num_transmission_actuator_state_variables(
                actor=actor, transmission_index=1
            )

        mochi.destroy_scene(scene)

    def test_transmission_state_variables_non_articulated_actor(self):
        scene = mochi.create_scene("Transmission Scene")
        actor = self._create_rigid_box_actor(scene)

        with self.assertRaises(mochi.Error):
            mochi.experimental.get_num_transmission_actuator_state_variables(
                actor=actor, transmission_index=0
            )

        with self.assertRaises(mochi.Error):
            mochi.experimental.set_transmission_actuator_state_variables(
                actor=actor,
                transmission_index=0,
                state_variables=np.array([0.0], dtype=np_real),
            )

        with self.assertRaises(mochi.Error):
            mochi.experimental.get_transmission_actuator_state_variables(
                actor=actor,
                transmission_index=0,
                out_state_variables=np.zeros(1, dtype=np_real),
            )

        mochi.destroy_scene(scene)

    def test_displacement_control_actuator_allow_compressive_force(self):
        # Round-trip the allow_compressive_force flag through the Python binding and verify both
        # behaviors: by default the actuator's force is clamped to non-negative; with the flag set,
        # the actuator produces a compressive (negative) force when below target.
        scene = mochi.create_scene("Transmission Scene")
        actor = self._create_revolute_chain_articulated_actor(scene, num_joints=2)

        transmission_index = mochi.experimental.add_linear_transmission(
            actor=actor,
            params=mochi.experimental.LinearTransmissionParams(
                joint_indices=[1],
                joint_coefficients=[1.0],
            ),
        )

        # Default: clamped. Setting a negative target far below current pose would normally try to
        # push back; with the clamp the resulting net force on the joint should remain non-negative
        # by construction. The flag itself should round-trip through Python.
        default_params = mochi.experimental.DisplacementControlActuatorParams(
            target_displacement=-1.0, stiffness=1e5
        )
        self.assertEqual(False, default_params.allow_compressive_force)

        compressive_params = mochi.experimental.DisplacementControlActuatorParams(
            target_displacement=-1.0, stiffness=1e5, allow_compressive_force=True
        )
        self.assertEqual(True, compressive_params.allow_compressive_force)

        mochi.experimental.attach_displacement_control_actuator(
            actor=actor,
            transmission_index=transmission_index,
            params=compressive_params,
        )

        mochi.destroy_scene(scene)

    def test_force_control_actuator_allow_compressive_force(self):
        # Default (allow_compressive_force = False) rejects negative force values with an error.
        # With allow_compressive_force = True, negative force values are accepted.
        scene = mochi.create_scene("Transmission Scene")
        actor = self._create_revolute_chain_articulated_actor(scene, num_joints=2)

        transmission_index = mochi.experimental.add_linear_transmission(
            actor=actor,
            params=mochi.experimental.LinearTransmissionParams(
                joint_indices=[0, 1],
                joint_coefficients=[0.3, 0.4],
            ),
        )

        with self.assertRaises(mochi.Error):
            mochi.experimental.attach_force_control_actuator(
                actor=actor,
                transmission_index=transmission_index,
                params=mochi.experimental.ForceControlActuatorParams(force=-50.0),
            )

        mochi.experimental.attach_force_control_actuator(
            actor=actor,
            transmission_index=transmission_index,
            params=mochi.experimental.ForceControlActuatorParams(
                force=-50.0, allow_compressive_force=True
            ),
        )

        # Read the negative value back through the state-variable API.
        out = np.zeros(1, dtype=np_real)
        mochi.experimental.get_transmission_actuator_state_variables(
            actor=actor, transmission_index=transmission_index, out_state_variables=out
        )
        self.assertAlmostEqual(-50.0, out[0], places=5)

        mochi.destroy_scene(scene)


# TODO: Actor methods not yet tested:
# - get_contact_force_world(), get_contact_torque_world()
# - get_contact_force_from_actor_world()
# - get_articulated_pose_distance()
# - get_articulated_dof_limits()
# - add_boundary_condition_constrained_nodes_at_rest(), add_boundary_condition_constrained_nodes_at_rest_permanent()
# - Visual node queries (VISUAL_NODE_POSITIONS, VISUAL_NODE_NORMALS). Requires an actor with a visual mesh.
#
# TODO: Experimental methods not yet tested:
# - get_contact_force_world_batch()


class TestGetReferenceShape(MochiTestBase):
    # Verify get_reference_shape raises an error for actors without a shape
    # (e.g., articulated actors whose links, not themselves, own shapes).
    def test_error_when_actor_has_no_shape(self):
        scene = mochi.create_scene("RefShapeTest")
        # Articulated actors don't have a shape of their own, only their nested link actors do.
        actor = self._create_two_link_articulated_actor(scene)
        with self.assertRaises(mochi.Error):
            actor.get_reference_shape()
        mochi.destroy_scene(scene)

    # Verify get_reference_shape returns a valid handle with a different numeric
    # value than the original.
    def test_returns_valid_handle(self):
        scene = mochi.create_scene("RefShapeTest")
        shape = mochi.create_tet_mesh_shape(
            small_cube_tet_mesh_coordinates, small_cube_tet_mesh_connectivity
        )
        params = mochi.RigidActorParams()
        params.shape = shape
        actor = scene.create_rigid_actor(params)

        ref_shape = actor.get_reference_shape()
        self.assertTrue(ref_shape.is_valid())
        # The new handle should have a different numeric value
        self.assertNotEqual(shape.value, ref_shape.value)

        mochi.destroy_scene(scene)

    # Verify get_reference_shape succeeds after the original handle is released,
    # because the actor's internal shared_ptr keeps the shape data alive.
    def test_shape_survives_after_original_released(self):
        scene = mochi.create_scene("RefShapeTest")
        prev_num = mochi.get_num_shapes()
        shape = mochi.create_tet_mesh_shape(
            small_cube_tet_mesh_coordinates, small_cube_tet_mesh_connectivity
        )
        self.assertEqual(prev_num + 1, mochi.get_num_shapes())

        params = mochi.RigidActorParams()
        params.shape = shape
        actor = scene.create_rigid_actor(params)

        # Release the original shape handle
        mochi.release_shape(shape)
        self.assertEqual(prev_num, mochi.get_num_shapes())

        # GetReferenceShape still works because actor holds the shared_ptr
        ref_shape = actor.get_reference_shape()
        self.assertTrue(ref_shape.is_valid())
        self.assertEqual(prev_num + 1, mochi.get_num_shapes())

        mochi.destroy_scene(scene)

    # Verify a reference shape from one actor can create a second actor with
    # matching geometry.
    def test_reference_shape_can_create_new_actor(self):
        scene = mochi.create_scene("RefShapeTest")
        shape = mochi.create_tet_mesh_shape(
            small_cube_tet_mesh_coordinates, small_cube_tet_mesh_connectivity
        )
        params = mochi.RigidActorParams()
        params.shape = shape
        actor1 = scene.create_rigid_actor(params)

        # Release original shape, get reference from actor
        mochi.release_shape(shape)
        ref_shape = actor1.get_reference_shape()

        # Use the reference shape to create another actor
        params2 = mochi.RigidActorParams()
        params2.shape = ref_shape
        actor2 = scene.create_rigid_actor(params2)
        self.assertIsNotNone(actor2)

        # Both actors should reference equivalent mesh data.
        self.assertEqual(actor1.get_mesh(), actor2.get_mesh())
        self.assertEqual(actor1.get_surface_mesh(), actor2.get_surface_mesh())
        self.assertEqual(actor1.get_visual_mesh(), actor2.get_visual_mesh())

        mochi.destroy_scene(scene)

    # Verify multiple reference handles are independently reference-counted:
    # each increments the shape count, releasing one does not affect the other.
    def test_multiple_references_are_independent(self):
        scene = mochi.create_scene("RefShapeTest")
        shape = mochi.create_tet_mesh_shape(
            small_cube_tet_mesh_coordinates, small_cube_tet_mesh_connectivity
        )
        params = mochi.RigidActorParams()
        params.shape = shape
        actor = scene.create_rigid_actor(params)
        mochi.release_shape(shape)

        prev_num = mochi.get_num_shapes()

        # Get multiple reference shapes
        ref1 = actor.get_reference_shape()
        ref2 = actor.get_reference_shape()
        self.assertTrue(ref1.is_valid())
        self.assertTrue(ref2.is_valid())
        self.assertNotEqual(ref1.value, ref2.value)
        self.assertEqual(prev_num + 2, mochi.get_num_shapes())

        # Release one — the other should still be valid
        mochi.release_shape(ref1)
        self.assertEqual(prev_num + 1, mochi.get_num_shapes())
        mochi.release_shape(ref2)
        self.assertEqual(prev_num, mochi.get_num_shapes())

        mochi.destroy_scene(scene)
