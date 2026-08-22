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

import os
import tempfile

from test.conftest import (
    assets_dir,
    mochi,
    MochiTestBase,
    requires_hdf5,
    requires_internal_assets,
    small_cube_tet_mesh_connectivity,
    small_cube_tet_mesh_coordinates,
)


class TestScene(MochiTestBase):
    def test_create_destroy_scene(self):
        mochi.destroy_scene(None)  # Safe no-op
        scene = mochi.create_scene("my_scene")  # ordinal
        self.assertTrue(mochi.is_valid_scene(scene))
        self.assertEqual("my_scene", scene.get_name())
        mochi.destroy_scene(scene)
        self.assertFalse(mochi.is_valid_scene(scene))
        scene = mochi.create_scene(name="Other Scene")
        self.assertTrue(mochi.is_valid_scene(scene))
        self.assertEqual("Other Scene", scene.get_name())
        mochi.destroy_scene(scene)
        self.assertFalse(mochi.is_valid_scene(scene))

    def test_scene_gravity(self):
        scene = mochi.create_scene(name="Other Scene")
        scene.set_gravity([0, -9.8, 0])  # ordinal
        self.assertEqual(mochi.Real3(0, -9.8, 0), scene.get_gravity())
        scene.set_gravity(gravity=[0, 0, 0])
        self.assertEqual(mochi.Real3(0, 0, 0), scene.get_gravity())
        mochi.destroy_scene(scene)

    @requires_internal_assets
    def test_scene_create_rigid_actor(self):
        scene = mochi.create_scene("My Scene")
        self.assertEqual(0, scene.get_num_actors())

        # Rigid actor params
        params = mochi.RigidActorParams()
        params.name = "My Name"
        params.layer = "My Layer"
        params.shape = mochi.create_plane_shape([0, 1, 0], 0.5)
        params.world_from_local.translation = [1, 2, 3]
        params.collider_type = mochi.ColliderType.PLANE
        params.contact.penalty_coefficient = 1.23e9
        params.is_static = True

        # Create actor and check it
        actor = scene.create_rigid_actor(params=params)
        handle = actor.get_handle()
        self.assertEqual(actor, scene.get_actor(handle))
        self.assertEqual(mochi.ActorType.RIGID, actor.get_type())
        self.assertEqual("My Name", actor.get_name())
        self.assertEqual("My Layer", actor.get_contact_layer())
        self.assertEqual(2.5, actor.get_aabb_world().max[1])
        self.assertTrue(actor.is_static())
        self.assertTrue(actor.has_root_transform())
        self.assertEqual(params.world_from_local, actor.get_root_transform())
        self.assertEqual(1.23e9, actor.get_contact_params().penalty_coefficient)
        self.assertEqual(1, scene.get_num_actors())

        # Test keyword argument overload with partial parameters.
        actor_kw = scene.create_rigid_actor(
            name="KeywordActor",
            center_of_mass=[0.1, 0.2, 0.3],
            moment_of_inertia=None,
            shape=mochi.load_shape_from_file(
                os.path.join(assets_dir, "cube/cube_minimal.mochi.json")
            ),
            is_static=False,
            world_from_local=mochi.TransformRT(translation=[5, 6, 7]),
        )

        default_params = mochi.RigidActorParams()
        self.assertEqual("KeywordActor", actor_kw.get_name())
        self.assertFalse(actor_kw.is_static())
        self.assertEqual(
            mochi.Real3(5, 6, 7), actor_kw.get_root_transform().translation
        )
        self.assertEqual(default_params.layer, actor_kw.get_contact_layer())
        self.assertEqual(
            default_params.contact.penalty_coefficient,
            actor_kw.get_contact_params().penalty_coefficient,
        )
        self.assertEqual(
            mochi.Real3(0.1, 0.2, 0.3), actor_kw.get_rigid_center_of_mass_local()
        )

        # Step
        scene.step(0.01)
        self.assertEqual(
            mochi.ConvergenceStatus.NONE,  # static actors do not report convergence status
            actor.get_convergence_status(),
        )
        self.assertNotEqual(
            mochi.ConvergenceStatus.NONE,  # dynamic actor must report convergence status
            actor_kw.get_convergence_status(),
        )

        # Cleanup
        scene.destroy_actor(None)  # Safe no-op
        scene.destroy_actor(actor)
        self.assertIsNone(scene.get_actor(handle))
        scene.destroy_actor(actor_kw)
        self.assertEqual(0, scene.get_num_actors())
        mochi.destroy_scene(scene)

    def test_scene_create_soft_actor(self):
        scene = mochi.create_scene("My Scene")
        self.assertEqual(0, scene.get_num_actors())

        # Create a soft cube
        params = mochi.SoftActorParams()
        params.name = "My Name"
        params.layer = "My Layer"
        params.shape = mochi.create_tet_mesh_shape(
            coordinates=small_cube_tet_mesh_coordinates,
            connectivity=small_cube_tet_mesh_connectivity,
        )
        params.world_from_local.translation = [1, 2, 3]
        params.contact.penalty_coefficient = 1.23e9

        actor = scene.create_soft_actor(params=params)
        handle = actor.get_handle()
        self.assertEqual(actor, scene.get_actor(handle))
        self.assertEqual(mochi.ActorType.SOFT, actor.get_type())
        self.assertEqual("My Name", actor.get_name())
        self.assertEqual("My Layer", actor.get_contact_layer())
        self.assertEqual(mochi.Real3(0.9, 1.9, 2.9), actor.get_aabb_world().min)
        self.assertEqual(mochi.Real3(1.1, 2.1, 3.1), actor.get_aabb_world().max)
        self.assertEqual(params.world_from_local, actor.get_root_transform())
        self.assertEqual(1.23e9, actor.get_contact_params().penalty_coefficient)
        self.assertEqual(1, scene.get_num_actors())

        # Test keyword argument overload with partial parameters.
        actor_kw = scene.create_soft_actor(
            name="KeywordSoft",
            shape=mochi.create_tet_mesh_shape(
                coordinates=small_cube_tet_mesh_coordinates,
                connectivity=small_cube_tet_mesh_connectivity,
            ),
            world_from_local=mochi.TransformRT(translation=[3, 4, 5]),
        )

        default_params = mochi.SoftActorParams()
        self.assertEqual("KeywordSoft", actor_kw.get_name())
        self.assertEqual(
            mochi.Real3(3, 4, 5), actor_kw.get_root_transform().translation
        )
        self.assertEqual(default_params.layer, actor_kw.get_contact_layer())

        # Step
        scene.step(0.01)
        self.assertEqual(
            mochi.ConvergenceStatus.CONVERGED, actor.get_convergence_status()
        )
        self.assertEqual(
            mochi.ConvergenceStatus.CONVERGED, actor_kw.get_convergence_status()
        )

        # Cleanup
        scene.destroy_actor(handle)  # Destroy using the ActorHandle this time
        self.assertIsNone(scene.get_actor(handle))
        scene.destroy_actor(handle)  # Safe no-op (already destroyed)
        self.assertIsNone(scene.get_actor(handle))
        scene.destroy_actor(actor_kw)
        self.assertEqual(0, scene.get_num_actors())
        mochi.destroy_scene(scene)

    def test_scene_create_articulated_actor(self):
        # Create scene and actor
        scene = mochi.create_scene("My Scene")
        actor = self._create_two_link_articulated_actor(scene)

        self.assertTrue(actor.get_handle().is_valid())
        self.assertEqual(actor, scene.get_actor(actor.get_handle()))
        self.assertEqual(mochi.ActorType.ARTICULATED, actor.get_type())
        self.assertEqual("MyArticulated", actor.get_name())

        # Check nested actors
        links = actor.get_nested_link_actors()
        self.assertEqual(2, len(links))
        for handle in links:
            link = scene.get_actor(handle)
            self.assertEqual(mochi.ActorType.RIGID, link.get_type())
            self.assertTrue(link.is_nested_link_actor())
            self.assertEqual("LinkLayer", link.get_contact_layer())
            self.assertEqual(1000.0, link.get_density())
            self.assertEqual(1.23e9, link.get_contact_params().penalty_coefficient)
        self.assertEqual(0, len(actor.get_nested_soft_actors()))

        # Step
        scene.step(0.01)
        self.assertEqual(
            mochi.ConvergenceStatus.CONVERGED, actor.get_convergence_status()
        )

        # Cleanup
        scene.destroy_actor(actor)
        mochi.destroy_scene(scene)

    def test_scene_get_set_joint_friction(self):
        # The joint-friction component is always present on articulated actors; seed nonzero
        # friction so the getter has non-default values to return.
        scene = mochi.create_scene("Joint Friction Scene")
        actor = self._create_revolute_chain_articulated_actor(
            scene,
            num_joints=2,
            friction=mochi.ArticulatedJointFrictionParams(viscous=0.5, coulomb=1.0),
        )

        # Getter returns the friction the actor was built with.
        friction = actor.get_articulated_joint_friction_params()
        self.assertEqual(2, len(friction))
        for f in friction:
            self.assertAlmostEqual(0.5, f.viscous)
            self.assertAlmostEqual(1.0, f.coulomb)

        # Setter round-trips new per-joint values.
        new_friction = [
            mochi.ArticulatedJointFrictionParams(viscous=1.5, coulomb=2.5),
            mochi.ArticulatedJointFrictionParams(viscous=3.0, coulomb=4.0),
        ]
        actor.set_articulated_joint_friction_params(new_friction)
        result = actor.get_articulated_joint_friction_params()
        self.assertEqual(2, len(result))
        self.assertEqual(new_friction[0], result[0])
        self.assertEqual(new_friction[1], result[1])

        # A size that does not match the joint count is rejected.
        with self.assertRaises(mochi.Error):
            actor.set_articulated_joint_friction_params(
                [mochi.ArticulatedJointFrictionParams(viscous=1.0)]
            )

        # Negative parameters are rejected.
        with self.assertRaises(mochi.Error):
            actor.set_articulated_joint_friction_params(
                [
                    mochi.ArticulatedJointFrictionParams(viscous=-1.0),
                    mochi.ArticulatedJointFrictionParams(viscous=1.0),
                ]
            )

        mochi.destroy_scene(scene)

    def test_scene_get_set_joint_inertia(self):
        # The joint-inertia component is always present on articulated actors; the getter returns
        # one coefficient per joint (default zero) and the setter round-trips new values.
        scene = mochi.create_scene("Joint Inertia Scene")
        actor = self._create_revolute_chain_articulated_actor(scene, num_joints=2)

        inertia = actor.get_articulated_joint_inertia_params()
        self.assertEqual(2, len(inertia))

        actor.set_articulated_joint_inertia_params([0.5, 1.5])
        result = actor.get_articulated_joint_inertia_params()
        self.assertEqual(2, len(result))
        self.assertAlmostEqual(0.5, result[0])
        self.assertAlmostEqual(1.5, result[1])

        # A size that does not match the joint count is rejected.
        with self.assertRaises(mochi.Error):
            actor.set_articulated_joint_inertia_params([1.0])

        # Negative coefficients are rejected.
        with self.assertRaises(mochi.Error):
            actor.set_articulated_joint_inertia_params([-1.0, 1.0])

        mochi.destroy_scene(scene)

    def test_scene_joint_friction_requires_articulated_actor(self):
        # The joint friction/inertia get/set are only valid on articulated actors.
        scene = mochi.create_scene("Rigid Scene")
        actor = self._create_rigid_box_actor(scene)
        with self.assertRaises(mochi.Error):
            actor.get_articulated_joint_friction_params()
        with self.assertRaises(mochi.Error):
            actor.set_articulated_joint_friction_params(
                [mochi.ArticulatedJointFrictionParams(viscous=1.0)]
            )
        with self.assertRaises(mochi.Error):
            actor.get_articulated_joint_inertia_params()
        with self.assertRaises(mochi.Error):
            actor.set_articulated_joint_inertia_params([1.0])
        mochi.destroy_scene(scene)

    @requires_internal_assets
    @requires_hdf5
    def test_scene_create_soft_skinned_actor(self):
        # Adapted from SoftCharacterMochi sample code:

        # Link shapes:
        link_shape_paths = [
            "soft_character/letters/m/bone0.mochi.h5",
            "soft_character/letters/m/bone1.mochi.h5",
            "soft_character/letters/m/bone2.mochi.h5",
            "soft_character/letters/m/bone3.mochi.h5",
        ]
        link_shapes = []
        for path in link_shape_paths:
            link_shapes.append(
                mochi.load_shape_from_file(file_path=os.path.join(assets_dir, path))
            )

        # Soft shape
        soft_shape = mochi.load_shape_from_file(
            file_path=os.path.join(
                assets_dir, "soft_character/letters/m/soft_skin.mochi.h5"
            )
        )

        # Joints
        joints = [
            mochi.ArticulatedJointParams(
                name="joint0",
                type=mochi.ArticulatedJointType.FREE,
            ),
            mochi.ArticulatedJointParams(
                name="joint1",
                type=mochi.ArticulatedJointType.SPHERICAL,
                parent_link_from_joint=mochi.TransformRT(
                    rotation=[0, 0, -0.42837992310523987, 0.9035987257957458],
                    translation=[0.12999999523162842, 0.15000000596046448, 0.0],
                ),
            ),
            mochi.ArticulatedJointParams(
                name="joint2",
                type=mochi.ArticulatedJointType.SPHERICAL,
                parent_link_from_joint=mochi.TransformRT(
                    rotation=[0, 0, 0.7741671204566956, 0.6329813003540039],
                    translation=[0.17723474553747215, 0.21676676734855108, 0.0],
                ),
            ),
            mochi.ArticulatedJointParams(
                name="joint3",
                type=mochi.ArticulatedJointType.SPHERICAL,
                parent_link_from_joint=mochi.TransformRT(
                    rotation=[0, 0, -0.42837992310523987, 0.9035987257957458],
                    translation=[-0.027507659919346283, -0.20333062840210003, 0.0],
                ),
            ),
        ]

        # Links
        links = [
            mochi.ArticulatedLinkParams(
                name="bone0",
                parent_link=-1,
                shape=link_shapes[0],
                layer="Links",
                collider_type=mochi.ColliderType.NONE,
                density=1000.0,
            ),
            mochi.ArticulatedLinkParams(
                name="bone1",
                parent_link=0,
                shape=link_shapes[1],
                layer="Links",
                collider_type=mochi.ColliderType.NONE,
                density=1000.0,
            ),
            mochi.ArticulatedLinkParams(
                name="bone2",
                parent_link=1,
                shape=link_shapes[2],
                layer="Links",
                collider_type=mochi.ColliderType.NONE,
                density=1000.0,
            ),
            mochi.ArticulatedLinkParams(
                name="bone3",
                parent_link=2,
                shape=link_shapes[3],
                layer="Links",
                collider_type=mochi.ColliderType.NONE,
                density=1000.0,
            ),
        ]

        # Soft params
        soft_params = mochi.SoftActorParams()
        soft_params.layer = "Soft"
        soft_params.name = "M_skin"
        soft_params.shape = soft_shape
        soft_params.material.density = 2000.0
        soft_params.material.neo_hookean.youngs_modulus = 12000.0
        soft_params.has_gravity = False
        soft_params.has_inertia = True
        soft_params.has_stress = False

        # Soft skinned params
        soft_skinned_params = mochi.SoftSkinnedActorParams()
        soft_skinned_params.has_gravity = False
        soft_skinned_params.has_inertia = False
        soft_skinned_params.has_stress = True
        soft_skinned_params.skeleton_params.world_from_root.translation = [
            -3.2,
            0.0,
            0.0,
        ]
        soft_skinned_params.skeleton_params.joints = joints
        soft_skinned_params.skeleton_params.links = links
        soft_skinned_params.soft_params = [soft_params]

        # Create scene and actor
        scene = mochi.create_scene("My Scene")
        actor = scene.create_soft_skinned_actor(soft_skinned_params)

        # Check nested actors
        self.assertEqual(1, len(actor.get_nested_soft_actors()))
        self.assertEqual(4, len(actor.get_nested_link_actors()))
        for handle in actor.get_nested_soft_actors():
            soft = scene.get_actor(handle)
            self.assertEqual(mochi.ActorType.SOFT, soft.get_type())
            self.assertFalse(soft.is_nested_link_actor())
        for handle in actor.get_nested_link_actors():
            link = scene.get_actor(handle)
            self.assertEqual(mochi.ActorType.RIGID, link.get_type())
            self.assertTrue(link.is_nested_link_actor())

        # Test keyword argument overload with partial parameters.
        kw_links = []
        for i in range(len(links)):
            kw_links.append(
                mochi.ArticulatedLinkParams(
                    name=links[i].name,
                    parent_link=links[i].parent_link,
                    parent_joint_from_link=links[i].parent_joint_from_link,
                    shape=link_shapes[i],
                    layer="Links2",
                )
            )
        actor_kw = scene.create_soft_skinned_actor(
            skeleton_params=mochi.ArticulatedActorParams(
                joints=joints,
                links=kw_links,
            ),
            soft_params=[
                mochi.SoftActorParams(
                    has_gravity=False,
                    shape=soft_shape,
                    layer="Soft2",
                )
            ],
        )

        # Shape handles can be released after actor creation.
        mochi.release_shape(soft_shape)
        for shape in link_shapes:
            mochi.release_shape(shape)

        self.assertTrue(actor_kw.get_handle().is_valid())
        for handle in actor_kw.get_nested_link_actors():
            link = scene.get_actor(handle)
            self.assertEqual("Links2", link.get_contact_layer())
            self.assertEqual(
                mochi.ContactParams().penalty_coefficient,
                link.get_contact_params().penalty_coefficient,
            )
        for handle in actor_kw.get_nested_soft_actors():
            soft = scene.get_actor(handle)
            self.assertEqual("Soft2", soft.get_contact_layer())
            self.assertEqual(
                mochi.SoftActorParams().material.density, soft.get_density()
            )

        # Step
        scene.step(0.01)

        # Cleanup
        scene.destroy_actor(actor)
        scene.destroy_actor(actor_kw)
        mochi.destroy_scene(scene)

    def test_scene_get_actor(self):
        # Look up an Actor using a default ActorHandle.
        # Other cases are tested with actor creation/destruction methods.
        scene = mochi.create_scene("My Scene")
        self.assertIsNone(scene.get_actor(mochi.ActorHandle()))
        mochi.destroy_scene(scene)

    def test_scene_for_each_actor(self):
        scene = mochi.create_scene("My Scene")

        # No actors yet
        actors_found = []
        scene.for_each_actor(lambda a: actors_found.append(a))
        self.assertEqual(0, len(actors_found))
        self.assertEqual(0, scene.get_num_actors())

        # Create one actor
        params = mochi.RigidActorParams()
        params.shape = mochi.create_plane_shape([0, 1, 0], 0)
        params.is_static = True
        actor = scene.create_rigid_actor(params)

        # Find one actor
        scene.for_each_actor(lambda a: actors_found.append(a))
        self.assertEqual(actor, actors_found[0])
        self.assertEqual(1, len(actors_found))
        self.assertEqual(1, scene.get_num_actors())

        mochi.destroy_scene(scene)

    def test_scene_solver_params(self):
        scene = mochi.create_scene("My Scene")
        params = mochi.SolverParams()
        params.non_linear_solver.max_iter = 123
        params.linear_solver.max_iter = 456
        scene.set_solver_params(params)
        params2 = scene.get_solver_params()
        self.assertEqual(123, params2.non_linear_solver.max_iter)
        self.assertEqual(456, params2.linear_solver.max_iter)
        mochi.destroy_scene(scene)

    def test_scene_step(self):
        scene = mochi.create_scene("My Scene")
        scene.step(0.123)  # ordinal
        self.assertEqual(0.123, scene.get_last_time_step())
        scene.step(time_step_sec=0.234)
        self.assertEqual(0.234, scene.get_last_time_step())
        mochi.destroy_scene(scene)

    def test_scene_get_solver_stats(self):
        scene = mochi.create_scene("My Scene")
        scene.step(0.1)
        solver_stats = scene.get_solver_stats()
        self.assertGreaterEqual(solver_stats.max_non_linear_iters, 0)
        self.assertGreaterEqual(solver_stats.residual_norm, 0.0)
        self.assertGreaterEqual(solver_stats.max_line_search_iters, 0)
        mochi.destroy_scene(scene)

    def test_scene_get_performance_stats(self):
        scene = mochi.create_scene("My Scene")
        scene.step(0.1)
        perf_stats = scene.get_performance_stats()
        self.assertGreater(perf_stats.total_step_duration_sec, 0.0)
        self.assertGreaterEqual(perf_stats.solve_step_duration_sec, 0.0)
        self.assertGreaterEqual(perf_stats.pre_step_callbacks_duration_sec, 0.0)
        self.assertGreaterEqual(perf_stats.post_step_callbacks_duration_sec, 0.0)
        self.assertGreaterEqual(perf_stats.recording_step_duration_sec, 0.0)
        mochi.destroy_scene(scene)

    def test_scene_single_island(self):
        scene = mochi.create_scene("My Scene")
        self.assertFalse(scene.get_force_single_island())
        scene.set_force_single_island(True)  # ordinal
        self.assertTrue(scene.get_force_single_island())
        scene.set_force_single_island(force_single_island=False)
        self.assertFalse(scene.get_force_single_island())
        mochi.destroy_scene(scene)

    @requires_hdf5
    def test_scene_recording(self):
        scene = mochi.create_scene("My Scene")
        self.assertFalse(scene.is_recording())

        # Make a temp file, to generate a temp file path.
        file, temp_path = tempfile.mkstemp(
            suffix=".recording.h5", prefix="mochi_physics_pybind_test_"
        )
        try:
            # Close and destroy the temp file
            os.close(file)
            self.assertTrue(os.path.exists(temp_path))
            os.unlink(temp_path)
            self.assertFalse(os.path.exists(temp_path))

            # Record a few steps
            params = mochi.RecordingParams()
            scene.start_recording(file_path=temp_path, params=params)
            scene.step(0.1)
            scene.step(0.2)
            scene.step(0.3)
            scene.stop_recording()

            # Make sure a file was created
            self.assertTrue(os.path.exists(temp_path))

            # Replace
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        mochi.destroy_scene(scene)

    def test_scene_contact_layers(self):
        scene = mochi.create_scene("My Scene")
        self.assertEqual(0, scene.get_num_contact_layers())
        self.assertTrue(scene.is_layer_contact_enabled("a", "b"))  # default

        scene.enable_layer_contact_asymmetric("a", "b", False)  # ordinal
        self.assertEqual(2, scene.get_num_contact_layers())
        self.assertFalse(scene.is_layer_contact_enabled("a", "b"))
        self.assertTrue(scene.is_layer_contact_enabled("b", "a"))

        scene.enable_layer_contact_asymmetric(layer_a="b", layer_b="a", enable=False)
        self.assertEqual(2, scene.get_num_contact_layers())
        self.assertFalse(scene.is_layer_contact_enabled("a", "b"))
        self.assertFalse(scene.is_layer_contact_enabled("b", "a"))

        scene.enable_layer_contact_symmetric(layer_a="c", layer_b="d", enable=True)
        self.assertEqual(4, scene.get_num_contact_layers())
        self.assertTrue(scene.is_layer_contact_enabled("c", "d"))
        self.assertTrue(scene.is_layer_contact_enabled("d", "c"))

        # Enumerate layer names
        layer_names = []
        scene.enumerate_contact_layer_names(lambda name: layer_names.append(name))
        self.assertEqual("abcd", "".join(sorted(layer_names)))

        mochi.destroy_scene(scene)

    def test_scene_step_callback(self):
        scene = mochi.create_scene("My Scene")
        history = []

        prestep = scene.register_pre_step_callback(
            "My PreStep",
            lambda info: (
                self.assertEqual(scene, info.scene),
                history.append(f"pre({info.time_step_sec})"),
            ),
            42,
        )  # ordinal

        poststep = scene.register_post_step_callback(
            "My PostStep",
            lambda info: (
                self.assertEqual(scene, info.scene),
                history.append(f"post({info.time_step_sec})"),
            ),
            42,
        )  # ordinal

        # Step the scene and trigger callbacks
        scene.step(0.1)
        scene.step(0.2)
        self.assertEqual("pre(0.1), post(0.1), pre(0.2), post(0.2)", ", ".join(history))
        history = []

        # Unregister
        scene.cancel_callback(prestep)
        scene.cancel_callback(poststep)

        # Step without callbacks
        scene.step(0.3)
        self.assertEqual("", ", ".join(history))
        history = []

        # Register again with named params
        prestep = scene.register_pre_step_callback(
            debug_name="My PreStep",
            callback=lambda info: (
                self.assertEqual(scene, info.scene),
                history.append(f"pre({info.time_step_sec})"),
            ),
            priority=42,
        )
        poststep = scene.register_post_step_callback(
            debug_name="My PostStep",
            callback=lambda info: (
                self.assertEqual(scene, info.scene),
                history.append(f"post({info.time_step_sec})"),
            ),
            priority=42,
        )

        scene.step(0.4)
        scene.step(0.5)
        self.assertEqual("pre(0.4), post(0.4), pre(0.5), post(0.5)", ", ".join(history))
        history = []

        mochi.destroy_scene(scene)

    def test_scene_capture_restore_state(self):
        scene = mochi.create_scene("My Scene")
        actor_a = self._create_rigid_box_actor(scene)
        actor_b = self._create_soft_box_actor(scene)

        state0 = scene.capture_state()
        self.assertTrue(state0.is_valid())
        transformA0 = actor_a.get_root_transform()
        transformB0 = actor_b.get_root_transform()

        for _ in range(10):
            scene.step(0.1)

        state1 = scene.capture_state()
        self.assertNotEqual(state0, state1)  # Different handles
        transformA1 = actor_a.get_root_transform()
        transformB1 = actor_b.get_root_transform()
        self.assertNotEqual(transformA0, transformA1)  # Actor moved
        self.assertNotEqual(transformB0, transformB1)  # Actor moved

        scene.restore_state(handle=state0, release_immediately=False)
        self.assertEqual(transformA0, actor_a.get_root_transform())
        self.assertEqual(transformB0, actor_b.get_root_transform())
        state0_again = scene.capture_state()
        self.assertTrue(state0_again.is_valid())
        self.assertNotEqual(state0, state0_again)  # Different handles
        self.assertTrue(scene.is_equal_state(state0, state0_again))  # Same state

        scene.restore_state(state1, False)  # Ordinal args this time
        self.assertEqual(transformA1, actor_a.get_root_transform())
        self.assertEqual(transformB1, actor_b.get_root_transform())
        state1_again = scene.capture_state()
        self.assertTrue(state1_again.is_valid())
        self.assertNotEqual(state1, state1_again)  # Different handles
        self.assertTrue(scene.is_equal_state(state1, state1_again))  # Same state

        scene.release_state(state0)
        self.assertFalse(
            scene.is_equal_state(state0, state0_again)
        )  # state0 is invalid now
        try:
            scene.restore_state(handle=state0, release_immediately=False)
            self.fail("Error exception expected")
        except mochi.Error:
            pass

        scene.step(0.01)
        self.assertNotEqual(transformA1, actor_a.get_root_transform())
        self.assertNotEqual(transformB1, actor_b.get_root_transform())

        # Restore state1 and release immediately
        scene.restore_state(handle=state1, release_immediately=True)
        self.assertEqual(transformA1, actor_a.get_root_transform())
        self.assertEqual(transformB1, actor_b.get_root_transform())

        # Fail to restore state1 after release
        scene.step(0.01)
        try:
            scene.restore_state(handle=state1, release_immediately=False)
            self.fail("Error exception expected")
        except mochi.Error:
            pass
        self.assertNotEqual(transformA1, actor_a.get_root_transform())  # not restored
        self.assertNotEqual(transformB1, actor_b.get_root_transform())  # not restored

        # state1_again is still valid
        scene.restore_state(handle=state1_again, release_immediately=False)
        self.assertEqual(transformA1, actor_a.get_root_transform())  # restored
        self.assertEqual(transformB1, actor_b.get_root_transform())  # restored

        scene.release_all_states()
        try:
            scene.restore_state(handle=state1_again, release_immediately=False)
            self.fail("Error exception expected")
        except mochi.Error:
            pass

        mochi.destroy_scene(scene)

    def test_scene_get_total_simulation_time(self):
        scene = mochi.create_scene("My Scene")

        # Initially zero
        self.assertEqual(0.0, scene.get_total_simulation_time())

        # Step and verify time accumulation
        scene.step(0.1)
        self.assertAlmostEqual(0.1, scene.get_total_simulation_time(), places=6)

        scene.step(0.2)
        self.assertAlmostEqual(0.3, scene.get_total_simulation_time(), places=6)

        scene.step(0.05)
        self.assertAlmostEqual(0.35, scene.get_total_simulation_time(), places=5)

        mochi.destroy_scene(scene)

    def test_convergence_status(self):
        scene = mochi.create_scene("Convergence")

        # Empty scene
        self.assertEqual(
            mochi.ConvergenceStatus.NONE, scene.get_solver_stats().convergence_status
        )

        # Static-only
        plane_shape = mochi.create_plane_shape([0, 1, 0], 0)
        s1 = scene.create_rigid_actor(shape=plane_shape, is_static=True)
        s2 = scene.create_rigid_actor(shape=plane_shape, is_static=True)
        scene.step(0.01)
        self.assertEqual(
            mochi.ConvergenceStatus.NONE, scene.get_solver_stats().convergence_status
        )

        # Static + dynamic (has_gravity=False + generous tolerance for reliable convergence)
        params = scene.get_solver_params()
        params.non_linear_solver.abs_tol = 1.0
        params.non_linear_solver.max_iter = 100
        scene.set_solver_params(params)
        scene.create_soft_actor(
            shape=mochi.create_tet_mesh_shape(
                coordinates=small_cube_tet_mesh_coordinates,
                connectivity=small_cube_tet_mesh_connectivity,
            ),
            has_gravity=False,
            world_from_local=mochi.TransformRT(translation=[0, 1, 0]),
        )
        scene.step(0.01)
        self.assertIsInstance(
            scene.get_solver_stats().convergence_status, mochi.ConvergenceStatus
        )
        self.assertEqual(
            mochi.ConvergenceStatus.CONVERGED,
            scene.get_solver_stats().convergence_status,
        )

        # Dynamic-only (destroy statics)
        scene.destroy_actor(s1)
        scene.destroy_actor(s2)
        scene.step(0.01)
        self.assertEqual(
            mochi.ConvergenceStatus.CONVERGED,
            scene.get_solver_stats().convergence_status,
        )

        mochi.destroy_scene(scene)


# TODO: Scene methods not yet tested:
# - capture_state_to_file()
# - get_debug_draw() and debug draw functionality
# - for_each_constraint()
# - enable_actor_contact_asymmetric(), enable_actor_contact_symmetric(), is_actor_contact_enabled()
