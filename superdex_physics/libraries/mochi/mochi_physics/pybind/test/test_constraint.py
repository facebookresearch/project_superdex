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

from test.conftest import mochi, MochiTestBase


class TestConstraint(MochiTestBase):
    def test_constraint_articulated_dofs_range(self):
        # TODO
        pass

    def test_constraint_articulated_rotation_dofs_target(self):
        # TODO
        pass

    def test_constraint_articulated_translation_dofs_target(self):
        # TODO
        pass

    def test_constraint_joint_rotation_range(self):
        scene = mochi.create_scene("My Scene")
        actor_a = self._create_rigid_box_actor(scene)
        actor_b = self._create_rigid_box_actor(scene)

        params = mochi.JointRotationRangeConstraintParams(
            damping=0.2,
            stiffness=0.1,
            saturation=0.3,
            ref_frame_rot_vec=[0.1, 0.2, 0.3],
            angle_range_x=[-1.0, 1.0],
            angle_range_y=[-0.5, 0.5],
            angle_range_z=[-0.8, 0.8],
            actor_a=actor_a.get_handle(),
            actor_b=actor_b.get_handle(),
        )
        params.range_around_rest = True
        constraint = scene.create_joint_rotation_range_constraint(params)
        self._check_constraint(
            scene,
            constraint,
            mochi.ConstraintType.JOINT_ROTATION_RANGE,
            [actor_a, actor_b],
        )
        scene.destroy_constraint(constraint)
        mochi.destroy_scene(scene)

    def test_constraint_joint_rotation_tracking(self):
        scene = mochi.create_scene("My Scene")
        actor_a = self._create_rigid_box_actor(scene)
        actor_b = self._create_rigid_box_actor(scene)

        params = mochi.JointRotationTrackingConstraintParams(
            saturation=0.3,
            actor_a=actor_a.get_handle(),
            stiffness=0.1,
            ref_frame_rot_vec=[0.1, 0.2, 0.3],
            actor_b=actor_b.get_handle(),
        )
        params.damping = 0.2
        constraint = scene.create_joint_rotation_tracking_constraint(params)
        self._check_constraint(
            scene,
            constraint,
            mochi.ConstraintType.JOINT_ROTATION_TRACKING,
            [actor_a, actor_b],
        )
        scene.destroy_constraint(constraint)
        mochi.destroy_scene(scene)

    def test_constraint_rigid_pivot_position(self):
        scene = mochi.create_scene("My Scene")
        actor = self._create_rigid_box_actor(scene)

        params = mochi.RigidPivotPositionConstraintParams(
            target_position=[1.0, 2.0, 3.0],
            damping=0.2,
            local_position=[0.1, 0.2, 0.3],
            stiffness=0.1,
            actor=actor.get_handle(),
        )
        params.saturation = 0.3
        constraint = scene.create_rigid_pivot_position_constraint(params)
        self._check_constraint(
            scene,
            constraint,
            mochi.ConstraintType.RIGID_PIVOT_POSITION,
            [actor],
        )
        scene.destroy_constraint(constraint)
        mochi.destroy_scene(scene)

    def test_constraint_rigid_pivot_to_rigid_target(self):
        scene = mochi.create_scene("My Scene")
        actor = self._create_rigid_box_actor(scene)

        params = mochi.RigidPivotToRigidTargetConstraintParams(
            target_transform=mochi.TransformRT(translation=[1, 2, 3]),
            damping=0.2,
            local_position=[0.1, 0.2, 0.3],
            stiffness=0.1,
            actor=actor.get_handle(),
        )
        params.saturation = 0.3
        constraint = scene.create_rigid_pivot_to_rigid_target_constraint(params)
        self._check_constraint(
            scene,
            constraint,
            mochi.ConstraintType.RIGID_PIVOT_TO_RIGID_TARGET,
            [actor],
        )
        scene.destroy_constraint(constraint)
        mochi.destroy_scene(scene)

    def test_constraint_rigid_pivot_rotation(self):
        scene = mochi.create_scene("My Scene")
        actor = self._create_rigid_box_actor(scene)

        params = mochi.RigidPivotRotationConstraintParams(
            target_rotation=[0.1, 0.2, 0.3],
            local_rotation=[0.0, 0.0, 0.0],
            damping=0.2,
            saturation=0.3,
            actor=actor.get_handle(),
        )
        params.stiffness = 0.1
        constraint = scene.create_rigid_pivot_rotation_constraint(params)
        self._check_constraint(
            scene,
            constraint,
            mochi.ConstraintType.RIGID_PIVOT_ROTATION,
            [actor],
        )
        scene.destroy_constraint(constraint)
        mochi.destroy_scene(scene)

    def test_constraint_rigid_prismatic_joint(self):
        scene = mochi.create_scene("My Scene")
        actor_a = self._create_rigid_box_actor(scene)
        actor_b = self._create_rigid_box_actor(scene)

        params = mochi.RigidPrismaticJointConstraintParams(
            free_axis=[1, 0, 0],
            actor_a=actor_a.get_handle(),
            stiffness=0.1,
            damping=0.2,
            saturation=0.3,
        )
        params.actor_b = actor_b.get_handle()
        constraint = scene.create_rigid_prismatic_joint_constraint(params)
        self._check_constraint(
            scene,
            constraint,
            mochi.ConstraintType.RIGID_PRISMATIC_JOINT,
            [actor_a, actor_b],
        )
        scene.destroy_constraint(constraint)

        # Repeat with optional min/max limits
        self.assertIsNone(params.min)
        self.assertIsNone(params.max)
        params.min = -1.0
        params.max = 1.0
        self.assertIsNotNone(params.min)
        self.assertIsNotNone(params.max)
        constraint = scene.create_rigid_prismatic_joint_constraint(params)
        self._check_constraint(
            scene,
            constraint,
            mochi.ConstraintType.RIGID_PRISMATIC_JOINT,
            [actor_a, actor_b],
        )
        scene.destroy_constraint(constraint)
        mochi.destroy_scene(scene)

    def test_constraint_rigid_spherical_joint(self):
        scene = mochi.create_scene("My Scene")
        actor_a = self._create_rigid_box_actor(scene)
        actor_b = self._create_rigid_box_actor(scene)

        params = mochi.RigidSphericalJointConstraintParams(
            actor_a=actor_a.get_handle(),
            stiffness=0.1,
            saturation=0.3,
            actor_b=actor_b.get_handle(),
            local_pos_a=[0, 0, 0],
            damping=0.2,
        )
        params.local_pos_b = [0, 0, 0]
        constraint = scene.create_rigid_spherical_joint_constraint(params)
        self._check_constraint(
            scene,
            constraint,
            mochi.ConstraintType.RIGID_SPHERICAL_JOINT,
            [actor_a, actor_b],
        )
        scene.destroy_constraint(constraint)
        mochi.destroy_scene(scene)

    def test_constraint_deformable_node_position(self):
        scene = mochi.create_scene("My Scene")
        actor = self._create_soft_box_actor(scene)

        params = mochi.DeformableNodePositionConstraintParams(
            saturation=0.3,
            actor=actor.get_handle(),
            stiffness=0.1,
            node_index=0,
            damping=0.2,
        )
        params.position = [1.0, 2.0, 3.0]
        constraint = scene.create_deformable_node_position_constraint(params)
        self._check_constraint(
            scene,
            constraint,
            mochi.ConstraintType.DEFORMABLE_NODE_POSITION,
            [actor],
        )
        scene.destroy_constraint(constraint)
        mochi.destroy_scene(scene)

    def test_constraint_deformable_node_to_rigid(self):
        scene = mochi.create_scene("My Scene")
        actor_rigid = self._create_rigid_box_actor(scene)
        actor_soft = self._create_soft_box_actor(scene)

        params = mochi.DeformableNodeToRigidConstraintParams(
            find_closest=True,
            rigid_local_pos=[0.1, 0.2, 0.3],
            saturation=0.3,
            deformable_actor=actor_soft.get_handle(),
            stiffness=0.1,
            damping=0.2,
            deformable_node_index=0,
            rigid_actor=actor_rigid.get_handle(),
        )
        params.fix_to_deformable_pos = False
        constraint = scene.create_deformable_node_to_rigid_constraint(params)
        self._check_constraint(
            scene,
            constraint,
            mochi.ConstraintType.DEFORMABLE_NODE_TO_RIGID,
            [actor_rigid, actor_soft],
        )
        scene.destroy_constraint(constraint)
        mochi.destroy_scene(scene)

    def test_constraint_deformable_node_to_deformable_node(self):
        scene = mochi.create_scene("My Scene")
        actor_a = self._create_soft_box_actor(scene)
        actor_b = self._create_soft_box_actor(scene)

        params = mochi.DeformableNodeToDeformableNodeConstraintParams(
            node_index_b=0,
            actor_a=actor_a.get_handle(),
            stiffness=0.1,
            find_closest=True,
            damping=0.2,
            saturation=0.3,
            actor_b=actor_b.get_handle(),
        )
        params.node_index_a = 0
        constraint = scene.create_deformable_node_to_deformable_node_constraint(params)
        self._check_constraint(
            scene,
            constraint,
            mochi.ConstraintType.DEFORMABLE_NODE_TO_DEFORMABLE_NODE,
            [actor_a, actor_b],
        )
        scene.destroy_constraint(constraint)
        mochi.destroy_scene(scene)

    def test_constraint_is_query_supported(self):
        scene = mochi.create_scene("My Scene")
        rigid_actor = self._create_rigid_box_actor(scene)
        rigid_actor_2 = self._create_rigid_box_actor(scene)
        params = mochi.JointRotationTrackingConstraintParams(
            actor_a=rigid_actor.get_handle(),
            actor_b=rigid_actor_2.get_handle(),
        )
        constraint = scene.create_joint_rotation_tracking_constraint(params)

        # Constraint supports constraint-specific queries
        self.assertTrue(constraint.is_query_supported(mochi.QueryType.CONSTRAINT_FORCE))

        # Constraint does not support actor-specific queries
        self.assertFalse(constraint.is_query_supported(mochi.QueryType.NODE_POSITIONS))
        self.assertFalse(constraint.is_query_supported(mochi.QueryType.ELASTIC_ENERGY))

        mochi.destroy_scene(scene)


# TODO: Constraint methods not yet tested:
# - create_articulated_dofs_range_constraint()
# - create_articulated_rotation_dofs_target_constraint()
# - create_articulated_translation_dofs_target_constraint()
# - Constraint query/update methods (set_target_position, set_target_rotation, etc.)
# - get_limit_min_values(), get_limit_max_values()
