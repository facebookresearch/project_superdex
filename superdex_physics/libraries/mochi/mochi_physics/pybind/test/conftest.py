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
import unittest

import mochi
import numpy as np

np_real = np.float64 if mochi.uses_double_precision() else np.float32

# Decorator that skips the test at runtime when MOCHI_USE_HDF5 is OFF.
requires_hdf5 = unittest.skipUnless(mochi.uses_hdf5(), "Requires MOCHI_USE_HDF5=ON")

# Tests using assets that are not shipped externally apply this decorator.
is_internal_build = os.environ.get("MOCHI_INTERNAL") == "1"
requires_internal_assets = unittest.skipUnless(
    is_internal_build,
    "Requires internal Mochi assets (MOCHI_INTERNAL=1)",
)

default_num_worker_threads = 0

# Initialize Mochi up front so each test doesn't have to do it.
mochi.initialize(num_worker_threads=default_num_worker_threads)


def find_repo_root() -> str | None:
    """Find the directory containing .buckconfig by walking up from current directory."""
    current_dir = os.path.abspath(os.getcwd())
    while current_dir != os.path.dirname(current_dir):
        if os.path.exists(os.path.join(current_dir, ".buckconfig")):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return None  # Not found


def find_assets_dir() -> str:
    # Case 1: Check for MOCHI_ASSETS_PATH environment variable (set by the test runner)
    env_path = os.environ.get("MOCHI_ASSETS_PATH")
    if env_path:
        return env_path if env_path.endswith("/") else env_path + "/"

    # Case 2: Look in the working directory
    if os.path.exists("./assets"):
        return "./assets/"

    # Case 3: Assume we're running in fbsource. Find the assets relative to the repo root.
    return os.path.join(find_repo_root(), "arvr/libraries/mochi/assets/")


assets_dir = find_assets_dir()

# fmt: off
# Minimal cube tet mesh (side length = 0.2)
small_cube_tet_mesh_coordinates = np.array([
    -0.1, -0.1, -0.1, # 0
    +0.1, -0.1, -0.1, # 1
    -0.1, +0.1, -0.1, # 2
    +0.1, +0.1, -0.1, # 3
    -0.1, -0.1, +0.1, # 4
    +0.1, -0.1, +0.1, # 5
    -0.1, +0.1, +0.1, # 6
    +0.1, +0.1, +0.1, # 7
], dtype=np_real)
small_cube_tet_mesh_connectivity = np.array([
    0, 1, 2, 4, # corner vert 0
    6, 7, 4, 2, # corner vert 6
    5, 4, 7, 1, # corner vert 5
    3, 2, 1, 7, # corner vert 3
    1, 2, 4, 7, # interior one
], dtype=np.int32)

# Minimal cube tri mesh (side length = 0.2)
small_cube_tri_mesh_coordinates = np.array([
    -0.1, -0.1, -0.1, # 0
    +0.1, -0.1, -0.1, # 1
    -0.1, +0.1, -0.1, # 2
    +0.1, +0.1, -0.1, # 3
    -0.1, -0.1, +0.1, # 4
    +0.1, -0.1, +0.1, # 5
    -0.1, +0.1, +0.1, # 6
    +0.1, +0.1, +0.1, # 7
], dtype=np_real)
small_cube_tri_mesh_connectivity = np.array([
    0, 2, 1, # back
    2, 3, 1, # back
    1, 3, 5, # right
    3, 7, 5, # right
    5, 7, 4, # front
    7, 6, 4, # front
    4, 6, 0, # left
    6, 2, 0, # left
    2, 6, 3, # top
    6, 7, 3, # top
    0, 1, 4, # bottom
    4, 1, 5, # bottom
], dtype=np.int32)
# fmt: on


def without_whitespace(s: str) -> str:
    return s.replace(" ", "").replace("\r", "").replace("\n", "")


class MochiTestBase(unittest.TestCase):
    """
    Shared test base class with helper methods used across multiple test files.
    These tests cover the Python bindings for the mochi_physics public API. The goal is to make sure that
    every public method and type is usable and that all arguments are passed to/from c++ correctly. The goal
    is NOT to exhaustively test mochi_physics simulation behavior, since it should be tested elsewhere.
    """

    def _create_two_link_articulated_actor(self, scene):
        # Rigid link shape
        rigid_shape = mochi.create_tet_mesh_shape(
            coordinates=small_cube_tet_mesh_coordinates,
            connectivity=small_cube_tet_mesh_connectivity,
        )

        link_contact = mochi.ContactParams()
        link_contact.penalty_coefficient = 1.23e9

        params = mochi.ArticulatedActorParams()
        params.name = "MyArticulated"
        params.joints = [
            mochi.ArticulatedJointParams(
                name="parent_joint",
                type=mochi.ArticulatedJointType.FREE,
            ),
            mochi.ArticulatedJointParams(
                name="child_joint",
                type=mochi.ArticulatedJointType.SPHERICAL,
            ),
        ]
        params.links = [
            mochi.ArticulatedLinkParams(
                name="parent",
                parent_link=-1,
                shape=rigid_shape,
                layer="LinkLayer",
                density=1000.0,
                contact=link_contact,
            ),
            mochi.ArticulatedLinkParams(
                name="child",
                parent_link=0,
                shape=rigid_shape,
                layer="LinkLayer",
                density=1000.0,
                contact=link_contact,
            ),
        ]

        actor = scene.create_articulated_actor(params)
        mochi.release_shape(rigid_shape)

        return actor

    def _expect_read_only_span(self, span, span_type, inner_type):
        self.assertEqual(span_type, type(span))
        test_value = inner_type()

        # Expect a non-empty span, that we can read but not write
        self.assertNotEqual(0, len(span))
        self.assertIsNotNone(span[0])
        self.assertEqual(type(span[0]), inner_type)
        try:
            # Try to modify the first element. Should throw a TypeError
            span[0] = test_value
            self.fail("TypeError exception expected")
        except TypeError:
            pass
        # should be convertible to a non-const list, or to numpy.array
        as_list = span.tolist()
        as_array = np.array(span)
        self.assertEqual(len(as_list), len(span))
        self.assertEqual(len(as_array), len(span))
        as_list[0] = test_value  # This works
        as_array[0] = test_value  # So does this

    def _create_revolute_chain_articulated_actor(
        self, scene, num_joints, friction=None
    ):
        rigid_shape = mochi.create_tet_mesh_shape(
            coordinates=small_cube_tet_mesh_coordinates,
            connectivity=small_cube_tet_mesh_connectivity,
        )

        joints = []
        links = []
        for i in range(num_joints):
            joint = mochi.ArticulatedJointParams(
                type=mochi.ArticulatedJointType.REVOLUTE,
                axis=[1, 0, 0],
            )
            if friction is not None:
                joint.friction = friction
            if i == 0:
                joint.parent_link_from_joint = mochi.TransformRT(
                    translation=mochi.Real3(1, 1, 1)
                )
            joints.append(joint)

            links.append(
                mochi.ArticulatedLinkParams(
                    parent_link=i - 1,
                    shape=rigid_shape,
                )
            )

        params = mochi.ArticulatedActorParams()
        params.joints = joints
        params.links = links
        return scene.create_articulated_actor(params)

    def _create_rigid_box_actor(
        self, scene, is_static=False, density=None, mass=None, com=None, moi=None
    ):
        params = mochi.RigidActorParams()
        params.shape = mochi.create_tet_mesh_shape(
            small_cube_tet_mesh_coordinates, small_cube_tet_mesh_connectivity
        )
        params.is_static = is_static
        if density:
            params.density = density
        if mass:
            params.mass = mass  # Set optional field. Overrides density.
        if com:
            params.center_of_mass = com
        if moi:
            params.moment_of_inertia = moi
        return scene.create_rigid_actor(params)

    def _create_soft_box_actor(self, scene):
        params = mochi.SoftActorParams()
        params.shape = mochi.create_tet_mesh_shape(
            small_cube_tet_mesh_coordinates, small_cube_tet_mesh_connectivity
        )
        return scene.create_soft_actor(params)

    def _check_constraint(self, scene, constraint, constraint_type, actors):  # noqa: C901
        # ConstraintType
        self.assertEqual(constraint_type, constraint.get_type())

        # ConstraintHandle
        handle = constraint.get_handle()
        self.assertTrue(handle.is_valid())
        self.assertEqual(constraint, scene.get_constraint(handle))

        # Stiffness
        self.assertAlmostEqual(0.1, constraint.get_stiffness())
        constraint.set_stiffness(stiffness=0.15)
        self.assertAlmostEqual(0.15, constraint.get_stiffness())

        # Damping
        self.assertAlmostEqual(0.2, constraint.get_damping())
        constraint.set_damping(damping=0.25)
        self.assertAlmostEqual(0.25, constraint.get_damping())

        # Saturation
        self.assertAlmostEqual(0.3, constraint.get_saturation())
        constraint.set_saturation(saturation=0.35)
        self.assertAlmostEqual(0.35, constraint.get_saturation())

        # Actors and DOFs
        dof_indicies_per_actor = []
        total_dofs = 0
        self.assertEqual(len(actors), constraint.get_num_actors())
        for i in range(len(actors)):
            self.assertEqual(actors[i], constraint.get_actor(actor_index=i))
            dof_indices = constraint.get_dof_indices_for_actor(actor_index=i)
            self._expect_read_only_span(dof_indices, mochi.SpanConstInt, int)
            dof_indicies_per_actor.append(dof_indices)
            total_dofs += len(dof_indices)

        # Queries
        myquery = constraint.register_query(mochi.QueryType.CONSTRAINT_FORCE)
        self.assertTrue(myquery.is_valid())
        scene.step(0.01)
        force = constraint.get_force()
        self._expect_read_only_span(force, mochi.SpanConstReal, float)
        constraint.set_stiffness(0)
        constraint.set_damping(0)
        scene.step(0.01)
        force = constraint.get_force()
        self.assertEqual(
            0, np.linalg.norm(force)
        )  # No force, because stiffness and damping are zero
        constraint.cancel_query(myquery)

        # These methods should be callable, but may only work for some types of constraints
        try:
            constraint.set_target_position(mochi.Real3())
        except mochi.Error:
            pass

        try:
            constraint.set_target_rotation(mochi.Quaternion())
        except mochi.Error:
            pass

        try:
            constraint.update_old_target()
        except mochi.Error:
            pass

        try:
            constraint.set_ref_relative_rotation(
                rotation_a=mochi.Quaternion(), rotation_b=mochi.Quaternion()
            )
        except mochi.Error:
            pass

        try:
            values = constraint.get_limit_min_values()
            self._expect_read_only_span(values, mochi.SpanConstReal, float)
        except mochi.Error:
            pass

        try:
            values = constraint.get_limit_max_values()
            self._expect_read_only_span(values, mochi.SpanConstReal, float)
        except mochi.Error:
            pass
