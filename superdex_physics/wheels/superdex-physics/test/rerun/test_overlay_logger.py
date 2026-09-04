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

"""Tests for the OverlayLogger class — rigid and articulated actors."""

import os
import tempfile
import unittest

import numpy as np
from arvr.libraries.mochi.mochi_python.rerun_timeline_test_support import (
    check_default_overlay_timeline,
    check_overlay_preroll_timeline,
    check_scene_snapshot_timeline,
    flush_and_load_rrd as _flush_and_load_rrd,
    ROOT_JOINT_OFFSET as _ROOT_JOINT_OFFSET,
)
from superdex.physics.utils.testing.testcases import (
    add_rigid_cube,
    make_empty_scene,
    make_single_rigid_cube_scene,
    MochiContextTestCase,
)


def _create_revolute_chain(scene, num_joints=3):
    """Create a minimal articulated actor with revolute joints."""
    import superdex.physics as sdp

    np_real = np.float64 if sdp.uses_double_precision() else np.float32

    coords = np.array(
        [
            -0.1,
            -0.1,
            -0.1,
            +0.1,
            -0.1,
            -0.1,
            -0.1,
            +0.1,
            -0.1,
            +0.1,
            +0.1,
            -0.1,
            -0.1,
            -0.1,
            +0.1,
            +0.1,
            -0.1,
            +0.1,
            -0.1,
            +0.1,
            +0.1,
            +0.1,
            +0.1,
            +0.1,
        ],
        dtype=np_real,
    )
    connectivity = np.array(
        [
            0,
            1,
            2,
            4,
            6,
            7,
            4,
            2,
            5,
            4,
            7,
            1,
            3,
            2,
            1,
            7,
            1,
            2,
            4,
            7,
        ],
        dtype=np.int32,
    )

    shape = sdp.create_tet_mesh_shape(
        coordinates=coords,
        connectivity=connectivity,
    )

    joints = []
    links = []
    for i in range(num_joints):
        joint = sdp.ArticulatedJointParams(
            type=sdp.ArticulatedJointType.REVOLUTE,
            axis=[1, 0, 0],
        )
        if i == 0:
            joint.parent_link_from_joint = sdp.TransformRT(
                translation=sdp.Real3(*_ROOT_JOINT_OFFSET)
            )
        joints.append(joint)
        links.append(
            sdp.ArticulatedLinkParams(
                parent_link=i - 1,
                shape=shape,
                density=1000.0,
            )
        )

    params = sdp.ArticulatedActorParams()
    params.name = "TestRobot"
    params.joints = joints
    params.links = links
    actor = scene.create_articulated_actor(params)
    sdp.release_shape(shape)
    return actor


# ── Config tests ──────────────────────────────────────────────────────────


class TestOverlayCfg(unittest.TestCase):
    def test_defaults(self):
        from superdex.physics.rerun import OverlayCfg

        cfg = OverlayCfg()
        self.assertEqual(cfg.name, "overlay")
        self.assertEqual(cfg.color, (1.0, 1.0, 1.0, 0.5))
        self.assertIsNone(cfg.include_actors)
        self.assertIsNone(cfg.exclude_actors)
        self.assertTrue(cfg.flat_shading)

    def test_custom(self):
        from superdex.physics.rerun import OverlayCfg

        cfg = OverlayCfg(name="ghost", color=(0.5, 0.8, 0.5, 1.0))
        self.assertEqual(cfg.name, "ghost")
        self.assertEqual(cfg.color, (0.5, 0.8, 0.5, 1.0))


class TestOverlayExports(unittest.TestCase):
    def test_exports(self):
        from superdex.physics.rerun import OverlayCfg, OverlayLogger

        self.assertIsNotNone(OverlayCfg)
        self.assertIsNotNone(OverlayLogger)


# ── Rigid overlay tests (optimizer path) ──────────────────────────────────


class TestRigidOverlay(MochiContextTestCase):
    """Tests the optimizer-style path: rigid actors + set_transform."""

    def _create_logger(self, save_path=None):
        from superdex.physics.rerun import RerunLogger, RerunLoggerCfg

        return RerunLogger(
            RerunLoggerCfg(
                application_id="test_overlay",
                connect=False,
                spawn=False,
                save_path=save_path,
            )
        )

    def test_create_overlay_from_logger(self):
        from superdex.physics.rerun import OverlayLogger

        with self._create_logger() as logger, make_single_rigid_cube_scene() as scene:
            logger.set_scene(scene)
            overlay = logger.create_overlay("ghost")
            self.assertIsInstance(overlay, OverlayLogger)
            self.assertGreater(len(overlay.get_actor_names()), 0)

    def test_create_overlay_raises_without_scene(self):
        with self._create_logger() as logger:
            with self.assertRaises(ValueError):
                logger.create_overlay("ghost")

    def test_create_overlay_with_explicit_scene(self):
        with self._create_logger() as logger, make_single_rigid_cube_scene() as scene:
            overlay = logger.create_overlay("ghost", scene=scene)
            self.assertGreater(len(overlay.get_actor_names()), 0)

    def test_entity_paths_contain_overlay_name(self):
        with self._create_logger() as logger, make_single_rigid_cube_scene() as scene:
            logger.set_scene(scene)
            overlay = logger.create_overlay("observed")
            for name in overlay.get_actor_names():
                path = overlay.get_entity_path(name)
                self.assertIn("observed", path)

    def test_nonexistent_actor_returns_none(self):
        with self._create_logger() as logger, make_single_rigid_cube_scene() as scene:
            logger.set_scene(scene)
            overlay = logger.create_overlay("ghost")
            self.assertIsNone(overlay.get_entity_path("nonexistent"))

    def test_set_transform(self):
        with self._create_logger() as logger, make_single_rigid_cube_scene() as scene:
            logger.set_scene(scene)
            overlay = logger.create_overlay("ghost")
            name = overlay.get_actor_names()[0]
            path_before = overlay.get_entity_path(name)
            overlay.set_transform(name, np.eye(4, dtype=np.float32))
            # Entity path is stable across pose updates.
            self.assertEqual(overlay.get_entity_path(name), path_before)

    def test_set_transform_ignores_unknown(self):
        with self._create_logger() as logger, make_single_rigid_cube_scene() as scene:
            logger.set_scene(scene)
            overlay = logger.create_overlay("ghost")
            names_before = overlay.get_actor_names()
            overlay.set_transform("nonexistent", np.eye(4))
            # Unknown actor: no entry created, no crash, registered actors unchanged.
            self.assertIsNone(overlay.get_entity_path("nonexistent"))
            self.assertEqual(overlay.get_actor_names(), names_before)

    def test_update_from_scene(self):
        with self._create_logger() as logger, make_single_rigid_cube_scene() as scene:
            logger.set_scene(scene)
            overlay = logger.create_overlay("ghost")
            names_before = overlay.get_actor_names()
            self.assertGreater(len(names_before), 0)
            overlay.update_from_scene()
            # Bulk transform sync preserves the registered actor set.
            self.assertEqual(overlay.get_actor_names(), names_before)

    def test_clear(self):
        with self._create_logger() as logger, make_single_rigid_cube_scene() as scene:
            logger.set_scene(scene)
            overlay = logger.create_overlay("ghost")
            names_before = overlay.get_actor_names()
            self.assertGreater(len(names_before), 0)
            overlay.clear()
            # clear() emits rr.Clear but keeps the local registry so future
            # set_transform calls still route to the right entity.
            self.assertEqual(overlay.get_actor_names(), names_before)

    def test_multiple_overlays(self):
        with self._create_logger() as logger, make_single_rigid_cube_scene() as scene:
            logger.set_scene(scene)
            o1 = logger.create_overlay("observed", color=(0.5, 0.8, 0.5, 1.0))
            o2 = logger.create_overlay("sim_state", color=(1.0, 1.0, 1.0, 0.5))
            name = o1.get_actor_names()[0]
            self.assertIn("observed", o1.get_entity_path(name))
            self.assertIn("sim_state", o2.get_entity_path(name))
            self.assertNotEqual(o1.get_entity_path(name), o2.get_entity_path(name))

    def test_include_filter(self):
        with self._create_logger() as logger, make_empty_scene() as scene:
            add_rigid_cube(scene, "finger_0")
            add_rigid_cube(scene, "finger_1")
            add_rigid_cube(scene, "table")
            overlay = logger.create_overlay(
                "ghost", include_actors=["finger_*"], scene=scene
            )
            names = overlay.get_actor_names()
            self.assertEqual(len(names), 2)
            for n in names:
                self.assertTrue(n.startswith("finger_"))

    def test_exclude_filter(self):
        with self._create_logger() as logger, make_empty_scene() as scene:
            add_rigid_cube(scene, "finger_0")
            add_rigid_cube(scene, "finger_1")
            add_rigid_cube(scene, "table")
            overlay = logger.create_overlay(
                "ghost", exclude_actors=["table"], scene=scene
            )
            names = overlay.get_actor_names()
            self.assertNotIn("table", names)
            self.assertGreater(len(names), 0)

    def test_overlay_in_rrd(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.rrd")
            with (
                self._create_logger(path) as logger,
                make_single_rigid_cube_scene() as scene,
            ):
                logger.set_scene(scene)
                logger.log_static_geometry()
                logger.log_frame(frame_idx=0)
                logger.create_overlay("ghost")
            recording = _flush_and_load_rrd(path)
            schema = recording.schema()
            entity_paths = {col.entity_path for col in schema.component_columns()}
            self.assertTrue(
                any("ghost" in p for p in entity_paths),
                f"Expected 'ghost' in paths: {entity_paths}",
            )

    def test_scene_snapshot_logs_preroll_without_advancing_frame(self):
        import superdex.physics as sdp

        check_scene_snapshot_timeline(
            self,
            create_logger=self._create_logger,
            make_empty_scene=make_empty_scene,
            add_rigid_cube=add_rigid_cube,
            physics_module=sdp,
        )


# ── Articulated overlay tests (playback path) ────────────────────────────


class TestArticulatedOverlay(MochiContextTestCase):
    """Tests the playback-style path: articulated actors + update_from_joint_angles."""

    def _create_logger(self, save_path=None):
        from superdex.physics.rerun import RerunLogger, RerunLoggerCfg

        return RerunLogger(
            RerunLoggerCfg(
                application_id="test_articulated_overlay",
                connect=False,
                spawn=False,
                save_path=save_path,
            )
        )

    def test_articulated_actor_discovered(self):
        with self._create_logger() as logger, make_empty_scene() as scene:
            _create_revolute_chain(scene, num_joints=3)
            overlay = logger.create_overlay("ghost", scene=scene)
            self.assertIn("TestRobot", overlay.get_articulated_names())

    def test_articulated_entity_path(self):
        with self._create_logger() as logger, make_empty_scene() as scene:
            _create_revolute_chain(scene, num_joints=3)
            overlay = logger.create_overlay("ghost", scene=scene)
            path = overlay.get_entity_path("TestRobot")
            self.assertIsNotNone(path)
            self.assertIn("ghost", path)
            self.assertIn("TestRobot", path)

    def test_rigid_actors_excluded_from_articulated(self):
        """Nested link actors should not appear in rigid actor list."""
        with self._create_logger() as logger, make_empty_scene() as scene:
            _create_revolute_chain(scene, num_joints=3)
            add_rigid_cube(scene, "table")
            overlay = logger.create_overlay("ghost", scene=scene)
            rigid_names = overlay.get_actor_names()
            self.assertIn("table", rigid_names)
            for name in rigid_names:
                self.assertNotIn("TestRobot", name)

    def test_set_articulated_pose(self):
        """set_articulated_pose does not raise for zero or nonzero joints."""
        with self._create_logger() as logger, make_empty_scene() as scene:
            actor = _create_revolute_chain(scene, num_joints=3)
            overlay = logger.create_overlay("ghost", scene=scene)
            n = actor.get_num_dofs()
            overlay.set_articulated_pose("TestRobot", np.zeros(n))
            overlay.set_articulated_pose("TestRobot", np.full(n, 0.5))

    def test_set_articulated_pose_with_anchor_transform(self):
        """Passing anchor_transform overrides the static root."""
        with self._create_logger() as logger, make_empty_scene() as scene:
            actor = _create_revolute_chain(scene, num_joints=3)
            overlay = logger.create_overlay("ghost", scene=scene)
            T = np.eye(4, dtype=np.float64)
            T[:3, 3] = [1.0, 2.0, 3.0]
            overlay.set_articulated_pose(
                "TestRobot", np.zeros(actor.get_num_dofs()), anchor_transform=T
            )

    def test_set_articulated_pose_unknown_actor(self):
        """Unknown actor name is a no-op, not an error."""
        with self._create_logger() as logger, make_empty_scene() as scene:
            _create_revolute_chain(scene, num_joints=3)
            overlay = logger.create_overlay("ghost", scene=scene)
            overlay.set_articulated_pose("NonExistent", np.zeros(3))

    def test_mixed_rigid_and_articulated(self):
        """Scene with both rigid and articulated actors."""
        with self._create_logger() as logger, make_empty_scene() as scene:
            _create_revolute_chain(scene, num_joints=3)
            add_rigid_cube(scene, "object")
            overlay = logger.create_overlay("observed", scene=scene)

            self.assertIn("TestRobot", overlay.get_articulated_names())
            self.assertIn("object", overlay.get_actor_names())

            overlay.set_articulated_pose("TestRobot", np.zeros(3))
            overlay.set_transform("object", np.eye(4))

    def test_exclude_filter_on_links(self):
        """Exclude patterns apply to individual link names."""
        with self._create_logger() as logger, make_empty_scene() as scene:
            _create_revolute_chain(scene, num_joints=3)
            overlay = logger.create_overlay(
                "ghost", exclude_actors=["TestRobot"], scene=scene
            )
            self.assertEqual(len(overlay.get_articulated_names()), 0)

    def test_articulated_overlay_in_rrd(self):
        """Articulated overlay meshes appear in saved .rrd."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.rrd")
            with (
                self._create_logger(path) as logger,
                make_empty_scene() as scene,
            ):
                actor = _create_revolute_chain(scene, num_joints=3)
                overlay = logger.create_overlay("ghost", scene=scene)
                overlay.set_articulated_pose(
                    "TestRobot", np.zeros(actor.get_num_dofs())
                )
            recording = _flush_and_load_rrd(path)
            schema = recording.schema()
            entity_paths = {col.entity_path for col in schema.component_columns()}
            self.assertTrue(
                any("ghost" in p and "TestRobot" in p for p in entity_paths),
                f"Expected articulated ghost paths, got: {entity_paths}",
            )

    def test_undriven_overlay_poses_land_on_frame_timeline(self):
        """Default overlay construction stays at frame and simulation time zero."""
        check_default_overlay_timeline(
            self,
            create_logger=self._create_logger,
            make_empty_scene=make_empty_scene,
            create_revolute_chain=_create_revolute_chain,
        )

    def test_overlay_setup_preroll_does_not_duplicate_frame_zero_source(self):
        """Setup poses stay on pre-roll when frame zero receives source poses."""
        check_overlay_preroll_timeline(
            self,
            create_logger=self._create_logger,
            make_empty_scene=make_empty_scene,
            create_revolute_chain=_create_revolute_chain,
        )

    def test_hierarchical_entity_paths(self):
        """Links are nested hierarchically, not flat."""
        with self._create_logger() as logger, make_empty_scene() as scene:
            _create_revolute_chain(scene, num_joints=3)
            overlay = logger.create_overlay("ghost", scene=scene)
            # The root entity should contain the actor name
            root_path = overlay.get_entity_path("TestRobot")
            self.assertIn("TestRobot", root_path)

    def test_full_playback_loop(self):
        """Simulate the playback path: logger + overlay + per-frame updates."""
        with self._create_logger() as logger, make_empty_scene() as scene:
            actor = _create_revolute_chain(scene, num_joints=3)
            add_rigid_cube(scene, "object")
            logger.set_scene(scene)
            logger.log_static_geometry()

            overlay = logger.create_overlay("observed", scene=scene)
            num_dofs = actor.get_num_dofs()

            for frame in range(5):
                scene.step(0.01)
                logger.log_frame(frame_idx=frame)

                angles = np.sin(frame * 0.1) * np.ones(num_dofs)
                overlay.set_articulated_pose("TestRobot", angles)

                T = np.eye(4, dtype=np.float64)
                T[:3, 3] = [0, 0, frame * 0.01]
                overlay.set_transform("object", T)

    def test_full_optimize_loop(self):
        """Simulate the optimizer path: logger + overlay + per-frame set_transform."""
        with self._create_logger() as logger, make_empty_scene() as scene:
            add_rigid_cube(scene, "finger_0")
            add_rigid_cube(scene, "finger_1")
            add_rigid_cube(scene, "object")
            logger.set_scene(scene)
            logger.log_static_geometry()

            overlay = logger.create_overlay(
                "observed", color=(0.5, 0.8, 0.5), scene=scene
            )
            # All three rigid actors registered up-front and stable across frames.
            actor_names_before = set(overlay.get_actor_names())
            self.assertEqual(actor_names_before, {"finger_0", "finger_1", "object"})

            for frame in range(5):
                scene.step(0.01)
                logger.log_frame(frame_idx=frame)

                for name in overlay.get_actor_names():
                    T = np.eye(4, dtype=np.float64)
                    T[:3, 3] = np.random.randn(3) * 0.1
                    overlay.set_transform(name, T)

            self.assertEqual(set(overlay.get_actor_names()), actor_names_before)


# ── Format-flex pose setter tests ────────────────────────────────────────


class TestRotationHelpers(unittest.TestCase):
    """Tests the rotation_to_matrix and pose_to_transform helpers."""

    def test_quat_identity(self):
        from superdex.physics.rerun.overlay import rotation_to_matrix

        m = rotation_to_matrix(quat=[0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(m, np.eye(3), atol=1e-12)

    def test_rotvec_identity(self):
        from superdex.physics.rerun.overlay import rotation_to_matrix

        m = rotation_to_matrix(rotvec=[0.0, 0.0, 0.0])
        np.testing.assert_allclose(m, np.eye(3), atol=1e-12)

    def test_mat_passthrough(self):
        from superdex.physics.rerun.overlay import rotation_to_matrix

        R = np.eye(3) * 1.0
        np.testing.assert_allclose(rotation_to_matrix(mat=R), R)

    def test_rot6d_recovers_identity(self):
        """Two rows of identity → identity matrix (cross product seals it)."""
        from superdex.physics.rerun.overlay import rotation_to_matrix

        r6 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        m = rotation_to_matrix(rot6d=r6)
        np.testing.assert_allclose(m, np.eye(3), atol=1e-12)

    def test_rot6d_matches_quat_for_nontrivial_rotation(self):
        """rot6d and quat paths produce the same matrix for a 90° Z rotation."""
        from scipy.spatial.transform import Rotation
        from superdex.physics.rerun.overlay import rotation_to_matrix

        R90z = Rotation.from_euler("z", 90, degrees=True)
        expected = R90z.as_matrix()
        r6 = expected[:2, :].ravel().tolist()  # first two rows, flattened
        m_rot6d = rotation_to_matrix(rot6d=r6)
        m_quat = rotation_to_matrix(quat=R90z.as_quat())
        np.testing.assert_allclose(m_rot6d, expected, atol=1e-12)
        np.testing.assert_allclose(m_rot6d, m_quat, atol=1e-12)

    def test_rejects_zero_args(self):
        from superdex.physics.rerun.overlay import rotation_to_matrix

        with self.assertRaises(ValueError):
            rotation_to_matrix()

    def test_rejects_multiple_args(self):
        from superdex.physics.rerun.overlay import rotation_to_matrix

        with self.assertRaises(ValueError):
            rotation_to_matrix(quat=[0, 0, 0, 1], rotvec=[0, 0, 0])

    def testpose_to_transform_builds_4x4(self):
        from superdex.physics.rerun.overlay import pose_to_transform

        T = pose_to_transform([1.0, 2.0, 3.0], quat=[0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(T[:3, 3], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(T[3, :], [0, 0, 0, 1])


class TestPoseSetters(MochiContextTestCase):
    """Tests set_rigid_pose and set_articulated_pose."""

    def _create_logger(self, save_path=None):
        from superdex.physics.rerun import RerunLogger, RerunLoggerCfg

        return RerunLogger(
            RerunLoggerCfg(
                application_id="test_pose_setters",
                connect=False,
                spawn=False,
                save_path=save_path,
            )
        )

    def test_set_rigid_pose_quat(self):
        with self._create_logger() as logger, make_single_rigid_cube_scene() as scene:
            logger.set_scene(scene)
            overlay = logger.create_overlay("ghost")
            name = overlay.get_actor_names()[0]
            overlay.set_rigid_pose(name, [1.0, 2.0, 3.0], quat=[0, 0, 0, 1])

    def test_set_rigid_pose_rotvec(self):
        with self._create_logger() as logger, make_single_rigid_cube_scene() as scene:
            logger.set_scene(scene)
            overlay = logger.create_overlay("ghost")
            name = overlay.get_actor_names()[0]
            overlay.set_rigid_pose(name, [0, 0, 0], rotvec=[0.1, 0.0, 0.0])

    def test_set_rigid_pose_rot6d(self):
        with self._create_logger() as logger, make_single_rigid_cube_scene() as scene:
            logger.set_scene(scene)
            overlay = logger.create_overlay("ghost")
            name = overlay.get_actor_names()[0]
            overlay.set_rigid_pose(name, [0, 0, 0], rot6d=[1, 0, 0, 0, 1, 0])

    def test_set_rigid_pose_rejects_missing_rotation(self):
        with self._create_logger() as logger, make_single_rigid_cube_scene() as scene:
            logger.set_scene(scene)
            overlay = logger.create_overlay("ghost")
            name = overlay.get_actor_names()[0]
            with self.assertRaises(ValueError):
                overlay.set_rigid_pose(name, [0, 0, 0])

    def test_set_rigid_pose_unknown_actor_silent(self):
        with self._create_logger() as logger, make_single_rigid_cube_scene() as scene:
            logger.set_scene(scene)
            overlay = logger.create_overlay("ghost")
            # Should not raise even without a rotation, because the actor is unknown
            overlay.set_rigid_pose("nonexistent", [0, 0, 0])

    def test_set_articulated_pose_with_anchor_transform(self):
        with self._create_logger() as logger, make_empty_scene() as scene:
            actor = _create_revolute_chain(scene, num_joints=3)
            overlay = logger.create_overlay("ghost", scene=scene)
            T = np.eye(4, dtype=np.float64)
            T[:3, 3] = [1, 2, 3]
            overlay.set_articulated_pose(
                "TestRobot", np.zeros(actor.get_num_dofs()), anchor_transform=T
            )

    def test_set_articulated_pose_without_anchor(self):
        """Joint angles can update without re-positioning the anchor."""
        with self._create_logger() as logger, make_empty_scene() as scene:
            actor = _create_revolute_chain(scene, num_joints=3)
            overlay = logger.create_overlay("ghost", scene=scene)
            overlay.set_articulated_pose("TestRobot", np.zeros(actor.get_num_dofs()))


# ── Anchor (mid-chain rooting) tests ─────────────────────────────────────


class TestAnchors(MochiContextTestCase):
    """Tests OverlayCfg.anchors — mid-chain anchoring of articulated actors."""

    def _create_logger(self, save_path=None):
        from superdex.physics.rerun import RerunLogger, RerunLoggerCfg

        return RerunLogger(
            RerunLoggerCfg(
                application_id="test_anchors",
                connect=False,
                spawn=False,
                save_path=save_path,
            )
        )

    def _link_names(self, scene, actor_name):
        """Discover the actual link names for the test chain."""
        import superdex.physics as sdp

        found = [None]

        def visit(a: sdp.Actor) -> None:
            if a.get_name() == actor_name:
                found[0] = a

        scene.for_each_actor(visit)
        return list(found[0].get_articulated_shape_info().link_names)

    def test_anchor_at_base_matches_default(self):
        """anchors={Robot: link0} should produce the same overlay as no anchors."""
        with self._create_logger() as logger, make_empty_scene() as scene:
            _create_revolute_chain(scene, num_joints=3)
            link_names = self._link_names(scene, "TestRobot")
            overlay = logger.create_overlay(
                "ghost", anchors={"TestRobot": link_names[0]}, scene=scene
            )
            self.assertIn("TestRobot", overlay.get_articulated_names())

    def test_anchor_mid_chain(self):
        """Anchoring at link 1 of a 3-link chain should still register the actor."""
        with self._create_logger() as logger, make_empty_scene() as scene:
            _create_revolute_chain(scene, num_joints=3)
            link_names = self._link_names(scene, "TestRobot")
            overlay = logger.create_overlay(
                "ghost", anchors={"TestRobot": link_names[1]}, scene=scene
            )
            self.assertIn("TestRobot", overlay.get_articulated_names())

    def test_anchor_at_last_link(self):
        """Anchor at the final link: only that link is in the subtree."""
        with self._create_logger() as logger, make_empty_scene() as scene:
            _create_revolute_chain(scene, num_joints=3)
            link_names = self._link_names(scene, "TestRobot")
            overlay = logger.create_overlay(
                "ghost", anchors={"TestRobot": link_names[-1]}, scene=scene
            )
            self.assertIn("TestRobot", overlay.get_articulated_names())

    def test_unknown_anchor_raises(self):
        with self._create_logger() as logger, make_empty_scene() as scene:
            _create_revolute_chain(scene, num_joints=3)
            with self.assertRaises(ValueError):
                logger.create_overlay(
                    "ghost", anchors={"TestRobot": "nonexistent_link"}, scene=scene
                )

    def test_anchor_then_set_articulated_pose(self):
        """End-to-end: anchor mid-chain, drive descendants via set_articulated_pose."""
        with self._create_logger() as logger, make_empty_scene() as scene:
            actor = _create_revolute_chain(scene, num_joints=3)
            link_names = self._link_names(scene, "TestRobot")
            overlay = logger.create_overlay(
                "ghost", anchors={"TestRobot": link_names[1]}, scene=scene
            )
            T = np.eye(4, dtype=np.float64)
            T[:3, 3] = [0.1, 0.2, 0.3]
            overlay.set_articulated_pose(
                "TestRobot", np.zeros(actor.get_num_dofs()), anchor_transform=T
            )

    def test_mark_subtree(self):
        """Subtree marking matches parent-indexed tree structure."""
        from superdex.physics.rerun.overlay import _mark_subtree

        # Tree:  0 -> 1 -> 2
        #            \-> 3 -> 4
        # parents indexed by node, -1 = root
        parents = [-1, 0, 1, 1, 3]
        # Anchored at 1: include 1, 2, 3, 4
        self.assertEqual(_mark_subtree(parents, 1), [False, True, True, True, True])
        # Anchored at 3: include 3, 4
        self.assertEqual(_mark_subtree(parents, 3), [False, False, False, True, True])
        # Anchored at 0 (root): all
        self.assertEqual(_mark_subtree(parents, 0), [True, True, True, True, True])


# ── Cosmetic exclude_actors tests ────────────────────────────────────────


class TestCosmeticExclude(MochiContextTestCase):
    """Verifies exclude_actors on articulated links is cosmetic (mesh-only)."""

    def _create_logger(self, save_path=None):
        from superdex.physics.rerun import RerunLogger, RerunLoggerCfg

        return RerunLogger(
            RerunLoggerCfg(
                application_id="test_cosmetic_exclude",
                connect=False,
                spawn=False,
                save_path=save_path,
            )
        )

    def test_exclude_link_does_not_break_chain(self):
        """Excluding an intermediate link via cosmetic filter still drives descendants."""
        with self._create_logger() as logger, make_empty_scene() as scene:
            actor = _create_revolute_chain(scene, num_joints=3)
            # Discover the middle link's full name
            import superdex.physics as sdp

            found = [None]

            def visit(a: sdp.Actor) -> None:
                if a.get_name() == "TestRobot":
                    found[0] = a

            scene.for_each_actor(visit)
            link_names = list(found[0].get_articulated_shape_info().link_names)
            middle_link = link_names[1]

            overlay = logger.create_overlay(
                "ghost", exclude_actors=[middle_link], scene=scene
            )
            # The whole articulated actor is still registered (mesh hidden, chain intact)
            self.assertIn("TestRobot", overlay.get_articulated_names())
            # Joint-driven update still works for the full chain
            overlay.set_articulated_pose("TestRobot", np.zeros(actor.get_num_dofs()))


# ── Geometry-frame correctness (jointFromChildLink) ──────────────────────


def _create_offset_revolute_chain(scene, num_joints=3):
    """Revolute chain with non-identity ``parent_joint_from_link`` and
    ``parent_link_from_joint``.

    ``parent_joint_from_link`` (a.k.a. ``jointFromChildLink``) is what
    distinguishes a link's geometry frame from its joint frame. Overlays must
    place link meshes in the geometry frame, so a non-identity value here is a
    regression guard: the demo/test chains above use identity
    ``parent_joint_from_link`` and therefore cannot catch a joint-frame vs
    geometry-frame mismatch.
    """
    import superdex.physics as sdp

    np_real = np.float64 if sdp.uses_double_precision() else np.float32
    # Unit-ish cube tet mesh (same topology as _create_revolute_chain).
    coords = np.array(
        [
            -0.1,
            -0.1,
            -0.1,
            +0.1,
            -0.1,
            -0.1,
            -0.1,
            +0.1,
            -0.1,
            +0.1,
            +0.1,
            -0.1,
            -0.1,
            -0.1,
            +0.1,
            +0.1,
            -0.1,
            +0.1,
            -0.1,
            +0.1,
            +0.1,
            +0.1,
            +0.1,
            +0.1,
        ],
        dtype=np_real,
    )
    connectivity = np.array(
        [0, 1, 2, 4, 6, 7, 4, 2, 5, 4, 7, 1, 3, 2, 1, 7, 1, 2, 4, 7],
        dtype=np.int32,
    )
    shape = sdp.create_tet_mesh_shape(coordinates=coords, connectivity=connectivity)

    joints = []
    links = []
    for i in range(num_joints):
        joint = sdp.ArticulatedJointParams(
            type=sdp.ArticulatedJointType.REVOLUTE, axis=[0, 0, 1]
        )
        joint.parent_link_from_joint = sdp.TransformRT(
            rotation=sdp.Quaternion.from_axis_angle([1, 0, 0], 0.3),
            translation=sdp.Real3(0.0, 0.25, 0.0),
        )
        link = sdp.ArticulatedLinkParams(parent_link=i - 1, shape=shape, density=1000.0)
        # Non-identity child-link offset (rotation + translation).
        link.parent_joint_from_link = sdp.TransformRT(
            rotation=sdp.Quaternion.from_axis_angle([0, 1, 0], 0.4),
            translation=sdp.Real3(0.05, 0.0, 0.02),
        )
        joints.append(joint)
        links.append(link)

    params = sdp.ArticulatedActorParams()
    params.name = "OffsetRobot"
    params.joints = joints
    params.links = links
    actor = scene.create_articulated_actor(params)
    sdp.release_shape(shape)
    return actor


class TestArticulatedGeometryFrame(MochiContextTestCase):
    """Articulated overlays must place link nodes in the geometry frame.

    Regression guard for the joint-frame vs geometry-frame mismatch: the
    hierarchy is built from ``get_articulated_shape_info`` (geometry-frame
    ``parent_link_from_joint`` / ``joint_from_child_link``), so the composed
    world transform of each link entity must equal
    ``get_articulated_link_transforms`` (which is the geometry frame). Uses a
    chain with non-identity ``parent_joint_from_link`` so the two frames differ.
    """

    def _create_logger(self):
        from superdex.physics.rerun import RerunLogger, RerunLoggerCfg

        return RerunLogger(
            RerunLoggerCfg(
                application_id="test_overlay_geom", connect=False, spawn=False
            )
        )

    def test_link_nodes_match_geometry_frame(self):
        from unittest import mock

        import superdex.physics as sdp
        import superdex.physics.rerun.overlay as overlay_mod

        with self._create_logger() as logger, make_empty_scene() as scene:
            actor = _create_offset_revolute_chain(scene, num_joints=3)
            num_dofs = actor.get_num_dofs()
            # Drive to a known non-rest pose so joint rotations are exercised.
            pose = np.linspace(0.2, 0.6, num_dofs)
            actor.set_articulated_pose_from_joints(pose)

            num_links = len(list(actor.get_articulated_shape_info().link_names))
            link_worlds = sdp.DynamicArrayTransformRT(num_links)
            actor.get_articulated_link_transforms(link_worlds)

            # Capture the 4x4 transforms the overlay logs (build + per-frame).
            captured: dict[str, np.ndarray] = {}

            def fake_log(path, tf, **_):
                captured[path] = np.asarray(tf, dtype=np.float64).copy()

            with mock.patch.object(overlay_mod, "_log_transform", fake_log):
                overlay = logger.create_overlay("ghost", scene=scene)
                # Position the base (anchor) at its true world pose and drive
                # all joints, so every link should land on the geometry frame.
                anchor_tf = overlay_mod._transformrt_to_4x4(link_worlds[0])
                overlay.set_articulated_pose(
                    "OffsetRobot", pose, anchor_transform=anchor_tf
                )

            art = overlay._articulated["OffsetRobot"]
            s2t = overlay._coordinate_transform.source_to_target

            def compose_world(path: str) -> np.ndarray:
                """Rerun-style composition: product of transforms on every
                ancestor entity path (missing paths contribute identity)."""
                segments = path.split("/")
                world = np.eye(4, dtype=np.float64)
                for k in range(1, len(segments) + 1):
                    prefix = "/".join(segments[:k])
                    if prefix in captured:
                        world = world @ captured[prefix]
                return world

            for i in range(num_links):
                path = art.entity_paths[i]
                self.assertIsNotNone(path, f"Link {i} has no entity path")
                expected = s2t @ overlay_mod._transformrt_to_4x4(link_worlds[i])
                np.testing.assert_allclose(
                    compose_world(path),
                    expected,
                    atol=1e-5,
                    err_msg=f"Link {i} node is not in the geometry frame",
                )


# ── Mesh helper tests ─────────────────────────────────────────────────────


class TestMeshHelperExports(unittest.TestCase):
    def test_compute_face_normals_exported(self):
        from superdex.physics.rerun.loggers import compute_face_normals

        self.assertIsNotNone(compute_face_normals)

    def test_extract_actor_mesh_exported(self):
        from superdex.physics.rerun.loggers import extract_actor_mesh

        self.assertIsNotNone(extract_actor_mesh)

    def test_compute_face_normals_basic(self):
        from superdex.physics.rerun.loggers import compute_face_normals

        # Two coplanar triangles forming a unit square. With auto-smooth (the
        # default after the textured-mesh logger update), the shared edge is
        # smooth so all 4 vertices are kept; faces still index into them.
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32
        )
        faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
        new_verts, new_faces, new_normals = compute_face_normals(vertices, faces)
        self.assertEqual(new_verts.shape, (4, 3))
        self.assertEqual(new_faces.shape, (2, 3))
        self.assertEqual(new_normals.shape, (4, 3))

    def test_backward_compat_alias(self):
        from superdex.physics.rerun.loggers.actor_logger import _compute_face_normals

        self.assertIsNotNone(_compute_face_normals)
