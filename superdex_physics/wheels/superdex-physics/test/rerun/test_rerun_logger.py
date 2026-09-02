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

"""Tests for the rerun logger integration."""

import os
import tempfile
import unittest

import numpy as np
from superdex.physics.utils.testing.testcases import (
    add_rigid_cube,
    make_empty_scene,
    make_single_rigid_cube_scene,
    MochiContextTestCase,
)


def _flush_and_load_rrd(save_path):
    """Flush the active rerun recording and load the .rrd file.

    Replaces the global recording to release the file sink, forces garbage
    collection, and waits briefly so the background writer thread finishes
    writing all buffered data to disk before reading the file back.
    """
    import gc
    import time

    import rerun as rr

    rr.init("test_flush", recording_id="flush")
    gc.collect()
    time.sleep(0.5)
    return rr.experimental.RrdReader(save_path).stream().collect()


class TestRerunImport(unittest.TestCase):
    """Tests that rerun can be imported directly."""

    def test_rerun_import(self):
        """Test that rerun can be imported."""
        import rerun as rr

        self.assertIsNotNone(rr, "Rerun module should be importable")
        self.assertTrue(hasattr(rr, "log"), "Rerun should have a log function")
        self.assertTrue(hasattr(rr, "Mesh3D"), "Rerun should have Mesh3D")

    def test_rerun_blueprint_import(self):
        """Test that rerun.blueprint can be imported."""
        import rerun.blueprint as rrb

        self.assertIsNotNone(rrb, "Rerun blueprint module should be importable")


class TestRerunLoggerModuleExports(unittest.TestCase):
    """Tests for the superdex.physics.rerun module exports."""

    def test_rerun_module_exports(self):
        """Test that the rerun module exports classes directly."""
        from superdex.physics.rerun import RerunLogger, RerunLoggerCfg

        self.assertIsNotNone(RerunLogger)
        self.assertIsNotNone(RerunLoggerCfg)


class TestRerunLoggerCfg(unittest.TestCase):
    """Tests for the RerunLoggerCfg configuration."""

    def test_default_config(self):
        """Test creating a default configuration."""
        from superdex.physics.rerun import RerunLoggerCfg

        cfg = RerunLoggerCfg()
        self.assertEqual(cfg.application_id, "mochi")
        self.assertIsNone(cfg.recording_id)
        self.assertFalse(cfg.spawn)
        self.assertTrue(cfg.connect)
        self.assertIsNone(cfg.connect_addr)
        self.assertIsNone(cfg.save_path)
        self.assertTrue(cfg.log_meshes)
        self.assertTrue(cfg.log_transforms)
        self.assertTrue(cfg.log_debug_draw)
        self.assertTrue(cfg.use_timeline)

    def test_custom_config(self):
        """Test creating a custom configuration."""
        from superdex.physics.rerun import RerunLoggerCfg

        cfg = RerunLoggerCfg(
            application_id="test_app",
            connect=False,
            spawn=True,
            log_meshes=False,
        )
        self.assertEqual(cfg.application_id, "test_app")
        self.assertFalse(cfg.connect)
        self.assertTrue(cfg.spawn)
        self.assertFalse(cfg.log_meshes)


class TestRerunLoggerIntegration(MochiContextTestCase):
    """Integration tests for the RerunLogger with Mochi scenes."""

    def test_logger_creation_without_connection(self):
        """Test creating a logger without connecting to a viewer."""
        from superdex.physics.rerun import RerunLogger, RerunLoggerCfg

        cfg = RerunLoggerCfg(
            application_id="test",
            connect=False,
            spawn=False,
        )
        with RerunLogger(cfg) as logger:
            self.assertIsNone(logger.get_scene())

    def test_logger_with_empty_scene(self):
        """Test logger with an empty scene."""
        from superdex.physics.rerun import RerunLogger, RerunLoggerCfg

        cfg = RerunLoggerCfg(
            application_id="test_empty_scene",
            connect=False,
            spawn=False,
        )
        with RerunLogger(cfg) as logger, make_empty_scene() as scene:
            logger.set_scene(scene)
            self.assertEqual(logger.get_scene(), scene)
            logger.log_static_geometry()
            logger.log_frame(frame_idx=0)

    def test_logger_with_single_cube(self):
        """Test logger with a single rigid cube scene."""
        from superdex.physics.rerun import RerunLogger, RerunLoggerCfg

        cfg = RerunLoggerCfg(
            application_id="test_single_cube",
            connect=False,
            spawn=False,
        )
        with RerunLogger(cfg) as logger, make_single_rigid_cube_scene() as scene:
            logger.set_scene(scene)
            self.assertEqual(logger.get_scene(), scene)
            logger.log_static_geometry()
            logger.log_frame(frame_idx=0)

    def test_logger_with_multiple_cubes(self):
        """Test logger with multiple cubes added dynamically."""
        from superdex.physics.rerun import RerunLogger, RerunLoggerCfg

        cfg = RerunLoggerCfg(
            application_id="test_multi_cube",
            connect=False,
            spawn=False,
        )
        with RerunLogger(cfg) as logger, make_empty_scene() as scene:
            logger.set_scene(scene)

            # Add first cube and log
            add_rigid_cube(scene, "Cube1")
            logger.log_frame(frame_idx=0)

            # Add second cube and log
            add_rigid_cube(scene, "Cube2")
            logger.log_frame(frame_idx=1)

    def test_logger_scene_switching(self):
        """Test switching scenes."""
        from superdex.physics.rerun import RerunLogger, RerunLoggerCfg

        cfg = RerunLoggerCfg(
            application_id="test_scene_switch",
            connect=False,
            spawn=False,
        )
        with RerunLogger(cfg) as logger:
            # First scene
            with make_single_rigid_cube_scene() as scene1:
                logger.set_scene(scene1)
                self.assertEqual(logger.get_scene(), scene1)
                logger.log_frame(frame_idx=0)

            # Second scene
            with make_empty_scene() as scene2:
                logger.set_scene(scene2)
                self.assertEqual(logger.get_scene(), scene2)
                logger.log_frame(frame_idx=1)

            # Clear scene
            logger.set_scene(None)
            self.assertIsNone(logger.get_scene())

    def test_logger_with_coordinate_system_preset(self):
        """Test logger with coordinate system preset."""
        from superdex.physics.rerun import RerunLogger, RerunLoggerCfg

        cfg = RerunLoggerCfg(
            application_id="test_coord_system",
            connect=False,
            spawn=False,
            coordinate_system="unity",
        )
        with RerunLogger(cfg) as logger, make_single_rigid_cube_scene() as scene:
            logger.set_scene(scene)
            logger.log_frame(frame_idx=0)

    def test_logger_multiple_frames(self):
        """Test logging multiple frames."""
        from superdex.physics.rerun import RerunLogger, RerunLoggerCfg

        cfg = RerunLoggerCfg(
            application_id="test_multi_frame",
            connect=False,
            spawn=False,
        )
        with RerunLogger(cfg) as logger, make_single_rigid_cube_scene() as scene:
            logger.set_scene(scene)
            for i in range(10):
                scene.step(0.01)  # 10ms time step
                logger.log_frame(frame_idx=i)


class TestRerunLoggerSaveToRrd(MochiContextTestCase):
    """Tests for saving and reading back .rrd recordings."""

    def _create_save_logger(self, save_path, app_id="test_save"):
        """Helper to create a logger that saves to an .rrd file."""
        from superdex.physics.rerun import RerunLogger, RerunLoggerCfg

        return RerunLogger(
            RerunLoggerCfg(
                application_id=app_id,
                connect=False,
                spawn=False,
                save_path=save_path,
            )
        )

    def test_save_to_rrd_creates_file(self):
        """Logger with save_path creates a valid .rrd file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.rrd")
            with self._create_save_logger(path) as logger:
                logger.log_static_geometry()
            _flush_and_load_rrd(path)  # Verify it loads without error
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)

    def test_rrd_contains_world_coordinates(self):
        """log_static_geometry() records ViewCoordinates at the world entity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.rrd")
            with self._create_save_logger(path) as logger:
                logger.log_static_geometry()
            recording = _flush_and_load_rrd(path)
            schema = recording.schema()
            entity_paths = {col.entity_path for col in schema.component_columns()}
            self.assertTrue(
                any("world" in p for p in entity_paths),
                f"Expected an entity path containing 'world', got: {entity_paths}",
            )

    def test_rrd_contains_actor_mesh(self):
        """Setting a scene and logging a frame records mesh entities."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.rrd")
            with (
                self._create_save_logger(path) as logger,
                make_single_rigid_cube_scene() as scene,
            ):
                logger.set_scene(scene)
                logger.log_static_geometry()
                logger.log_frame(frame_idx=0)
            recording = _flush_and_load_rrd(path)
            schema = recording.schema()
            entity_paths = {col.entity_path for col in schema.component_columns()}
            # Should have world path and at least one actor mesh path
            self.assertGreater(
                len(entity_paths),
                1,
                f"Expected multiple entity paths (world + actor), got: {entity_paths}",
            )

    def test_rrd_multiple_frames(self):
        """Logging 10 frames results in frame timeline values 0-9."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.rrd")
            with (
                self._create_save_logger(path) as logger,
                make_single_rigid_cube_scene() as scene,
            ):
                logger.set_scene(scene)
                for i in range(10):
                    scene.step(0.01)
                    logger.log_frame(frame_idx=i)
            recording = _flush_and_load_rrd(path)
            frames_set: set[int] = set()
            for chunk in recording.stream().filter(has_timeline="frame"):
                batch = chunk.to_record_batch()
                if "frame" in batch.schema.names:
                    frames_set.update(batch.column("frame").to_pylist())
            frames = sorted(frames_set)
            self.assertEqual(frames, list(range(10)))


class TestRerunLoggerResetScene(MochiContextTestCase):
    """Tests for reset_scene() method."""

    def _create_logger(self, **kwargs):
        """Helper to create a logger without connection."""
        from superdex.physics.rerun import RerunLogger, RerunLoggerCfg

        cfg_kwargs = {"application_id": "test_reset", "connect": False, "spawn": False}
        cfg_kwargs.update(kwargs)
        return RerunLogger(RerunLoggerCfg(**cfg_kwargs))

    def test_reset_scene_resets_frame_count(self):
        """reset_scene() resets the internal frame counter to 0."""
        with self._create_logger() as logger, make_single_rigid_cube_scene() as scene:
            logger.set_scene(scene)
            for i in range(5):
                logger.log_frame(frame_idx=i)
            self.assertEqual(logger._frame_count, 5)
            logger.reset_scene(scene=scene)
            self.assertEqual(logger._frame_count, 0)

    def test_reset_scene_rebases_sim_time(self):
        """reset_scene() zeroes sim_time against the already-stepped clock.

        Mirrors an environment reset: the scene has been stepped (settling)
        before the episode is recorded, so the raw scene clock is nonzero by
        the time the first frame is logged.
        """
        with self._create_logger() as logger, make_single_rigid_cube_scene() as scene:
            for _ in range(5):
                scene.step(0.1)  # "settling" — clock is now 0.5
            self.assertAlmostEqual(scene.get_total_simulation_time(), 0.5)

            logger.reset_scene(scene=scene)
            logger.log_frame(frame_idx=0)
            self.assertAlmostEqual(logger.get_sim_time(), 0.0)

            scene.step(0.1)
            logger.log_frame(frame_idx=1)
            self.assertAlmostEqual(logger.get_sim_time(), 0.1)

    def test_sim_time_is_zero_before_first_frame(self):
        """get_sim_time() reports 0 until a frame has been logged."""
        with self._create_logger() as logger, make_single_rigid_cube_scene() as scene:
            logger.set_scene(scene)
            scene.step(0.1)
            self.assertAlmostEqual(logger.get_sim_time(), 0.0)
            logger.reset_scene(scene=scene)
            self.assertAlmostEqual(logger.get_sim_time(), 0.0)

    def test_sim_time_holds_between_frames(self):
        """get_sim_time() reports the last logged frame, not the live clock.

        Callers may buffer a value at log time and emit it later (after the
        scene has moved on), so this must not track the scene continuously.
        """
        with self._create_logger() as logger, make_single_rigid_cube_scene() as scene:
            logger.reset_scene(scene=scene)
            logger.log_frame(frame_idx=0)
            scene.step(0.1)
            self.assertAlmostEqual(logger.get_sim_time(), 0.0)

    def test_set_scene_drops_a_stale_origin(self):
        """Rebinding to another scene must not keep the old scene's origin.

        The origin belongs to the scene it came from; reusing it against a
        younger scene would stamp a negative sim_time.
        """
        with self._create_logger() as logger, make_single_rigid_cube_scene() as scene1:
            for _ in range(5):
                scene1.step(0.1)
            logger.reset_scene(scene=scene1)  # origin = 0.5
            with make_single_rigid_cube_scene() as scene2:
                logger.set_scene(scene2)  # fresh scene, clock at 0
                scene2.step(0.1)
                logger.log_frame(frame_idx=0)
                self.assertAlmostEqual(logger.get_sim_time(), 0.1)

    def test_reset_scene_without_scene_uses_zero_origin(self):
        """reset_scene() with no scene, ever, leaves the origin at 0."""
        with self._create_logger() as logger:
            logger.reset_scene()
            self.assertAlmostEqual(logger.get_sim_time(), 0.0)

    def test_sim_time_reaches_the_rrd_timeline(self):
        """The rebased value is recorded, not merely tracked in Python."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sim_time.rrd")
            with (
                self._create_logger(save_path=path) as logger,
                make_single_rigid_cube_scene() as scene,
            ):
                for _ in range(5):
                    scene.step(0.1)  # "settling" — clock is now 0.5
                logger.reset_scene(scene=scene)
                for i in range(3):
                    logger.log_frame(frame_idx=i)
                    scene.step(0.1)

            recording = _flush_and_load_rrd(path)
            sim_times = set()
            for chunk in recording.stream().filter(has_timeline="sim_time"):
                batch = chunk.to_record_batch()
                if "sim_time" not in batch.schema.names:
                    continue
                for value in batch.column("sim_time").to_pylist():
                    if value is None:
                        continue
                    sim_times.add(
                        round(
                            value.total_seconds()
                            if hasattr(value, "total_seconds")
                            else value / 1e9,
                            6,
                        )
                    )
            # Episode-relative: 0.0/0.1/0.2, not the raw 0.5/0.6/0.7.
            self.assertEqual(sorted(sim_times), [0.0, 0.1, 0.2])

    def test_sim_time_rebases_again_on_each_reset(self):
        """Every reset_scene() re-zeroes, so episodes don't accumulate time."""
        with self._create_logger() as logger, make_single_rigid_cube_scene() as scene:
            for _ in range(2):  # two "episodes" on one monotonic scene clock
                logger.reset_scene(scene=scene)
                logger.log_frame(frame_idx=0)
                self.assertAlmostEqual(logger.get_sim_time(), 0.0)
                scene.step(0.1)
                logger.log_frame(frame_idx=1)
                self.assertAlmostEqual(logger.get_sim_time(), 0.1)

    def test_set_scene_only_callers_keep_absolute_clock(self):
        """Without reset_scene(), sim_time stays the raw scene clock.

        The origin only moves at reset_scene(), so single-session callers that
        use set_scene() + log_frame() are unaffected by the rebase.
        """
        with self._create_logger() as logger, make_single_rigid_cube_scene() as scene:
            logger.set_scene(scene)
            scene.step(0.1)
            logger.log_frame(frame_idx=0)
            self.assertAlmostEqual(logger.get_sim_time(), 0.1)

    def test_reset_scene_with_new_scene(self):
        """reset_scene(scene) replaces the current scene."""
        with self._create_logger() as logger:
            with make_single_rigid_cube_scene() as scene1:
                logger.set_scene(scene1)
                self.assertEqual(logger.get_scene(), scene1)

                with make_empty_scene() as scene2:
                    logger.reset_scene(scene=scene2)
                    self.assertEqual(logger.get_scene(), scene2)

    def test_reset_scene_without_scene_keeps_current(self):
        """reset_scene() without a scene argument keeps the current scene."""
        with self._create_logger() as logger, make_single_rigid_cube_scene() as scene:
            logger.set_scene(scene)
            logger.log_frame(frame_idx=0)
            logger.reset_scene()
            self.assertEqual(logger.get_scene(), scene)

    def test_reset_scene_can_log_after_reset(self):
        """Frames can be logged after reset_scene()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.rrd")
            with (
                self._create_logger(save_path=path) as logger,
                make_single_rigid_cube_scene() as scene,
            ):
                logger.set_scene(scene)
                logger.log_frame(frame_idx=0)
                logger.reset_scene(scene=scene)
                logger.log_frame(frame_idx=0)
                logger.log_frame(frame_idx=1)
            recording = _flush_and_load_rrd(path)
            schema = recording.schema()
            entity_paths = {col.entity_path for col in schema.component_columns()}
            self.assertTrue(
                any("world" in p for p in entity_paths),
                f"Expected entity paths with 'world', got: {entity_paths}",
            )

    def test_reset_scene_new_recording_returns_stream(self):
        """reset_scene(new_recording=True) returns a RecordingStream."""
        import rerun as rr

        with (
            self._create_logger() as logger,
            make_single_rigid_cube_scene() as scene,
        ):
            logger.set_scene(scene)
            rec = logger.reset_scene(
                scene=scene,
                new_recording=True,
                recording_id="test_ep",
            )
            self.assertIsNotNone(rec)
            # The new recording should be the active global recording. In
            # rerun 0.26 each RecordingStream accessor returns a fresh wrapper
            # object, so compare recording IDs rather than object identity.
            self.assertIsNotNone(rr.get_global_data_recording())
            self.assertEqual(rr.get_recording_id(), rr.get_recording_id(rec))

    def test_reset_scene_without_new_recording_returns_none(self):
        """reset_scene() without new_recording returns None."""
        with self._create_logger() as logger, make_single_rigid_cube_scene() as scene:
            logger.set_scene(scene)
            result = logger.reset_scene(scene=scene)
            self.assertIsNone(result)

    def test_per_episode_save_paths(self):
        """Two reset_scene calls with different save_paths produce two .rrd files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = os.path.join(tmpdir, "ep1.rrd")
            path2 = os.path.join(tmpdir, "ep2.rrd")
            with (
                self._create_logger() as logger,
                make_single_rigid_cube_scene() as scene,
            ):
                logger.set_scene(scene)

                # Episode 1: 3 frames
                logger.reset_scene(
                    scene=scene,
                    new_recording=True,
                    recording_id="ep1",
                    save_path=path1,
                )
                for i in range(3):
                    scene.step(0.01)
                    logger.log_frame(frame_idx=i)

                # Episode 2: 7 frames
                logger.reset_scene(
                    scene=scene,
                    new_recording=True,
                    recording_id="ep2",
                    save_path=path2,
                )
                for i in range(7):
                    scene.step(0.01)
                    logger.log_frame(frame_idx=i)

            # Both files must exist and contain scene data.
            for path in (path1, path2):
                self.assertTrue(os.path.exists(path), f"Expected .rrd file at {path}")
                recording = _flush_and_load_rrd(path)
                entity_paths = {
                    col.entity_path for col in recording.schema().component_columns()
                }
                self.assertTrue(
                    any("world" in p for p in entity_paths),
                    f"Expected 'world' entities in {path}, got: {entity_paths}",
                )

    def test_per_episode_frame_isolation(self):
        """Each episode recording has its own frame timeline starting at 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = os.path.join(tmpdir, "ep1.rrd")
            path2 = os.path.join(tmpdir, "ep2.rrd")
            with (
                self._create_logger() as logger,
                make_single_rigid_cube_scene() as scene,
            ):
                logger.set_scene(scene)

                # Episode 1: frames 0-9
                logger.reset_scene(
                    scene=scene,
                    new_recording=True,
                    recording_id="ep1",
                    save_path=path1,
                )
                for i in range(10):
                    scene.step(0.01)
                    logger.log_frame(frame_idx=i)

                # Episode 2: frames 0-2 (must NOT inherit episode 1's timeline)
                logger.reset_scene(
                    scene=scene,
                    new_recording=True,
                    recording_id="ep2",
                    save_path=path2,
                )
                for i in range(3):
                    scene.step(0.01)
                    logger.log_frame(frame_idx=i)

            rec1 = _flush_and_load_rrd(path1)
            frames1_set: set[int] = set()
            for chunk in rec1.stream().filter(has_timeline="frame"):
                batch = chunk.to_record_batch()
                if "frame" in batch.schema.names:
                    frames1_set.update(batch.column("frame").to_pylist())
            frames1 = sorted(frames1_set)
            rec2 = _flush_and_load_rrd(path2)
            frames2_set: set[int] = set()
            for chunk in rec2.stream().filter(has_timeline="frame"):
                batch = chunk.to_record_batch()
                if "frame" in batch.schema.names:
                    frames2_set.update(batch.column("frame").to_pylist())
            frames2 = sorted(frames2_set)

            self.assertEqual(frames1, list(range(10)))
            self.assertEqual(frames2, list(range(3)))

    def test_save_path_override_receives_new_data(self):
        """reset_scene(save_path=override) writes new frames to override, not cfg path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "cfg.rrd")
            override_path = os.path.join(tmpdir, "override.rrd")
            with (
                self._create_logger(save_path=cfg_path) as logger,
                make_single_rigid_cube_scene() as scene,
            ):
                logger.set_scene(scene)
                # The constructor already saved an initial recording to cfg_path.
                # Now start a new recording that should save to override_path.
                logger.reset_scene(
                    scene=scene,
                    new_recording=True,
                    save_path=override_path,
                )
                for i in range(5):
                    scene.step(0.01)
                    logger.log_frame(frame_idx=i)

            # The override path must have the 5 frames we logged.
            rec = _flush_and_load_rrd(override_path)
            frames_set: set[int] = set()
            for chunk in rec.stream().filter(has_timeline="frame"):
                batch = chunk.to_record_batch()
                if "frame" in batch.schema.names:
                    frames_set.update(batch.column("frame").to_pylist())
            frames = sorted(frames_set)
            self.assertEqual(frames, list(range(5)))


def _make_cube_mesh():
    """Unit cube: 8 vertices, 12 triangles. All dihedral angles are 90 degrees."""
    verts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],  # bottom
            [4, 6, 5],
            [4, 7, 6],  # top
            [0, 4, 5],
            [0, 5, 1],  # front
            [2, 6, 7],
            [2, 7, 3],  # back
            [0, 3, 7],
            [0, 7, 4],  # left
            [1, 5, 6],
            [1, 6, 2],  # right
        ],
        dtype=np.int32,
    )
    return verts, faces


def _make_coplanar_strip():
    """Two coplanar triangles sharing an edge. Dihedral angle = 0 degrees."""
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [1.5, 1, 0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    return verts, faces


class TestAutoSmoothNormals(unittest.TestCase):
    """Tests for angle-based auto-smooth normal computation."""

    def test_cube_splits_sharp_edges(self):
        """Cube has 90-degree edges (sharp) and coplanar diagonals (smooth).

        Each cube face has 2 coplanar triangles sharing a smooth diagonal,
        forming 6 smoothing groups. Each of the 8 vertices touches 3 groups,
        so the output is 8 * 3 = 24 vertices.
        """
        from superdex.physics.rerun.loggers.actor_logger import _compute_face_normals

        verts, faces = _make_cube_mesh()
        new_v, new_f, new_n = _compute_face_normals(verts, faces)
        self.assertEqual(new_v.shape[0], 24)
        self.assertEqual(new_n.shape[0], 24)
        self.assertEqual(new_f.shape, (len(faces), 3))

    def test_coplanar_no_splits(self):
        """Two coplanar triangles (0-degree dihedral) share the middle edge."""
        from superdex.physics.rerun.loggers.actor_logger import _compute_face_normals

        verts, faces = _make_coplanar_strip()
        new_v, new_f, new_n = _compute_face_normals(verts, faces)
        self.assertEqual(new_v.shape[0], 4)
        np.testing.assert_allclose(new_n, [[0, 0, 1]] * 4, atol=1e-6)

    def test_threshold_zero_splits_noncoplanar(self):
        """threshold=0 splits all edges with any dihedral angle > 0.

        A tetrahedron has 4 faces with ~70.5-degree dihedral angles on every
        edge, so threshold=0 makes them all sharp. Each vertex appears in 3
        faces → 4 * 3 = 12 output vertices (F * 3).
        """
        from superdex.physics.rerun.loggers.actor_logger import _compute_face_normals

        verts = np.array(
            [[0, 0, 0], [1, 0, 0], [0.5, 0.866, 0], [0.5, 0.289, 0.816]],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2], [0, 3, 1], [1, 3, 2], [0, 2, 3]], dtype=np.int32)
        new_v, new_f, new_n = _compute_face_normals(
            verts, faces, angle_threshold_deg=0.0
        )
        self.assertEqual(new_v.shape[0], len(faces) * 3)

    def test_threshold_180_all_smooth(self):
        """threshold=180 means no edge is sharp."""
        from superdex.physics.rerun.loggers.actor_logger import _compute_face_normals

        verts, faces = _make_cube_mesh()
        new_v, new_f, new_n = _compute_face_normals(
            verts, faces, angle_threshold_deg=180.0
        )
        self.assertEqual(new_v.shape[0], 8)

    def test_degenerate_triangle_no_crash(self):
        """Zero-area triangle should not crash or produce NaN."""
        from superdex.physics.rerun.loggers.actor_logger import _compute_face_normals

        verts = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        new_v, new_f, new_n = _compute_face_normals(verts, faces)
        self.assertFalse(np.any(np.isnan(new_n)))

    def test_single_triangle(self):
        """Single triangle: all boundary edges are sharp."""
        from superdex.physics.rerun.loggers.actor_logger import _compute_face_normals

        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        new_v, new_f, new_n = _compute_face_normals(verts, faces)
        self.assertEqual(new_v.shape[0], 3)
        np.testing.assert_allclose(new_n, [[0, 0, 1]] * 3, atol=1e-6)

    def test_output_dtypes(self):
        """Output arrays have correct dtypes."""
        from superdex.physics.rerun.loggers.actor_logger import _compute_face_normals

        verts, faces = _make_cube_mesh()
        new_v, new_f, new_n = _compute_face_normals(verts, faces)
        self.assertEqual(new_v.dtype, np.float32)
        self.assertEqual(new_f.dtype, np.int32)
        self.assertEqual(new_n.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
