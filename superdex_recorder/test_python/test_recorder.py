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

from __future__ import annotations

import gc
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import rerun.experimental as rr_experimental
import superdex.physics as sdp
import superdex.recorder as recorder
from superdex.recorder.reader import EpisodeReader

from . import rrd_verify

# The Rerun wheel does not expose its experimental API to static type checkers.
_rr_experimental: Any = rr_experimental


def stored_value_types(path: Path, components: Sequence[str]) -> dict[str, pa.DataType]:
    """Return the Arrow element type that each component is stored with."""
    types = {}
    for chunk in (
        _rr_experimental.RrdReader(path).stream().filter(components=components)
    ):
        schema = chunk.to_record_batch().schema
        for component in components:
            index = schema.get_field_index(component)
            if index >= 0:
                types[component] = schema.field(index).type.value_type
    return types


class RecorderTest(unittest.TestCase):
    def make_recorder(self, directory: Path) -> recorder.Recorder:
        return recorder.Recorder(recorder.RecorderConfig(directory))

    def make_metadata(self) -> recorder.EpisodeMetadata:
        return recorder.EpisodeMetadata(
            properties={
                "/": "slash",
                "\\/": "escaped slash",
                "build revision": "abc123",
            },
        )

    def test_committed_rrd_contains_episode_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            directory = Path(directory_string)
            episode = self.make_recorder(directory).begin_episode(self.make_metadata())
            episode.commit()

            self.assertTrue(episode.output_path.exists())
            recording = EpisodeReader(episode.output_path)
            self.assertEqual(
                recording.read_static_value("superdex.episode.coordinate_space"),
                "FLU",
            )
            self.assertEqual(
                recording.read_static_value("superdex.episode.units_per_meter"),
                1.0,
            )
            self.assertEqual(
                recording.read_static_value("superdex.episode.schema_version"),
                "1.0",
            )
            property_keys = recording.component_samples("superdex.episode.property_key")
            self.assertCountEqual(
                [sample.value[0] for sample in property_keys],
                ["/", "\\/", "build revision"],
            )
            # `/` escapes to `\/`, so the key `\/` must still get its own path.
            self.assertCountEqual(
                {sample.entity_path for sample in property_keys},
                {
                    r"/episode/properties/\/",
                    r"/episode/properties/\\\/",
                    r"/episode/properties/build\ revision",
                },
            )

    def test_context_exit_without_commit_discards_episode(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            directory = Path(directory_string)
            with self.make_recorder(directory).begin_episode(self.make_metadata()):
                pass
            self.assertEqual(list(directory.iterdir()), [])

    def test_exception_in_context_propagates_and_discards_episode(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            directory = Path(directory_string)
            error = RuntimeError("controller failed")
            with self.assertRaises(RuntimeError) as raised:
                with self.make_recorder(directory).begin_episode() as episode:
                    raise error
            self.assertIs(raised.exception, error)
            self.assertFalse(episode.is_active)
            self.assertEqual(list(directory.iterdir()), [])

    def test_physics_recorder_owns_episode_for_its_lifetime(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            directory = Path(directory_string)
            sdp.initialize(num_worker_threads=0)
            try:
                scene = sdp.create_scene("recorder lifetime test")
                episode = self.make_recorder(directory).begin_episode()
                physics = recorder.PhysicsRecorder(episode, scene)
                partial_files = list(directory.glob("*.rrd.partial"))
                self.assertEqual(len(partial_files), 1)

                del episode
                gc.collect()
                self.assertTrue(partial_files[0].exists())

                del physics
                gc.collect()
                self.assertEqual(list(directory.iterdir()), [])
            finally:
                sdp.shutdown()

    def test_duplicate_actor_names_get_distinct_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            directory = Path(directory_string)
            sdp.initialize(num_worker_threads=0)
            try:
                scene = sdp.create_scene("duplicate actor names test")
                dynamic_shape = sdp.create_tri_mesh_shape(
                    coordinates=[
                        0,
                        0,
                        0,
                        0.1,
                        0,
                        0,
                        0,
                        0.1,
                        0,
                        0,
                        0,
                        0.1,
                    ],
                    connectivity=[0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3],
                )
                dynamic_actor = scene.create_rigid_actor(
                    name="cube",
                    shape=dynamic_shape,
                    density=1000,
                )
                static_shape = sdp.create_plane_shape(normal=[0, 0, 1], distance=0)
                static_actor = scene.create_rigid_actor(
                    name="cube",
                    shape=static_shape,
                    is_static=True,
                )
                self.assertIsNotNone(dynamic_actor)
                self.assertIsNotNone(static_actor)

                episode = self.make_recorder(directory).begin_episode()
                physics = recorder.PhysicsRecorder(episode, scene)
                physics.record(0, 0.0)
                scene.step(0.02)
                physics.record(1, scene.get_total_simulation_time())
                episode.commit()

                recording = EpisodeReader(episode.output_path)
                actor_ids = recording.component_samples("superdex.physics.actor_id")
                self.assertCountEqual(
                    [sample.value[0] for sample in actor_ids],
                    ["cube", "cube_1"],
                )
                self.assertEqual(len({sample.entity_path for sample in actor_ids}), 2)
                static_by_path = {
                    sample.entity_path: bool(sample.value[0])
                    for sample in recording.component_samples(
                        "superdex.physics.is_static"
                    )
                }
                self.assertCountEqual(static_by_path.values(), [False, True])
                root_poses = recording.component_samples(
                    "superdex.physics.root_translation_xyz"
                )
                for actor_path, is_static in static_by_path.items():
                    sample_frames = [
                        sample.timelines["frame"]
                        for sample in root_poses
                        if sample.entity_path == actor_path
                    ]
                    self.assertEqual(sample_frames, [0] if is_static else [0, 1])
                com_translations = recording.component_samples(
                    "superdex.physics.com_translation_xyz"
                )
                self.assertEqual(
                    {sample.entity_path for sample in com_translations},
                    {
                        path
                        for path, is_static in static_by_path.items()
                        if not is_static
                    },
                )
                self.assertEqual(
                    com_translations[-1].value,
                    dynamic_actor.get_center_of_mass_transform().translation.tolist(),
                )
                param_names = recording.read_static_value(
                    "superdex.physics.contact_param_names"
                )
                contact_params = {
                    sample.entity_path: sample.value
                    for sample in recording.component_samples(
                        "superdex.physics.contact_params"
                    )
                }
                self.assertEqual(contact_params.keys(), static_by_path.keys())
                for actor_path, is_static in static_by_path.items():
                    actor = static_actor if is_static else dynamic_actor
                    assert actor is not None
                    live = actor.get_contact_params()
                    self.assertEqual(
                        contact_params[actor_path],
                        [float(getattr(live, name)) for name in param_names],
                    )
                densities = recording.component_samples("superdex.physics.density")
                masses = recording.component_samples("superdex.physics.mass")
                assert dynamic_actor is not None
                self.assertEqual(
                    [sample.value for sample in densities],
                    [[dynamic_actor.get_density()]],
                )
                self.assertEqual(
                    [sample.value for sample in masses], [[dynamic_actor.get_mass()]]
                )
                step_durations = recording.component_samples(
                    "superdex.physics.step_dt_sec"
                )
                self.assertEqual(
                    [sample.timelines["frame"] for sample in step_durations], [0, 1]
                )
                self.assertAlmostEqual(step_durations[0].value[0], 0.01)
                self.assertAlmostEqual(step_durations[1].value[0], 0.02)
                self.assertEqual(
                    recording.component_samples(
                        "superdex.physics.contact_force_on_a_world_xyz"
                    ),
                    [],
                )
            finally:
                sdp.shutdown()

    def test_task_bodies_are_recorded_under_spawn_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            directory = Path(directory_string)
            sdp.initialize(num_worker_threads=0)
            scene = None
            physics = None
            try:
                scene = sdp.create_scene("spawn names")
                actors = {}
                for name in ("square", "left", "right", "floor", "a/b", "a.b"):
                    actor = scene.create_rigid_actor(
                        name=name,
                        shape=sdp.create_plane_shape(normal=[0, 0, 1], distance=0),
                        is_static=True,
                    )
                    assert actor is not None
                    actors[name] = actor
                config = recorder.PhysicsRecorderConfig()
                config.spawn_names = [
                    (actors["square"].get_handle(), "Square"),
                    (actors["left"].get_handle(), "Pair"),
                    (actors["right"].get_handle(), "Pair"),
                ]
                episode = self.make_recorder(directory).begin_episode()
                physics = recorder.PhysicsRecorder(episode, scene, config)
                physics.record(0, 0.0)
                episode.commit()

                recording = EpisodeReader(episode.output_path)
                ids = {
                    sample.entity_path: sample.value[0]
                    for sample in recording.component_samples(
                        "superdex.physics.actor_id"
                    )
                }
                self.assertCountEqual(
                    ids.values(),
                    ["Square", "Pair.left", "Pair.right", "floor", "a.b", "a.b_1"],
                )
                self.assertEqual(
                    {
                        ids[sample.entity_path]: sample.value[0]
                        for sample in recording.component_samples(
                            "superdex.physics.spawn_name"
                        )
                    },
                    {"Square": "Square", "Pair.left": "Pair", "Pair.right": "Pair"},
                )
                self.assertCountEqual(
                    [
                        sample.value[0]
                        for sample in recording.component_samples(
                            "superdex.physics.runtime_name"
                        )
                        if ids[sample.entity_path].startswith("a.b")
                    ],
                    ["a/b", "a.b"],
                )

                config.spawn_names = [(actors["floor"].get_handle(), "")]
                with self.assertRaisesRegex(sdp.Error, "Spawn names must not be empty"):
                    recorder.PhysicsRecorder(
                        self.make_recorder(directory).begin_episode(), scene, config
                    )
            finally:
                physics = None
                gc.collect()
                if scene is not None:
                    sdp.destroy_scene(scene)
                sdp.shutdown()

    def test_per_frame_values_keep_the_scene_precision(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            sdp.initialize(num_worker_threads=0)
            scene = None
            physics = None
            try:
                scene = sdp.create_scene("precision")
                floor = scene.create_rigid_actor(
                    name="floor",
                    shape=sdp.create_plane_shape(normal=[0, 0, 1], distance=0),
                    is_static=True,
                )
                assert floor is not None
                episode = self.make_recorder(Path(directory_string)).begin_episode()
                physics = recorder.PhysicsRecorder(episode, scene)
                physics.record(0, 0.0)
                episode.commit()

                real_type = pa.from_numpy_dtype(np.asarray(scene.get_gravity()).dtype)
                self.assertEqual(
                    stored_value_types(
                        episode.output_path,
                        ["superdex.physics.step_dt_sec", "superdex.physics.is_static"],
                    ),
                    {
                        "superdex.physics.step_dt_sec": real_type,
                        "superdex.physics.is_static": pa.float64(),
                    },
                )
            finally:
                physics = None
                gc.collect()
                if scene is not None:
                    sdp.destroy_scene(scene)
                sdp.shutdown()

    def test_each_entity_has_one_static_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            sdp.initialize(num_worker_threads=0)
            scene = None
            physics = None
            try:
                scene = sdp.create_scene("static chunks")
                cube = scene.create_rigid_actor(
                    name="cube",
                    shape=sdp.create_tri_mesh_shape(
                        coordinates=[0, 0, 0, 0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1],
                        connectivity=[0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3],
                    ),
                    density=1000,
                )
                floor = scene.create_rigid_actor(
                    name="floor",
                    shape=sdp.create_plane_shape(normal=[0, 0, 1], distance=0),
                    is_static=True,
                )
                assert cube is not None and floor is not None
                episode = self.make_recorder(Path(directory_string)).begin_episode()
                physics = recorder.PhysicsRecorder(episode, scene)
                physics.record(0, 0.0)
                episode.commit()

                stream = _rr_experimental.RrdReader(episode.output_path).stream()
                static_chunks = [chunk for chunk in stream if chunk.is_static]
                entity_paths = [chunk.entity_path for chunk in static_chunks]
                self.assertCountEqual(entity_paths, set(entity_paths))
                columns = {
                    chunk.entity_path: chunk.to_record_batch().schema.names
                    for chunk in static_chunks
                }
                self.assertIn("Mesh3D:vertex_positions", columns["/world/cube"])
                self.assertIn("superdex.episode.started_utc_ns", columns["/episode"])
                self.assertIn("superdex.episode.committed_utc_ns", columns["/episode"])
                actor_columns = columns["/data/physics/actors/cube"]
                self.assertIn("superdex.physics.actor_id", actor_columns)
                self.assertIn("superdex.physics.contact_params", actor_columns)
            finally:
                physics = None
                gc.collect()
                if scene is not None:
                    sdp.destroy_scene(scene)
                sdp.shutdown()

    def test_meshes_are_flat_shaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            sdp.initialize(num_worker_threads=0)
            scene = None
            physics = None
            try:
                scene = sdp.create_scene("flat shading")
                tetrahedron = scene.create_rigid_actor(
                    name="tetrahedron",
                    shape=sdp.create_tri_mesh_shape(
                        coordinates=[0, 0, 0, 0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1],
                        connectivity=[0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3],
                    ),
                    density=1000,
                )
                assert tetrahedron is not None
                episode = self.make_recorder(Path(directory_string)).begin_episode()
                physics = recorder.PhysicsRecorder(episode, scene)
                physics.record(0, 0.0)
                episode.commit()

                recording = EpisodeReader(episode.output_path)
                self.assertEqual(
                    recording.component_samples("Mesh3D:triangle_indices"), []
                )
                [positions] = recording.component_samples(
                    "Mesh3D:vertex_positions", entity_path="world/tetrahedron"
                )
                [normals] = recording.component_samples(
                    "Mesh3D:vertex_normals", entity_path="world/tetrahedron"
                )
                corners = np.asarray(positions.value, dtype=np.float64).reshape(
                    -1, 3, 3
                )
                vertex_normals = np.asarray(normals.value, dtype=np.float64).reshape(
                    -1, 3, 3
                )
                # Each face has its own three vertices, all with the face's normal.
                self.assertEqual(corners.shape, (4, 3, 3))
                self.assertEqual(vertex_normals.shape, corners.shape)
                face_normals = np.cross(
                    corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
                )
                face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True)
                np.testing.assert_allclose(
                    vertex_normals,
                    np.repeat(face_normals[:, np.newaxis], 3, axis=1),
                    atol=1e-6,
                )
            finally:
                physics = None
                gc.collect()
                if scene is not None:
                    sdp.destroy_scene(scene)
                sdp.shutdown()

    def test_soft_actors_record_their_surface_at_every_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            sdp.initialize(num_worker_threads=0)
            scene = None
            physics = None
            try:
                scene = sdp.create_scene("soft surface")
                tetrahedron = scene.create_soft_actor(
                    name="tetrahedron",
                    shape=sdp.create_tet_mesh_shape(
                        coordinates=[0, 0, 0, 0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1],
                        connectivity=[0, 1, 2, 3],
                    ),
                )
                assert tetrahedron is not None
                # Moving one corner deforms the tetrahedron.
                tetrahedron.set_node_velocities_local(
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
                )
                episode = self.make_recorder(Path(directory_string)).begin_episode()
                physics = recorder.PhysicsRecorder(episode, scene)
                surfaces = []
                for frame in range(3):
                    if frame > 0:
                        scene.step(0.01)
                    physics.record(frame, scene.get_total_simulation_time())
                    surfaces.append(
                        np.array(tetrahedron.get_surface_mesh_node_positions_local())
                    )
                triangles = np.asarray(tetrahedron.get_surface_mesh().connectivity)
                episode.commit()

                rrd_verify.assert_rerun_accepts(episode.output_path)
                recording = EpisodeReader(episode.output_path)
                path = "world/tetrahedron/surface"
                [indices] = recording.component_samples(
                    "Mesh3D:triangle_indices", entity_path=path
                )
                self.assertTrue(indices.is_static)
                np.testing.assert_array_equal(np.ravel(indices.value), triangles)
                samples = recording.component_samples_many(
                    ("Mesh3D:vertex_positions", "Mesh3D:vertex_normals"),
                    entity_path=path,
                )
                positions = samples["Mesh3D:vertex_positions"]
                self.assertEqual(
                    [sample.timelines["frame"] for sample in positions], [0, 1, 2]
                )
                np.testing.assert_allclose(
                    [np.ravel(sample.value) for sample in positions],
                    surfaces,
                    atol=1e-6,
                )
                self.assertEqual(len(samples["Mesh3D:vertex_normals"]), 3)
                # The surface's parent entity carries the root transform.
                transforms = recording.component_samples(
                    "Transform3D:translation", entity_path="world/tetrahedron"
                )
                self.assertEqual(transforms[0].timelines["frame"], 0)
            finally:
                physics = None
                gc.collect()
                if scene is not None:
                    sdp.destroy_scene(scene)
                sdp.shutdown()

    def test_reader_returns_only_requested_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            sdp.initialize(num_worker_threads=0)
            try:
                scene = sdp.create_scene("frames")
                episode = self.make_recorder(Path(directory_string)).begin_episode()
                physics = recorder.PhysicsRecorder(episode, scene)
                for frame in range(5):
                    if frame > 0:
                        scene.step(0.01)
                    physics.record(frame, scene.get_total_simulation_time())
                episode.commit()

                recording = EpisodeReader(episode.output_path)
                samples = recording.component_samples(
                    "superdex.physics.step_dt_sec", frames={1, 3}
                )
                self.assertEqual(
                    [sample.timelines["frame"] for sample in samples], [1, 3]
                )
                self.assertEqual(
                    recording.component_samples(
                        "superdex.episode.schema_version", frames={1}
                    ),
                    [],
                )
            finally:
                sdp.shutdown()

    def test_reader_keeps_nanosecond_simulation_times(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            sdp.initialize(num_worker_threads=0)
            try:
                scene = sdp.create_scene("times")
                episode = self.make_recorder(Path(directory_string)).begin_episode()
                physics = recorder.PhysicsRecorder(episode, scene)
                times_sec = []
                for frame in range(3):
                    if frame > 0:
                        # Not a whole number of microseconds, which a timedelta would truncate.
                        scene.step(0.0010003)
                    times_sec.append(scene.get_total_simulation_time())
                    physics.record(frame, times_sec[-1])
                episode.commit()

                samples = EpisodeReader(episode.output_path).component_samples(
                    "superdex.physics.step_dt_sec"
                )
                for sample, time_sec in zip(samples, times_sec, strict=True):
                    self.assertAlmostEqual(
                        sample.timelines["sim_time"], time_sec * 1e9, delta=1
                    )
            finally:
                sdp.shutdown()

    def test_records_simulated_contact_forces(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            sdp.initialize(num_worker_threads=0)
            scene = None
            physics = None
            try:
                scene = sdp.create_scene("contact")
                scene.set_gravity([0, -10, 0])
                h = 0.05
                cube_shape = sdp.create_tri_mesh_shape(
                    coordinates=[
                        c
                        for z in (-h, h)
                        for y in (-h, h)
                        for x in (-h, h)
                        for c in (x, y, z)
                    ],
                    connectivity=[
                        *(0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6),
                        *(0, 1, 5, 0, 5, 4, 2, 6, 3, 3, 6, 7),
                        *(0, 4, 2, 2, 4, 6, 1, 3, 5, 3, 7, 5),
                    ],
                )
                cube = scene.create_rigid_actor(
                    name="cube",
                    shape=cube_shape,
                    density=1000,
                    world_from_local=sdp.TransformRT(translation=[0, h, 0]),
                    collider_type=sdp.ColliderType.BOX,
                )
                scene.create_rigid_actor(
                    name="ground",
                    shape=sdp.create_plane_shape(normal=[0, 1, 0], distance=0),
                    is_static=True,
                )
                assert cube is not None
                cube.register_query(sdp.QueryType.TOTAL_CONTACT_FORCE)
                episode = self.make_recorder(Path(directory_string)).begin_episode()
                config = recorder.PhysicsRecorderConfig()
                config.record_contact_forces = True
                physics = recorder.PhysicsRecorder(episode, scene, config)
                physics.record(0, scene.get_total_simulation_time())
                expected_forces: dict[int, list[float]] = {}
                for frame in (1, 2, 3):
                    scene.step(0.01)
                    physics.record(frame, scene.get_total_simulation_time())
                    expected_forces[frame] = cube.get_contact_force_world().tolist()
                episode.commit()
                rrd_verify.assert_rerun_accepts(episode.output_path)
                self.assertGreater(expected_forces[3][1], 1.0)

                recording = EpisodeReader(episode.output_path)
                self.assertEqual(
                    recording.read_static_value(
                        "superdex.physics.contact_force_source"
                    ),
                    "simulation",
                )
                self.assertEqual(
                    recording.read_static_value("superdex.physics.contact_body_paths"),
                    ["world/cube", "world/ground"],
                )
                contacts = recording.component_samples_many(
                    (
                        "superdex.physics.contact_body_a_indices",
                        "superdex.physics.contact_body_b_indices",
                        "superdex.physics.contact_force_on_a_world_xyz",
                    ),
                    entity_path="data/physics/contacts",
                )
                arrows = recording.component_samples(
                    "Arrows3D:vectors", entity_path="world/contact_forces"
                )
                rows = list(
                    zip(
                        contacts["superdex.physics.contact_body_a_indices"],
                        contacts["superdex.physics.contact_body_b_indices"],
                        contacts["superdex.physics.contact_force_on_a_world_xyz"],
                        arrows,
                        strict=True,
                    )
                )
                # Frame 0 precedes the first step, so it has no contacts.
                self.assertEqual([row[2].timelines["frame"] for row in rows], [1, 2, 3])
                for body_a, body_b, forces, arrow in rows:
                    self.assertEqual(len(arrow.value), len(body_a.value))
                    self.assertEqual(set(body_a.value), {0.0})
                    self.assertEqual(set(body_b.value), {1.0})
                    recorded_force = [sum(forces.value[axis::3]) for axis in range(3)]
                    frame = forces.timelines["frame"]
                    assert isinstance(frame, int)
                    expected_force = expected_forces[frame]
                    for actual, expected in zip(
                        recorded_force, expected_force, strict=True
                    ):
                        self.assertAlmostEqual(actual, expected, delta=1e-4)
            finally:
                physics = None
                gc.collect()
                if scene is not None:
                    sdp.destroy_scene(scene)
                sdp.shutdown()
