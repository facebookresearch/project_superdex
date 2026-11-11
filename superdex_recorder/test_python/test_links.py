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
from pathlib import Path
from typing import Any

import numpy as np
import rerun.experimental as rr_experimental
import superdex.physics as sdp
import superdex.recorder as recorder
import superdex.robotics as sdr
from superdex.physics.paths import resolve_asset
from superdex.physics.utils.scene_helpers import load_bot_scene
from superdex.recorder.reader import EpisodeReader

from . import rrd_verify

# The Rerun wheel does not expose its experimental API to static type checkers.
_rr_experimental: Any = rr_experimental

BOT_SCENE_PATH = resolve_asset(
    "bots/internal/tests/bot_scene/test_bot_scene.mochi_bot_scene"
)
ROBOT_NAME = "t3_left"
HAND_ROOT_LINK = "dg5f_link_palm"
KNOB_NAME = "Knob_M8_50mm_4273T84"
# The Robotiq 2F-85 gripper alone with a free root, and on a fixed-base arm.
ROBOTIQ_BOTS = (
    "bots/grippers/2f_85/2f_85.superdex_bot",
    "bots/arm_hand_combos/fr3_v2_2f_85/fr3_v2_2f_85.superdex_bot",
)
NUM_FRAMES = 10
MIDDLE_FRAME = NUM_FRAMES // 2
TIME_STEP = 0.005
# Recorded transforms are float32, and a link's pose composes about 20 of them.
TOLERANCE = 1e-5
# A wrist with a one-segment thumb and a two-segment index finger.
KEYPOINT_NAMES = ["wrist", "thumb_tip", "index_knuckle", "index_tip"]
KEYPOINT_PARENTS = [-1, 0, 0, 2]
HAND_COLOR = (10, 200, 30, 255)
HAND_RADIUS = 0.005
TRACKED_HAND_PATH = f"/world/shadows/{ROBOT_NAME}/tracked_hand"


def recorded_world_transforms(
    path: Path, frame: int | None = None
) -> tuple[dict[str, sdp.TransformRT], dict[str, int]]:
    """Compose each entity's last recorded transform, up to frame if given, with its
    ancestors', like the viewer does, and count each entity's transform samples."""
    samples = EpisodeReader(path).component_samples_many(
        ("Transform3D:translation", "Transform3D:quaternion"),
        frames=None if frame is None else range(frame + 1),
    )
    translations = {
        sample.entity_path: np.array(sample.value, dtype=np.float64).reshape(-1)
        for sample in samples["Transform3D:translation"]
    }
    rotations = {
        sample.entity_path: np.array(sample.value, dtype=np.float64).reshape(-1)
        for sample in samples["Transform3D:quaternion"]
    }
    counts: dict[str, int] = {}
    for sample in samples["Transform3D:translation"]:
        counts[sample.entity_path] = counts.get(sample.entity_path, 0) + 1
    world = {}
    for entity_path in translations:
        parts = entity_path.lstrip("/").split("/")
        transform = sdp.TransformRT()
        for depth in range(1, len(parts) + 1):
            prefix = "/" + "/".join(parts[:depth])
            if prefix in translations:
                transform = transform * sdp.TransformRT(
                    rotation=rotations[prefix].tolist(),
                    translation=translations[prefix].tolist(),
                )
        world[entity_path] = transform
    return world, counts


def recorded_link_names(path: Path, actor_name: str) -> dict[str, str]:
    """Map each recorded link entity of the named actor to its link name."""
    recording = EpisodeReader(path)
    actor_path = next(
        sample.entity_path.replace("/data/physics/actors/", "/world/", 1)
        for sample in recording.component_samples("superdex.physics.runtime_name")
        if sample.value[0] == actor_name
    )
    return {
        sample.entity_path: sample.value[0]
        for sample in recording.component_samples("superdex.physics.link_name")
        if sample.entity_path.startswith(f"{actor_path}/links/")
    }


def link_actors(scene: sdp.Scene, robot: sdp.Actor) -> dict[str, sdp.Actor]:
    links = {}
    for handle in robot.get_nested_link_actors():
        link = scene.get_actor(handle)
        assert link is not None
        links[link.get_name().rsplit("/", 1)[-1]] = link
    return links


def link_poses(links: dict[str, sdp.Actor]) -> dict[str, sdp.TransformRT]:
    return {name: link.get_root_transform() for name, link in links.items()}


def converge_each_step(scene: sdp.Scene) -> None:
    # The restore doesn't reproduce every solver input bitwise, so with the default stopping
    # criteria a restored step's solve can stop an iteration apart from the recorded one's.
    params = scene.get_solver_params()
    solver = params.non_linear_solver
    solver.abs_tol = 0.0
    solver.rel_tol = 0.0
    solver.rel_step_tol = 0.0
    solver.max_iter = 20
    params.non_linear_solver = solver
    scene.set_solver_params(params)


def fixed_link_names(robot: sdp.Actor) -> set[str]:
    shape_info = robot.get_articulated_shape_info()
    link_names = list(shape_info.link_names)
    # Cycle joints follow the links' joints.
    joint_types = list(shape_info.joint_types)[: len(link_names)]
    return {
        str(name)
        for name, joint_type in zip(link_names, joint_types, strict=True)
        if joint_type == sdp.ArticulatedJointType.HARD
    }


class LinkRecordingTest(unittest.TestCase):
    def setUp(self) -> None:
        sdp.initialize(num_worker_threads=0)
        self.addCleanup(sdp.shutdown)
        self.addCleanup(gc.collect)
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.episode_recorder = recorder.Recorder(recorder.RecorderConfig(directory))
        self.robotics_context = sdr.create_context()
        self.bot_scene = load_bot_scene(str(BOT_SCENE_PATH), self.robotics_context)
        scene = self.bot_scene.get_scene()
        assert scene is not None
        self.scene = scene
        bot = self.bot_scene.get_bot(ROBOT_NAME)
        assert bot is not None
        self.bot = bot
        robot = bot.get_articulated_actor()
        assert robot is not None
        self.robot = robot
        self.links = link_actors(scene, robot)
        self.addCleanup(self.release_scene)

    def release_scene(self) -> None:
        # Mochi objects must be released before Mochi shuts down.
        del self.links, self.robot, self.bot, self.scene, self.bot_scene
        del self.robotics_context

    def assert_transform_near(
        self, actual: sdp.TransformRT, expected: sdp.TransformRT
    ) -> None:
        np.testing.assert_allclose(
            actual.translation, expected.translation, atol=TOLERANCE
        )
        actual_rotation = np.asarray(actual.rotation.tolist())
        rotation = np.asarray(expected.rotation.tolist())
        # q and -q are the same rotation.
        self.assertLessEqual(
            min(
                np.abs(actual_rotation - rotation).max(),
                np.abs(actual_rotation + rotation).max(),
            ),
            TOLERANCE,
        )

    def assert_links_match(
        self,
        world: dict[str, sdp.TransformRT],
        link_paths: dict[str, str],
        poses: dict[str, sdp.TransformRT],
    ) -> None:
        self.assertEqual(set(link_paths.values()), set(poses))
        for path, name in link_paths.items():
            with self.subTest(link=name):
                self.assert_transform_near(world[path], poses[name])

    def assert_poses_near(
        self, actual: dict[str, sdp.TransformRT], expected: dict[str, sdp.TransformRT]
    ) -> None:
        self.assertEqual(set(actual), set(expected))
        for name, pose in actual.items():
            with self.subTest(link=name):
                self.assert_transform_near(pose, expected[name])

    def create_robot_scene(self, bot_path: str) -> tuple[sdp.Scene, sdp.Actor]:
        scene = sdp.create_scene("robot")
        self.addCleanup(sdp.destroy_scene, scene)
        scene.set_gravity([0, 0, -9.81])
        prefab = sdr.load_bot_prefab_from_file(str(resolve_asset(bot_path)))
        bot = sdr.create_bot(scene, prefab, self.robotics_context)
        assert bot is not None
        self.addCleanup(sdr.destroy_bot, scene, bot)
        robot = bot.get_articulated_actor()
        assert robot is not None
        return scene, robot

    def record_frames(
        self, scene: sdp.Scene, links: dict[str, sdp.Actor]
    ) -> tuple[Path, list[dict[str, sdp.TransformRT]]]:
        """Record frames 0 to NUM_FRAMES, and return the episode's path and Mochi's link
        poses at each frame."""
        episode = self.episode_recorder.begin_episode()
        physics = recorder.PhysicsRecorder(episode, scene)
        poses = []
        for frame in range(NUM_FRAMES + 1):
            if frame > 0:
                scene.step(TIME_STEP)
            physics.record(frame, scene.get_total_simulation_time())
            poses.append(link_poses(links))
        episode.commit()
        return episode.output_path, poses

    def record_tracked_hand(
        self,
    ) -> tuple[Path, list[list[list[float]] | None], dict[str, sdp.TransformRT]]:
        """Record the scene with a command shadow and a tracked hand that repeats
        samples and loses tracking. Return the episode's path, the hand's keypoints at
        each frame, or None where untracked, and Mochi's link poses at the last frame."""
        real_dtype = np.asarray(self.scene.get_gravity()).dtype
        keypoints = np.array(
            [[0.0, 0.0, 0.0], [0.03, 0.02, 0.0], [0.0, 0.04, 0.0], [0.0, 0.08, 0.01]]
        )
        hand_samples = [keypoints, keypoints, None, None, keypoints + 0.01]
        episode = self.episode_recorder.begin_episode()
        physics = recorder.PhysicsRecorder(episode, self.scene)
        command = recorder.RobotShadow(episode, self.bot, "command")
        hand = recorder.SkeletonShadow(
            episode,
            self.bot,
            "tracked_hand",
            KEYPOINT_NAMES,
            KEYPOINT_PARENTS,
            label="tracked hand",
            color_rgba=HAND_COLOR,
            keypoint_radius_meters=HAND_RADIUS,
        )
        pose = np.zeros(self.robot.get_num_dofs(), dtype=real_dtype)
        for frame, sample in enumerate(hand_samples):
            if frame > 0:
                self.scene.step(TIME_STEP)
            time_sec = self.scene.get_total_simulation_time()
            physics.record(frame, time_sec)
            self.robot.get_articulated_pose(pose)
            command.record(frame, time_sec, self.robot.get_root_transform(), pose)
            if sample is None:
                hand.record_untracked(frame, time_sec)
            else:
                hand.record(frame, time_sec, sample.astype(real_dtype).reshape(-1))
        episode.commit()
        return (
            episode.output_path,
            [
                None if sample is None else sample.astype(np.float32).tolist()
                for sample in hand_samples
            ],
            link_poses(self.links),
        )

    def test_robot_links_compose_to_mochi_link_poses(self) -> None:
        episode = self.episode_recorder.begin_episode()
        physics = recorder.PhysicsRecorder(episode, self.scene)
        middle_poses: dict[str, sdp.TransformRT] = {}
        for frame in range(NUM_FRAMES + 1):
            if frame > 0:
                self.scene.step(TIME_STEP)
            physics.record(frame, self.scene.get_total_simulation_time())
            if frame == MIDDLE_FRAME:
                middle_poses = link_poses(self.links)
        episode.commit()

        world, counts = recorded_world_transforms(episode.output_path)
        link_paths = recorded_link_names(episode.output_path, ROBOT_NAME)
        fixed_links = fixed_link_names(self.robot)
        self.assertEqual(set(link_paths.values()), set(self.links))
        self.assertGreater(len(fixed_links), 0)
        # A link that moves is recorded again.
        self.assertTrue(any(counts[path] > 1 for path in link_paths))
        middle_world, _ = recorded_world_transforms(episode.output_path, MIDDLE_FRAME)
        with self.subTest(frame=MIDDLE_FRAME):
            self.assert_links_match(middle_world, link_paths, middle_poses)
        for path, name in link_paths.items():
            with self.subTest(link=name):
                self.assert_transform_near(
                    world[path], self.links[name].get_root_transform()
                )
                # A link on a fixed joint never moves relative to its parent.
                if name in fixed_links:
                    self.assertEqual(counts[path], 1)

        stream = _rr_experimental.RrdReader(episode.output_path).stream()
        static_chunks = [chunk for chunk in stream if chunk.is_static]
        entity_paths = [chunk.entity_path for chunk in static_chunks]
        self.assertCountEqual(entity_paths, set(entity_paths))
        columns = {
            chunk.entity_path: chunk.to_record_batch().schema.names
            for chunk in static_chunks
        }
        palm_path = next(
            path for path, name in link_paths.items() if name == HAND_ROOT_LINK
        )
        for component in (
            "superdex.physics.link_name",
            "Mesh3D:vertex_positions",
            "superdex.physics.contact_params",
        ):
            self.assertIn(component, columns[palm_path])

    def test_robot_shadows_compose_to_mochi_link_poses(self) -> None:
        episode = self.episode_recorder.begin_episode()
        link_names = [
            str(name) for name in self.robot.get_articulated_shape_info().link_names
        ]
        shadows = {
            "full": recorder.RobotShadow(episode, self.bot, "full"),
            "hand": recorder.RobotShadow(
                episode,
                self.bot,
                "hand",
                subtree_root_link_index=link_names.index(HAND_ROOT_LINK),
            ),
        }
        pose = np.zeros(
            self.robot.get_num_dofs(), dtype=np.asarray(self.scene.get_gravity()).dtype
        )
        self.robot.get_articulated_pose(pose)
        time_sec = self.scene.get_total_simulation_time()
        shadows["full"].record(0, time_sec, self.robot.get_root_transform(), pose)
        shadows["hand"].record(
            0, time_sec, self.links[HAND_ROOT_LINK].get_root_transform(), pose
        )
        episode.commit()

        world, _ = recorded_world_transforms(episode.output_path)
        for shadow_id, expected_links in (
            ("full", set(self.links)),
            ("hand", None),
        ):
            prefix = f"/world/shadows/{ROBOT_NAME}/{shadow_id}/links/"
            shadow_links = {
                path.rsplit("/", 1)[-1]: transform
                for path, transform in world.items()
                if path.startswith(prefix)
            }
            if expected_links is not None:
                self.assertEqual(set(shadow_links), expected_links)
            self.assertIn(HAND_ROOT_LINK, shadow_links)
            for name, transform in shadow_links.items():
                with self.subTest(shadow=shadow_id, link=name):
                    self.assert_transform_near(
                        transform, self.links[name].get_root_transform()
                    )

    def test_skeleton_shadows_record_keypoints_and_their_topology(self) -> None:
        path, hand_samples, _ = self.record_tracked_hand()

        rrd_verify.assert_rerun_accepts(path)
        recording = EpisodeReader(path)
        positions = recording.component_samples(
            "Points3D:positions", entity_path=TRACKED_HAND_PATH
        )
        # Repeated samples are skipped, and an untracked frame stores an empty batch.
        self.assertEqual(
            [(sample.timelines["frame"], sample.value) for sample in positions],
            [(0, hand_samples[0]), (2, []), (4, hand_samples[4])],
        )
        # The topology is constant, so it is stored once, in the entity's static row.
        static_chunks = [
            chunk
            for chunk in _rr_experimental.RrdReader(path).stream()
            if chunk.is_static and chunk.entity_path == TRACKED_HAND_PATH
        ]
        [[constants]] = [chunk.to_record_batch().to_pylist() for chunk in static_chunks]
        [skeleton] = constants["AnnotationContext:context"][0]
        description = skeleton["class_description"]
        self.assertEqual(description["info"]["label"], "tracked hand")
        self.assertEqual(
            [
                (keypoint["id"], keypoint["label"])
                for keypoint in description["keypoint_annotations"]
            ],
            list(enumerate(KEYPOINT_NAMES)),
        )
        self.assertEqual(
            [
                (connection["keypoint0"], connection["keypoint1"])
                for connection in description["keypoint_connections"]
            ],
            [(0, 1), (0, 2), (2, 3)],
        )
        self.assertEqual(
            constants["Points3D:class_ids"],
            [skeleton["class_id"]] * len(KEYPOINT_NAMES),
        )
        self.assertEqual(
            constants["Points3D:keypoint_ids"], list(range(len(KEYPOINT_NAMES)))
        )
        self.assertEqual(constants["Points3D:show_labels"], [False])
        np.testing.assert_allclose(constants["Points3D:radii"], [HAND_RADIUS])
        red, green, blue, alpha = HAND_COLOR
        self.assertEqual(
            description["info"]["color"], red << 24 | green << 16 | blue << 8 | alpha
        )
        self.assertEqual(constants["Points3D:colors"], [description["info"]["color"]])
        self.assertNotIn("Points3D:positions", constants)

    def test_skeleton_shadow_record_takes_flat_or_per_keypoint_xyz(self) -> None:
        real_dtype = np.asarray(self.scene.get_gravity()).dtype
        keypoints = np.linspace(0.0, 0.11, 12, dtype=real_dtype).reshape(4, 3)
        episode = self.episode_recorder.begin_episode()
        flat, rows = [
            recorder.SkeletonShadow(
                episode, self.bot, shadow_id, KEYPOINT_NAMES, KEYPOINT_PARENTS
            )
            for shadow_id in ("flat", "rows")
        ]

        flat.record(0, 0.0, keypoints.reshape(-1))
        rows.record(0, 0.0, keypoints)
        with self.assertRaisesRegex(sdp.Error, "must be XYZ triples"):
            rows.record(1, TIME_STEP, np.zeros(10, dtype=real_dtype))
        # As many values as 4 keypoints, but not in rows of XYZ.
        with self.assertRaisesRegex(sdp.Error, "must be XYZ triples"):
            rows.record(1, TIME_STEP, np.zeros((6, 2), dtype=real_dtype))
        with self.assertRaisesRegex(sdp.Error, "one position per keypoint"):
            rows.record(1, TIME_STEP, keypoints[:3])
        episode.commit()

        positions = EpisodeReader(episode.output_path).component_samples(
            "Points3D:positions"
        )
        self.assertEqual(
            {sample.entity_path: sample.value for sample in positions},
            {
                f"/world/shadows/{ROBOT_NAME}/{shadow_id}": keypoints.astype(
                    np.float32
                ).tolist()
                for shadow_id in ("flat", "rows")
            },
        )

    def test_a_stored_24_keypoint_sample_takes_under_400_bytes(self) -> None:
        real_dtype = np.asarray(self.scene.get_gravity()).dtype
        num_keypoints = 24
        episode = self.episode_recorder.begin_episode()
        hand = recorder.SkeletonShadow(
            episode,
            self.bot,
            "tracked_hand",
            [f"keypoint_{index}" for index in range(num_keypoints)],
            list(range(-1, num_keypoints - 1)),
        )
        rng = np.random.default_rng(0)

        for frame in range(NUM_FRAMES):
            hand.record(
                frame,
                frame * TIME_STEP,
                rng.random((num_keypoints, 3)).astype(real_dtype),
            )
        episode.commit()

        batches = [
            chunk.to_record_batch()
            for chunk in _rr_experimental.RrdReader(episode.output_path).stream()
            if not chunk.is_static and chunk.entity_path == TRACKED_HAND_PATH
        ]
        self.assertEqual(sum(batch.num_rows for batch in batches), NUM_FRAMES)
        # Decoded Arrow bytes, not the LZ4-compressed bytes in the file.
        self.assertLess(sum(batch.nbytes for batch in batches) / NUM_FRAMES, 400)

    def test_restoring_and_rebuilding_ignore_skeleton_shadows(self) -> None:
        path, _, last_poses = self.record_tracked_hand()

        rebuilt = recorder.rebuild_scene(
            EpisodeReader(path), BOT_SCENE_PATH, self.robotics_context
        )
        scene = rebuilt.bot_scene.get_scene()
        bot = rebuilt.bot_scene.get_bot(ROBOT_NAME)
        assert scene is not None and bot is not None
        robot = bot.get_articulated_actor()
        assert robot is not None
        restorer = recorder.EpisodeRestorer(EpisodeReader(path), scene)
        restorer.restore_frame(restorer.frames[-1])

        self.assert_poses_near(link_poses(link_actors(scene, robot)), last_poses)

    def test_robot_id_is_the_recorded_physics_actor_name(self) -> None:
        episode = self.episode_recorder.begin_episode()
        physics = recorder.PhysicsRecorder(episode, self.scene)
        robotics = recorder.RoboticsRecorder(episode, self.bot)
        episode.commit()
        del physics, robotics

        recording = EpisodeReader(episode.output_path)
        robot_ids = [
            sample.value[0]
            for sample in recording.component_samples("superdex.robotics.robot_id")
        ]
        prefab_names = [
            sample.value[0]
            for sample in recording.component_samples(
                "superdex.robotics.robot_prefab_name"
            )
        ]
        runtime_names = {
            sample.value[0]
            for sample in recording.component_samples("superdex.physics.runtime_name")
        }
        self.assertEqual(robot_ids, [self.robot.get_name()])
        self.assertIn(self.robot.get_name(), runtime_names)
        self.assertEqual(prefab_names, [self.bot.get_name()])

    def test_robots_with_the_same_name_are_rejected(self) -> None:
        # A different bot given the same name, so its recorded constants differ.
        prefab = sdr.load_bot_prefab_from_file(str(resolve_asset(ROBOTIQ_BOTS[0])))
        prefab.name = self.robot.get_name()
        twin = sdr.create_bot(self.scene, prefab, self.robotics_context)
        assert twin is not None
        self.addCleanup(sdr.destroy_bot, self.scene, twin)
        twin_robot = twin.get_articulated_actor()
        assert twin_robot is not None
        self.assertEqual(twin_robot.get_name(), self.robot.get_name())
        self.assertNotEqual(twin_robot.get_num_dofs(), self.robot.get_num_dofs())
        episode = self.episode_recorder.begin_episode()
        recorder.RoboticsRecorder(episode, self.bot)
        recorder.RobotShadow(episode, self.bot, "full")

        with self.assertRaisesRegex(sdp.Error, "A series already exists"):
            recorder.RoboticsRecorder(episode, twin)
        with self.assertRaisesRegex(sdp.Error, "A series already exists"):
            recorder.RobotShadow(episode, twin, "full")
        episode.commit()

        num_dofs = EpisodeReader(episode.output_path).component_samples(
            "superdex.robotics.num_dofs"
        )
        self.assertEqual(
            [sample.value for sample in num_dofs], [[self.robot.get_num_dofs()]]
        )

    def test_free_root_links_compose_with_a_moved_root_joint(self) -> None:
        knob_bot = self.bot_scene.get_bot(KNOB_NAME)
        assert knob_bot is not None
        knob = knob_bot.get_articulated_actor()
        assert knob is not None
        self.assertEqual(
            knob.get_articulated_shape_info().joint_types[0],
            sdp.ArticulatedJointType.FREE,
        )
        links = link_actors(self.scene, knob)
        spawn_poses = link_poses(links)
        # The free joint's translation [m] and rotation vector [rad].
        knob.set_articulated_pose_from_joints([0.01, -0.02, 0.03, 0.3, -0.2, 0.5])
        poses = link_poses(links)
        self.assertTrue(
            any(
                not np.allclose(pose.translation, spawn_poses[name].translation)
                for name, pose in poses.items()
            )
        )
        episode = self.episode_recorder.begin_episode()
        physics = recorder.PhysicsRecorder(episode, self.scene)
        physics.record(0, self.scene.get_total_simulation_time())
        episode.commit()

        world, _ = recorded_world_transforms(episode.output_path)
        self.assert_links_match(
            world, recorded_link_names(episode.output_path, KNOB_NAME), poses
        )

    def test_robotiq_gripper_links_compose_to_mochi_link_poses(self) -> None:
        for bot_path in ROBOTIQ_BOTS:
            with self.subTest(bot=bot_path):
                scene, robot = self.create_robot_scene(bot_path)
                # Each finger is a closed four-bar linkage.
                self.assertEqual(len(robot.get_articulated_shape_info().cycles), 2)
                episode_path, poses = self.record_frames(
                    scene, link_actors(scene, robot)
                )

                link_paths = recorded_link_names(episode_path, robot.get_name())
                for frame in (MIDDLE_FRAME, NUM_FRAMES):
                    world, _ = recorded_world_transforms(episode_path, frame)
                    with self.subTest(frame=frame):
                        self.assert_links_match(world, link_paths, poses[frame])
                _, counts = recorded_world_transforms(episode_path)
                fixed_links = fixed_link_names(robot)
                self.assertGreater(len(fixed_links), 0)
                self.assertEqual(
                    {
                        name: counts[path]
                        for path, name in link_paths.items()
                        if name in fixed_links
                    },
                    dict.fromkeys(fixed_links, 1),
                )
                self.assertTrue(any(counts[path] > 1 for path in link_paths))

    def test_robotiq_gripper_restores_its_recorded_frames(self) -> None:
        for bot_path in ROBOTIQ_BOTS:
            with self.subTest(bot=bot_path):
                scene, robot = self.create_robot_scene(bot_path)
                converge_each_step(scene)
                episode_path, poses = self.record_frames(
                    scene, link_actors(scene, robot)
                )

                restored_scene, restored_robot = self.create_robot_scene(bot_path)
                converge_each_step(restored_scene)
                restored_links = link_actors(restored_scene, restored_robot)
                restorer = recorder.EpisodeRestorer(
                    EpisodeReader(episode_path), restored_scene
                )
                for frame in (MIDDLE_FRAME, NUM_FRAMES):
                    restorer.restore_frame(frame)
                    with self.subTest(frame=frame):
                        self.assert_poses_near(link_poses(restored_links), poses[frame])
                # The restored velocities reproduce the next step.
                restorer.restore_frame(MIDDLE_FRAME)
                restored_scene.step(TIME_STEP)
                with self.subTest(frame=MIDDLE_FRAME + 1):
                    self.assert_poses_near(
                        link_poses(restored_links), poses[MIDDLE_FRAME + 1]
                    )
