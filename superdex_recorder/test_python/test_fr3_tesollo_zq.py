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
from typing import NamedTuple

import superdex.physics as sdp
import superdex.recorder as recorder
import superdex.robotics as sdr
from superdex.physics.paths import resolve_asset
from superdex.physics.utils.scene_helpers import find_actor, load_bot_scene
from superdex.recorder.reader import EpisodeReader


BOT_SCENE = "bots/internal/tests/zq_fr3_tesollo/fr3_tesollo_zq.mochi_bot_scene"
CHECKPOINT_FRAMES = (40, 60, 120, 180, 240, 300)
RESUME_START_FRAME = 180
RESUME_END_FRAME = 240
# A grasp resumed from a checkpoint stays this close to the live run: the block's
# translation [m] and the joint positions. Without the restored velocities, each check
# fails. Restoring sets the pose controller's target velocity to the restored joint
# velocities, so the first resumed step extrapolates its old target from them instead
# of using the live previous target. Checked with backward Euler only: a restore starts
# BDF2 and BDF3 afresh, which costs them up to about 0.7 mm here.
BLOCK_RESUME_TOLERANCE = 2e-4
JOINT_RESUME_TOLERANCE = 5e-4


class LoadedScene(NamedTuple):
    robotics_context: sdr.RoboticsContext
    bot_scene: sdr.BotScene
    scene: sdp.Scene
    actor: sdp.Actor
    block: sdp.Actor
    controller: sdr.ControllerMochiArticulatedPose


class Rollout(NamedTuple):
    path: Path
    poses: tuple[list[float], list[float], list[float]]
    heights: dict[int, float]
    block_translations: dict[int, list[float]]
    joint_positions: dict[int, list[float]]


def max_error(actual: list[float], expected: list[float]) -> float:
    return max(abs(a - e) for a, e in zip(actual, expected, strict=True))


def create_pose_controller(bot: sdr.Bot) -> sdr.ControllerMochiArticulatedPose:
    tracking = sdp.PoseTrackingParams(stiffness=1000, damping=100)
    controller = bot.create_controller("MOCHI_ARTICULATED_POSE")
    if not isinstance(controller, sdr.ControllerMochiArticulatedPose):
        raise RuntimeError("Failed to create the articulated pose controller.")
    controller.set_params(
        sdr.ControllerMochiArticulatedPoseParams(
            pose_controller_params=sdp.PoseControllerParams(
                joint_tracking=sdp.DynamicArrayPoseTrackingParams([tracking])
            )
        )
    )
    controller.initialize(True)
    return controller


def load_test_scene(method: sdp.IntegrationMethod) -> LoadedScene:
    robotics_context = sdr.create_context()
    bot_scene = load_bot_scene(str(resolve_asset(BOT_SCENE)), robotics_context)
    scene = bot_scene.get_scene()
    bot = bot_scene.get_bot("robot")
    actor = bot.get_articulated_actor() if bot is not None else None
    if scene is None or bot is None or actor is None:
        raise RuntimeError("FR3 and Tesollo test bot was not loaded.")
    solver_params = scene.get_solver_params()
    solver_params.integration_method = method
    scene.set_solver_params(solver_params)
    return LoadedScene(
        robotics_context,
        bot_scene,
        scene,
        actor,
        find_actor(scene, "*PickupBlock"),
        create_pose_controller(bot),
    )


def get_trajectory_poses(
    actor: sdp.Actor,
) -> tuple[list[float], list[float], list[float]]:
    pose = sdp.DynamicArrayReal(actor.get_num_dofs())
    actor.get_articulated_pose(pose)
    open_pose = list(pose)
    closed_pose = open_pose[:7] + [
        0.5,
        -1.2,
        0.9,
        0.9,
        -0.1,
        1.0,
        1.0,
        1.0,
        -0.03,
        1.0,
        1.0,
        1.0,
        0.03,
        1.0,
        1.0,
        1.0,
        0.08,
        1.0,
        1.0,
        1.0,
    ]
    lifted_pose = closed_pose.copy()
    lifted_pose[3] += 0.25
    return open_pose, closed_pose, lifted_pose


def interpolate_pose(start: list[float], end: list[float], alpha: float) -> list[float]:
    return [(1 - alpha) * a + alpha * b for a, b in zip(start, end)]


def trajectory_target(
    frame: int,
    open_pose: list[float],
    closed_pose: list[float],
    lifted_pose: list[float],
) -> list[float]:
    if frame <= 30:
        return interpolate_pose(open_pose, closed_pose, frame / 30)
    if frame <= 60:
        return closed_pose
    return interpolate_pose(closed_pose, lifted_pose, min(1.0, (frame - 60) / 180))


def get_joint_positions(actor: sdp.Actor) -> list[float]:
    positions = sdp.DynamicArrayReal(actor.get_num_dofs())
    actor.get_articulated_pose(positions)
    return list(positions)


class Fr3TesolloZqTest(unittest.TestCase):
    def record_rollout(self, loaded: LoadedScene, directory: Path) -> Rollout:
        poses = get_trajectory_poses(loaded.actor)
        observation = sdr.ControllerMochiArticulatedPoseObsv()
        root = loaded.actor.get_root_transform()
        episode = recorder.Recorder(recorder.RecorderConfig(directory)).begin_episode()
        physics = recorder.PhysicsRecorder(episode, loaded.scene)
        physics.record(0, loaded.scene.get_total_simulation_time())
        heights = {}
        block_translations = {}
        joint_positions = {}
        for frame in range(1, 301):
            loaded.controller.compute_output(
                observation,
                sdr.ControllerMochiArticulatedPoseTarget(
                    world_from_root=root, pose_dofs=trajectory_target(frame, *poses)
                ),
            )
            loaded.scene.step(0.005)
            physics.record(frame, loaded.scene.get_total_simulation_time())
            if frame in CHECKPOINT_FRAMES:
                heights[frame] = loaded.block.get_root_transform().translation[2]
            block_translations[frame] = (
                loaded.block.get_root_transform().translation.tolist()
            )
            joint_positions[frame] = get_joint_positions(loaded.actor)
        episode.commit_sync()
        return Rollout(
            episode.output_path, poses, heights, block_translations, joint_positions
        )

    def resume_errors(
        self,
        loaded: LoadedScene,
        restorer: recorder.EpisodeRestorer,
        rollout: Rollout,
        zero_velocity: bool,
    ) -> tuple[float, float]:
        """The largest block translation and joint position errors over the resume."""
        restorer.restore_frame(RESUME_START_FRAME)
        if zero_velocity:
            loaded.actor.set_articulated_joint_velocities(
                [0.0] * loaded.actor.get_num_dofs()
            )
            loaded.block.set_velocity([0, 0, 0], [0, 0, 0])
        root = loaded.actor.get_root_transform()
        block_error = joint_error = 0.0
        for frame in range(RESUME_START_FRAME + 1, RESUME_END_FRAME + 1):
            loaded.controller.compute_output(
                sdr.ControllerMochiArticulatedPoseObsv(),
                sdr.ControllerMochiArticulatedPoseTarget(
                    world_from_root=root,
                    pose_dofs=trajectory_target(frame, *rollout.poses),
                ),
            )
            loaded.scene.step(0.005)
            block_error = max(
                block_error,
                max_error(
                    loaded.block.get_root_transform().translation.tolist(),
                    rollout.block_translations[frame],
                ),
            )
            joint_error = max(
                joint_error,
                max_error(
                    get_joint_positions(loaded.actor), rollout.joint_positions[frame]
                ),
            )
        return block_error, joint_error

    def test_grasp_with_backward_euler(self) -> None:
        self.assert_grasp_restores(sdp.IntegrationMethod.BACKWARD_EULER)

    def test_grasp_with_bdf2(self) -> None:
        self.assert_grasp_restores(sdp.IntegrationMethod.BDF2)

    def test_grasp_with_bdf3(self) -> None:
        self.assert_grasp_restores(sdp.IntegrationMethod.BDF3)

    def test_resumed_grasp_with_backward_euler(self) -> None:
        self.assert_grasp_resumes(sdp.IntegrationMethod.BACKWARD_EULER)

    def test_robotics_record_fails_after_the_robot_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            sdp.initialize(num_worker_threads=0)
            loaded = None
            controllers = None
            try:
                loaded = load_test_scene(sdp.IntegrationMethod.BACKWARD_EULER)
                bot = loaded.bot_scene.get_bot("robot")
                assert bot is not None
                with recorder.Recorder(
                    recorder.RecorderConfig(Path(directory_string))
                ).begin_episode() as episode:
                    controllers = recorder.RoboticsRecorder(episode, bot)
                    controllers.record(0, 0.0)

                    loaded.bot_scene.remove_bot("robot")

                    with self.assertRaisesRegex(sdp.Error, "destroyed"):
                        controllers.record(1, 0.005)
            finally:
                controllers = None
                loaded = None
                gc.collect()
                sdp.shutdown()

    def assert_grasp_restores(self, method: sdp.IntegrationMethod) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            sdp.initialize(num_worker_threads=0)
            source = None
            reconstructed = None
            try:
                source = load_test_scene(method)
                rollout = self.record_rollout(source, Path(directory_string))
                reconstructed = load_test_scene(method)
                restorer = recorder.EpisodeRestorer(
                    EpisodeReader(rollout.path), reconstructed.scene
                )
                for checkpoint in CHECKPOINT_FRAMES:
                    restorer.restore_frame(checkpoint)
                    self.assertAlmostEqual(
                        reconstructed.block.get_root_transform().translation[2],
                        rollout.heights[checkpoint],
                        delta=1e-6,
                    )
            finally:
                source = None
                reconstructed = None
                gc.collect()
                sdp.shutdown()

    def assert_grasp_resumes(self, method: sdp.IntegrationMethod) -> None:
        with tempfile.TemporaryDirectory() as directory_string:
            sdp.initialize(num_worker_threads=0)
            source = None
            reconstructed = None
            try:
                source = load_test_scene(method)
                rollout = self.record_rollout(source, Path(directory_string))
                self.assertGreater(rollout.heights[300], rollout.heights[60] + 0.1)
                reconstructed = load_test_scene(method)
                restorer = recorder.EpisodeRestorer(
                    EpisodeReader(rollout.path), reconstructed.scene
                )
                block_error, joint_error = self.resume_errors(
                    reconstructed, restorer, rollout, zero_velocity=False
                )
                self.assertLess(block_error, BLOCK_RESUME_TOLERANCE)
                self.assertLess(joint_error, JOINT_RESUME_TOLERANCE)
                block_error, joint_error = self.resume_errors(
                    reconstructed, restorer, rollout, zero_velocity=True
                )
                self.assertGreater(block_error, BLOCK_RESUME_TOLERANCE)
                self.assertGreater(joint_error, JOINT_RESUME_TOLERANCE)
            finally:
                source = None
                reconstructed = None
                gc.collect()
                sdp.shutdown()
