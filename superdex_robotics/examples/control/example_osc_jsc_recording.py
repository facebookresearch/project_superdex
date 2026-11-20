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

r"""Record the combined OSC and JSC control example to one self-contained RRD.

Usage:
    cd <path_to_superdex_robotics>
    python3 examples/control/example_osc_jsc_recording.py \
        --output-dir /tmp/superdex_recordings --steps 800
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt
import superdex.physics as sdp
import superdex.recorder as sdrec
import superdex.robotics as sdr
from superdex.physics.paths import resolve_asset

# OSC controls the arm chain ending at the hand mounting flange.
ARM_BASE_LINK = "fr3_link0"
ARM_EE_LINK = "fr3_link8"
# This prefix identifies the arm DOFs masked out of the JSC output.
ARM_JOINT_PREFIX = "fr3_joint"
# The non-thumb knuckles; finger 1 is the thumb.
KNUCKLE_JOINTS = (
    "dg5f_joint_2_2",
    "dg5f_joint_3_2",
    "dg5f_joint_4_2",
    "dg5f_joint_5_3",
)
COMMAND_SHADOW_COLOR = (77, 191, 242, 115)

GROUND_COORDINATES = (
    (-2.0, -2.0, -0.05),
    (2.0, -2.0, -0.05),
    (2.0, 2.0, -0.05),
    (-2.0, 2.0, -0.05),
    (-2.0, -2.0, 0.0),
    (2.0, -2.0, 0.0),
    (2.0, 2.0, 0.0),
    (-2.0, 2.0, 0.0),
)
GROUND_CONNECTIVITY = (
    (0, 2, 1),
    (0, 3, 2),
    (4, 5, 6),
    (4, 6, 7),
    (0, 1, 5),
    (0, 5, 4),
    (1, 2, 6),
    (1, 6, 5),
    (2, 3, 7),
    (2, 7, 6),
    (3, 0, 4),
    (3, 4, 7),
)


def get_default_bot_path() -> str:
    """Resolve the default FR3 and DG5F-short bot asset."""
    return str(
        resolve_asset(
            "bots/arm_hand_combos/fr3_dg5f_short/right/fr3_dg5f_short_right.superdex_bot"
        )
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "recordings",
        help="Directory in which to atomically publish the committed RRD.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=800,
        help="Number of 200 Hz control transitions to record.",
    )
    parser.add_argument(
        "--debugger",
        action="store_true",
        help="Also attach the SuperDex Physics Debugger while recording.",
    )
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    return args


def _spawn_robot(
    scene: sdp.Scene,
    robotics_context: sdr.RoboticsContext,
    bot_path: str,
) -> tuple[sdr.Bot, sdr.BotPrefab, sdp.Actor]:
    """Load and instantiate the example robot."""
    bot_prefab = sdr.load_bot_prefab_from_file(bot_path)
    # Neither controller compensates for gravity, so disable it before spawning.
    for index in range(len(bot_prefab.links)):
        bot_prefab.links[index].has_gravity = False

    bot = sdr.create_bot(scene, bot_prefab, robotics_context)
    if bot is None:
        raise RuntimeError("Failed to create the example robot.")
    bot_actor = bot.get_articulated_actor()
    if bot_actor is None:
        sdr.destroy_bot(scene, bot)
        raise RuntimeError("The example robot has no articulated actor.")
    return bot, bot_prefab, bot_actor


def _controlled_dofs(
    bot_prefab: sdr.BotPrefab,
    num_root_dofs: int,
) -> tuple[npt.NDArray[np.int32], list[int]]:
    """Map arm and knuckle joints into actor DOF indices."""
    # Bot DOF space follows the prefab's joint order, skipping the root joint, and every
    # moving joint on this bot is a 1-DoF revolute joint, so the indices run consecutively.
    arm_dofs: list[int] = []
    joint_name_to_dof: dict[str, int] = {}
    dof = 0
    for joint in bot_prefab.joints:
        if joint.type != sdp.ArticulatedJointType.REVOLUTE:
            continue
        joint_name_to_dof[joint.name] = dof
        if joint.name.startswith(ARM_JOINT_PREFIX):
            arm_dofs.append(dof)
        dof += 1

    return (
        np.array(arm_dofs, dtype=np.int32) + num_root_dofs,
        [joint_name_to_dof[name] + num_root_dofs for name in KNUCKLE_JOINTS],
    )


def run_initialized(args: argparse.Namespace, bot_path: str) -> Path:
    """Run and record the example while the process-wide physics runtime is active."""
    np_real = np.float64 if sdp.uses_double_precision() else np.float32

    scene = sdp.create_scene("Recorded OSC + JSC Control Example")
    scene.set_gravity([0, 0, -9.81])

    robotics_context = sdr.create_context()
    bot, bot_prefab, bot_actor = _spawn_robot(scene, robotics_context, bot_path)
    try:
        ground_shape = sdp.create_tri_mesh_shape(
            coordinates=np.asarray(GROUND_COORDINATES, dtype=np.float32).flatten(),
            connectivity=np.asarray(GROUND_CONNECTIVITY, dtype=np.int32).flatten(),
        )
        ground = scene.create_rigid_actor(
            name="ground", shape=ground_shape, is_static=True
        )
        if ground is None:
            raise RuntimeError("Failed to create the example ground actor.")

        num_dofs = bot_actor.get_num_dofs()
        all_dof_indices = np.arange(num_dofs, dtype=np.int32)
        num_root_dofs = bot_actor.get_articulated_shape_info().dof_info[0].get_size()
        arm_dof_indices, knuckle_dofs = _controlled_dofs(bot_prefab, num_root_dofs)

        osc = bot.create_controller("BASIC_OSC_PD")
        if not isinstance(osc, sdr.ControllerBasicOscPd):
            raise RuntimeError("Failed to create the OSC controller.")
        bot_name = bot.get_name()
        osc.initialize(f"{bot_name}/{ARM_BASE_LINK}", f"{bot_name}/{ARM_EE_LINK}")
        osc_params = osc.get_params()
        osc_params.kp_p = 900.0
        osc_params.kd_p = 75.0
        osc_params.kp_r = 30.0
        osc_params.kd_r = 3.0
        osc_params.max_translation_error = 0.05
        osc_params.max_rotation_error = 0.4
        osc_params.b_apply_max_osc_torque_normalization = True
        osc.set_params(osc_params)

        jsc = bot.create_controller("BASIC_JSC_PD")
        if not isinstance(jsc, sdr.ControllerBasicJscPd):
            raise RuntimeError("Failed to create the JSC controller.")
        jsc_params = sdr.ControllerBasicJscPdParams()
        jsc_params.kp = np.full(num_dofs, 3.0, dtype=np_real)
        jsc_params.kd = np.full(num_dofs, 0.2, dtype=np_real)
        jsc_params.saturation = np.full(num_dofs, 2.0, dtype=np_real)
        jsc_params.deadband = np.zeros(num_dofs, dtype=np_real)
        jsc.set_params(jsc_params)

        default_pose = sdp.DynamicArrayReal(num_dofs)
        bot_actor.get_articulated_pose(default_pose)
        hold_pose = np.array(default_pose, dtype=np_real)
        # The FR3 root is fixed by a Hard weld, so this transform is constant.
        world_from_root = osc.get_current_observations_from_mochi().world_from_root

        root_position = np.asarray(world_from_root.translation, dtype=float)
        circle_center = np.array([root_position[0] + 0.5, root_position[1], 0.45])
        circle_radius = 0.12
        circle_period = 4.0
        ee_down = sdp.Quaternion.rotation_x(np.pi)
        sweep_period = 2.0
        sweep_mid = np.radians(30.0)
        sweep_amplitude = np.radians(30.0)
        finger_phase_offset = np.radians(30.0)
        time_step = 1.0 / 200.0

        # Must come before attach(), which starts the debug server.
        sdp.get_debug_server().set_coordinate_space(
            sdp.CoordinateSpace(axes=sdp.CoordinateSpaceAxes.FLU)
        )
        if args.debugger:
            sdp.debugger.attach()

        recorder = sdrec.Recorder(sdrec.RecorderConfig(args.output_dir))
        metadata = sdrec.EpisodeMetadata(
            properties={
                "description": "OSC + JSC control",
                "task": "superdex_robotics.example_osc_jsc_control",
                "bot_asset": bot_path,
                "control_frequency_hz": "200",
            },
        )

        with recorder.begin_episode(metadata) as episode:
            physics = sdrec.PhysicsRecorder(episode, scene)
            controls = sdrec.RoboticsRecorder(episode, bot)
            command_shadow = sdrec.RobotShadow(
                episode,
                bot,
                "command",
                osc.get_ee_link_index(),
                COMMAND_SHADOW_COLOR,
            )
            physics.record(0, scene.get_total_simulation_time())

            for frame in range(args.steps):
                t = scene.get_total_simulation_time()

                theta = 2.0 * np.pi * t / circle_period
                world_from_target_ee = sdp.TransformRT()
                world_from_target_ee.translation = [
                    circle_center[0] + circle_radius * np.cos(theta),
                    circle_center[1] + circle_radius * np.sin(theta),
                    circle_center[2],
                ]
                world_from_target_ee.rotation = ee_down
                osc_target = sdr.ControllerBasicOscPdTarget(
                    root_from_target_ee=world_from_root.inverse() * world_from_target_ee
                )

                target_pose = hold_pose.copy()
                for finger, knuckle_dof in enumerate(knuckle_dofs):
                    target_pose[knuckle_dof] = sweep_mid + sweep_amplitude * np.sin(
                        2.0 * np.pi * t / sweep_period + finger * finger_phase_offset
                    )
                jsc_target = sdr.ControllerBasicJscPdTarget(target_pose=target_pose)

                arm_tau = np.array(
                    osc.compute_output(
                        osc.get_current_observations_from_mochi(), osc_target
                    ),
                    dtype=np_real,
                )
                jsc_observation = jsc.get_current_observations_from_mochi()
                jsc_observation.dt = time_step
                hand_tau = np.array(
                    jsc.compute_output(jsc_observation, jsc_target), dtype=np_real
                )
                # Each controller emits all DOFs; apply OSC to the arm and JSC to the hand.
                hand_tau[arm_dof_indices] = 0.0
                applied_effort = arm_tau + hand_tau
                command_shadow.record(
                    frame,
                    t,
                    world_from_target_ee * osc_params.ee_link_from_ee.inverse(),
                    target_pose,
                )
                bot_actor.set_external_forces_on_dofs(
                    dof_indices=all_dof_indices,
                    force_values=applied_effort,
                )
                # Records the forces just set as this frame's applied effort.
                controls.record(frame, t)
                scene.step(time_step)
                physics.record(frame + 1, scene.get_total_simulation_time())

                if args.debugger:
                    scene.update_debugger()

            episode.commit_sync()
            return episode.output_path
    finally:
        sdr.destroy_bot(scene, bot)


def main() -> None:
    """Run the OSC and JSC controller pair and commit one episode RRD."""
    args = parse_args()
    bot_path = get_default_bot_path()
    sdp.initialize(num_worker_threads=0)
    try:
        output_path = run_initialized(args, bot_path)
    finally:
        sdp.shutdown()

    print(f"Recorded {args.steps} transitions to {output_path}")


if __name__ == "__main__":
    main()
