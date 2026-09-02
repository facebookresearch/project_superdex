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

"""Example: Bimanual OSC control tracing mirrored circles

Runs one operational-space PD (OSC) controller per arm on an OpenArm V2.0: a
fixed torso with a 7-DoF arm and a two-finger gripper on each side. Each gripper
traces a vertical circle around its own starting position, and the two circles
are swept in opposite directions, so the hands mirror each other. Meanwhile a
joint-space PD (JSC) pinches all four fingers open and closed, five times per
arm revolution.

Driving two arms needs no coordination machinery: the two OSC chains are
disjoint, one arm each, so every controller gets its own target and its own
torque vector and the vectors simply add. The JSC spans the whole actor, so only
its finger entries are kept.

Usage:
    cd <path_to_superdex_robotics>
    python3 examples/control/example_bimanual_control.py
"""

import numpy as np
import superdex.physics as sdp
import superdex.robotics as sdr

# The build's `real` type. A pose handed to a controller Target is copied into the
# Target's own storage, so matching the dtype here keeps that a straight copy rather
# than an element-by-element conversion.
np_real = np.float64 if sdp.uses_double_precision() else np.float32
from superdex.physics.paths import resolve_asset

# Each OSC controller acts on the chain of joints between these two links. The
# base is the arm's mount on the torso and the end-effector is the gripper's root,
# so the chain spans exactly that arm's seven revolute joints. A bot's link actors
# are named "<bot_name>/<link_name>", so we prefix these at runtime.
SIDES = ("left", "right")
ARM_BASE_LINK = "openarm_{side}_base_link"
ARM_EE_LINK = "openarm_{side}_ee_base_link"

# The two gripper finger joints per side are named "openarm_<side>_finger_joint<n>".
# They are the only DOFs outside the two OSC chains.
FINGER_JOINT_TOKEN = "finger_joint"

# Fraction of each finger joint's travel to sweep, so the pinch stops just short
# of the hard stops.
FINGER_TRAVEL_FRACTION = 0.9


def get_default_bot_path() -> str:
    """Resolve path to the default OpenArm V2.0 .superdex_bot file."""
    return str(
        resolve_asset("bots/arm_hand_combos/openarm_v20/openarm_v20.superdex_bot")
    )


def main() -> None:
    """Load a two-armed bot and trace a mirrored pair of circles with the grippers."""
    bot_path = get_default_bot_path()

    # Initialize the physics engine before creating scenes or actors.
    # num_worker_threads=0 runs single-threaded; pass -1 to auto-select.
    sdp.initialize(num_worker_threads=0)

    # Create an empty scene. SuperDex robots use a Z-up convention, so gravity
    # points down the -Z axis.
    scene = sdp.create_scene("Bimanual Control Example")
    scene.set_gravity([0, 0, -9.81])

    bot_prefab = sdr.load_bot_prefab_from_file(bot_path)

    # Cheap "gravity compensation": disable gravity on every link before
    # spawning. Neither controller has a gravity term, so the arms would
    # otherwise sag off their targets.
    for i in range(len(bot_prefab.links)):
        bot_prefab.links[i].has_gravity = False

    # Find the finger DOFs and how far each one may travel. This numbers them in bot
    # DOF space: the prefab's joint order, skipping the root joint, and every moving
    # joint on this bot is a 1-DoF revolute joint, so the indices run consecutively.
    # Joint limits are stored per axis, so we read the component the joint actually
    # rotates about.
    finger_dofs = []
    finger_travel = []
    dof = 0
    for i in range(len(bot_prefab.joints)):
        joint = bot_prefab.joints[i]
        if joint.type != sdp.ArticulatedJointType.REVOLUTE:
            continue
        if FINGER_JOINT_TOKEN in joint.name:
            axis = int(np.argmax(np.abs(np.asarray(joint.axis, dtype=float))))
            finger_dofs.append(dof)
            finger_travel.append(
                max(
                    abs(float(joint.min_limit[axis])), abs(float(joint.max_limit[axis]))
                )
            )
        dof += 1
    finger_travel = np.array(finger_travel, dtype=np.float32)

    # Instantiate the prefab as a live Bot. The robotics context tracks every
    # bot and controller you create.
    robotics_context = sdr.create_context()
    bot = sdr.create_bot(scene, bot_prefab, robotics_context)
    bot_actor = bot.get_articulated_actor()

    # Add a static ground plane for the robot to rest on (normal points up, +Z).
    plane_shape = sdp.create_plane_shape(normal=[0, 0, 1], distance=0)
    scene.create_rigid_actor(name="ground", shape=plane_shape, is_static=True)

    num_dofs = bot_actor.get_num_dofs()
    all_dof_indices = np.arange(num_dofs, dtype=np.int32)

    # The loop above numbered the fingers in bot DOF space, which never includes the
    # root joint's DOFs. Everything from here on indexes the actor, where the root's
    # DOFs come first, so shift the indices across that gap. This torso is welded to
    # the world and contributes none, but reading the count off the actor keeps the
    # mapping correct for a bot on a free-floating base, which contributes six.
    num_root_dofs = bot_actor.get_articulated_shape_info().dof_info[0].get_size()
    finger_dof_indices = np.array(finger_dofs, dtype=np.int32) + num_root_dofs

    # --- One OSC controller per arm --------------------------------------------
    # initialize() resolves the base/end-effector links by name and figures out
    # which DOFs lie between them. The two chains are disjoint, so the two
    # controllers never contend for a DOF.
    bot_name = bot.get_name()
    arm_oscs = []
    for side in SIDES:
        osc = bot.create_controller("BASIC_OSC_PD")
        osc.initialize(
            f"{bot_name}/{ARM_BASE_LINK.format(side=side)}",
            f"{bot_name}/{ARM_EE_LINK.format(side=side)}",
        )
        osc_params = osc.get_params()
        osc_params.kp_p = 900.0
        osc_params.kd_p = 75.0
        osc_params.kp_r = 5.0
        osc_params.kd_r = 0.05
        osc_params.max_translation_error = 0.05  # [m]
        osc_params.max_rotation_error = 0.4  # [rad]
        osc_params.b_apply_max_osc_torque_normalization = True
        osc.set_params(osc_params)
        arm_oscs.append(osc)

    # --- JSC controller pinching the gripper fingers ---------------------------
    # JSC has no notion of a sub-chain: its gains, its target pose and its output
    # are all sized to the full actor. We only harvest its finger entries below.
    jsc = bot.create_controller("BASIC_JSC_PD")
    jsc_params = sdr.ControllerBasicJscPdParams()
    jsc_params.kp = np.full(num_dofs, 0.5, dtype=np.float32)  # position gain [Nm/rad]
    jsc_params.kd = np.full(num_dofs, 0.005, dtype=np.float32)  # damping gain [Nms/rad]
    jsc_params.saturation = np.full(num_dofs, 0.5, dtype=np.float32)  # torque clamp
    jsc_params.deadband = np.zeros(num_dofs, dtype=np.float32)
    jsc.set_params(jsc_params)

    default_pose = sdp.DynamicArrayReal(num_dofs)
    bot_actor.get_articulated_pose(default_pose)
    hold_pose = np.array(default_pose, dtype=np_real)

    # Each finger swings between fully closed (0) and its open limit. The limits
    # record only the magnitude of the travel, so the direction comes from the pose
    # the gripper spawned in: both fingers of a hand share a sign there, which is
    # what closes them symmetrically onto each other.
    finger_open = (
        np.sign(hold_pose[finger_dof_indices]) * FINGER_TRAVEL_FRACTION * finger_travel
    )

    # --- Circle targets --------------------------------------------------------
    # The torso is welded to the world, so world_from_root is constant and can be
    # captured once to convert world-frame targets into the root frame OSC expects.
    start_obsv = [osc.get_current_observations_from_mochi() for osc in arm_oscs]
    root_from_world = start_obsv[0].world_from_root.inverse()

    # Each circle is centered on its own gripper's spawn pose and the orientation
    # is held at its spawn value, so the arms only have to translate.
    ee_start_positions = [
        np.asarray(obsv.world_from_ee_link.translation, dtype=float)
        for obsv in start_obsv
    ]
    ee_start_rotations = [obsv.world_from_ee_link.rotation for obsv in start_obsv]

    # The circles lie in the vertical plane spanned by world Y (across) and Z
    # (up), centered on each gripper's spawn position. Sweeping the two arms in
    # opposite directions mirrors them: the hands rise and fall together while
    # swinging apart and back together.
    circle_radius = 0.12  # [m]
    circle_period = 4.0  # [s] per revolution
    sweep_directions = (1.0, -1.0)

    # The pinch runs five times per arm revolution.
    pinch_period = 0.8  # [s] per open-close cycle

    # Simulation time step in seconds.
    time_step = 1.0 / 200.0

    # Declare the scene's coordinate convention so the debugger renders it the
    # right way up: SuperDex is X-forward, Y-left, Z-up (FLU). Must come before
    # attach(), which starts the server.
    sdp.get_debug_server().set_coordinate_space(
        sdp.CoordinateSpace(axes=sdp.CoordinateSpaceAxes.FLU)
    )

    # Launch and connect the SuperDex Physics Debugger, a separate desktop app for
    # viewing and interacting with the simulation. The loop runs until you close
    # the debugger; attach() returns False if it can't connect.
    if sdp.debugger.attach():
        while sdp.debugger.is_attached():
            t = scene.get_total_simulation_time()

            # Pinch: sweep the fingers closed -> open -> closed. The JSC target is
            # full length, so start from the spawn pose and overwrite the fingers.
            target_pose = np.array(hold_pose, dtype=np_real)
            target_pose[finger_dof_indices] = (
                0.5 * finger_open * (1.0 - np.cos(2.0 * np.pi * t / pinch_period))
            )

            # The JSC output spans every DOF, so keep only its finger entries --
            # the arm DOFs belong to the OSCs.
            jsc_obsv = jsc.get_current_observations_from_mochi()
            jsc_obsv.dt = time_step
            jsc_tau = np.asarray(
                jsc.compute_output(
                    jsc_obsv,
                    sdr.ControllerBasicJscPdTarget(target_pose=target_pose),
                ),
                dtype=np.float32,
            )
            tau = np.zeros(num_dofs, dtype=np.float32)
            tau[finger_dof_indices] = jsc_tau[finger_dof_indices]

            # Each OSC output is also full length but is already zero outside its
            # own arm chain, so the three contributions just add up.
            for osc, start_pos, start_rot, direction in zip(
                arm_oscs, ee_start_positions, ee_start_rotations, sweep_directions
            ):
                theta = direction * 2.0 * np.pi * t / circle_period
                world_from_target_ee = sdp.TransformRT(
                    rotation=start_rot,
                    translation=[
                        start_pos[0],
                        start_pos[1] + circle_radius * np.sin(theta),
                        start_pos[2] + circle_radius * np.cos(theta),
                    ],
                )
                tau += np.asarray(
                    osc.compute_output(
                        osc.get_current_observations_from_mochi(),
                        sdr.ControllerBasicOscPdTarget(
                            root_from_target_ee=root_from_world * world_from_target_ee
                        ),
                    ),
                    dtype=np.float32,
                )

            bot_actor.set_external_forces_on_dofs(
                dof_indices=all_dof_indices,
                force_values=tau,
            )
            scene.step(time_step)

    # Tear down: destroy the bot, then shut the engine down cleanly.
    sdr.destroy_bot(scene, bot)
    sdp.shutdown()
    print("Simulation complete.")


if __name__ == "__main__":
    main()
