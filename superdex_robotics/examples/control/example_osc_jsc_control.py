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

"""Example: Combining OSC and JSC control on a single bot

Runs the OSC and JSC examples at the same time on one arm-hand combo: an FR3 arm
with a DG5F short hand at its tip. OSC tracks a planar circle with the wrist
while JSC waves the hand's knuckles.

Two controllers on one actor means two torque vectors on one actor. Both
controllers return a vector sized to the *whole* actor, so combining them is a
sum -- but only if each contributes zero on the DOFs it does not own. OSC does
that for free: it zeros every DOF outside the base-to-end-effector chain. JSC
does not; it spans every DOF, so we zero the arm entries of its output by hand
before summing. Each step we build both targets, compute both torques, mask and
sum them, apply the result, and step the simulation.

Usage:
    cd <path_to_superdex_robotics>
    python3 examples/control/example_osc_jsc_control.py
"""

import numpy as np
import superdex.physics as sdp
import superdex.robotics as sdr

# The build's `real` type. A pose handed to a controller Target is copied into the
# Target's own storage, so matching the dtype here keeps that a straight copy rather
# than an element-by-element conversion.
np_real = np.float64 if sdp.uses_double_precision() else np.float32
from superdex.physics.paths import resolve_asset

# The OSC controller acts on the chain of joints between these two links. A
# bot's link actors are named "<bot_name>/<link_name>", so we prefix these at
# runtime with bot.get_name(). fr3_link8 is the arm's tool flange, which is
# where the hand is attached.
ARM_BASE_LINK = "fr3_link0"
ARM_EE_LINK = "fr3_link8"

# The arm's joints all share this prefix, which is how we tell arm DOFs (owned
# by OSC) from hand DOFs (owned by JSC).
ARM_JOINT_PREFIX = "fr3_joint"

# The non-thumb knuckle joints (finger 1 is the thumb).
KNUCKLE_JOINTS = (
    "dg5f_joint_2_2",
    "dg5f_joint_3_2",
    "dg5f_joint_4_2",
    "dg5f_joint_5_3",
)


def get_default_bot_path() -> str:
    """Resolve path to the default FR3 + DG5F short (right) .superdex_bot file."""
    return str(
        resolve_asset(
            "bots/arm_hand_combos/fr3_dg5f_short/right/fr3_dg5f_short_right.superdex_bot"
        )
    )


def main() -> None:
    """Load an arm-hand combo and drive the arm with OSC and the hand with JSC."""
    bot_path = get_default_bot_path()

    # Initialize the physics engine before creating scenes or actors.
    # num_worker_threads=0 runs single-threaded; pass -1 to auto-select.
    sdp.initialize(num_worker_threads=0)

    # Create an empty scene. SuperDex robots use a Z-up convention, so gravity
    # points down the -Z axis.
    scene = sdp.create_scene("OSC + JSC Control Example")
    scene.set_gravity([0, 0, -9.81])

    bot_prefab = sdr.load_bot_prefab_from_file(bot_path)

    # Cheap "gravity compensation": disable gravity on every link before
    # spawning. Neither BASIC_OSC_PD nor BASIC_JSC_PD has a gravity term, so
    # both the arm and the fingers would otherwise sag off their targets.
    for i in range(len(bot_prefab.links)):
        bot_prefab.links[i].has_gravity = False

    # Split the DOFs into the two disjoint sets the controllers own. This numbers them
    # in bot DOF space: the prefab's joint order, skipping the root joint, and every
    # moving joint on this bot is a 1-DoF revolute joint, so the indices run
    # consecutively.
    arm_dofs = []
    joint_name_to_dof = {}
    dof = 0
    for i in range(len(bot_prefab.joints)):
        joint = bot_prefab.joints[i]
        if joint.type != sdp.ArticulatedJointType.REVOLUTE:
            continue
        joint_name_to_dof[joint.name] = dof
        if joint.name.startswith(ARM_JOINT_PREFIX):
            arm_dofs.append(dof)
        dof += 1

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

    # The loop above numbered the joints in bot DOF space, which never includes the
    # root joint's DOFs. Everything from here on indexes the actor, where the root's
    # DOFs come first, so shift the indices across that gap. This arm is welded to the
    # world and contributes none, but reading the count off the actor keeps the mapping
    # correct for a bot on a free-floating base, which contributes six.
    num_root_dofs = bot_actor.get_articulated_shape_info().dof_info[0].get_size()
    arm_dof_indices = np.array(arm_dofs, dtype=np.int32) + num_root_dofs
    knuckle_dofs = [joint_name_to_dof[name] + num_root_dofs for name in KNUCKLE_JOINTS]

    # --- OSC controller on the arm ---------------------------------------------
    # initialize() resolves the base/end-effector links by name and figures out
    # which DOFs lie between them, i.e. the arm joints.
    osc = bot.create_controller("BASIC_OSC_PD")
    bot_name = bot.get_name()
    osc.initialize(f"{bot_name}/{ARM_BASE_LINK}", f"{bot_name}/{ARM_EE_LINK}")

    osc_params = osc.get_params()
    osc_params.kp_p = 900.0
    osc_params.kd_p = 75.0
    osc_params.kp_r = 30.0
    osc_params.kd_r = 3.0
    osc_params.max_translation_error = 0.05  # [m]
    osc_params.max_rotation_error = 0.4  # [rad]
    osc_params.b_apply_max_osc_torque_normalization = True
    osc.set_params(osc_params)

    # --- JSC controller on the hand --------------------------------------------
    # JSC has no notion of a sub-chain: its gains, its target pose and its output
    # are all sized to the full actor, arm DOFs included.
    jsc = bot.create_controller("BASIC_JSC_PD")
    jsc_params = sdr.ControllerBasicJscPdParams()
    jsc_params.kp = np.full(num_dofs, 3.0, dtype=np.float32)  # position gain [Nm/rad]
    jsc_params.kd = np.full(num_dofs, 0.2, dtype=np.float32)  # damping gain [Nms/rad]
    jsc_params.saturation = np.full(num_dofs, 2.0, dtype=np.float32)  # torque clamp
    jsc_params.deadband = np.zeros(num_dofs, dtype=np.float32)
    jsc.set_params(jsc_params)

    # The default pose is the JSC hold target; only the knuckles move off it.
    default_pose = sdp.DynamicArrayReal(num_dofs)
    bot_actor.get_articulated_pose(default_pose)
    hold_pose = np.array(default_pose, dtype=np_real)

    # --- Targets ---------------------------------------------------------------
    # The FR3 base is an intrinsically fixed Hard weld, so world_from_root is
    # constant and can be captured once to convert world-frame targets into the
    # root frame OSC expects.
    world_from_root = osc.get_current_observations_from_mochi().world_from_root

    # Circle lies in a horizontal plane 0.45 m above the ground and 0.5 m in
    # front of the robot base along +X.
    root_pos = np.asarray(world_from_root.translation, dtype=float)
    circle_center = np.array([root_pos[0] + 0.5, root_pos[1], 0.45])
    circle_radius = 0.12  # [m]
    circle_period = 4.0  # [s] per revolution

    # Keep the hand pointing straight down (its z-axis into the ground): a
    # 180-degree rotation about world X flips local +Z to world -Z.
    ee_down = sdp.Quaternion.rotation_x(np.pi)

    # Knuckle sweep: each knuckle oscillates between 0 and 60 degrees, with a
    # 30-degree phase offset between fingers and a 2 s period.
    sweep_period = 2.0  # [s]
    sweep_mid = np.radians(30.0)  # midpoint so the swing spans 0..60 deg
    sweep_amplitude = np.radians(30.0)  # [rad]
    finger_phase_offset = np.radians(30.0)  # [rad] between successive fingers

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

            # OSC target: a point on the circle, hand oriented into the ground.
            theta = 2.0 * np.pi * t / circle_period
            world_from_target_ee = sdp.TransformRT()
            world_from_target_ee.translation = [
                circle_center[0] + circle_radius * np.cos(theta),
                circle_center[1] + circle_radius * np.sin(theta),
                circle_center[2],
            ]
            world_from_target_ee.rotation = ee_down
            # OSC targets are expressed in the actor root frame.
            target_root_from_ee = world_from_root.inverse() * world_from_target_ee

            # JSC target: the default pose everywhere, with the four knuckles
            # driven by a phase-shifted sine.
            target_pose = np.array(hold_pose, dtype=np_real)
            for finger, knuckle_dof in enumerate(knuckle_dofs):
                target_pose[knuckle_dof] = sweep_mid + sweep_amplitude * np.sin(
                    2.0 * np.pi * t / sweep_period + finger * finger_phase_offset
                )

            # np.array (not np.asarray) because the spans the controllers return
            # are read-only views onto their internal buffers.
            # Each controller reads its own observations off the simulation; the
            # JSC additionally needs the control period, which cannot be harvested.
            osc_obsv = osc.get_current_observations_from_mochi()
            arm_tau = np.array(
                osc.compute_output(
                    osc_obsv,
                    sdr.ControllerBasicOscPdTarget(
                        root_from_target_ee=target_root_from_ee
                    ),
                ),
                dtype=np.float32,
            )
            jsc_obsv = jsc.get_current_observations_from_mochi()
            jsc_obsv.dt = time_step
            hand_tau = np.array(
                jsc.compute_output(
                    jsc_obsv,
                    sdr.ControllerBasicJscPdTarget(target_pose=target_pose),
                ),
                dtype=np.float32,
            )

            # Both torque vectors span the whole actor. OSC already zeros
            # everything outside its arm chain, but JSC does not, so we zero its
            # arm entries here -- otherwise it would fight OSC over those DOFs.
            # With the two now disjoint, the combined torque is just their sum.
            hand_tau[arm_dof_indices] = 0.0
            bot_actor.set_external_forces_on_dofs(
                dof_indices=all_dof_indices,
                force_values=arm_tau + hand_tau,
            )
            scene.step(time_step)

    # Tear down: destroy the bot, then shut the engine down cleanly.
    sdr.destroy_bot(scene, bot)
    sdp.shutdown()
    print("Simulation complete.")


if __name__ == "__main__":
    main()
