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

"""Example: JSC control waving a hand's fingers

A basic demonstration of the joint-space PD (JSC) controller. It loads a DG5F
hand, welds it to the world, and holds every joint at its default pose except the
four non-thumb knuckles, which sweep back and forth with a phase-shifted sine.

JSC works in joint space: you give it a target joint pose (one angle per DOF) and
it applies per-joint PD torques to reach it. Each step we build the target pose
(default everywhere, sinusoidal on the knuckles), compute the JSC torques, apply
them to the articulated actor, and step the simulation.

Usage:
    cd <path_to_superdex_robotics>
    python3 examples/control/example_jsc_control.py
"""

import numpy as np
import superdex.physics as physics
import superdex.robotics as robotics

# The build's `real` type. A pose handed to a controller Target is copied into the
# Target's own storage, so matching the dtype here keeps that a straight copy rather
# than an element-by-element conversion.
np_real = np.float64 if physics.uses_double_precision() else np.float32
from superdex.physics.paths import resolve_asset

# The non-thumb knuckle joints (finger 1 is the thumb). For each of the other
# four fingers, this is its metacarpophalangeal (knuckle) flexion joint.
KNUCKLE_JOINTS = (
    "dg5f_joint_2_2",
    "dg5f_joint_3_2",
    "dg5f_joint_4_2",
    "dg5f_joint_5_3",
)


def get_default_bot_path() -> str:
    """Resolve path to the default DG5F (long, right) hand .superdex_bot file."""
    return str(resolve_asset("bots/hands/dg5f_long/right/dg5f_long_right.superdex_bot"))


def main() -> None:
    """Load a hand, attach a JSC controller, and wave its knuckles."""
    bot_path = get_default_bot_path()

    # Initialize the physics engine before creating scenes or actors.
    # num_worker_threads=0 runs single-threaded; pass -1 to auto-select.
    physics.initialize(num_worker_threads=0)

    # Create an empty scene. SuperDex robots use a Z-up convention, so gravity
    # points down the -Z axis.
    scene = physics.create_scene("JSC Control Example")
    scene.set_gravity([0, 0, -9.81])

    bot_prefab = robotics.load_bot_prefab_from_file(bot_path)

    # The hand ships with a free (6-DoF) world joint at index 0. Make it a Hard
    # (0-DoF weld) joint so the hand is rigidly fixed to the world.
    bot_prefab.joints[0].type = physics.ArticulatedJointType.HARD

    # Map each actuated joint name to its DOF index. DOFs follow the prefab joint
    # order; after the weld above every remaining moving joint is a 1-DoF revolute
    # joint, so we assign them consecutive indices. The weld is also what lets these
    # index the actor arrays below directly: a root joint's DOFs lead the actor's,
    # and a welded root has none, so nothing is skipped ahead of the first knuckle.
    joint_name_to_dof = {}
    dof = 0
    for i in range(len(bot_prefab.joints)):
        joint = bot_prefab.joints[i]
        if joint.type == physics.ArticulatedJointType.REVOLUTE:
            joint_name_to_dof[joint.name] = dof
            dof += 1
    knuckle_dofs = [joint_name_to_dof[name] for name in KNUCKLE_JOINTS]

    # Instantiate the prefab as a live Bot. The robotics context tracks every bot
    # and controller you create.
    robotics_context = robotics.create_context()
    bot = robotics.create_bot(scene, bot_prefab, robotics_context)
    bot_actor = bot.get_articulated_actor()

    num_dofs = bot_actor.get_num_dofs()
    all_dof_indices = np.arange(num_dofs, dtype=np.int32)

    # --- JSC controller --------------------------------------------------------
    # create_controller attaches the controller to the bot. JSC spans every DOF;
    # per-joint gains (kp/kd) are the same length as the pose.
    jsc = bot.create_controller("BASIC_JSC_PD")
    jsc_params = robotics.ControllerBasicJscPdParams()
    jsc_params.kp = np.full(num_dofs, 3.0, dtype=np.float32)  # position gain [Nm/rad]
    jsc_params.kd = np.full(num_dofs, 0.2, dtype=np.float32)  # damping gain [Nms/rad]
    jsc_params.saturation = np.full(num_dofs, 2.0, dtype=np.float32)  # torque clamp
    jsc_params.deadband = np.zeros(num_dofs, dtype=np.float32)
    jsc.set_params(jsc_params)

    # Read the hand's default joint pose. get_articulated_pose fills a
    # DynamicArrayReal, which matches the engine's float precision; copy it to a
    # numpy array to build per-step targets.
    default_pose = physics.DynamicArrayReal(num_dofs)
    bot_actor.get_articulated_pose(default_pose)
    hold_pose = np.array(default_pose, dtype=np_real)

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
    physics.get_debug_server().set_coordinate_space(
        physics.CoordinateSpace(axes=physics.CoordinateSpaceAxes.FLU)
    )

    # Launch and connect the SuperDex Physics Debugger, a separate desktop app for
    # viewing and interacting with the simulation. The loop runs until you close
    # the debugger; attach() returns False if it can't connect.
    if physics.debugger.attach():
        while physics.debugger.is_attached():
            # Start from the default pose, then drive the four knuckles with a
            # phase-shifted sine.
            t = scene.get_total_simulation_time()
            target_pose = np.array(hold_pose, dtype=np_real)
            for finger, knuckle_dof in enumerate(knuckle_dofs):
                target_pose[knuckle_dof] = sweep_mid + sweep_amplitude * np.sin(
                    2.0 * np.pi * t / sweep_period + finger * finger_phase_offset
                )

            # Read this step's robot state off the simulation, then supply the
            # control period, which the harvester cannot know.
            obsv = jsc.get_current_observations_from_mochi()
            obsv.dt = time_step
            tau = np.asarray(
                jsc.compute_output(
                    obsv,
                    robotics.ControllerBasicJscPdTarget(target_pose=target_pose),
                ),
                dtype=np.float32,
            )
            bot_actor.set_external_forces_on_dofs(
                dof_indices=all_dof_indices,
                force_values=tau,
            )
            scene.step(time_step)

    # Tear down: destroy the bot, then shut the engine down cleanly.
    robotics.destroy_bot(scene, bot)
    physics.shutdown()
    print("Simulation complete.")


if __name__ == "__main__":
    main()
