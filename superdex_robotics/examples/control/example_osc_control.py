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

"""Example: OSC control tracking a planar circle

A basic demonstration of the operational-space PD (OSC) controller. It loads an
FR3 v2 arm and drives the end-effector around a horizontal circle while keeping
the tool pointing straight down.

OSC works in Cartesian task space: you give it a target end-effector pose and it
solves for the arm joint torques that move the end-effector there. Each step we
build a target pose (a point on the circle, tool pointing down), compute the OSC
torques, apply them to the articulated actor, and step the simulation.

Usage:
    cd <path_to_superdex_robotics>
    python3 examples/control/example_osc_control.py
"""

import numpy as np
import superdex.physics as physics
import superdex.robotics as robotics
from superdex.physics.paths import resolve_asset

# The OSC controller acts on the chain of joints between these two links on the
# FR3 arm. A bot's link actors are named "<bot_name>/<link_name>", so we prefix
# these at runtime with bot.get_name().
ARM_BASE_LINK = "fr3_link0"
ARM_EE_LINK = "fr3_link8"


def get_default_bot_path() -> str:
    """Resolve path to the default FR3 v2 arm .superdex_bot file."""
    return str(resolve_asset("bots/arms/fr3_v2/fr3_v2.superdex_bot"))


def main() -> None:
    """Load an arm, attach an OSC controller, and track a planar circle."""
    bot_path = get_default_bot_path()

    # Initialize the physics engine before creating scenes or actors.
    # num_worker_threads=0 runs single-threaded; pass -1 to auto-select.
    physics.initialize(num_worker_threads=0)

    # Create an empty scene. SuperDex robots use a Z-up convention, so gravity
    # points down the -Z axis.
    scene = physics.create_scene("OSC Control Example")
    scene.set_gravity([0, 0, -9.81])

    # Load the robot and instantiate it as a live Bot. The robotics context
    # tracks every bot and controller you create.
    bot_prefab = robotics.load_bot_prefab_from_file(bot_path)

    # Cheap "gravity compensation": disable gravity on every link before spawning.
    # BASIC_OSC_PD is a pure task-space PD with no gravity term, so if the arm has
    # to fight gravity it sags off the target. With gravity off, holding the
    # default end-effector frame is (nearly) load-free.
    for i in range(len(bot_prefab.links)):
        bot_prefab.links[i].has_gravity = False

    robotics_context = robotics.create_context()
    bot = robotics.create_bot(scene, bot_prefab, robotics_context)
    bot_actor = bot.get_articulated_actor()

    # Add a static ground plane for the robot to rest on (normal points up, +Z).
    plane_shape = physics.create_plane_shape(normal=[0, 0, 1], distance=0)
    scene.create_rigid_actor(name="ground", shape=plane_shape, is_static=True)

    num_dofs = bot_actor.get_num_dofs()
    all_dof_indices = np.arange(num_dofs, dtype=np.int32)

    # --- OSC controller on the arm ---------------------------------------------
    # create_controller attaches the controller to the bot. initialize() resolves
    # the base/end-effector links by name and figures out which DOFs lie between
    # them (the arm joints).
    osc = bot.create_controller("BASIC_OSC_PD")
    bot_name = bot.get_name()
    osc.initialize(f"{bot_name}/{ARM_BASE_LINK}", f"{bot_name}/{ARM_EE_LINK}")

    # Task-space PD gains: kp_*/kd_* for position (_p) and rotation (_r).
    osc_params = osc.get_params()
    osc_params.kp_p = 900.0
    osc_params.kd_p = 75.0
    osc_params.kp_r = 30.0
    osc_params.kd_r = 3.0
    # Error-magnitude normalization (on by default) clamps how far the target may
    # pull before the force saturates, so these caps must be positive.
    osc_params.max_translation_error = 0.05  # [m]
    osc_params.max_rotation_error = 0.4  # [rad]
    osc_params.b_apply_max_osc_torque_normalization = True
    osc.set_params(osc_params)

    # --- Circle target ---------------------------------------------------------
    # Read the arm's default frames once. The FR3 bot file has an intrinsically
    # fixed base (its root joint is a Hard weld by default), so world_from_root is
    # constant and can be captured once to convert world-frame targets into the
    # root frame OSC expects.
    obsv = osc.get_current_observations_from_mochi()
    world_from_root = obsv.world_from_root

    # Circle lies in a horizontal plane (its normal is the world up-axis) 0.45 m
    # above the ground and 0.5 m in front of the robot base along +X.
    root_pos = np.asarray(world_from_root.translation, dtype=float)
    circle_center = np.array([root_pos[0] + 0.5, root_pos[1], 0.45])
    circle_radius = 0.12  # [m]
    circle_period = 4.0  # [s] per revolution

    # Keep the end-effector pointing straight down (its z-axis into the ground):
    # a 180-degree rotation about world X flips local +Z to world -Z.
    ee_down = physics.Quaternion.rotation_x(np.pi)

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
            # Target end-effector pose in the world frame: a point on the circle,
            # oriented so the EE z-axis points into the ground.
            theta = 2.0 * np.pi * scene.get_total_simulation_time() / circle_period
            world_from_target_ee = physics.TransformRT()
            world_from_target_ee.translation = [
                circle_center[0] + circle_radius * np.cos(theta),
                circle_center[1] + circle_radius * np.sin(theta),
                circle_center[2],
            ]
            world_from_target_ee.rotation = ee_down

            # OSC targets are expressed in the actor root frame.
            target_root_from_ee = world_from_root.inverse() * world_from_target_ee

            # OSC returns a full-length torque vector for the actor, so we can
            # apply it directly.
            # Read this step's robot state off the simulation.
            obsv = osc.get_current_observations_from_mochi()
            arm_tau = np.asarray(
                osc.compute_output(
                    obsv,
                    robotics.ControllerBasicOscPdTarget(
                        root_from_target_ee=target_root_from_ee
                    ),
                ),
                dtype=np.float32,
            )
            bot_actor.set_external_forces_on_dofs(
                dof_indices=all_dof_indices,
                force_values=arm_tau,
            )
            scene.step(time_step)

    # Tear down: destroy the bot, then shut the engine down cleanly.
    robotics.destroy_bot(scene, bot)
    physics.shutdown()
    print("Simulation complete.")


if __name__ == "__main__":
    main()
