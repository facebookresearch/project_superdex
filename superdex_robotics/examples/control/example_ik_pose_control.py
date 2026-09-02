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

"""Example: IK + pose control tracking a planar circle

Drives an FR3 v2 arm around the same horizontal circle as the OSC example, tool
pointing straight down, but reaches it a different way: inverse kinematics picks
the joint angles and Mochi's articulated pose controller holds the arm there.

Mochi's IK solver runs on its own scene, so the arm is loaded twice: once into
the simulated scene and once into a kinematic twin that only the solver touches.
Each step we put position and rotation targets on the twin's end-effector, solve
for a joint configuration that reaches them, and hand that configuration to the
pose controller as its target. Only the simulated scene is ever stepped.

Usage:
    cd <path_to_superdex_robotics>
    python3 examples/control/example_ik_pose_control.py
"""

import numpy as np
import superdex.physics as sdp
import superdex.robotics as sdr
from superdex.physics.paths import resolve_asset

# The IK targets are placed on this link, the FR3's tool flange. A bot's link
# actors are named "<bot_name>/<link_name>".
ARM_EE_LINK = "fr3_link8"

# Per-joint stiffness and damping for the pose controller, tuned for this arm and
# shipped alongside it.
POSE_CONTROLLER_PARAMS = "bots/arms/fr3_v2/control/fr3_v2_pose.superdex_controller"

# Objective weights for the two IK targets, [N/m] and [Nm/rad]. The solver stops
# once the objective gradient falls below its absolute tolerance, so the weights
# have to be large enough that a millimetre of error still registers. Their
# relative magnitude sets the position/rotation tradeoff when both cannot be met.
IK_POSITION_WEIGHT = 1.0e4
IK_ROTATION_WEIGHT = 1.0e2

# Keep the end-effector pointing straight down (its z-axis into the ground). IK
# rotation targets are rotation vectors (axis * angle), so this is a half turn
# about world X, which flips local +Z to world -Z.
EE_DOWN_ROTATION_VECTOR = [np.pi, 0.0, 0.0]


def get_default_bot_path() -> str:
    """Resolve path to the default FR3 v2 arm .superdex_bot file."""
    return str(resolve_asset("bots/arms/fr3_v2/fr3_v2.superdex_bot"))


def create_arm(
    scene: sdp.Scene, bot_path: str, robotics_context: sdr.RoboticsContext
) -> sdr.Bot:
    """Load the arm into a scene, with gravity disabled on every link.

    Nothing here compensates for gravity, so switching it off is what lets the arm
    hold exactly the configuration IK asks for.
    """
    bot_prefab = sdr.load_bot_prefab_from_file(bot_path)
    for i in range(len(bot_prefab.links)):
        bot_prefab.links[i].has_gravity = False
    return sdr.create_bot(scene, bot_prefab, robotics_context)


def main() -> None:
    """Load an arm, solve IK for a circle, and track it with a pose controller."""
    bot_path = get_default_bot_path()

    # Initialize the physics engine before creating scenes or actors.
    # num_worker_threads=0 runs single-threaded; pass -1 to auto-select.
    sdp.initialize(num_worker_threads=0)

    # Create an empty scene. SuperDex robots use a Z-up convention, so gravity
    # points down the -Z axis.
    scene = sdp.create_scene("IK + Pose Control Example")
    scene.set_gravity([0, 0, -9.81])

    # The robotics context tracks every bot and controller you create.
    robotics_context = sdr.create_context()
    bot = create_arm(scene, bot_path, robotics_context)
    bot_actor = bot.get_articulated_actor()

    # Add a static ground plane for the robot to rest on (normal points up, +Z).
    plane_shape = sdp.create_plane_shape(normal=[0, 0, 1], distance=0)
    scene.create_rigid_actor(name="ground", shape=plane_shape, is_static=True)

    # --- IK solver -------------------------------------------------------------
    # The solver takes ownership of the scene it is given and reshapes it for
    # quasistatic solving, so it gets a scene of its own holding nothing but a
    # second copy of the arm. Resolve the end-effector link before handing the
    # scene over; targets are addressed by link actor handle.
    ik_scene = sdp.create_scene("IK Solver Scene")
    ik_bot = create_arm(ik_scene, bot_path, robotics_context)
    ik_actor = ik_bot.get_articulated_actor()
    ik_ee_handle = next(
        handle
        for handle in ik_actor.get_nested_link_actors()
        if ik_scene.get_actor(handle).get_name().endswith(f"/{ARM_EE_LINK}")
    )
    ik_solver = sdp.experimental.create_ik_solver(ik_scene)

    # Create the two targets once and keep the position one: the solver hands back
    # a constraint whose target is updated in place each step. The rotation target
    # never changes, so it is set up and left alone.
    ik_position_target = ik_solver.create_position_target(
        ik_ee_handle,
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        IK_POSITION_WEIGHT,
    )
    ik_solver.create_rotation_target(
        ik_ee_handle,
        [0.0, 0.0, 0.0],
        EE_DOWN_ROTATION_VECTOR,
        IK_ROTATION_WEIGHT,
    )

    # Scratch buffer for reading the solved configuration back out.
    ik_pose = sdp.DynamicArrayReal(ik_actor.get_num_dofs())

    # --- Pose controller -------------------------------------------------------
    # The pose controller is an implicit PD: instead of handing torques back for
    # you to apply as external forces, its per-joint stiffness and damping become
    # part of the system the solver resolves during the step. Being solved rather
    # than applied, it stays stable at gains an explicit PD could not hold -- so
    # compute_output applies the control itself and returns nothing.
    #
    # An articulation may only carry one pose controller, and installing it on the
    # actor is what initialize() does, so the params are set first.
    pose_controller = bot.create_controller("MOCHI_ARTICULATED_POSE")
    pose_controller.set_params(
        sdr.ControllerMochiArticulatedPoseParams.load_from_file(
            str(resolve_asset(POSE_CONTROLLER_PARAMS))
        )
    )
    pose_controller.initialize(True)

    # The target pairs a root transform with the non-root joint DOFs. The FR3 base
    # is welded, so the root transform is constant and only the DOFs change.
    pose_obsv = sdr.ControllerMochiArticulatedPoseObsv()
    pose_target = sdr.ControllerMochiArticulatedPoseTarget()
    pose_target.world_from_root = bot_actor.get_root_transform()

    # --- Circle target ---------------------------------------------------------
    # IK targets are given in the world frame, so unlike OSC -- which wants them in
    # the actor root frame -- no conversion is needed.
    root_position = np.asarray(bot_actor.get_root_transform().translation, dtype=float)

    # Circle lies in a horizontal plane 0.45 m above the ground and 0.5 m in front
    # of the robot base along +X.
    circle_center = np.array([root_position[0] + 0.5, root_position[1], 0.45])
    circle_radius = 0.12  # [m]
    circle_period = 4.0  # [s] per revolution

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
    #
    # This example has two scenes, and the debugger starts every scene paused and
    # plays only the selected one: solving IK steps the IK scene, so nothing moves
    # until both are running. Press play on the IK scene, then switch to this
    # example's scene and press play again. Unchecking "Start Paused" when
    # connecting avoids the dance and brings both up running.
    if sdp.debugger.attach():
        while sdp.debugger.is_attached():
            theta = 2.0 * np.pi * scene.get_total_simulation_time() / circle_period
            ik_position_target.set_target_position(
                [
                    circle_center[0] + circle_radius * np.cos(theta),
                    circle_center[1] + circle_radius * np.sin(theta),
                    circle_center[2],
                ]
            )
            ik_solver.solve_ik()

            # The solution lands in the twin's joint pose, which becomes the pose
            # controller's target for the simulated arm.
            ik_actor.get_articulated_pose(ik_pose)
            pose_target.pose_dofs = ik_pose
            pose_controller.compute_output(pose_obsv, pose_target)

            scene.step(time_step)

    # Tear down: release the IK targets and the twin, then destroy the solver
    # (which destroys the scene it owns), then the simulated bot and the engine.
    ik_solver.clear_position_target(ik_ee_handle)
    ik_solver.clear_rotation_target(ik_ee_handle)
    sdr.destroy_bot(ik_scene, ik_bot)
    sdp.experimental.destroy_ik_solver(ik_solver)
    sdr.destroy_bot(scene, bot)
    sdp.shutdown()
    print("Simulation complete.")


if __name__ == "__main__":
    main()
