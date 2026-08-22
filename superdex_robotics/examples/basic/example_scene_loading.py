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

"""Example: Building a scene from separate bots and a task prefab

Demonstrates assembling a workcell out of independent pieces rather than loading
one monolithic scene file: two arm-and-hand bots, each placed at its own root
transform and started in its own joint pose, plus a task prefab -- a box of
blocks -- positioned in front of them.

The two halves come from different kinds of asset. A `.superdex_bot` is a robot
description, loaded into a prefab whose placement (`world_from_root`) and
starting joint angles (`default_pose`) you set before instantiating it. A
`.mochi_prefab` is a bundle of scene geometry, added straight to the scene with
its own placement. Neither knows about the other; the scene is just what you put
in it.

Usage:
    cd <path_to_superdex_robotics>
    python3 examples/basic/example_scene_loading.py
"""

import math
from typing import NamedTuple

import superdex.physics as physics
import superdex.robotics as robotics
from superdex.physics.paths import get_assets_root, resolve_asset


class BotPlacement(NamedTuple):
    """Where a bot goes and how it starts out."""

    asset: str
    rotation: list[float]  # world-from-root quaternion, XYZW
    translation: list[float]  # world-from-root position [m]
    initial_pose: list[float]  # one angle per DOF [rad]


# The two bots face each other across the workspace, mounted at the same height.
BOTS = (
    BotPlacement(
        asset="bots/arm_hand_combos/fr3_dg5f_short/left/fr3_dg5f_short_left.superdex_bot",
        rotation=[-0.436878, 0.022288, -0.242939, 0.865807],
        translation=[-0.773603, 0.050683, 0.650259],
        initial_pose=[
            0.130710,
            -1.414085,
            -0.399909,
            -2.727225,
            0.441916,
            2.804378,
            -0.322869,
            -0.499898,
            0.954225,
            0.132193,
            0.129680,
            0.135242,
            0.391550,
            0.335378,
            0.161290,
            0.049379,
            0.283850,
            0.167437,
            0.171763,
            0.007906,
            0.179695,
            0.170909,
            0.234395,
            -0.272505,
            -0.120150,
            0.141308,
            0.258700,
        ],
    ),
    BotPlacement(
        asset="bots/arm_hand_combos/fr3_dg5f_short/right/fr3_dg5f_short_right.superdex_bot",
        rotation=[0.436878, 0.022288, 0.242939, 0.865807],
        translation=[-0.773602, -0.050681, 0.650259],
        initial_pose=[
            -0.194073,
            -1.600604,
            0.488709,
            -2.522844,
            0.077551,
            2.294475,
            -0.378717,
            0.499898,
            -0.954225,
            0.132193,
            0.129680,
            -0.135242,
            0.391550,
            0.335378,
            0.161290,
            -0.049379,
            0.283850,
            0.167437,
            0.171763,
            -0.007906,
            0.179695,
            0.170909,
            0.234395,
            0.272505,
            0.120150,
            0.141308,
            0.258700,
        ],
    ),
)

# A box holding a pile of loose blocks, for the robots to work on. The prefab is
# authored with its long axis along its own X and all of its contents in the
# positive octant, so it is yawed to face the robots and then shifted back to
# bring the box in front of them.
TASK_PREFAB = "prefabs/box_and_blocks/box_and_blocks.mochi_prefab"
TASK_PREFAB_YAW = math.radians(90.0)  # [rad] about world Z
TASK_PREFAB_OFFSET_Y = -0.25  # [m] the yaw puts the prefab's own X along world Y

# Uniform joint-space gains for holding each bot's starting pose. A single
# tracking entry is broadcast to every controllable joint in the articulation.
POSE_STIFFNESS = 1.0e3
POSE_DAMPING = 1.0e2


def main() -> None:
    """Load two bots and a task prefab into one scene."""
    assets_root = str(get_assets_root())

    # Initialize the physics engine before creating scenes or actors.
    # num_worker_threads=0 runs single-threaded; pass -1 to auto-select.
    physics.initialize(num_worker_threads=0)

    # Create an empty scene. SuperDex robots use a Z-up convention, so gravity
    # points down the -Z axis.
    scene = physics.create_scene("Scene Loading Example")
    scene.set_gravity([0, 0, -9.81])

    # Add a static ground plane (normal points up, +Z).
    plane_shape = physics.create_plane_shape(normal=[0, 0, 1], distance=0)
    scene.create_rigid_actor(name="ground", shape=plane_shape, is_static=True)

    # --- The bots --------------------------------------------------------------
    # Each bot is loaded from its own .superdex_bot file into a prefab. Placement
    # and starting pose are fields on that prefab, so set them before creating the
    # bot: create_bot builds the articulated actor at world_from_root and seeds it
    # with default_pose. The robotics context tracks every bot you create.
    robotics_context = robotics.create_context()
    bots = []
    for placement in BOTS:
        bot_prefab = robotics.load_bot_prefab_from_file(
            str(resolve_asset(placement.asset))
        )
        bot_prefab.world_from_root = physics.TransformRT(
            rotation=placement.rotation,
            translation=placement.translation,
        )
        bot_prefab.default_pose = placement.initial_pose

        bot = robotics.create_bot(scene, bot_prefab, robotics_context)
        bots.append(bot)

        # Use Mochi's implicit articulated pose controller to hold the pose that
        # create_bot just applied. Empty Cartesian tracking arrays leave link
        # position and rotation control disabled.
        tracking = physics.PoseTrackingParams(
            stiffness=POSE_STIFFNESS,
            damping=POSE_DAMPING,
        )
        pose_params = physics.PoseControllerParams(
            joint_tracking=physics.DynamicArrayPoseTrackingParams([tracking])
        )
        pose_controller = bot.create_controller("MOCHI_ARTICULATED_POSE")
        pose_controller.set_params(
            robotics.ControllerMochiArticulatedPoseParams(
                pose_controller_params=pose_params
            )
        )
        pose_controller.initialize(True)

        print(
            f"Bot: {bot_prefab.name} ({bot.get_articulated_actor().get_num_dofs()} DOFs)"
        )

    # --- The task prefab -------------------------------------------------------
    # A .mochi_prefab drops a whole bundle of actors into the scene in one call.
    # PrefabParams carries its placement and a name prefix, so the actors land as
    # "box_and_blocks/<actor name>". Paths inside the prefab resolve against
    # root_path.
    prefab_result = physics.prefab.add_to_scene(
        prefab_path=str(resolve_asset(TASK_PREFAB)),
        root_path=assets_root,
        scene=scene,
        params=physics.prefab.PrefabParams(
            name="box_and_blocks",
            rotation=physics.Quaternion.rotation_z(TASK_PREFAB_YAW),
            translation=[0.0, TASK_PREFAB_OFFSET_Y, BOTS[0].translation[2]],
        ),
    )
    print(f"Task prefab: {len(prefab_result.actors)} actors")

    # Simulate at 200 Hz (each step advances 1/200 of a second). The pose
    # controllers hold the robots at their starting configurations while the loose
    # blocks remain free to fall under gravity.
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
            scene.step(time_step)

    # Tear down: destroy the bots, then shut the engine down cleanly.
    for bot in bots:
        robotics.destroy_bot(scene, bot)
    physics.shutdown()
    print("Simulation complete.")


if __name__ == "__main__":
    main()
