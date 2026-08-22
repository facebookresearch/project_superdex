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

"""Example: Loading a robot from a URDF with `superdex.robotics`

Loads a robot directly from a .urdf file and simulates it. Collision meshes can
be plain .stl/.obj/.ply/.off files — no preprocessing is required; an SDF
collider is baked on the fly when the bot is created.

PITFALL: The runtime URDF loader only understands *mesh* collision geometry and
silently ignores primitive shapes (<box>, <cylinder>, <sphere>). The bundled FR3
URDF defines its gripper fingers as <box> primitives, so the loaded fingers have
no collision geometry.

PITFALL: Baking an SDF on the fly assumes each collision mesh is a closed
(watertight) surface. Raw URDF meshes often are not, so loading may warn, e.g.:

    Actor 'fr3v2_1/fr3v2_1_link7': The collider mesh is not topologically closed
    (not a closed surface). The SDF sign may be incorrect, causing unreliable
    collision detection near the open boundaries.

The default FR3 URDF trips this on some links — another reason to prefer
SuperDex Studio, whose importer produces watertight colliders.

NOTE: For production assets, prefer importing URDFs through SuperDex Studio. Its
importer remeshes collision (or visual) geometry, bakes watertight SDFs offline, and
emits a .superdex_bot with preprocessed assets. This runtime path is best for quick
prototyping.

Usage:
    cd <path_to_superdex_robotics>
    python3 examples/basic/example_urdf_loading.py
    python3 examples/basic/example_urdf_loading.py /path/to/robot.urdf
"""

import argparse

import superdex.physics as physics
import superdex.robotics as robotics
from superdex.physics.paths import resolve_asset


def get_default_urdf_path() -> str:
    """Resolve the path to the bundled FR3 test URDF."""
    return str(resolve_asset("test/urdf/fr3v2_1_urdf/robots/fr3v2_1_franka_hand.urdf"))


def main() -> None:
    """Load a robot from a URDF file and simulate it."""
    parser = argparse.ArgumentParser(description="URDF bot loading example")
    parser.add_argument(
        "path",
        type=str,
        nargs="?",
        default=None,
        help="Path to a .urdf file (default: fr3v2_1_franka_hand.urdf)",
    )
    args = parser.parse_args()

    urdf_path = args.path if args.path else get_default_urdf_path()

    # Initialize the physics engine before creating scenes or actors.
    # num_worker_threads=0 runs single-threaded; pass -1 to auto-select.
    physics.initialize(num_worker_threads=0)

    # Create an empty scene. SuperDex robots use a Z-up convention, so gravity
    # points down the -Z axis.
    scene = physics.create_scene("URDF Bot Loading Example")
    scene.set_gravity([0, 0, -9.81])

    # Parse the URDF into a bot prefab: a template describing links and joints.
    # Collision mesh paths are kept as-is; no mesh conversion happens here.
    bot_prefab = robotics.load_bot_prefab_from_urdf_file(urdf_path)

    # URDF loading adds a free "world_joint" at index 0, leaving the base free to
    # fall. Make it a Hard (0-DoF weld) joint to anchor the base to the world.
    # The arm still sags under gravity since there is no controller.
    bot_prefab.joints[0].type = physics.ArticulatedJointType.HARD

    # Instantiate the prefab as a live Bot in the scene. This builds the robot's
    # articulated actor and bakes each link's SDF collider from its mesh. The
    # robotics context tracks every bot and controller you create.
    robotics_context = robotics.create_context()
    bot = robotics.create_bot(scene, bot_prefab, robotics_context)
    bot_actor = bot.get_articulated_actor()

    # Add a static ground plane for the robot to rest on (normal points up, +Z).
    plane_shape = physics.create_plane_shape(normal=[0, 0, 1], distance=0)
    scene.create_rigid_actor(name="ground", shape=plane_shape, is_static=True)

    # Print a quick summary of the loaded robot.
    print(f"Robot: {bot_prefab.name}")
    print(f"  Links: {len(bot_prefab.links)}")
    print(f"  Joints: {len(bot_prefab.joints)}")
    print(f"  DOFs: {bot_actor.get_num_dofs()}")

    # Simulate at 60 Hz (each step advances 1/60 of a second).
    time_step = 1.0 / 60.0

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

    # Tear down: destroy the bot, then shut the engine down cleanly.
    robotics.destroy_bot(scene, bot)
    physics.shutdown()
    print("Simulation complete.")


if __name__ == "__main__":
    main()
