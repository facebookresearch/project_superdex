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

"""Example: Loading a robot with `superdex.robotics`

Demonstrates loading a robot from a `.superdex_bot` file using the SuperDex
bindings and simulating it in a physics scene.

Usage:
    cd <path_to_superdex_robotics>
    python3 examples/basic/example_bot_loading.py
    python3 examples/basic/example_bot_loading.py /path/to/my_robot.superdex_bot
"""

import argparse

import superdex.physics as sdp
import superdex.robotics as sdr
from superdex.physics.paths import resolve_asset


def get_default_bot_path() -> str:
    """Resolve path to the default FR3 robot .superdex_bot file."""
    return str(resolve_asset("bots/arms/fr3/fr3.superdex_bot"))


def main() -> None:
    """Load a robot from a .superdex_bot file and simulate it."""
    parser = argparse.ArgumentParser(description="Bot loading example")
    parser.add_argument(
        "path",
        type=str,
        nargs="?",
        default=None,
        help="Path to a .superdex_bot file (default: fr3.superdex_bot)",
    )
    args = parser.parse_args()

    bot_path = args.path if args.path else get_default_bot_path()

    # Initialize the physics engine before creating scenes or actors.
    # num_worker_threads=0 runs single-threaded; pass -1 to auto-select.
    sdp.initialize(num_worker_threads=0)

    # Create an empty scene. SuperDex robots use a Z-up convention, so gravity
    # points down the -Z axis.
    scene = sdp.create_scene("Bot Loading Example")
    scene.set_gravity([0, 0, -9.81])

    # A .superdex_bot is a self-contained robot description. Loading it returns a
    # "prefab": a reusable template describing the robot's links and joints.
    bot_prefab = sdr.load_bot_prefab_from_file(bot_path)

    # Instantiate the prefab as a live Bot in the scene. This builds the robot's
    # articulated actor and seeds its default pose. The robotics context tracks
    # every bot and controller you create.
    robotics_context = sdr.create_context()
    bot = sdr.create_bot(scene, bot_prefab, robotics_context)
    bot_actor = bot.get_articulated_actor()

    # Add a static ground plane for the robot to rest on (normal points up, +Z).
    plane_shape = sdp.create_plane_shape(normal=[0, 0, 1], distance=0)
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
    sdp.get_debug_server().set_coordinate_space(
        sdp.CoordinateSpace(axes=sdp.CoordinateSpaceAxes.FLU)
    )

    # Launch and connect the SuperDex Physics Debugger, a separate desktop app for
    # viewing and interacting with the simulation. The loop runs until you close
    # the debugger; attach() returns False if it can't connect.
    if sdp.debugger.attach():
        while sdp.debugger.is_attached():
            scene.step(time_step)

    # Tear down: destroy the bot, then shut the engine down cleanly.
    sdr.destroy_bot(scene, bot)
    sdp.shutdown()
    print("Simulation complete.")


if __name__ == "__main__":
    main()
