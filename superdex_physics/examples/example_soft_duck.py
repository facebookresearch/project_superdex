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

"""Example: Soft Bodies

Simulates a deformable duck falling onto a ground plane using FEM (Finite Element
Method). Demonstrates creating soft body actors from tetrahedral meshes.
"""

import superdex.physics as sdp
from superdex.physics import Actor, Scene
from superdex.physics.paths import resolve_asset

# SuperDex Physics uses fully-implicit integration, enabling substantially larger
# stable time steps than explicit or semi-implicit methods. This simple example uses
# 16.7 ms per step. In practice, 10–25 ms steps remain robust across a broad range of
# simulations, including complex non-convex contact and deformable actors such as
# soft bodies, shells, and rods.
TIME_STEP = 1.0 / 60.0  # [s]


def create_soft_duck_simulation() -> tuple[Scene, Actor, Actor]:
    """Create a soft duck physics simulation.

    Note: physics.initialize() must be called before this function.

    Returns:
        tuple: (scene, soft_duck_actor, rigid_plane_actor)
    """
    # Create scene
    scene = sdp.create_scene("Soft Duck Scene")

    # SuperDex Physics uses Y-up coordinates and defaults to gravity along -Y:
    # (0, -9.8, 0). Set it explicitly for illustration.
    # Note: SuperDex Robotics uses Z-up coordinates. Its examples and SuperDex
    # Studio's default configuration set gravity along -Z.
    scene.set_gravity([0, -9.8, 0])

    # Create shapes
    shape_path = str(resolve_asset("duck/duck_1899.mochi.h5"))
    tet_mesh_shape = sdp.load_shape_from_file(
        file_path=shape_path,
    )
    plane_shape = sdp.create_plane_shape(normal=[0, 1, 0], distance=-0.5)

    # Create soft duck
    soft_duck_actor = scene.create_soft_actor(
        name="duck",
        shape=tet_mesh_shape,
        world_from_local=sdp.TransformRT(translation=[-0.5, 0.5, -1.0]),
    )

    # Create ground plane
    rigid_plane_actor = scene.create_rigid_actor(
        name="ground", shape=plane_shape, is_static=True
    )

    return scene, soft_duck_actor, rigid_plane_actor


def cleanup_simulation(
    scene: Scene, soft_duck_actor: Actor, rigid_plane_actor: Actor
) -> None:
    """Demonstrate actor, scene, and global resource cleanup."""
    # Destroying individual actors is not required before destroying their scene;
    # these calls are included to demonstrate the actor lifecycle.
    scene.destroy_actor(soft_duck_actor)
    scene.destroy_actor(rigid_plane_actor)

    # Destroying this scene is not required immediately before shutdown; this
    # call demonstrates how to release a scene while SuperDex Physics remains active.
    sdp.destroy_scene(scene)

    # Shutdown releases any remaining global resources. It is shown explicitly
    # so this example can be initialized again in the same process.
    sdp.shutdown()


def main() -> None:
    """Main function that runs the interactive soft duck simulation."""

    # Initialize SuperDex Physics and specify the number of worker threads to
    # use for simulation:
    #   0 = single-threaded (no additional threads beyond the calling thread)
    #  -1 = let SuperDex Physics choose based on your hardware
    #   N = use N worker threads in addition to the calling thread
    #
    # Run single-threaded to keep the example simple. For scenes with a large
    # number of DoFs (e.g. scenes with high-resolution soft bodies), running
    # with multiple threads will improve performance.
    sdp.initialize(num_worker_threads=0)

    # Create simulation
    scene, soft_duck_actor, rigid_plane_actor = create_soft_duck_simulation()

    # Launch and attach the remote debugger for visualization and interaction.
    if sdp.debugger.attach():
        # Simulate until the debugger detaches.
        while sdp.debugger.is_attached():
            scene.step(TIME_STEP)
        print("Simulation complete.")

    cleanup_simulation(scene, soft_duck_actor, rigid_plane_actor)


if __name__ == "__main__":
    main()
