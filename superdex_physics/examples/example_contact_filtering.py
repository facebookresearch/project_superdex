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

"""Example: Contact Filtering

Demonstrates contact filtering using layers and per-actor settings. Shows how
to selectively enable/disable contact between groups of objects and between
specific actor pairs.
"""

import superdex.physics as physics
from superdex.physics import Actor, Scene
from superdex.physics.paths import resolve_asset


def create_contact_filtering_simulation() -> tuple[Scene, list[Actor]]:
    """Create a contact filtering demonstration simulation.

    Note: physics.initialize() must be called before this function.

    Returns:
        tuple: (scene, list of all actors)
    """
    # Create scene
    scene = physics.create_scene("Contact Filtering Scene")

    # Create shapes
    # - Cube is loaded from file
    # - Platforms use an implicit plane shape (no mesh data needed)
    cube_shape = physics.load_shape_from_file(
        file_path=str(resolve_asset("cube/cube_mesh.mochi.h5")),
        bake_scale=[0.3, 0.3, 0.3],
    )
    platform_shape = physics.create_plane_shape(normal=[0, 1, 0], distance=0.0)

    actors = []

    # Create static platforms:
    # - platform_1 in layer_1
    # - platform_2 in layer_2
    platform_1 = scene.create_rigid_actor(
        name="platform_1",
        layer="layer_1",
        shape=platform_shape,
        is_static=True,
        world_from_local=physics.TransformRT(translation=[0, 0.5, 0]),
    )
    actors.append(platform_1)

    platform_2 = scene.create_rigid_actor(
        name="platform_2",
        layer="layer_2",
        shape=platform_shape,
        is_static=True,
        world_from_local=physics.TransformRT(translation=[0, 0, 0]),
    )
    actors.append(platform_2)

    # Create two stacks of cubes, each in a different layer.
    # Note: By default, contact between layers is ENABLED and contact between
    # actors is ENABLED. Contact only occurs if BOTH are enabled.
    cube_size = 0.3
    stack_spacing = 1.0
    cube_vertical_gap = 0.01
    stack_base_height = 1.5
    cube_names = ["bottom", "middle", "top"]

    # Stack 1 goes in layer_1 and stack 2 in layer_2. What each one lands on is
    # decided by the layer filter set up below.
    cubes = {}
    for stack_idx, (layer, x_pos) in enumerate(
        [("layer_1", -stack_spacing / 2), ("layer_2", stack_spacing / 2)], start=1
    ):
        for cube_idx, name in enumerate(cube_names):
            y_pos = (
                cube_size * (cube_idx + 0.5)
                + cube_vertical_gap * cube_idx
                + stack_base_height
            )
            cube_name = f"{name}_cube_{stack_idx}"
            cube = scene.create_rigid_actor(
                name=cube_name,
                layer=layer,
                shape=cube_shape,
                world_from_local=physics.TransformRT(translation=[x_pos, y_pos, 0]),
            )
            actors.append(cube)
            cubes[cube_name] = cube

    # Layer-based filtering: Disable contact between layer_1 and layer_2.
    # Stack 1 (layer_1) will land on the higher platform_1 (layer_1).
    # Stack 2 (layer_2) will pass through platform_1 and land on platform_2 (layer_2).
    # Note: This example uses symmetric filtering, which affects contact checks
    # in both directions. For one-directional filtering, use
    # enable_layer_contact_asymmetric(layer_a, layer_b, enable=False) instead.
    scene.enable_layer_contact_symmetric("layer_1", "layer_2", enable=False)

    # Actor-based filtering: Disable contact between bottom_cube_2 and middle_cube_2.
    # This demonstrates per-actor filtering: middle_cube_2 will fall through
    # bottom_cube_2, but top_cube_2 still lands on middle_cube_2.
    # Note: This example uses symmetric filtering, which affects contact checks
    # in both directions. For one-directional filtering, use
    # enable_actor_contact_asymmetric(
    #     colliding, collider, enable=False, include_nested_actors=physics.IncludeNestedActors.NO
    # ) instead.
    scene.enable_actor_contact_symmetric(
        cubes["bottom_cube_2"].get_handle(),
        cubes["middle_cube_2"].get_handle(),
        enable=False,
        include_nested_actors=physics.IncludeNestedActors.NO,
    )

    return scene, actors


def cleanup_simulation(scene: Scene, actors: list[Actor]):
    """Clean up simulation resources.

    Args:
        scene: Physics scene
        actors: List of all actors to destroy
    """
    # This is how you destroy individual actors.
    # Not necessary if you're going to destroy the whole scene.
    for actor in actors:
        scene.destroy_actor(actor)

    # This is how you destroy an individual scene and everything in it.
    # Not necessary if you're shutting down.
    physics.destroy_scene(scene)

    # Shut down SuperDex Physics.
    # Not necessary unless you want to call initialize() again with
    # different values. Shown here just for completeness.
    physics.shutdown()


def main():
    """Main function that runs the interactive contact filtering simulation."""

    # Initialize SuperDex Physics and specify the number of worker threads to
    # use for simulation:
    #   0 = single-threaded (no additional threads beyond the calling thread)
    #  -1 = let SuperDex Physics choose based on your hardware
    #   N = use N worker threads in addition to the calling thread
    #
    # Run single-threaded to keep the example simple. For scenes with a large
    # number of DoFs (e.g. scenes with high-resolution soft bodies), running
    # with multiple threads will improve performance.
    physics.initialize(num_worker_threads=0)

    # Create simulation
    scene, actors = create_contact_filtering_simulation()

    # Simulation time step in seconds
    time_step = 1.0 / 60.0

    # Launch and attach the remote debugger for visualization and interaction.
    if not physics.debugger.attach():
        cleanup_simulation(scene, actors)
        return

    # Simulate until the debugger detaches
    while physics.debugger.is_attached():
        scene.step(time_step)

    # Clean up simulation
    cleanup_simulation(scene, actors)
    print("Simulation complete.")


if __name__ == "__main__":
    main()
