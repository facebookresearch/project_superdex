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

"""Example: Soft Body with Visual Mesh

Simulates two identical soft ducks side by side. One uses an embedded visual
mesh for rendering, while the other renders the boundary surface of its coarse
tetrahedral simulation mesh.

The visual mesh vertices are embedded in the tetrahedral elements using
barycentric coordinates, so the visual mesh follows the simulated deformation.
"""

import superdex.physics as physics
from superdex.physics import Actor, Scene
from superdex.physics.paths import resolve_asset

TIME_STEP = 1.0 / 60.0  # [s]


def create_soft_duck_visual_mesh_simulation() -> tuple[Scene, Actor, Actor, Actor]:
    """Create a side-by-side visual-mesh comparison.

    Note: physics.initialize() must be called before this function.

    Returns:
        tuple: (scene, visual_mesh_actor, simulation_mesh_actor, ground_actor)
    """
    scene = physics.create_scene("Soft Body Visual Mesh Scene")

    # This asset contains both a coarse tetrahedral simulation mesh and an
    # embedded triangular visual mesh.
    shape_with_visual_mesh = physics.load_shape_from_file(
        file_path=str(resolve_asset("duck/duck_coarse.mochi.h5")),
    )

    # Create a second shape from only the tetrahedral simulation mesh. Without
    # an embedded visual mesh, the debugger renders its derived boundary surface.
    simulation_mesh = physics.get_shape_mesh(shape_with_visual_mesh)
    shape_without_visual_mesh = physics.create_mesh_shape(simulation_mesh)

    visual_mesh_actor = scene.create_soft_actor(
        name="duck_with_visual_mesh",
        shape=shape_with_visual_mesh,
        world_from_local=physics.TransformRT(translation=[-1.0, 0.5, -0.5]),
    )
    simulation_mesh_actor = scene.create_soft_actor(
        name="duck_without_visual_mesh",
        shape=shape_without_visual_mesh,
        world_from_local=physics.TransformRT(translation=[0.0, 0.5, -0.5]),
    )

    plane_shape = physics.create_plane_shape(normal=[0, 1, 0], distance=0.0)
    ground_actor = scene.create_rigid_actor(
        name="ground", shape=plane_shape, is_static=True
    )

    return scene, visual_mesh_actor, simulation_mesh_actor, ground_actor


def cleanup_simulation(
    scene: Scene,
    visual_mesh_actor: Actor,
    simulation_mesh_actor: Actor,
    ground_actor: Actor,
) -> None:
    """Demonstrate actor, scene, and global resource cleanup."""
    # Destroying individual actors is not required before destroying their scene;
    # these calls are included to demonstrate the actor lifecycle.
    scene.destroy_actor(visual_mesh_actor)
    scene.destroy_actor(simulation_mesh_actor)
    scene.destroy_actor(ground_actor)

    # Destroying this scene is not required immediately before shutdown; this
    # call demonstrates how to release a scene while SuperDex Physics remains active.
    physics.destroy_scene(scene)

    # Shutdown releases any remaining global resources. It is shown explicitly
    # so this example can be initialized again in the same process.
    physics.shutdown()


def main() -> None:
    """Run the interactive visual-mesh comparison."""
    # Run single-threaded to keep the example simple. Use -1 to let SuperDex
    # Physics select a worker count, or a positive value to choose it explicitly.
    physics.initialize(num_worker_threads=0)

    scene, visual_mesh_actor, simulation_mesh_actor, ground_actor = (
        create_soft_duck_visual_mesh_simulation()
    )

    if physics.debugger.attach():
        while physics.debugger.is_attached():
            scene.step(TIME_STEP)
        print("Simulation complete.")

    cleanup_simulation(scene, visual_mesh_actor, simulation_mesh_actor, ground_actor)


if __name__ == "__main__":
    main()
