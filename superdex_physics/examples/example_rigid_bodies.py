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

"""Example: Rigid Bodies

Simulates a sphere and cube falling onto a table under gravity. Demonstrates
four ways to create rigid body actors:
1. Implicit shape function (ground plane)
2. Mesh shape loaded from file (sphere)
3. Mesh shape defined programmatically (cube)
4. Actor loaded from a prefab file (table)
"""

import math

import numpy as np
import superdex.physics as physics
from superdex.physics import Actor, Scene
from superdex.physics.paths import resolve_asset, resolve_asset_root
from superdex.physics.utils.scene_helpers import find_actor

# SuperDex Physics uses fully-implicit integration, enabling substantially larger
# stable time steps than explicit or semi-implicit methods. This simple example uses
# 16.7 ms per step. In practice, 10–25 ms steps remain robust across a broad range of
# simulations, including complex non-convex contact and deformable actors such as
# soft bodies, shells, and rods.
TIME_STEP = 1.0 / 60.0  # [s]


def create_coarse_cube_shape(size: float) -> physics.ShapeHandle:
    """Create a cube mesh shape from vertex coordinates and face connectivity.

    This function demonstrates creating a mesh shape programmatically using
    create_tri_mesh_shape(). Alternatively, you could load a cube mesh from
    a file using

    cube_shape = physics.load_shape_from_file(
        file_path=str(resolve_asset("cube/cube_mesh.mochi.h5")),
        bake_scale=[size, size, size]
    )

    Note:
        This mesh is intentionally coarse (8 vertices, 12 triangles) to keep the
        example simple. In practice, use finer meshes for accurate contact
        simulation.

    Args:
        size: The side length of the cube in meters.

    Returns:
        ShapeHandle for the cube mesh shape.
    """
    half = size / 2.0

    # Define the 8 vertices of a cube centered at origin
    coordinates = np.array(
        [
            [-half, -half, -half],  # 0: back-bottom-left
            [half, -half, -half],  # 1: back-bottom-right
            [half, half, -half],  # 2: back-top-right
            [-half, half, -half],  # 3: back-top-left
            [-half, -half, half],  # 4: front-bottom-left
            [half, -half, half],  # 5: front-bottom-right
            [half, half, half],  # 6: front-top-right
            [-half, half, half],  # 7: front-top-left
        ],
        dtype=np.float32,
    ).flatten()

    # Define the 12 triangular faces (2 triangles per cube face)
    # Counter-clockwise winding order for outward-facing normals
    connectivity = np.array(
        [
            # Back face (-Z)
            [0, 2, 1],
            [0, 3, 2],
            # Front face (+Z)
            [4, 5, 6],
            [4, 6, 7],
            # Left face (-X)
            [0, 4, 7],
            [0, 7, 3],
            # Right face (+X)
            [1, 2, 6],
            [1, 6, 5],
            # Bottom face (-Y)
            [0, 1, 5],
            [0, 5, 4],
            # Top face (+Y)
            [2, 3, 7],
            [2, 7, 6],
        ],
        dtype=np.int32,
    ).flatten()

    return physics.create_tri_mesh_shape(
        coordinates=coordinates, connectivity=connectivity
    )


def create_rigid_bodies_simulation() -> tuple[Scene, Actor, Actor, Actor, Actor]:
    """Create a rigid bodies physics simulation.

    Note: physics.initialize() must be called before this function.

    Returns:
        tuple: (scene, sphere_actor, cube_actor, ground_actor, table_actor)
    """
    # Create scene
    scene = physics.create_scene("Rigid Bodies Scene")

    # SuperDex Physics uses Y-up coordinates and defaults to gravity along -Y:
    # (0, -9.8, 0). Set it explicitly for illustration.
    # Note: SuperDex Robotics uses Z-up coordinates. Its examples and SuperDex
    # Studio's default configuration set gravity along -Z.
    scene.set_gravity([0, -9.8, 0])

    # Create shapes using different methods:
    # - Ground plane uses implicit shape function (no mesh data needed)
    # - Sphere is loaded from a mesh file (icosphere with 3 subdivisions)
    # - Cube uses explicit mesh vertices and faces defined programmatically
    # - Table is directly loaded from a prefab file
    plane_shape = physics.create_plane_shape(normal=[0, 1, 0], distance=-1.0)
    sphere_shape = physics.load_shape_from_file(
        file_path=str(resolve_asset("sphere/icosphere_3subdiv.1.mochi.json")),
        bake_scale=[0.2, 0.2, 0.2],  # Scale to radius 0.2
    )
    cube_shape = create_coarse_cube_shape(size=0.4)

    # Create dynamic sphere (mesh-based shape loaded from file)
    sphere_actor = scene.create_rigid_actor(
        name="sphere",
        shape=sphere_shape,
        is_static=False,  # Dynamic actor (default)
        density=1000.0,  # Density in kg/m³
        world_from_local=physics.TransformRT(translation=[-0.5, 0.2, 0]),
        collider_type=physics.ColliderType.SPHERE,
    )

    # Create dynamic cube (mesh-based shape)
    cube_actor = scene.create_rigid_actor(
        name="cube",
        shape=cube_shape,
        is_static=False,  # Dynamic actor (default)
        density=1000.0,  # Density in kg/m³
        world_from_local=physics.TransformRT(translation=[0.5, 0.2, 0]),
        collider_type=physics.ColliderType.BOX,
    )

    # Create static ground plane (implicit shape)
    ground_actor = scene.create_rigid_actor(
        name="ground",
        shape=plane_shape,
        is_static=True,
    )

    # Create table from a prefab file.
    # Prefabs are JSON files that declaratively define actors and their properties.
    # They are general-purpose: a single prefab can contain any combination of
    # supported actor types, constraints, controllers, and even nested references
    # to other prefabs.
    # Here we load a simple prefab containing one static rigid actor (a table).
    table_prefab_path = str(resolve_asset("table/table.mochi_scene"))
    physics.prefab.add_to_scene(
        prefab_path=table_prefab_path,
        root_path=str(resolve_asset_root("table/table.mochi_scene")),
        scene=scene,
        params=physics.prefab.PrefabParams(
            name="tablePrefab",
            # Turn the table 90 degrees to make it horizontal
            rotation=physics.Quaternion.rotation_x(-90 * math.pi / 180),
            translation=[0, -1.0, 0],
        ),
    )

    # Find the table actor created by the prefab.
    # When a prefab name is specified in PrefabParams, actors are accessed via
    # "prefabName/actorName", where "actorName" is the actor name defined in the
    # JSON. When no prefab name is specified, actors are accessed via "actorName".
    table_actor = find_actor(scene, "tablePrefab/Table")

    return scene, sphere_actor, cube_actor, ground_actor, table_actor


def cleanup_simulation(
    scene: Scene,
    sphere_actor: Actor,
    cube_actor: Actor,
    ground_actor: Actor,
    table_actor: Actor,
) -> None:
    """Demonstrate actor, scene, and global resource cleanup."""
    # Destroying individual actors is not required before destroying their scene;
    # these calls are included to demonstrate the actor lifecycle.
    scene.destroy_actor(sphere_actor)
    scene.destroy_actor(cube_actor)
    scene.destroy_actor(ground_actor)
    scene.destroy_actor(table_actor)

    # Destroying this scene is not required immediately before shutdown; this
    # call demonstrates how to release a scene while SuperDex Physics remains active.
    physics.destroy_scene(scene)

    # Shutdown releases any remaining global resources. It is shown explicitly
    # so this example can be initialized again in the same process.
    physics.shutdown()


def main() -> None:
    """Main function that runs the interactive rigid bodies simulation."""

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
    scene, sphere_actor, cube_actor, ground_actor, table_actor = (
        create_rigid_bodies_simulation()
    )

    # Launch and attach the remote debugger for visualization and interaction.
    if physics.debugger.attach():
        # Simulate until the debugger detaches.
        while physics.debugger.is_attached():
            scene.step(TIME_STEP)
        print("Simulation complete.")

    cleanup_simulation(scene, sphere_actor, cube_actor, ground_actor, table_actor)


if __name__ == "__main__":
    main()
