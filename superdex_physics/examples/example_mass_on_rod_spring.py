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

"""Example: Mass on Rod Spring

Simulates a rigid cube hanging from a helical spring modeled as an elastic rod.

It demonstrates:
- Loading rod geometry from a file (helix shape with a tubular visual mesh)
- Creating a rod actor with custom material parameters
- Constraining one end with a position constraint
- Attaching a rigid cube mass to the far end using both position and rotation constraints
- Sizing constraint stiffnesses by dimensional analysis of the rod's own stiffnesses

See website/docs/examples/tendons_rods/mass_on_rod_spring.md.
"""

import math

import superdex.physics as physics
from superdex.physics import Actor, Scene
from superdex.physics.paths import resolve_asset


# Material and geometry constants
RADIUS = 1e-2  # Rod cross-section radius [m]
SHEAR_MODULUS = 1e9  # Shear modulus [Pa]
YOUNGS_MODULUS = 1e9  # Young's modulus [Pa]
DENSITY = 1e3  # Material density [kg/m^3]
CUBE_SIZE = 0.2  # Rigid cube size [m]

# Length scale [m] relating the rod's stiffness coefficients to the constraint
# stiffnesses. Holding it fixed gives the constraints a fixed compliance, independent
# of how finely the rod is discretized. Making it proportional to the element size
# instead would let the constraints converge to hard constraints under refinement.
CONSTRAINT_STIFFNESS_LENGTH_SCALE = 1.0  # [m]


def create_mass_on_rod_spring_simulation() -> tuple[Scene, Actor, Actor]:
    """Create a physics simulation with a rod spring and mass.

    Note: physics.initialize() must be called before this function.

    Returns:
        tuple: (scene, rod_actor, cube_actor)
    """
    # Create physics scene
    scene = physics.create_scene("Mass on Rod Spring Scene")

    # Load the polyline shape from file (helix). The asset's coil radius tapers
    # smoothly to zero at both ends, so the spring is a single analytical space
    # curve terminating on the x axis, and it carries a tubular visual mesh.
    helix_path = str(resolve_asset("rods/helix_with_visual.mochi.h5"))
    shape = physics.load_shape_from_file(helix_path, bake_scale=[1.0, 1.0, 1.0])

    # Calculate material parameters for circular cross-section
    area = math.pi * RADIUS**2
    polar_moment_of_inertia = 0.5 * math.pi * RADIUS**4
    second_moment_of_area = 0.25 * math.pi * RADIUS**4
    torsion_constant = 0.5 * math.pi * RADIUS**4

    # Stiffness coefficients: EA [N], GJ [N*m^2] and EI [N*m^2]
    axial_stiffness = YOUNGS_MODULUS * area
    torsional_stiffness = SHEAR_MODULUS * torsion_constant
    flexural_stiffness = YOUNGS_MODULUS * second_moment_of_area

    # Create rod material parameters
    material_params = physics.experimental.RodMaterialParams(
        linear_density=DENSITY * area,
        linear_rotational_inertia=DENSITY * polar_moment_of_inertia,
        axial_stiffness=axial_stiffness,
        torsional_stiffness=torsional_stiffness,
        flexural_stiffness=[flexural_stiffness, flexural_stiffness],
    )

    # Create rod actor params
    rod_params = physics.experimental.RodActorParams(
        name="Spring",
        shape=shape,
        world_from_local=physics.TransformRT(),
        material=material_params,
    )

    # Create rod actor
    rod_actor = physics.experimental.create_rod_actor(scene, rod_params)

    # Reference node positions of the rod, flattened as [x0, y0, z0, x1, y1, z1, ...].
    # They are in the actor's local frame, which coincides with the world frame here
    # because world_from_local is the identity.
    coordinates = list(rod_actor.get_mesh().coordinates)
    num_nodes = len(coordinates) // 3
    last_node_index = num_nodes - 1
    last_element_index = num_nodes - 2
    rod_near_end_position = coordinates[0:3]
    rod_far_end_position = coordinates[-3:]

    # Constraint stiffnesses derived from the rod's own stiffness coefficients, so they
    # stay consistent if the cross-section or material changes. Dividing by a length
    # turns EA [N] into [N/m] and GJ [N*m^2] into [N*m/rad], the units the translation
    # and rotation constraints expect.
    position_stiffness = axial_stiffness / CONSTRAINT_STIFFNESS_LENGTH_SCALE
    rotation_stiffness = torsional_stiffness / CONSTRAINT_STIFFNESS_LENGTH_SCALE

    # Apply position constraint to fix the first node
    scene.create_deformable_node_position_constraint(
        actor=rod_actor.get_handle(),
        node_index=0,
        position=rod_near_end_position,
        stiffness=position_stiffness,
    )

    # Create rigid cube at the far end of the rod
    cube_half_extent = 0.5 * CUBE_SIZE
    # Load cube mesh from asset file (scaled to CUBE_SIZE)
    cube_mesh_path = str(resolve_asset("cube/cube_mesh.mochi.json"))
    cube_shape = physics.load_shape_from_file(
        cube_mesh_path, bake_scale=[CUBE_SIZE, CUBE_SIZE, CUBE_SIZE]
    )

    # Position cube so that its -x face coincides with the rod far end
    cube_position = [
        rod_far_end_position[0],
        rod_far_end_position[1] - cube_half_extent,
        rod_far_end_position[2] - cube_half_extent,
    ]

    cube_actor = scene.create_rigid_actor(
        name="Mass",
        layer="Object",
        shape=cube_shape,
        density=DENSITY,
        collider_type=physics.ColliderType.BOX,
        world_from_local=physics.TransformRT(cube_position),
    )

    # Create DeformableNodeToRigidConstraint to attach the last node of the rod
    # to the -x face center of the cube
    scene.create_deformable_node_to_rigid_constraint(
        deformable_actor=rod_actor.get_handle(),
        rigid_actor=cube_actor.get_handle(),
        deformable_node_index=last_node_index,
        fix_to_deformable_pos=True,
        stiffness=position_stiffness,
    )

    # Create rotation constraint: last rod element to the rigid cube
    scene.create_rod_element_rotation_to_rigid_constraint(
        rigid_actor=cube_actor.get_handle(),
        rod_actor=rod_actor.get_handle(),
        element_index=last_element_index,
        ref_frame_rot_vec=[0.0, 0.0, 0.0],
        stiffness=rotation_stiffness,
    )

    return scene, rod_actor, cube_actor


def main():
    """Main function that runs the mass on rod spring simulation."""
    # Initialize SuperDex Physics.
    physics.initialize(num_worker_threads=0)

    # Create simulation
    scene, _, _ = create_mass_on_rod_spring_simulation()

    # Simulation time step in seconds
    time_step = 1.0 / 60.0

    # Launch and attach the remote debugger for visualization and interaction.
    if not physics.debugger.attach():
        physics.shutdown()
        return

    # Simulate until the debugger detaches
    while physics.debugger.is_attached():
        scene.step(time_step)

    # Shut down SuperDex Physics.
    physics.shutdown()
    print("Simulation complete.")


if __name__ == "__main__":
    main()
