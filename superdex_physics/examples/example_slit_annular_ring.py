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

"""Simulate the slit annular ring nonlinear shell benchmark documented by Sze et al.
in https://doi.org/10.1016/j.finel.2003.11.001 and used to evaluate shell formulations
in many other references.

A transverse load opens the ring's slit while the other slit edge is clamped.
The tracked outer-edge displacement is compared with the benchmark's steady solution.
"""

from __future__ import annotations

import numpy as np
import superdex.physics as physics
from superdex.physics import Actor, Scene
from superdex.physics.paths import resolve_asset

np_real = np.float64 if physics.uses_double_precision() else np.float32

# Node indices for the benchmark mesh in slit_annular_ring.mochi.h5.
FIXED_NODES = np.array(
    [0, 1, 65, 66, 130, 131, 195, 196, 260, 261],
    dtype=np.int32,
)
LOADED_NODES = np.array([64, 129, 194, 259, 324], dtype=np.int32)
TRACKED_NODE = 324

# Shell material parameters.
YOUNGS_MODULUS = 210e6  # [Pa]
POISSONS_RATIO = 0.0
DENSITY = 1.0e3  # [kg/m^3]
THICKNESS = 3.0e-3  # [m]
MASS_DAMPING_COEFFICIENT = 0.3  # [1/s]

# Loading and reference response.
TOTAL_TRANSVERSE_FORCE = 0.32  # [N]
REFERENCE_TRACKED_Z_DISPLACEMENT = 1.75  # [m]

# Time integration and terminal reporting.
TIME_STEP = 1.0 / 30.0  # [s]
REPORT_INTERVAL = 1.0  # [s]
REPORT_INTERVAL_STEPS = round(REPORT_INTERVAL / TIME_STEP)


def create_simulation() -> tuple[Scene, Actor]:
    """Create the slit annular ring shell simulation."""
    scene = physics.create_scene("Slit Annular Ring")
    scene.set_gravity([0, 0, 0])

    shape = physics.load_shape_from_file(
        str(resolve_asset("samples/slit_annular_ring.mochi.h5"))
    )
    # Derive shell stiffness and inertia from familiar 3D material properties.
    material = physics.experimental.shell_material_params_from3d_isotropic(
        youngs_modulus3d=YOUNGS_MODULUS,
        poissons_ratio3d=POISSONS_RATIO,
        density3d=DENSITY,
        thickness=THICKNESS,
    )
    # Use heavy mass damping to drive the dynamics toward the static solution.
    material.mass_damping_coefficient = MASS_DAMPING_COEFFICIENT
    actor = physics.experimental.create_shell_actor(
        scene,
        physics.experimental.ShellActorParams(
            name="SlitAnnularRing",
            shape=shape,
            material=material,
            world_from_local=physics.TransformRT(),
            has_gravity=False,
        ),
    )
    if actor is None:
        raise RuntimeError("Failed to create the slit annular ring shell actor.")

    # Fix the first two nodes on each radial ring at their reference positions.
    coordinates = np.asarray(
        list(physics.get_shape_mesh(shape).coordinates), dtype=np_real
    )
    fixed_positions = coordinates.reshape(-1, 3)[FIXED_NODES].reshape(-1)
    actor.add_boundary_condition_nodes_world(
        node_indices=FIXED_NODES, node_positions_world=fixed_positions
    )

    # Each shell node has x, y, z DoFs; load the z DoF of each slit-edge node.
    loaded_dofs = 3 * LOADED_NODES + 2
    force_values = np.full(
        len(LOADED_NODES), TOTAL_TRANSVERSE_FORCE / len(LOADED_NODES), dtype=np_real
    )
    actor.set_external_forces_on_dofs(
        dof_indices=loaded_dofs, force_values=force_values
    )
    return scene, actor


def get_tracked_displacement(actor: Actor) -> list[float]:
    """Return the tracked node's three displacement components."""
    offset = 3 * TRACKED_NODE
    displacements = actor.get_displacements()
    return [displacements[offset + axis] for axis in range(3)]


def main() -> None:
    """Run the interactive simulation and report the tracked-node displacement."""
    # Run on the calling thread.
    physics.initialize(num_worker_threads=0)
    try:
        scene, actor = create_simulation()

        # Launch the debugger for visualization and interactive playback control.
        if not physics.debugger.attach():
            return

        print(
            "Displacement telemetry will appear after the simulation starts.",
            flush=True,
        )
        # Advance until the debugger detaches, reporting once per simulated second.
        step_count = 0
        while physics.debugger.is_attached():
            scene.step(TIME_STEP)
            step_count += 1
            if step_count % REPORT_INTERVAL_STEPS == 0:
                displacement_z = get_tracked_displacement(actor)[2]
                print(
                    f"t={scene.get_total_simulation_time():.1f} s: node {TRACKED_NODE} "
                    f"u_z={displacement_z:.6g} m; steady reference="
                    f"{REFERENCE_TRACKED_Z_DISPLACEMENT:.6g} m",
                    flush=True,
                )

        displacement = get_tracked_displacement(actor)
        print(f"Node {TRACKED_NODE} final displacement [m]: {displacement}")
    finally:
        physics.shutdown()


if __name__ == "__main__":
    main()
