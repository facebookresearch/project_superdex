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

"""Example: Damping Parameter Sweep

Drops two parallel rows of ducks, built from the same tetrahedral mesh asset,
onto a static ground plane:

- A row of soft actors sweeping the stiffness-damping coefficient,
  which dissipates energy within the deforming material.
- A row of rigid actors sweeping the normal viscous contact damping
  coefficient, which dissipates energy during impact with the ground.
"""

from __future__ import annotations

import superdex.physics as physics
from superdex.physics import Actor, Scene
from superdex.physics.paths import resolve_asset

# Use a small time step to resolve dynamics with minimal numerical dissipation.
TIME_STEP = 1.0 / 300.0  # [s]

# A single scale factor sizes the demonstration scene. It is baked into the duck
# shape and also drives the layout of the two rows.
DUCK_SCALE = 0.5
DROP_HEIGHT = DUCK_SCALE  # [m]
COLUMN_SPACING = 1.5 * DUCK_SCALE  # [m]
SOFT_ROW_Z = -2.0 * DUCK_SCALE  # [m]
RIGID_ROW_Z = 0.0  # [m]

# Stiffness-damping coefficients swept by the soft row [s].
STIFFNESS_DAMPING_COEFFICIENTS = (0.001, 0.002, 0.003, 0.004)  # [s]

# Normal viscous contact damping coefficients swept by the rigid row [s/m].
# These are the effective coefficients of each duck-ground pair, not the values
# stored on the actors; see GROUND_NORMAL_DAMPING_COEFFICIENT.
EFFECTIVE_NORMAL_DAMPING_COEFFICIENTS = (0.1, 0.2, 0.3, 0.4)  # [s/m]

# Dissipation coefficients are combined over a contact pair as a geometric
# mean, so the ground must carry a non-zero coefficient for any normal damping
# to act. Fixing it at unity makes each duck's own coefficient the square of the
# effective coefficient of its pair with the ground.
GROUND_NORMAL_DAMPING_COEFFICIENT = 1.0  # [s/m]


def _column_x(index: int, count: int) -> float:
    """Center a row of `count` evenly spaced columns on the origin."""
    return (index - 0.5 * (count - 1)) * COLUMN_SPACING


def create_damping_sweep_simulation() -> tuple[Scene, list[Actor], list[Actor]]:
    """Create the damping parameter sweep simulation.

    Note: physics.initialize() must be called before this function.

    Returns:
        tuple: (scene, soft_duck_actors, rigid_duck_actors)
    """
    scene = physics.create_scene("Damping Sweep Scene")

    # BDF2 is second-order accurate and A-stable. It introduces far less
    # numerical dissipation than the default first-order backward Euler, so the
    # damping seen in the simulation is dominated by the physical damping
    # parameters swept below rather than by the time integrator.
    solver_params = scene.get_solver_params()
    solver_params.integration_method = physics.IntegrationMethod.BDF2
    scene.set_solver_params(solver_params)

    # Static ground plane at y = 0, carrying the reference contact damping
    # coefficient that the rigid ducks' coefficients are paired against.
    plane_shape = physics.create_plane_shape(normal=[0, 1, 0], distance=0)
    scene.create_rigid_actor(
        name="ground",
        shape=plane_shape,
        is_static=True,
        contact=physics.ContactParams(
            normal_viscous_damping_coefficient=GROUND_NORMAL_DAMPING_COEFFICIENT
        ),
    )

    # Both rows are built from the same tetrahedral mesh asset. Loading it once
    # lets every actor share a single shape, including its baked collider data.
    duck_shape = physics.load_shape_from_file(
        file_path=str(resolve_asset("duck/duck_coarse.mochi.h5")),
        bake_scale=[DUCK_SCALE] * 3,
    )

    num_columns = len(STIFFNESS_DAMPING_COEFFICIENTS)
    soft_duck_actors = [
        scene.create_soft_actor(
            name=f"soft_duck_{1e3 * stiffness_damping:g}ms",
            shape=duck_shape,
            world_from_local=physics.TransformRT(
                translation=[_column_x(index, num_columns), DROP_HEIGHT, SOFT_ROW_Z]
            ),
            material=physics.SoftMaterialParams(
                stiffness_damping_coefficient=stiffness_damping
            ),
        )
        for index, stiffness_damping in enumerate(STIFFNESS_DAMPING_COEFFICIENTS)
    ]

    num_columns = len(EFFECTIVE_NORMAL_DAMPING_COEFFICIENTS)
    rigid_duck_actors = [
        scene.create_rigid_actor(
            name=f"rigid_duck_{effective_damping:g}spm",
            shape=duck_shape,
            world_from_local=physics.TransformRT(
                translation=[_column_x(index, num_columns), DROP_HEIGHT, RIGID_ROW_Z]
            ),
            contact=physics.ContactParams(
                normal_viscous_damping_coefficient=(
                    effective_damping**2 / GROUND_NORMAL_DAMPING_COEFFICIENT
                )
            ),
        )
        for index, effective_damping in enumerate(EFFECTIVE_NORMAL_DAMPING_COEFFICIENTS)
    ]

    return scene, soft_duck_actors, rigid_duck_actors


def main():
    """Main function that runs the interactive damping sweep simulation."""

    # Select the number of worker threads automatically for performance with
    # multiple soft actors.
    physics.initialize(num_worker_threads=-1)

    scene, _, _ = create_damping_sweep_simulation()

    # Launch and attach the remote debugger for visualization and interaction.
    if not physics.debugger.attach():
        physics.shutdown()
        return

    # Simulate until the debugger detaches
    while physics.debugger.is_attached():
        scene.step(TIME_STEP)

    physics.shutdown()
    print("Simulation complete.")


if __name__ == "__main__":
    main()
