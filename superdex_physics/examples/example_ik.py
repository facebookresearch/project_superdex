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

"""Example: Inverse kinematics

Solves inverse kinematics on a five-link articulation with mixed joint types.
The solver alternates between a rotation target and a position target on the
last link, re-solving each iteration so the arm snaps to a fresh random goal.

SuperDex Physics solves IK by reusing the physics engine as a quasistatic
optimizer, so the solve runs inside a real scene and respects every constraint
already in it. The root joint here is ``Free``, so two rigid pivot constraints
pin the base; without them the optimizer would reach each target by translating
the whole articulation instead of bending it.

The solver takes ownership of its scene and reconfigures it (infinite timestep,
no gravity, no friction), which makes the scene unusable for dynamic
simulation. This example is IK-only and creates a single dedicated scene.

Keep the debugger attached to watch the arm track each new target; the loop
runs until the debugger detaches.

Usage:
    cd superdex/superdex_physics
    python3 examples/example_ik.py
"""

from __future__ import annotations

import random
import time

import superdex.physics as physics
from superdex.physics import Actor, Scene
from superdex.physics.paths import resolve_asset, resolve_asset_root

PREFAB_PATH = "articulated/mixed/mixed_articulation.mochi_prefab"

# Link 0 is the base, whose inbound jointA is Free; the pivot constraints below
# hold it in place. Link 4 is the end effector, reached through jointE
# (Spherical). jointB, jointC and jointD are revolute, for 12 DoFs in total.
ROOT_LINK = 0

# Uniform scale applied when the prefab is instantiated. Every length constant
# below is expressed in terms of it, because prefab scale bakes into the mesh
# geometry and joint translations but not into anything defined here.
PREFAB_SCALE = 2.0

ROOT_PIVOT_LOCAL_POSITION = [0.05 * PREFAB_SCALE, 0, 0]  # [m]
# Prefab scale deliberately leaves constraint stiffness alone, so this keeps its
# authored value. Gravity is off and the solve is quasistatic, so the heavier
# scaled base does not need a stiffer pin.
ROOT_PIVOT_STIFFNESS = 1e4  # [N/m] and [N*m/rad]

# The unscaled end-effector box is 0.1 m long, so a position target offset half a
# length past the centre of mass aims at its tip rather than its middle.
END_EFFECTOR_TIP_OFFSET = 0.05 * PREFAB_SCALE  # [m]

# World targets are sampled around the origin, where the pinned base sits. The
# position range stays well inside the arm's reach so many solves converge; the
# rotation range is wider because orientation is cheaper to satisfy.
POSITION_TARGET_RANGE = 0.1 * PREFAB_SCALE  # [m]
ROTATION_TARGET_RANGE = 0.5  # [rad], scale invariant
TARGET_WEIGHT = 1.0

# Use an intentionally conservative Newton iteration cap for illustration
# purposes. In the tested example workload, this cap and the default cap
# produced the same reachability count. Reachability uses separate target-error
# thresholds, which can be met without satisfying Newton's residual tolerances.
MAX_ITER = 500

REPORT_INTERVAL = 10
RANDOM_SEED = 0

# The solve is quasistatic, so the arm snaps to each new target in one step.
# Holding a target for a second makes that motion followable instead of a blur.
TARGET_HOLD_SECONDS = 1.0  # [s]


def load_articulation(scene: Scene) -> Actor:
    """Load the mixed-joint articulation and pin its free-floating base."""
    prefab_params = physics.prefab.PrefabParams()
    prefab_params.scale = PREFAB_SCALE
    result = physics.prefab.add_to_scene(
        prefab_path=str(resolve_asset(PREFAB_PATH)),
        root_path=str(resolve_asset_root(PREFAB_PATH)),
        scene=scene,
        params=prefab_params,
    )
    articulation = result.actors[0]

    # The links are boxes that overlap at the joints; without this they would
    # push each other apart during the solve.
    scene.enable_layer_contact_symmetric("RigidLink", "RigidLink", enable=False)

    root = articulation.get_nested_link_actors()[ROOT_LINK]

    position_params = physics.RigidPivotPositionConstraintParams()
    position_params.local_position = ROOT_PIVOT_LOCAL_POSITION
    position_params.target_position = [0, 0, 0]
    position_params.actor = root
    position_params.stiffness = ROOT_PIVOT_STIFFNESS
    scene.create_rigid_pivot_position_constraint(position_params)

    rotation_params = physics.RigidPivotRotationConstraintParams()
    rotation_params.local_rotation = [0, 0, 0]
    rotation_params.target_rotation = [0, 0, 0]
    rotation_params.actor = root
    rotation_params.stiffness = ROOT_PIVOT_STIFFNESS
    scene.create_rigid_pivot_rotation_constraint(rotation_params)

    return articulation


def set_rotation_target(ik_solver, end_effector, rng: random.Random) -> list[float]:
    """Replace the active target with a random end-effector orientation."""
    # Rotation targets are rotation vectors -- an axis scaled by the angle in
    # radians -- not Euler angles or a quaternion.
    target = [
        rng.uniform(-ROTATION_TARGET_RANGE, ROTATION_TARGET_RANGE) for _ in range(3)
    ]
    ik_solver.clear_position_target(end_effector)
    ik_solver.create_rotation_target(end_effector, [0, 0, 0], target, TARGET_WEIGHT)
    return target


def set_position_target(
    ik_solver, scene: Scene, end_effector, rng: random.Random
) -> list[float]:
    """Replace the active target with a random end-effector tip position."""
    target = [
        rng.uniform(-POSITION_TARGET_RANGE, POSITION_TARGET_RANGE) for _ in range(3)
    ]
    ik_solver.clear_rotation_target(end_effector)

    com_local = scene.get_actor(end_effector).get_rigid_center_of_mass_local()
    tip_local = [
        END_EFFECTOR_TIP_OFFSET + com_local[0],
        com_local[1],
        com_local[2],
    ]
    ik_solver.create_position_target(end_effector, tip_local, target, TARGET_WEIGHT)
    return target


def run_interactive(ik_solver, scene: Scene, end_effector) -> None:
    """Alternate position and rotation targets while the debugger is attached."""
    if not physics.debugger.attach():
        return

    rng = random.Random(RANDOM_SEED)
    solves = 0
    reached = 0
    converged = 0
    want_rotation = True

    while physics.debugger.is_attached():
        if want_rotation:
            target = set_rotation_target(ik_solver, end_effector, rng)
            kind = "rotation [rad]"
        else:
            target = set_position_target(ik_solver, scene, end_effector, rng)
            kind = "position [m]"
        want_rotation = not want_rotation

        # Returns True only when every active target is met within the solver's
        # error thresholds. A random target may simply be out of reach.
        if ik_solver.solve_ik():
            reached += 1
        solves += 1

        # Whether the Newton solve itself converged, which is independent of
        # whether the target it converged towards was reachable.
        status = scene.get_solver_stats().convergence_status
        if status == physics.ConvergenceStatus.CONVERGED:
            converged += 1

        if solves % REPORT_INTERVAL == 0:
            print(
                f"solve {solves}: {kind} target="
                f"{[round(value, 3) for value in target]}, "
                f"{converged}/{solves} converged, {reached}/{solves} reached"
            )

        # Detaching during the hold is only noticed once it elapses.
        time.sleep(TARGET_HOLD_SECONDS)

    print(
        f"IK solved {solves} times; {converged} solves converged, "
        f"{reached} targets reached."
    )


def main() -> None:
    """Run the interactive inverse-kinematics tutorial."""
    physics.initialize(num_worker_threads=0)

    # The solver reconfigures and takes ownership of this scene.
    ik_scene = physics.create_scene("Inverse Kinematics Scene")
    ik_scene.set_gravity([0, 0, 0])
    articulation = load_articulation(ik_scene)
    # SpanConstActorHandle does not support negative indexing.
    links = articulation.get_nested_link_actors()
    end_effector = links[len(links) - 1]

    ik_solver = physics.experimental.create_ik_solver(ik_scene)
    ik_solver_params = ik_solver.get_solver_params()
    ik_solver_params.max_iter = MAX_ITER
    ik_solver.set_solver_params(ik_solver_params)

    run_interactive(ik_solver, ik_scene, end_effector)

    # Destroys ik_scene as well; do not destroy the scene separately.
    physics.experimental.destroy_ik_solver(ik_solver)

    physics.shutdown()
    print("Simulation complete.")


if __name__ == "__main__":
    main()
