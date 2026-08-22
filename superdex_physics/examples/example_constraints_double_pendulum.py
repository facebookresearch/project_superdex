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

"""Example: Constraints

A double pendulum on a moving base, used as a guided tour of
the SuperDex Physics ``Constraint`` API. The scene has two rigid links:

* ``Link1`` is pinned to a fixed world point by a *rigid pivot-position*
  constraint (a 3-DoF ball joint to a world anchor). Rotation is free, so the
  link swings about the anchor. The anchor is animated over time to create a
  moving base.
* ``Link2`` hangs off the free end of ``Link1`` via a *rigid spherical joint*
  (a body-to-body ball joint).

One minimal scene therefore covers a world anchor, a body-to-body joint, and a
time-varying target, while exercising the full generic ``Constraint``
interface: creation, enumeration, introspection, parameter tuning, per-step
diagnostics (deviation / force), target animation, and destruction.

Note: modeling a double pendulum with constraints (as done here) is the
preferred approach when joints are added or removed at runtime. For a fixed
joint structure, prefer an articulated actor instead -- it is more efficient
and robust.

The console walks through the one-shot API sections at startup. It then
attaches the debugger for visualization as the simulation animates the moving
base and reports constraint forces and pivot deviation to the console. Finally,
it runs a short timeline that weakens the joint via stiffness, removes the
middle joint, and auto-destroys the remaining constraint by removing its actor.
"""

from __future__ import annotations

import math

import numpy as np
import superdex.physics as physics
from superdex.physics import Actor, Constraint, Scene
from superdex.physics.paths import resolve_asset

# --- Locked scene spec -------------
# Links are slender boxes, long axis along +X, released from horizontal.
LINK_LENGTH = 0.25  # [m] long (X) dimension
LINK_THICKNESS = 0.025  # [m] cross-section (Y, Z)
DENSITY = 1000.0  # [kg/m^3] -> mass = 0.25 * 0.025 * 0.025 * 1000 = 0.15625 kg / link

# World anchor that Link1's upper end is pinned to.
PIVOT_ANCHOR = [0.0, 0.5, 0.0]  # [m]

# Spring-damper parameters shared by both constraints.
STIFFNESS = 2.5e4  # [N/m]
DAMPING = 3.5  # [N*s/m]  (~3% of the joint-mode critical damping)

# Moving-base animation: the anchor traces a small circle in the XZ plane.
TARGET_RADIUS = 0.075  # [m]
TARGET_PERIOD = 4.0  # [s]

TIME_STEP = 1.0 / 60.0  # [s]

# Scripted-timeline trigger times [s].
T_WEAKEN = 8.0  # [s]
T_RESTORE = 11.0  # [s]
T_REMOVE_JOINT = 14.0  # [s]
T_DESTROY_ACTOR = 18.0  # [s]

# The cube mesh spans [0, 1]^3 (corner-anchored), so after baking the link
# scale a link occupies local [0, L] x [0, w] x [0, w]. The centerline
# end-faces (where we attach the joints) are therefore at local (x, w/2, w/2).
_HALF_W = LINK_THICKNESS / 2.0  # [m]
END_A = [0.0, _HALF_W, _HALF_W]  # [m] -X end (toward the pivot / parent)
END_B = [LINK_LENGTH, _HALF_W, _HALF_W]  # [m] +X end (toward the child)

# world_from_local places the box corner, so a link's origin is its attach point
# minus END_A. Link1 attaches at the pivot, Link2 one link-length further in +X.
LINK1_ORIGIN = [a - e for a, e in zip(PIVOT_ANCHOR, END_A)]  # [m]
LINK2_ORIGIN = [LINK1_ORIGIN[0] + LINK_LENGTH, LINK1_ORIGIN[1], LINK1_ORIGIN[2]]  # [m]


def build_scene() -> tuple[Scene, Actor, Actor]:
    """Build the double-pendulum scene in code.

    Note: physics.initialize() must be called before this function.

    Returns:
        tuple: (scene, link1, link2)
    """
    scene = physics.create_scene("Constraints Scene")

    # Both links share the same box shape. The cube mesh is corner-anchored, so
    # baking the scale gives a box occupying local [0, L] x [0, w] x [0, w].
    link_shape = physics.load_shape_from_file(
        file_path=str(resolve_asset("cube/cube_mesh.mochi.json")),
        bake_scale=[LINK_LENGTH, LINK_THICKNESS, LINK_THICKNESS],
    )

    # Link1's end-A sits on the pivot anchor; Link1's end-B and Link2's end-A
    # coincide one link-length further along +X.
    link1 = scene.create_rigid_actor(
        name="Link1",
        layer="Link",
        shape=link_shape,
        is_static=False,
        density=DENSITY,
        world_from_local=physics.TransformRT(translation=LINK1_ORIGIN),
        collider_type=physics.ColliderType.BOX,
    )
    link2 = scene.create_rigid_actor(
        name="Link2",
        layer="Link",
        shape=link_shape,
        is_static=False,
        density=DENSITY,
        world_from_local=physics.TransformRT(translation=LINK2_ORIGIN),
        collider_type=physics.ColliderType.BOX,
    )

    # The two links overlap at the shared joint point; disable contact between
    # them so the joint does not fight a contact response.
    scene.enable_layer_contact_symmetric("Link", "Link", enable=False)

    return scene, link1, link2


def create_constraints(
    scene: Scene, link1: Actor, link2: Actor
) -> tuple[Constraint, Constraint]:
    """Create the pivot-position and spherical-joint constraints.

    Returns:
        tuple: (pivot, spherical)
    """
    # C1: pin Link1's upper end (local END_A) to the world anchor. This is the
    # only constraint with a Real3 position target, which we animate later.
    pivot_params = physics.RigidPivotPositionConstraintParams(
        actor=link1.get_handle(),
        local_position=END_A,
        target_position=PIVOT_ANCHOR,
        stiffness=STIFFNESS,
        damping=DAMPING,
        saturation=-1.0,  # negative disables the force cap
    )
    pivot = scene.create_rigid_pivot_position_constraint(pivot_params)

    # C2: tie Link1's lower end (END_B) to Link2's upper end (END_A).
    spherical_params = physics.RigidSphericalJointConstraintParams(
        actor_a=link1.get_handle(),
        actor_b=link2.get_handle(),
        local_pos_a=END_B,
        local_pos_b=END_A,
        stiffness=STIFFNESS,
        damping=DAMPING,
        saturation=-1.0,
    )
    spherical = scene.create_rigid_spherical_joint_constraint(spherical_params)

    return pivot, spherical


def enumerate_constraints(scene: Scene) -> None:
    """Walk every constraint in the scene via ForEachConstraint."""
    types: list[physics.ConstraintType] = []
    scene.for_each_constraint(lambda c: types.append(c.get_type()))
    print(f"Scene has {len(types)} constraint(s): {[str(t) for t in types]}")


def introspect_constraint(scene: Scene, constraint: Constraint, label: str) -> None:
    """Print the type-agnostic introspection surface of a constraint."""
    handle = constraint.get_handle()
    # A handle is a stable, safe reference; look the constraint back up with it.
    assert scene.get_constraint(handle) == constraint

    print(f"[{label}] type={constraint.get_type()} handle_valid={handle.is_valid()}")

    num_actors = constraint.get_num_actors()
    print(f"[{label}] num_actors={num_actors}  (1 = world anchor, 2 = body-to-body)")
    for i in range(num_actors):
        actor = constraint.get_actor(actor_index=i)
        dofs = list(constraint.get_dof_indices_for_actor(actor_index=i))
        print(f"[{label}]   actor[{i}]={actor.get_name()!r} dof_indices={dofs}")


def tune_parameters(constraint: Constraint, label: str) -> None:
    """Demonstrate the get/set parameter accessors, then restore the values."""
    stiffness = constraint.get_stiffness()
    damping = constraint.get_damping()
    saturation = constraint.get_saturation()
    print(
        f"[{label}] stiffness={stiffness:g} damping={damping:g} saturation={saturation:g}"
    )

    # Setters validate their input (stiffness/damping must be finite and >= 0).
    constraint.set_stiffness(stiffness=stiffness * 2.0)
    constraint.set_damping(damping=damping * 2.0)
    assert constraint.get_stiffness() == stiffness * 2.0

    # Restore the locked spec values.
    constraint.set_stiffness(stiffness=stiffness)
    constraint.set_damping(damping=damping)
    constraint.set_saturation(saturation=saturation)


def demonstrate_type_specific_api(pivot: Constraint, spherical: Constraint) -> None:
    """Show that the interface is uniform but capabilities are type-specific.

    Every constraint exposes the same methods, but many only apply to certain
    constraint types. Inapplicable calls raise ``physics.Error`` rather than
    doing something surprising, so callers can probe capabilities gracefully.
    """
    # The spherical joint has no position target -> graceful error.
    try:
        spherical.set_target_position(physics.Real3(*PIVOT_ANCHOR))
        print("[spherical] set_target_position: accepted (unexpected)")
    except physics.Error:
        print("[spherical] set_target_position: not supported (as expected)")

    # Neither constraint tracks a rotation target or exposes DoF limits.
    for label, c in (("pivot", pivot), ("spherical", spherical)):
        try:
            c.set_target_rotation(physics.Quaternion())
            print(f"[{label}] set_target_rotation: accepted (unexpected)")
        except physics.Error:
            print(f"[{label}] set_target_rotation: not supported (as expected)")
        try:
            c.get_limit_min_values()
            print(f"[{label}] get_limit_min_values: accepted (unexpected)")
        except physics.Error:
            print(f"[{label}] get_limit_min_values: not supported (as expected)")

    # Constraints support constraint-specific queries (force) but not
    # actor-specific ones (node positions).
    print(
        "[pivot] CONSTRAINT_FORCE supported: "
        f"{pivot.is_query_supported(physics.QueryType.CONSTRAINT_FORCE)}; "
        "NODE_POSITIONS supported: "
        f"{pivot.is_query_supported(physics.QueryType.NODE_POSITIONS)}"
    )


def linear_force_magnitude(constraint: Constraint) -> float:
    """Return the magnitude [N] of the constraint's translational force.

    ``get_force`` returns a flattened generalized force (for the pivot: 3 force
    [N] + 3 torque [N*m]); the first three entries are the translational force
    on the first actor, which is what we report.
    """
    force = np.asarray(constraint.get_force(), dtype=float)
    return float(np.linalg.norm(force[:3]))


def animate_anchor(pivot: Constraint, sim_time: float) -> None:
    """Drive the pivot's world target along a small circle in the XZ plane."""
    omega = 2.0 * math.pi / TARGET_PERIOD
    x = PIVOT_ANCHOR[0] + TARGET_RADIUS * math.cos(omega * sim_time)
    z = PIVOT_ANCHOR[2] + TARGET_RADIUS * math.sin(omega * sim_time)
    # Setting the target each step lets the damping term track the target
    # velocity. (Use update_old_target() only when teleporting the target, to
    # suppress the resulting damping transient.)
    pivot.set_target_position(physics.Real3(x, PIVOT_ANCHOR[1], z))


def report_diagnostics(
    sim_time: float, pivot: Constraint | None, spherical: Constraint | None
) -> None:
    """Read and print each constraint's force and the pivot deviation.

    get_force() needs a CONSTRAINT_FORCE query registered before stepping;
    get_deviation() needs no query and reports the current position error [m]
    per constrained dimension (printed here as a magnitude in mm).
    """
    pivot_force = linear_force_magnitude(pivot) if pivot else 0.0
    spherical_force = linear_force_magnitude(spherical) if spherical else 0.0
    deviation_mm = 0.0
    if pivot is not None:
        deviation = np.asarray(pivot.get_deviation(), dtype=float)
        deviation_mm = float(np.linalg.norm(deviation)) * 1.0e3
    print(
        f"t={sim_time:5.1f}s  pivot_force={pivot_force:7.2f} N  "
        f"spherical_force={spherical_force:7.2f} N  "
        f"pivot_deviation={deviation_mm:6.2f} mm"
    )


def run_interactive(
    scene: Scene, pivot: Constraint | None, spherical: Constraint | None, link1: Actor
) -> None:
    """Attach the debugger, then animate the base, report diagnostics, and
    run the scripted timeline until the debugger detaches."""
    assert pivot is not None and spherical is not None  # both alive at start
    # Register a force query on each constraint BEFORE stepping; the force is
    # computed during Step and read back with get_force().
    pivot.register_query(physics.QueryType.CONSTRAINT_FORCE)
    spherical.register_query(physics.QueryType.CONSTRAINT_FORCE)

    sim_time = 0.0
    next_report_time = 1.0
    weakened = restored = joint_removed = actor_destroyed = False

    # Launch and attach the remote debugger for visualization and interaction.
    if not physics.debugger.attach():
        return

    # Simulate until the debugger detaches.
    while physics.debugger.is_attached():
        # --- Scripted timeline -----------------------------------------------
        if pivot is not None and not weakened and sim_time >= T_WEAKEN:
            # Lower the pivot stiffness so the spring goes soft: holding the
            # ~3.06 N weight now needs a large deviation, so the base sags
            # visibly (3.06 N / 50 N/m ~ 0.06 m).
            pivot.set_stiffness(stiffness=5.0e1)
            weakened = True
            print(f"t={sim_time:.1f}s: weakened pivot stiffness to 50 N/m")
        if pivot is not None and not restored and sim_time >= T_RESTORE:
            pivot.set_stiffness(stiffness=STIFFNESS)  # restore the stiff pin
            restored = True
            print(f"t={sim_time:.1f}s: restored pivot stiffness")
        if spherical is not None and not joint_removed and sim_time >= T_REMOVE_JOINT:
            # Destroying the middle joint detaches Link2, which falls away.
            scene.destroy_constraint(spherical)
            spherical = None
            joint_removed = True
            print(f"t={sim_time:.1f}s: destroyed spherical joint; Link2 detached")
            enumerate_constraints(scene)
        if pivot is not None and not actor_destroyed and sim_time >= T_DESTROY_ACTOR:
            # Destroying an actor auto-destroys the constraints attached to it:
            # removing Link1 also removes the pivot constraint.
            scene.destroy_actor(link1)
            pivot = None
            actor_destroyed = True
            print(f"t={sim_time:.1f}s: destroyed Link1; its pivot auto-destroyed")
            enumerate_constraints(scene)

        # --- Drive the moving base -----------------------------------------------
        if pivot is not None:
            animate_anchor(pivot, sim_time)

        scene.step(TIME_STEP)
        sim_time += TIME_STEP

        # --- Read per-step diagnostics (reported ~once per second) ---------------
        if sim_time >= next_report_time:
            next_report_time += 1.0
            report_diagnostics(sim_time, pivot, spherical)


def main() -> None:
    """Run the interactive constraints tutorial."""
    # 0 = single-threaded (simplest); -1 = auto; N = N worker threads.
    physics.initialize(num_worker_threads=0)

    scene, link1, link2 = build_scene()

    # --- Lifecycle: create ---------------------------------------------------
    pivot, spherical = create_constraints(scene, link1, link2)

    # --- Lifecycle: enumerate the whole scene --------------------------------
    enumerate_constraints(scene)

    # --- Introspection: type, handle, actors, DoFs ---------------------------
    introspect_constraint(scene, pivot, "pivot")
    introspect_constraint(scene, spherical, "spherical")

    # --- Parameters: get / set stiffness, damping, saturation ----------------
    tune_parameters(pivot, "pivot")
    tune_parameters(spherical, "spherical")

    # --- Type-specific API + graceful errors ---------------------------------
    demonstrate_type_specific_api(pivot, spherical)

    # --- Run: animate the base, report force/deviation, scripted timeline ----
    run_interactive(scene, pivot, spherical, link1)

    # Destroying the scene frees everything it contains.
    physics.destroy_scene(scene)
    physics.shutdown()
    print("Simulation complete.")


if __name__ == "__main__":
    main()
