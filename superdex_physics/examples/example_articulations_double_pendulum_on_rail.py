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

"""Example: Articulations

A double pendulum on a rail, built in code and used as a guided tour of
the SuperDex Physics *articulated actor* API. A single articulated actor
forms a serial chain:

    ceiling -[Hard]-> RailHousing -[Prismatic]-> Cart -[Revolute]-> UpperArm
            -[Spherical]-> LowerArm

A cart slides on a horizontal rail; hanging from it is a double pendulum (a
revolute upper hinge and a spherical lower joint) whose tip strikes a ball
resting on the ground. This is the programmatic twin of the
``assets/samples/articulations_double_pendulum_on_rail.mochi_scene`` prefab.

The console walks the one-shot API sections at startup (build-time modeling,
introspection, forward-kinematics reads, direct state manipulation, live joint
modeling, mass/root queries, the end-effector Jacobian, and contact control).
The debugger then attaches for visualization while the simulation runs a
short scripted timeline (freeze the rail via a boundary condition -> release ->
push the cart with an external DoF force -> damp the pendulum by raising joint
friction), printing the rail position and the upper-hinge angle once per
simulated second.
"""

from __future__ import annotations

import numpy as np
import superdex.physics as physics
from superdex.physics import Actor, Scene
from superdex.physics.paths import resolve_asset

np_real = np.float64 if physics.uses_double_precision() else np.float32

# --- Locked scene spec (mirrors the .mochi_scene prefab) ---------------------
# Frame: +Y up, rail along +X. The root/CeilingWeld is anchored at the ceiling.
ROOT_HEIGHT = 0.75  # [m] height of the welded rail housing

# Link box sizes [m]. The cube mesh spans [0, 1]^3 (corner-anchored), so a baked
# link occupies local [0, sx] x [0, sy] x [0, sz]. To center a box on its joint
# in X/Z and hang it down in -Y, a link uses parent_joint_from_link
# [-sx/2, -sy, -sz/2], and the child joint at the box's bottom-center uses
# parent_link_from_joint [+sx/2, 0, +sz/2]. The welded RailHousing is the
# exception: it is centered on its joint in Y too, so it uses -sy/2.
RAIL_SCALE = [0.5, 0.05, 0.05]  # [m] horizontal rail housing (welded)
CART_SCALE = [0.075, 0.075, 0.075]  # [m] slider on the rail
ARM_SCALE = [0.03, 0.3, 0.03]  # [m] each pendulum arm (rod along Y)

BALL_RADIUS = 0.05  # [m]
BALL_POSITION = [0.15, 0.05, 0.0]  # [m] rests on the ground, within the tip's reach

# DoF layout of the 4 joints: Hard(0) + Prismatic(1) + Revolute(1) + Spherical(3).
NUM_DOFS = 5
RAIL_DOF = 0  # prismatic rail position [m]
UPPER_HINGE_DOF = 1  # revolute upper-hinge angle [rad]
# Seed velocity per DoF: a small rail drift plus an upper-hinge kick to start
# the double pendulum swinging.
SEED_JOINT_VELOCITIES = [0.3, 4.2, 0.0, 0.0, 0.0]  # rail [m/s]; rotations [rad/s]

TIME_STEP = 1.0 / 60.0  # [s]

# Scripted-timeline trigger times [s].
T_FREEZE_RAIL = 6.0  # [s]
T_RELEASE_RAIL = 10.0  # [s]
T_PUSH = 14.0  # [s]
T_STOP_PUSH = 16.0  # [s]
T_DAMP = 20.0  # [s]

RAIL_PUSH_FORCE = 1.0  # [N] external force applied to the rail DoF
DAMP_VISCOUS = 0.022  # [N*m*s/rad] pendulum-joint friction for the damping event


def _make_joints() -> list[physics.ArticulatedJointParams]:
    """Build the 4 inbound joints of the chain (joints[i] drives links[i])."""
    return [
        # Root weld: fixes the rail housing to the world. Contributes 0 DoFs.
        physics.ArticulatedJointParams(
            name="CeilingWeld",
            type=physics.ArticulatedJointType.HARD,
        ),
        # Horizontal rail: the cart slides along +X, clamped by joint limits. It
        # is the only joint that carries viscous friction and an armature inertia.
        physics.ArticulatedJointParams(
            name="Rail",
            type=physics.ArticulatedJointType.PRISMATIC,
            axis=[1, 0, 0],
            parent_link_from_joint=physics.TransformRT(
                translation=[RAIL_SCALE[0] / 2, 0, RAIL_SCALE[2] / 2]
            ),
            min_limit=[-0.2, 0, 0],  # scalar limit times axis
            max_limit=[0.2, 0, 0],  # scalar limit times axis
            limit_stiffness=250.0,
            limit_damping=8.8,
            friction=physics.ArticulatedJointFrictionParams(viscous=0.018),
            inertia=0.125,
        ),
        # Upper pendulum hinge: revolute about Z, swings in the vertical X-Y plane.
        physics.ArticulatedJointParams(
            name="UpperSwing",
            type=physics.ArticulatedJointType.REVOLUTE,
            axis=[0, 0, -1],
            parent_link_from_joint=physics.TransformRT(
                translation=[CART_SCALE[0] / 2, 0, CART_SCALE[2] / 2]
            ),
        ),
        # Lower pendulum joint: a free 3-DoF ball-and-socket.
        physics.ArticulatedJointParams(
            name="LowerSwing",
            type=physics.ArticulatedJointType.SPHERICAL,
            parent_link_from_joint=physics.TransformRT(
                translation=[ARM_SCALE[0] / 2, 0, ARM_SCALE[2] / 2]
            ),
        ),
    ]


def _make_links(
    rail_shape: physics.ShapeHandle,
    cart_shape: physics.ShapeHandle,
    arm_shape: physics.ShapeHandle,
) -> list[physics.ArticulatedLinkParams]:
    """Build the 4 links. Each box is centered on its inbound joint and hangs -Y.

    Note on per-link modeling: every link sets ``density`` here, but a link can
    instead be given an explicit ``mass`` + ``center_of_mass`` +
    ``moment_of_inertia``, or opt out of gravity with ``has_gravity=False``.
    """
    return [
        physics.ArticulatedLinkParams(
            name="RailHousing",
            parent_link=-1,
            parent_joint_from_link=physics.TransformRT(
                translation=[-RAIL_SCALE[0] / 2, -RAIL_SCALE[1] / 2, -RAIL_SCALE[2] / 2]
            ),
            shape=rail_shape,
            collider_type=physics.ColliderType.BOX,
            layer="Pendulum",
            density=1000.0,
        ),
        physics.ArticulatedLinkParams(
            name="Cart",
            parent_link=0,
            parent_joint_from_link=physics.TransformRT(
                translation=[-CART_SCALE[0] / 2, -CART_SCALE[1], -CART_SCALE[2] / 2]
            ),
            shape=cart_shape,
            collider_type=physics.ColliderType.BOX,
            layer="Pendulum",
            density=1000.0,
        ),
        physics.ArticulatedLinkParams(
            name="UpperArm",
            parent_link=1,
            parent_joint_from_link=physics.TransformRT(
                translation=[-ARM_SCALE[0] / 2, -ARM_SCALE[1], -ARM_SCALE[2] / 2]
            ),
            shape=arm_shape,
            collider_type=physics.ColliderType.BOX,
            layer="Pendulum",
            density=1000.0,
        ),
        physics.ArticulatedLinkParams(
            name="LowerArm",  # the striking tip
            parent_link=2,
            parent_joint_from_link=physics.TransformRT(
                translation=[-ARM_SCALE[0] / 2, -ARM_SCALE[1], -ARM_SCALE[2] / 2]
            ),
            shape=arm_shape,
            collider_type=physics.ColliderType.BOX,
            layer="EndEffector",
            density=2000.0,  # heavier tip for a firmer strike
        ),
    ]


def build_scene() -> tuple[Scene, Actor, Actor]:
    """Build the double-pendulum-on-rail scene in code.

    Note: physics.initialize() must be called before this function.

    Returns:
        tuple: (scene, articulation, ball)
    """
    scene = physics.create_scene("Articulations Scene")

    # One baked shape per distinct link size (the cube mesh is corner-anchored,
    # so baking the scale gives a box occupying local [0, scale]).
    def box(scale: list[float]) -> physics.ShapeHandle:
        return physics.load_shape_from_file(
            file_path=str(resolve_asset("cube/cube_fine_mesh.mochi.json")),
            bake_scale=scale,
        )

    rail_shape = box(RAIL_SCALE)
    cart_shape = box(CART_SCALE)
    arm_shape = box(ARM_SCALE)  # shared by UpperArm and LowerArm

    params = physics.ArticulatedActorParams(name="DoublePendulumOnRail")
    params.world_from_root = physics.TransformRT(translation=[0, ROOT_HEIGHT, 0])
    params.joints = _make_joints()
    params.links = _make_links(rail_shape, cart_shape, arm_shape)
    articulation = scene.create_articulated_actor(params)

    # The actor holds its own reference to each shape, so release our local
    # handles now (shapes are reference-counted and freed at the last release).
    for shape in (rail_shape, cart_shape, arm_shape):
        physics.release_shape(shape)

    # Seed initial joint velocities to set the double pendulum swinging.
    articulation.set_articulated_joint_velocities(
        velocities=np.array(SEED_JOINT_VELOCITIES, dtype=np_real)
    )

    # Ball on the ground, offset in +X within reach of the swinging/sliding tip.
    # The unit-radius icosphere renders as a real mesh; the analytic Sphere
    # collider handles contact.
    ball_shape = physics.load_shape_from_file(
        file_path=str(resolve_asset("sphere/icosphere_4subdiv.1.mochi.json")),
        bake_scale=[BALL_RADIUS, BALL_RADIUS, BALL_RADIUS],
    )
    ball = scene.create_rigid_actor(
        name="Ball",
        layer="Ball",
        shape=ball_shape,
        is_static=False,
        density=500.0,
        world_from_local=physics.TransformRT(translation=BALL_POSITION),
        collider_type=physics.ColliderType.SPHERE,
    )
    physics.release_shape(ball_shape)

    # Static ground plane (layer "Environment", matching the prefab).
    plane_shape = physics.create_plane_shape(normal=[0, 1, 0], distance=0)
    scene.create_rigid_actor(
        name="Ground", layer="Environment", shape=plane_shape, is_static=True
    )
    physics.release_shape(plane_shape)

    # Contact filter: only the end-effector tip hits the ball (and the ball
    # rests on the ground). Disable everything else.
    scene.enable_layer_contact_symmetric("Pendulum", "Pendulum", enable=False)
    scene.enable_layer_contact_symmetric("Pendulum", "Ball", enable=False)
    scene.enable_layer_contact_symmetric("Pendulum", "EndEffector", enable=False)
    scene.enable_layer_contact_symmetric("Pendulum", "Environment", enable=False)
    scene.enable_layer_contact_symmetric("EndEffector", "Environment", enable=False)

    return scene, articulation, ball


def print_introspection(scene: Scene, articulation: Actor) -> None:
    """Tour the topology / introspection API of an articulated actor."""
    print(f"num_dofs = {articulation.get_num_dofs()}")

    # Each link is itself a queryable rigid sub-actor.
    link_handles = articulation.get_nested_link_actors()
    names = [scene.get_actor(h).get_name() for h in link_handles]
    print(f"nested link actors ({len(link_handles)}): {names}")

    # One-stop dump of the articulation structure. All per-link arrays are
    # indexed the same way: joints[i] is the inbound joint of links[i], and
    # parents[i] is the *parent link index* of link i (-1 for the root).
    info = articulation.get_articulated_shape_info()
    for i in range(len(info.link_names)):
        print(
            f"  link[{i}] {info.link_names[i]!r}: parent_link={info.parents[i]}, "
            f"inbound_joint={info.joint_names[i]!r} ({info.joint_types[i]})"
        )

    # Joint limits are exposed as inspectable constraints (the prismatic joints
    # carry limits; the free revolute/spherical joints do not).
    limit_constraints = articulation.get_articulated_joint_limit_constraints()
    print(f"joint-limit constraints: {len(limit_constraints)}")


def print_read_state(articulation: Actor) -> None:
    """Tour the forward-kinematics read API (pose / link transforms / velocity)."""
    num_dofs = articulation.get_num_dofs()
    num_links = len(articulation.get_nested_link_actors())

    pose = physics.DynamicArrayReal(num_dofs)
    articulation.get_articulated_pose(pose)
    print(f"pose (joint-space DoFs) = {[round(x, 3) for x in pose]}")

    transforms = physics.DynamicArrayTransformRT(num_links)
    articulation.get_articulated_link_transforms(transforms)
    tip = transforms[num_links - 1].translation
    print(f"LowerArm world transform origin = {[round(x, 3) for x in tip]}")

    velocities = physics.DynamicArrayReal(num_dofs)
    articulation.get_articulated_joint_velocities(velocities)
    print(f"joint velocities = {[round(x, 3) for x in velocities]}")


def demonstrate_pose_manipulation(articulation: Actor) -> None:
    """Tour the direct state-manipulation API, then restore the rest state."""
    num_dofs = articulation.get_num_dofs()
    num_links = len(articulation.get_nested_link_actors())

    # Set the pose directly from joint-space DoFs (e.g. slide the rail by hand).
    poked = np.zeros(num_dofs, dtype=np_real)
    poked[RAIL_DOF] = 0.1
    poked[UPPER_HINGE_DOF] = 0.5
    articulation.set_articulated_pose_from_joints(pose=poked)

    # Pose-space math: build a joint-space delta and apply it to the current
    # pose with add_articulated_delta_to_pose (the delta lives in the tangent
    # space, so spherical DoFs compose correctly), then set the actor to the
    # result.
    delta = np.zeros(num_dofs, dtype=np_real)
    delta[UPPER_HINGE_DOF] = 0.1
    nudged = np.zeros(num_dofs, dtype=np_real)
    articulation.add_articulated_delta_to_pose(
        pose=poked, delta_dofs=delta, out_pose=nudged
    )
    articulation.set_articulated_pose_from_joints(pose=nudged)
    print(f"pose after +0.1 rad on the upper hinge = {[round(x, 3) for x in nudged]}")

    # Set the pose from link transforms instead (IK-style): read the current
    # link transforms, nudge the end-effector tip in +X, and write them back.
    transforms = physics.DynamicArrayTransformRT(num_links)
    articulation.get_articulated_link_transforms(transforms)
    world_from_links = [
        physics.TransformRT(transforms[i].rotation, transforms[i].translation)
        for i in range(num_links)
    ]
    world_from_links[num_links - 1].translation += [0.01, 0.0, 0.0]
    articulation.set_articulated_pose_from_links(world_from_links=world_from_links)

    # Restore a clean rest pose for the interactive run.
    articulation.set_articulated_pose_from_joints(
        pose=np.zeros(num_dofs, dtype=np_real)
    )


def demonstrate_live_joint_modeling(articulation: Actor) -> None:
    """Read and round-trip the per-joint friction and armature-inertia params."""
    friction = articulation.get_articulated_joint_friction_params()
    print(
        "joint viscous friction = "
        f"{[round(f.viscous, 3) for f in friction]} (per joint)"
    )
    # Round-trip: the setter accepts exactly what the getter returned, unchanged.
    articulation.set_articulated_joint_friction_params(friction)

    inertia = articulation.get_articulated_joint_inertia_params()
    print(f"joint armature inertia = {[round(x, 3) for x in inertia]} (per joint)")
    articulation.set_articulated_joint_inertia_params(list(inertia))


def demonstrate_mass_and_root(scene: Scene, articulation: Actor) -> None:
    """Tour mass / root-transform queries (state-preserving).

    Mass and the root transform are whole-articulation queries. Center-of-mass
    and linear/angular velocity are *per-rigid-body* queries, so they live on
    the nested link sub-actors rather than on the compound articulation (calling
    them on the articulation raises a graceful physics.Error). The articulated
    equivalent of set_velocity is set_articulated_joint_velocities.
    """
    print(f"total mass = {articulation.get_mass():.4f} kg")

    root = articulation.get_root_transform()
    print(f"root translation = {[round(x, 3) for x in root.translation]}")

    # Center of mass is a rigid-body query -> read it from a nested link sub-actor.
    link_handles = articulation.get_nested_link_actors()
    lower_arm = scene.get_actor(link_handles[len(link_handles) - 1])
    com = lower_arm.get_center_of_mass_transform()
    print(
        f"LowerArm center-of-mass translation = {[round(x, 3) for x in com.translation]}"
    )

    # set_root_transform teleports the whole articulation; demonstrate and restore.
    moved = physics.TransformRT(
        rotation=root.rotation,
        translation=[
            root.translation[0] + 0.05,
            root.translation[1],
            root.translation[2],
        ],
    )
    articulation.set_root_transform(moved)
    articulation.set_root_transform(root)


def demonstrate_jacobian(scene: Scene, articulation: Actor) -> None:
    """Read the end-effector Jacobian (joint motion -> link motion)."""
    link_handles = articulation.get_nested_link_actors()
    end_effector = scene.get_actor(link_handles[len(link_handles) - 1])
    jacobian = end_effector.get_articulated_jacobian()
    # Flattened 6 x num_dofs (3 translation + 3 rotation rows per joint DoF).
    print(
        f"end-effector Jacobian has {len(jacobian)} entries "
        f"(= 6 x {articulation.get_num_dofs()} DoFs)"
    )


def demonstrate_contact_control(scene: Scene, articulation: Actor, ball: Actor) -> None:
    """Tour the contact-layer and per-actor contact controls."""
    print(f"num contact layers = {scene.get_num_contact_layers()}")

    layer_names: list[str] = []
    scene.enumerate_contact_layer_names(lambda name: layer_names.append(name))
    print(f"contact layers = {sorted(layer_names)}")

    print(
        "EndEffector<->Ball enabled: "
        f"{scene.is_layer_contact_enabled('EndEffector', 'Ball')}; "
        "Pendulum<->Ball enabled: "
        f"{scene.is_layer_contact_enabled('Pendulum', 'Ball')}"
    )

    # Finest-grained control: enable contact for the specific LowerArm sub-actor
    # against the ball (already enabled by the EndEffector<->Ball layer pair, so
    # this is illustrative rather than a behavior change).
    link_handles = articulation.get_nested_link_actors()
    lower_arm = link_handles[len(link_handles) - 1]
    scene.enable_actor_contact_symmetric(
        lower_arm,
        ball.get_handle(),
        enable=True,
        include_nested_actors=physics.IncludeNestedActors.NO,
    )


def reset_to_initial_state(articulation: Actor) -> None:
    """Restore the intended starting state before the interactive run."""
    num_dofs = articulation.get_num_dofs()
    articulation.set_articulated_pose_from_joints(
        pose=np.zeros(num_dofs, dtype=np_real)
    )
    articulation.set_articulated_joint_velocities(
        velocities=np.array(SEED_JOINT_VELOCITIES, dtype=np_real)
    )


def _rail_position(articulation: Actor, num_dofs: int) -> float:
    pose = physics.DynamicArrayReal(num_dofs)
    articulation.get_articulated_pose(pose)
    return float(pose[RAIL_DOF])


def _damp_pendulum_joints(articulation: Actor) -> None:
    """Raise the viscous friction on the revolute + spherical joints."""
    friction = list(articulation.get_articulated_joint_friction_params())
    for i in (2, 3):  # UpperSwing (revolute), LowerSwing (spherical)
        friction[i] = physics.ArticulatedJointFrictionParams(viscous=DAMP_VISCOUS)
    articulation.set_articulated_joint_friction_params(friction)


def run_interactive(scene: Scene, articulation: Actor) -> None:
    """Attach the debugger, then run the scripted timeline, printing state
    once per second until the debugger detaches."""
    num_dofs = articulation.get_num_dofs()

    sim_time = 0.0
    next_report = 1.0  # [s] print the state once per simulated second
    frozen = released = pushing = pushed = damped = False

    # Launch and attach the remote debugger for visualization and interaction.
    if not physics.debugger.attach():
        return

    # Simulate until the debugger detaches.
    while physics.debugger.is_attached():
        # --- Scripted timeline -----------------------------------------------
        if not frozen and sim_time >= T_FREEZE_RAIL:
            # Pin the rail DoF at its current position with a boundary
            # condition; the cart stops sliding while the pendulum swings.
            articulation.add_boundary_condition_dofs_world(
                dof_indices=np.array([RAIL_DOF], dtype=np.int32),
                dof_values_world=np.array(
                    [_rail_position(articulation, num_dofs)], dtype=np_real
                ),
            )
            frozen = True
            print(f"t={sim_time:.1f}s: froze the rail DoF (boundary condition)")
        if not released and sim_time >= T_RELEASE_RAIL:
            articulation.clear_boundary_conditions()
            released = True
            print(f"t={sim_time:.1f}s: released the rail DoF")
        if not pushing and sim_time >= T_PUSH:
            # Actuate directly: apply a constant force to the rail DoF.
            articulation.set_external_forces_on_dofs(
                dof_indices=np.array([RAIL_DOF], dtype=np.int32),
                force_values=np.array([RAIL_PUSH_FORCE], dtype=np_real),
            )
            pushing = True
            print(f"t={sim_time:.1f}s: pushing the cart (+X external DoF force)")
        if not pushed and sim_time >= T_STOP_PUSH:
            articulation.clear_external_forces()
            pushed = True
            print(f"t={sim_time:.1f}s: cleared the external force")
        if not damped and sim_time >= T_DAMP:
            _damp_pendulum_joints(articulation)
            damped = True
            print(f"t={sim_time:.1f}s: raised pendulum-joint friction (damping)")

        scene.step(TIME_STEP)
        sim_time += TIME_STEP

        # Print the rail position and upper-hinge angle once per second.
        if sim_time >= next_report:
            pose = physics.DynamicArrayReal(num_dofs)
            articulation.get_articulated_pose(pose)
            print(
                f"t={sim_time:.1f}s: rail={pose[RAIL_DOF]:.3f} m, "
                f"upper hinge={pose[UPPER_HINGE_DOF]:.3f} rad"
            )
            next_report += 1.0


def main() -> None:
    """Run the interactive articulations tutorial."""
    # 0 = single-threaded (simplest); -1 = auto; N = N worker threads.
    physics.initialize(num_worker_threads=0)

    scene, articulation, ball = build_scene()

    # --- One-shot API tour (logged once at startup) --------------------------
    print_introspection(scene, articulation)
    print_read_state(articulation)
    demonstrate_pose_manipulation(articulation)
    demonstrate_live_joint_modeling(articulation)
    demonstrate_mass_and_root(scene, articulation)
    demonstrate_jacobian(scene, articulation)
    demonstrate_contact_control(scene, articulation, ball)

    # --- Run: attach the debugger, run the scripted timeline -----------------
    reset_to_initial_state(articulation)
    run_interactive(scene, articulation)

    # Destroying the scene frees everything it contains.
    physics.destroy_scene(scene)
    physics.shutdown()
    print("Simulation complete.")


if __name__ == "__main__":
    main()
