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

"""Example: Soft Skinned Articulations

A double pendulum carrying a soft body, built in code and used as a guided tour
of the SuperDex Physics *soft-skinned articulated actor* API. A two-bone
revolute skeleton is anchored to a world pivot:

    world -[Revolute]-> UpperArm -[Revolute]-> LowerArm -[soft attached]-> SoftArm

Configuration of the actor:
    (1) same top arm of the double pendulum (0.25 m),
    (2) then a second arm with half the length (0.125 m),
    (3) attached to this a soft skinned actor (0.1 m).

The soft body is a tetrahedral mesh spanning X=0.375 to 0.475 m in the skeleton rest
frame (total rigid length 0.25+0.125=0.375, plus soft 0.1 = 0.475). Its first cross-
section nodes (4 nodes at X=0.375) are listed as ``constrainedNodes`` in the shape
JSON and are bound to the second link via ``SoftSkinnedActorParams.softAttachLinks``.
Contact between the soft and its attachment link is automatically disabled.

The soft is a **colliding actor only**: it probes contact against other actors, but
does not act as collider. Here a ball resting on the ground is struck by the swinging
soft. This is the programmatic twin of the
``assets/samples/articulations_soft_skinned_double_pendulum.mochi_scene`` prefab.

The console walks the one-shot API sections at startup (introspection, constrained
nodes, contact control). The debugger then attaches for visualization while
the simulation runs a short scripted timeline (report the soft's world bounds,
joint angles, and the contact force from the ball once per simulated second),
until the debugger detaches.
"""

from __future__ import annotations

import numpy as np
import superdex.physics as physics
from superdex.physics import Actor, Scene
from superdex.physics.paths import resolve_asset

np_real = np.float64 if physics.uses_double_precision() else np.float32

# --- Locked scene spec (mirrors the .mochi_scene prefab) ---------------------
# Frame: +Y up. The pivot is anchored in the world by joint_0's
# parent_link_from_joint (world_from_root is left at identity). The two arms lie
# along local +X at rest and swing about Z revolute joints (in the X-Y plane).
ROOT_HEIGHT = 0.5  # [m] pivot height; chosen so the fully-extended tip (rigid 0.375 + soft 0.10 = 0.475) stays just above ground.

ARM_LENGTH = 0.25  # [m] length of top arm along +X (L)
SECOND_ARM_LENGTH = 0.125  # [m] half length
SOFT_LENGTH = 0.10  # [m] slightly shorter than half to avoid ground contact
ARM_WIDTH = 0.025  # [m] cross-section of each rigid arm (w)
ARM_SCALE = [ARM_LENGTH, ARM_WIDTH, ARM_WIDTH]  # [m]
SECOND_ARM_SCALE = [SECOND_ARM_LENGTH, ARM_WIDTH, ARM_WIDTH]  # [m]

# Soft tet mesh: authored as rod along +X from 0.375 to 0.475 with constrained nodes at X=0.375.
SOFT_ASSET = "samples/articulations_parts/soft.mochi.json"

BALL_RADIUS = 0.05  # [m]
BALL_POSITION = [0.05, 0.05, 0.0]  # [m] rests on the ground, within the tip's reach

NUM_DOFS = 2
SEED_JOINT_VELOCITIES = [4.2, 0.0]  # [rad/s]

TIME_STEP = 1.0 / 120.0  # [s]


def _make_joints() -> list[physics.ArticulatedJointParams]:
    """Build the 2 inbound joints of the chain (joints[i] drives links[i])."""
    return [
        physics.ArticulatedJointParams(
            name="joint_0",
            type=physics.ArticulatedJointType.REVOLUTE,
            axis=[0, 0, -1],
            parent_link_from_joint=physics.TransformRT(translation=[0, ROOT_HEIGHT, 0]),
        ),
        physics.ArticulatedJointParams(
            name="joint_1",
            type=physics.ArticulatedJointType.REVOLUTE,
            axis=[0, 0, -1],
            parent_link_from_joint=physics.TransformRT(
                translation=[ARM_LENGTH, ARM_WIDTH / 2, ARM_WIDTH / 2]
            ),
        ),
    ]


def _make_links(
    arm_shape: physics.ShapeHandle, second_arm_shape: physics.ShapeHandle
) -> list[physics.ArticulatedLinkParams]:
    """Build the 2 arm links. Second arm is half length."""
    return [
        physics.ArticulatedLinkParams(
            name="UpperArm",
            parent_link=-1,
            parent_joint_from_link=physics.TransformRT(
                translation=[0, -ARM_WIDTH / 2, -ARM_WIDTH / 2]
            ),
            shape=arm_shape,
            collider_type=physics.ColliderType.BOX,
            layer="Pendulum",
            density=1000.0,
        ),
        physics.ArticulatedLinkParams(
            name="LowerArm",
            parent_link=0,
            parent_joint_from_link=physics.TransformRT(
                translation=[0, -ARM_WIDTH / 2, -ARM_WIDTH / 2]
            ),
            shape=second_arm_shape,
            collider_type=physics.ColliderType.BOX,
            layer="Pendulum",
            density=1000.0,
        ),
    ]


def build_scene() -> tuple[Scene, Actor, Actor]:
    """Build the soft-skinned-double-pendulum scene in code.

    Note: physics.initialize() must be called before this function.

    Returns:
        tuple: (scene, soft_skinned_actor, ball)
    """
    scene = physics.create_scene("Soft Skinned Articulations Scene")

    arm_shape = physics.load_shape_from_file(
        file_path=str(resolve_asset("cube/cube_fine_mesh.mochi.json")),
        bake_scale=ARM_SCALE,
    )
    second_arm_shape = physics.load_shape_from_file(
        file_path=str(resolve_asset("cube/cube_fine_mesh.mochi.json")),
        bake_scale=SECOND_ARM_SCALE,
    )

    # Soft shape: tet rod X=0.375..0.475 with constrainedNodes=[0,1,2,3] at the attachment end.
    soft_shape = physics.load_shape_from_file(
        file_path=str(resolve_asset(SOFT_ASSET)),
        bake_scale=[1, 1, 1],
    )

    # Skeleton params (articulated part)
    skeleton_params = physics.ArticulatedActorParams(name="SoftSkinnedDoublePendulum")
    skeleton_params.joints = _make_joints()
    skeleton_params.links = _make_links(arm_shape, second_arm_shape)

    # Soft params (deformable part) - must have has_gravity=False, constrained via JSON + attach link.
    # Use unposed elasticity (has_stress=True), which is accurate because skinning is rigid.
    soft_params = physics.SoftActorParams(
        name="SoftArm",
        shape=soft_shape,
        layer="Soft",
        has_gravity=False,
        has_inertia=False,
        has_stress=True,
    )
    # Optional: tune material - softer than default to show large deformation.
    soft_params.material = physics.SoftMaterialParams()
    soft_params.material.type = physics.SoftMaterialType.NEO_HOOKEAN
    soft_params.material.neo_hookean.youngs_modulus = 1.5e4
    soft_params.material.density = 500.0

    # Soft-skinned actor params - attach soft to LowerArm via softAttachLinks.
    # Use posed gravity and inertia (has_gravity, has_inertia on the articulated actor);
    # physically correct.
    # Enable colliding links as their geometry is not wrapped by the soft actor.
    ss_params = physics.SoftSkinnedActorParams(
        skeleton_params=skeleton_params,
        soft_params=[soft_params],
        soft_attach_links=["LowerArm"],
        has_gravity=True,
        has_inertia=True,
        has_stress=False,
        enable_colliding_links=True,
    )

    soft_skinned_actor = scene.create_soft_skinned_actor(ss_params)
    physics.release_shape(arm_shape)
    physics.release_shape(second_arm_shape)
    physics.release_shape(soft_shape)

    # Seed initial joint velocities to set the double pendulum swinging.
    soft_skinned_actor.set_articulated_joint_velocities(
        velocities=np.array(SEED_JOINT_VELOCITIES, dtype=np_real)
    )

    # Ball on the ground, offset in +X within reach of the swinging soft.
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

    # Static ground plane (layer "Environment").
    plane_shape = physics.create_plane_shape(normal=[0, 1, 0], distance=0)
    scene.create_rigid_actor(
        name="Ground", layer="Environment", shape=plane_shape, is_static=True
    )
    physics.release_shape(plane_shape)

    # Contact filter: only the soft is struck by the ball. Disable per-link "Pendulum" colliders
    # against everything, and keep the soft from interacting with the ground. Contact between
    # LowerArm and SoftArm is automatically disabled via softAttachLinks.
    scene.enable_layer_contact_symmetric("Pendulum", "Pendulum", enable=False)
    scene.enable_layer_contact_symmetric("Pendulum", "Ball", enable=False)
    scene.enable_layer_contact_symmetric("Pendulum", "Environment", enable=False)
    scene.enable_layer_contact_symmetric("Soft", "Environment", enable=False)

    return scene, soft_skinned_actor, ball


def print_introspection(scene: Scene, actor: Actor) -> None:
    """Tour the topology / introspection API for soft-skinned actor."""
    print(f"articulated actor num_dofs = {actor.get_num_dofs()}")

    link_handles = actor.get_nested_link_actors()
    soft_handles = actor.get_nested_soft_actors()
    link_names = [scene.get_actor(h).get_name() for h in link_handles]
    soft_names = [scene.get_actor(h).get_name() for h in soft_handles]
    print(f"nested link actors ({len(link_handles)}): {link_names}")
    print(f"nested soft actors ({len(soft_handles)}): {soft_names}")

    info = actor.get_articulated_shape_info()
    for i in range(len(info.link_names)):
        print(
            f"  link[{i}] {info.link_names[i]!r}: parent_link={info.parents[i]}, "
            f"inbound_joint={info.joint_names[i]!r} ({info.joint_types[i]})"
        )

    # Show that soft is attached to LowerArm via constrained nodes
    print(
        "softAttachLinks = ['LowerArm'] - soft 'SoftArm' constrained to link 'LowerArm'"
    )
    # The soft shape itself carries constrainedNodes=[0,1,2,3] at X=0.375 end.

    probes = [
        physics.QueryType.SURFACE_NODE_POSITIONS,
        physics.QueryType.CONTACT_POINTS,
        physics.QueryType.TOTAL_CONTACT_FORCE,
        physics.QueryType.NODE_POSITIONS,
    ]
    for q in probes:
        print(
            f"  is_query_supported({q}) on articulated actor = {actor.is_query_supported(q)}"
        )
    # Also probe the nested soft actor
    if soft_handles:
        soft_actor = scene.get_actor(soft_handles[0])
        for q in probes:
            print(
                f"  is_query_supported({q}) on nested soft = {soft_actor.is_query_supported(q)}"
            )


def demonstrate_soft_surface(scene: Scene, actor: Actor) -> None:
    """Read the deformed soft surface / volume."""
    # Query nodes on the nested soft actor - data computed during step, so register then step
    soft_handles = actor.get_nested_soft_actors()
    if not soft_handles:
        return
    soft_actor = scene.get_actor(soft_handles[0])
    # Soft actor's AABB is available before step.
    aabb = soft_actor.get_aabb_world()
    print(
        "soft world AABB after first step (nested soft): "
        f"min={[round(v, 3) for v in aabb.min]}, "
        f"max={[round(v, 3) for v in aabb.max]}"
    )
    # Register node positions query on nested soft (data computed during step)
    pos_query = soft_actor.register_query(physics.QueryType.NODE_POSITIONS)
    scene.step(TIME_STEP)
    # Read the deformed node positions computed during the step (3 values per node).
    node_positions = soft_actor.get_node_positions_local()
    num_nodes = len(node_positions) // 3
    print(
        f"soft deformed nodes: {num_nodes} nodes; "
        f"node[0] local=[{node_positions[0]:.3f}, {node_positions[1]:.3f}, "
        f"{node_positions[2]:.3f}] m"
    )
    # The soft shape itself carries constrainedNodes=[0,1,2,3] at X=0.375 end.
    print("soft constrained nodes: [0,1,2,3] (attachment end)")
    soft_actor.cancel_query(pos_query)


def demonstrate_contact_control(scene: Scene, actor: Actor) -> None:
    """Tour the contact-layer controls around the soft actor."""
    print(f"num contact layers = {scene.get_num_contact_layers()}")
    layer_names: list[str] = []
    scene.enumerate_contact_layer_names(lambda name: layer_names.append(name))
    print(f"contact layers = {sorted(layer_names)}")
    print(
        "Soft<->Ball enabled: "
        f"{scene.is_layer_contact_enabled('Soft', 'Ball')}; "
        "Pendulum<->Ball enabled: "
        f"{scene.is_layer_contact_enabled('Pendulum', 'Ball')}"
    )
    link_handles = actor.get_nested_link_actors()
    print(
        f"per-link colliders present: {len(link_handles)} links, but disabled vs Ball/Environment"
    )
    print("Contact between LowerArm and SoftArm auto-disabled via softAttachLinks")


def run_interactive(scene: Scene, actor: Actor, ball: Actor) -> None:
    """Attach the debugger, then run the scripted timeline, printing the
    soft's world bounds, joint angles, and the ball contact force once per
    second until the debugger detaches."""
    sim_time = 0.0
    next_report = 1.0

    if not physics.debugger.attach():
        return

    # Contact queries are supported on the nested soft actor, not the top-level articulated
    # actor.
    soft_handles = actor.get_nested_soft_actors()
    soft_actor = scene.get_actor(soft_handles[0]) if soft_handles else None
    if soft_actor is None:
        return

    contact_points = soft_actor.register_query(physics.QueryType.CONTACT_POINTS)
    contact_force = soft_actor.register_query(physics.QueryType.TOTAL_CONTACT_FORCE)

    while physics.debugger.is_attached():
        scene.step(TIME_STEP)
        sim_time += TIME_STEP

        if sim_time >= next_report:
            pose = physics.DynamicArrayReal(NUM_DOFS)
            actor.get_articulated_pose(pose)
            aabb = soft_actor.get_aabb_world()
            num_contacts = len(soft_actor.get_contact_points_world())
            force = soft_actor.get_contact_force_from_actor_world(ball)
            force_mag = float(np.linalg.norm(np.array([force[0], force[1], force[2]])))
            print(
                f"t={sim_time:.1f}s: joints=[{pose[0]:.2f}, {pose[1]:.2f}] rad, "
                f"soft min Y={aabb.min[1]:.3f} m, "
                f"ball contacts={num_contacts}, force={force_mag:.2f} N"
            )
            next_report += 1.0

    soft_actor.cancel_query(contact_points)
    soft_actor.cancel_query(contact_force)


def main() -> None:
    """Run the interactive soft-skinned-articulations tutorial."""
    physics.initialize(num_worker_threads=0)
    scene, actor, ball = build_scene()
    print_introspection(scene, actor)
    demonstrate_soft_surface(scene, actor)
    demonstrate_contact_control(scene, actor)
    run_interactive(scene, actor, ball)
    physics.destroy_scene(scene)
    physics.shutdown()
    print("Simulation complete.")


if __name__ == "__main__":
    main()
