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

"""Example: Skinned Articulations

A double pendulum carrying a skinned surface, built in code and used as a guided
tour of the SuperDex Physics *skinned articulated actor* API. A two-bone
revolute skeleton is anchored to a world pivot:

    world -[Revolute]-> UpperArm -[Revolute]-> LowerArm

A single triangle mesh (the *skin*) is bound to the two links via linear-blend
skinning (LBS): vertices near the upper arm follow link 0, vertices past the
elbow follow link 1, and a smooth weight ramp blends them across the joint so
the one continuous surface bends as the pendulum swings.

The skin is **colliding only**: it probes contact against other actors, but does
not act as collider. The links play the collider role. Here a ball resting on the
ground is struck by the swinging skin. This is the programmatic twin of the
``assets/samples/articulations_skinned_double_pendulum.mochi_scene`` prefab, and
both load the same shared skin asset so they stay in sync.

The console walks the one-shot API sections at startup (introspection and
query-support probing, the skin surface read-back, how the surface deforms when
the joints move, and contact control). The debugger then attaches for
visualization while the simulation runs a short scripted timeline (report the
skin's world bounds, joint angles, and the contact force from the ball once per
simulated second, then damp the joints), until the debugger detaches.
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
#
# This example stores the pivot in joint_0 and leaves world_from_root at identity
# to mirror the reference prefab. Either placement is valid: world_from_root
# moves the links and skin together without changing their alignment.
ROOT_HEIGHT = 0.5  # [m] pivot height; chosen so the fully-extended tip reaches
# the floor ball (arm total length 2*L = 0.5 m).

ARM_LENGTH = 0.25  # [m] length of each arm along +X (L)
ARM_WIDTH = 0.025  # [m] cross-section of each arm (w)
# The cube mesh spans [0, 1]^3 (corner-anchored), so a baked arm occupies local
# [0, L] x [0, w] x [0, w]. parent_joint_from_link [0, -w/2, -w/2] centers the
# bar on its inbound joint in Y/Z while leaving it to extend along +X.
ARM_SCALE = [ARM_LENGTH, ARM_WIDTH, ARM_WIDTH]  # [m]

# Shared skin asset: a thin rectangular tube along +X spanning both arms in the
# root-local rest frame, with per-node LBS weights binding to link indices 0/1.
SKIN_ASSET = "samples/articulations_parts/skin.mochi.json"

BALL_RADIUS = 0.05  # [m]
BALL_POSITION = [0.05, 0.05, 0.0]  # [m] rests on the ground, within the tip's reach

# DoF layout: Revolute(1) + Revolute(1).
NUM_DOFS = 2
# Seed velocity per DoF: an upper-hinge kick to start the double pendulum
# swinging chaotically so the skin sweeps down into the ball.
SEED_JOINT_VELOCITIES = [4.2, 0.0]  # [rad/s]

TIME_STEP = 1.0 / 120.0  # [s]

# Scripted-timeline trigger time [s].
T_DAMP = 12.0  # [s]
DAMP_VISCOUS = 0.022  # [N*m*s/rad] joint friction for the damping event


def _make_joints() -> list[physics.ArticulatedJointParams]:
    """Build the 2 inbound joints of the chain (joints[i] drives links[i])."""
    return [
        # Upper hinge: revolute about Z, anchored at the world pivot (the pivot
        # height is baked into this joint so world_from_root stays identity).
        physics.ArticulatedJointParams(
            name="joint_0",
            type=physics.ArticulatedJointType.REVOLUTE,
            axis=[0, 0, -1],
            parent_link_from_joint=physics.TransformRT(translation=[0, ROOT_HEIGHT, 0]),
        ),
        # Elbow: revolute about Z, at the far end of the upper arm on its
        # centerline (link-local [L, w/2, w/2]).
        physics.ArticulatedJointParams(
            name="joint_1",
            type=physics.ArticulatedJointType.REVOLUTE,
            axis=[0, 0, -1],
            parent_link_from_joint=physics.TransformRT(
                translation=[ARM_LENGTH, ARM_WIDTH / 2, ARM_WIDTH / 2]
            ),
        ),
    ]


def _make_links(arm_shape: physics.ShapeHandle) -> list[physics.ArticulatedLinkParams]:
    """Build the 2 arm links. Each box is centered on its inbound joint in Y/Z
    and extends along +X. Both links keep a per-link Box collider on layer
    "Pendulum" -- these remain the probing colliders; the skin is a collidee."""
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
            shape=arm_shape,
            collider_type=physics.ColliderType.BOX,
            layer="Pendulum",
            density=1000.0,
        ),
    ]


def build_scene() -> tuple[Scene, Actor, Actor]:
    """Build the skinned-double-pendulum scene in code.

    Note: physics.initialize() must be called before this function.

    Returns:
        tuple: (scene, articulation, ball)
    """
    scene = physics.create_scene("Skinned Articulations Scene")

    # One baked cube shape shared by both arm links (corner-anchored, so baking
    # the scale gives a box occupying local [0, scale]).
    arm_shape = physics.load_shape_from_file(
        file_path=str(resolve_asset("cube/cube_fine_mesh.mochi.json")),
        bake_scale=ARM_SCALE,
    )

    # The skin: a skinned triangle mesh. Its rest coordinates are authored in the
    # skeleton's link-local rest frame (cross-section centered on the arm
    # centerline at y=z=w/2) so the tube overlays the two arms at rest.
    # bake_scale=1 keeps the authored units (baking only moves positions, never
    # the weights).
    skin_shape = physics.load_shape_from_file(
        file_path=str(resolve_asset(SKIN_ASSET)),
        bake_scale=[1, 1, 1],
    )

    params = physics.ArticulatedActorParams(name="SkinnedDoublePendulum")
    params.joints = _make_joints()
    params.links = _make_links(arm_shape)
    # Attach the skinned surface. The skin is a contact collidee. Two optional
    # tuning knobs (left at their defaults here) bound its per-step contact cost:
    # pairing boundary_element_type=P1Q1 with boundary_subsampling reduces the
    # number of contact-integration samples on the skin surface.
    params.skin = physics.ArticulatedSkinParams(
        shape=skin_shape,
        layer="Skin",
    )
    articulation = scene.create_articulated_actor(params)

    # The actor holds its own reference to each shape, so release our local
    # handles now (shapes are reference-counted and freed at the last release).
    physics.release_shape(arm_shape)
    physics.release_shape(skin_shape)

    # Seed initial joint velocities to set the double pendulum swinging.
    articulation.set_articulated_joint_velocities(
        velocities=np.array(SEED_JOINT_VELOCITIES, dtype=np_real)
    )

    # Ball on the ground, offset in +X within reach of the swinging skin. The
    # unit-radius icosphere renders as a real mesh; the analytic Sphere collider
    # handles contact (the ball is the collider, the skin the collidee).
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

    # Contact filter: only the skin is struck by the ball (which rests on the
    # ground). Disable the per-link "Pendulum" colliders against everything, and
    # keep the skin from interacting with the ground.
    scene.enable_layer_contact_symmetric("Pendulum", "Pendulum", enable=False)
    scene.enable_layer_contact_symmetric("Pendulum", "Skin", enable=False)
    scene.enable_layer_contact_symmetric("Pendulum", "Ball", enable=False)
    scene.enable_layer_contact_symmetric("Pendulum", "Environment", enable=False)
    scene.enable_layer_contact_symmetric("Skin", "Environment", enable=False)

    return scene, articulation, ball


def print_introspection(scene: Scene, articulation: Actor) -> None:
    """Tour the topology / introspection API and probe skin query support."""
    print(f"num_dofs = {articulation.get_num_dofs()}")

    # Each link is itself a queryable rigid sub-actor.
    link_handles = articulation.get_nested_link_actors()
    names = [scene.get_actor(h).get_name() for h in link_handles]
    print(f"nested link actors ({len(link_handles)}): {names}")

    info = articulation.get_articulated_shape_info()
    for i in range(len(info.link_names)):
        print(
            f"  link[{i}] {info.link_names[i]!r}: parent_link={info.parents[i]}, "
            f"inbound_joint={info.joint_names[i]!r} ({info.joint_types[i]})"
        )

    # The skin surface is exposed on the articulation actor itself. Probe which
    # queries it supports: the skin is a *surface* collidee, so surface-node and
    # contact queries apply, while volumetric/soft queries (NODE_POSITIONS,
    # ELEMENTS_DEFORMATION_GRADIENT) and VISUAL_NODE_POSITIONS do not.
    probes = [
        physics.QueryType.SURFACE_NODE_POSITIONS,
        physics.QueryType.SURFACE_NODE_NORMALS,
        physics.QueryType.CONTACT_POINTS,
        physics.QueryType.TOTAL_CONTACT_FORCE,
        physics.QueryType.NODE_POSITIONS,
        physics.QueryType.VISUAL_NODE_POSITIONS,
        physics.QueryType.ELEMENTS_DEFORMATION_GRADIENT,
    ]
    for q in probes:
        print(f"  is_query_supported({q}) = {articulation.is_query_supported(q)}")


def _skin_x_extent(positions: np.ndarray) -> tuple[float, float]:
    """Return the (min, max) local X of the flat [x, y, z, ...] node array."""
    xs = positions[0::3]
    return float(xs.min()), float(xs.max())


def demonstrate_skin_surface(scene: Scene, articulation: Actor) -> None:
    """Read the deformed skin surface via the SURFACE_NODE_* queries.

    Query data is computed during ``scene.step``, so register the query, step,
    then read it back. The reference surface mesh (topology) is available
    directly via ``get_surface_mesh``.
    """
    mesh = articulation.get_surface_mesh()
    print(
        f"skin surface mesh: {mesh.get_num_nodes()} nodes, "
        f"{mesh.get_num_elements()} triangles"
    )

    pos_query = articulation.register_query(physics.QueryType.SURFACE_NODE_POSITIONS)
    nrm_query = articulation.register_query(physics.QueryType.SURFACE_NODE_NORMALS)
    scene.step(TIME_STEP)

    positions = np.array(articulation.get_surface_mesh_node_positions_local())
    normals = np.array(articulation.get_surface_mesh_node_normals_local())
    x_min, x_max = _skin_x_extent(positions)
    print(
        f"skin surface node positions (local): {len(positions) // 3} nodes, "
        f"X extent = [{x_min:.3f}, {x_max:.3f}] m"
    )
    print(f"skin surface normals: {len(normals) // 3} nodes (3 values each)")

    aabb = articulation.get_aabb_world()
    print(
        "skin world AABB: "
        f"min={[round(v, 3) for v in aabb.min]}, "
        f"max={[round(v, 3) for v in aabb.max]}"
    )

    # Count how many skin surface nodes fall inside a local AABB covering the
    # lower arm (boundary_only uses the surface-node query registered above). The
    # skin's local frame has the pivot at y=ROOT_HEIGHT, so the arms lie near
    # that height along +X.
    volume = physics.Aabb(
        min=[0.2, ROOT_HEIGHT - 0.05, -0.05], max=[0.55, ROOT_HEIGHT + 0.05, 0.05]
    )
    hits: list[int] = []
    articulation.query_nodes_in_volume_local(
        volume, True, lambda node, _pos: hits.append(node)
    )
    print(f"skin nodes in the lower-arm half-volume = {len(hits)}")

    articulation.cancel_query(pos_query)
    articulation.cancel_query(nrm_query)


def demonstrate_contact_control(scene: Scene, articulation: Actor) -> None:
    """Tour the contact-layer controls around the skin collidee."""
    print(f"num contact layers = {scene.get_num_contact_layers()}")

    layer_names: list[str] = []
    scene.enumerate_contact_layer_names(lambda name: layer_names.append(name))
    print(f"contact layers = {sorted(layer_names)}")

    print(
        "Skin<->Ball enabled: "
        f"{scene.is_layer_contact_enabled('Skin', 'Ball')}; "
        "Pendulum<->Ball enabled: "
        f"{scene.is_layer_contact_enabled('Pendulum', 'Ball')}"
    )

    # The per-link box colliders are still present (they are the probing
    # colliders); the skin is the collidee that the ball strikes.
    link_handles = articulation.get_nested_link_actors()
    print(f"per-link colliders still present: {len(link_handles)} links")


def demonstrate_non_skinned_rejection(scene: Scene) -> None:
    """Teaching moment: a skin built from a *non-skinned* shape is rejected.

    The skin must carry per-node skinning weights and link indices. Attaching a
    plain triangle mesh (no skinning data) fails gracefully with a physics.Error.
    """
    plain_tri = physics.create_tri_mesh_shape(
        coordinates=[0, 0, 0, 1, 0, 0, 0, 1, 0],
        connectivity=[0, 1, 2],
    )
    link_shape = physics.load_shape_from_file(
        file_path=str(resolve_asset("cube/cube_fine_mesh.mochi.json")),
        bake_scale=ARM_SCALE,
    )
    params = physics.ArticulatedActorParams(name="BadSkin")
    params.joints = [
        physics.ArticulatedJointParams(
            name="joint_0", type=physics.ArticulatedJointType.REVOLUTE, axis=[0, 0, -1]
        )
    ]
    params.links = [
        physics.ArticulatedLinkParams(
            name="Arm",
            parent_link=-1,
            shape=link_shape,
            collider_type=physics.ColliderType.BOX,
            layer="Pendulum",
            density=1000.0,
        )
    ]
    params.skin = physics.ArticulatedSkinParams(shape=plain_tri, layer="Skin")
    try:
        scene.create_articulated_actor(params)
        print("non-skinned skin: accepted (unexpected)")
    except physics.Error:
        print("non-skinned skin: rejected (as expected)")
    finally:
        physics.release_shape(plain_tri)
        physics.release_shape(link_shape)


def _damp_joints(articulation: Actor) -> None:
    """Raise the viscous friction on both revolute joints to settle the motion."""
    friction = [
        physics.ArticulatedJointFrictionParams(viscous=DAMP_VISCOUS)
        for _ in range(NUM_DOFS)
    ]
    articulation.set_articulated_joint_friction_params(friction)


def run_interactive(scene: Scene, articulation: Actor, ball: Actor) -> None:
    """Attach the debugger, then run the scripted timeline, printing the
    skin's world bounds, joint angles, and the ball contact force once per
    second until the debugger detaches."""
    sim_time = 0.0
    next_report = 1.0  # [s] print state once per simulated second
    damped = False

    # Launch and attach the remote debugger for visualization and interaction.
    if not physics.debugger.attach():
        return

    # Register contact queries so we can read the force the ball exerts on the
    # skin during the run (computed each step).
    contact_points = articulation.register_query(physics.QueryType.CONTACT_POINTS)
    contact_force = articulation.register_query(physics.QueryType.TOTAL_CONTACT_FORCE)

    # Simulate until the debugger detaches.
    while physics.debugger.is_attached():
        if not damped and sim_time >= T_DAMP:
            _damp_joints(articulation)
            damped = True
            print(f"t={sim_time:.1f}s: raised joint friction (damping)")

        scene.step(TIME_STEP)
        sim_time += TIME_STEP

        if sim_time >= next_report:
            pose = physics.DynamicArrayReal(NUM_DOFS)
            articulation.get_articulated_pose(pose)
            aabb = articulation.get_aabb_world()
            num_contacts = len(articulation.get_contact_points_world())
            force = articulation.get_contact_force_from_actor_world(ball)
            force_mag = float(np.linalg.norm(np.array([force[0], force[1], force[2]])))
            print(
                f"t={sim_time:.1f}s: joints=[{pose[0]:.2f}, {pose[1]:.2f}] rad, "
                f"skin min Y={aabb.min[1]:.3f} m, "
                f"ball contacts={num_contacts}, force={force_mag:.2f} N"
            )
            next_report += 1.0

    articulation.cancel_query(contact_points)
    articulation.cancel_query(contact_force)


def main() -> None:
    """Run the interactive skinned-articulations tutorial."""
    # 0 = single-threaded (simplest); -1 = auto; N = N worker threads.
    physics.initialize(num_worker_threads=0)

    scene, articulation, ball = build_scene()

    # --- One-shot API tour (logged once at startup) --------------------------
    print_introspection(scene, articulation)
    demonstrate_skin_surface(scene, articulation)
    demonstrate_contact_control(scene, articulation)
    demonstrate_non_skinned_rejection(scene)

    # --- Run: attach the debugger, run the scripted timeline -----------------
    run_interactive(scene, articulation, ball)

    # Destroying the scene frees everything it contains.
    physics.destroy_scene(scene)
    physics.shutdown()
    print("Simulation complete.")


if __name__ == "__main__":
    main()
