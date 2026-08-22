---
title: Inverse Kinematics
sidebar_position: 5
---

# Inverse Kinematics

:::caution Experimental
The IK solver is part of the experimental API. It may change without notice.
:::

This example solves inverse kinematics on a five-link articulation with mixed joint types, alternating between a rotation target and a position target on the last link so the arm snaps to a fresh random goal each iteration.

SuperDex Physics does not ship a dedicated IK backend. It reuses the physics engine as a quasistatic optimizer, so the solve runs inside a real scene and automatically respects everything already in it — joint limits, contacts, and any constraint the scene holds.

**Source**: `examples/example_ik.py`

For the objective function, the Newton solve, the full solver parameter reference and collision-aware IK, see [Inverse Kinematics](../../concepts/inverse_kinematics.mdx). To track targets with implicit PD constraints during a dynamic simulation, rather than solving for a static pose, see [Pose Controller](../../concepts/pose_controller.mdx) and the [Pose Controller example](./pose_controller.md).

## Implementation

### Create the IK Scene

The solver reconfigures its scene into a quasistatic optimizer — infinite timestep, no gravity, no friction, no inertia — which makes that scene useless for rendering or dynamic simulation. A production application therefore keeps a second scene for display and copies the solved pose across. This example is IK-only, so one dedicated scene is enough.

```python
ik_scene = physics.create_scene("Inverse Kinematics Scene")
ik_scene.set_gravity([0, 0, 0])
articulation = load_articulation(ik_scene)

ik_solver = physics.experimental.create_ik_solver(ik_scene)
solver_params = ik_solver.get_solver_params()
solver_params.max_iter = MAX_ITER
ik_solver.set_solver_params(solver_params)
```

`create_ik_solver` takes ownership of the scene: the scene must not be used directly afterwards, and destroying the solver destroys it. See [Tips and Best Practices](../../concepts/inverse_kinematics.mdx#tips-and-best-practices) for the dual-scene pattern.

### Load the Articulation

The prefab supplies a five-link chain whose inbound joints cover three of the four joint types, for twelve degrees of freedom in total:

```text
0: base   / jointA (Free)      -> DoFs 0-5
1: link1  / jointB (Revolute)  -> DoF 6    (axis Y)
2: link2  / jointC (Revolute)  -> DoF 7    (axis Z)
3: link3  / jointD (Revolute)  -> DoF 8    (axis X)
4: link4  / jointE (Spherical) -> DoFs 9-11
```

```python
prefab_params = physics.prefab.PrefabParams()
prefab_params.scale = PREFAB_SCALE
result = physics.prefab.add_to_scene(
    prefab_path=str(resolve_asset(PREFAB_PATH)),
    root_path=str(resolve_asset_root(PREFAB_PATH)),
    scene=scene,
    params=prefab_params,
)
articulation = result.actors[0]

scene.enable_layer_contact_symmetric("RigidLink", "RigidLink", enable=False)
```

The links are boxes that overlap at the joints. Because the solve honours contact like any other scene constraint, leaving self-collision on would have neighbouring links push each other apart. See [Contact Filtering](../contact_constraints/contact_filtering.md) for the filtering levels available.

### Pin the Root

`jointA` is a `Free` joint, so the base carries six unconstrained degrees of freedom. Left alone, the optimizer would satisfy every target by translating and rotating the whole articulation instead of bending it — a perfect score, and useless. Two rigid pivot constraints hold the base still:

```python
ROOT_PIVOT_LOCAL_POSITION = [0.05 * PREFAB_SCALE, 0, 0]  # [m]
ROOT_PIVOT_STIFFNESS = 1e4  # [N/m] and [N*m/rad]

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
```

These are ordinary scene constraints, not IK targets — the solver never sees them as goals, it simply has to satisfy them alongside everything else in the scene. They are compliant penalty constraints rather than hard ones, so they resist motion of the base with a finite stiffness instead of forbidding it outright; raise `ROOT_PIVOT_STIFFNESS` if the base still drifts under a stronger pull. An articulation with a fixed base needs none of this; pin the root only when the root joint is `Free`.

### Select the End Effector

Every target in this example is attached to the last link in the chain.

```python
links = articulation.get_nested_link_actors()
# SpanConstActorHandle does not support negative indexing.
end_effector = links[len(links) - 1]
```

### Alternate Position and Rotation Targets

Each actor holds at most one position target and one rotation target, and creating a second of the same kind replaces the first. This example wants exactly one active at a time, so each iteration clears the other kind before creating its own.

A rotation target is a rotation vector — an axis scaled by the angle in radians — not Euler angles or a quaternion:

```python
ROTATION_TARGET_RANGE = 0.5  # [rad]

target = [
    rng.uniform(-ROTATION_TARGET_RANGE, ROTATION_TARGET_RANGE) for _ in range(3)
]
ik_solver.clear_position_target(end_effector)
ik_solver.create_rotation_target(end_effector, [0, 0, 0], target, TARGET_WEIGHT)
```

A position target takes a local point on the link and a world-space goal. Offsetting the local point from the link's centre of mass rather than its origin aims at the tip of the box instead of its middle:

```python
POSITION_TARGET_RANGE = 0.1 * PREFAB_SCALE  # [m]
END_EFFECTOR_TIP_OFFSET = 0.05 * PREFAB_SCALE  # [m]

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
```

Targets are sampled in a small box around the origin, where the pinned base sits. The position range is the tighter of the two because a Cartesian goal is easier to place out of reach than an orientation is. The `weight` argument maps to the constraint stiffness. With a single active target its absolute value does not matter; with several active at once, their relative values set the priority between them.

### Run Interactively

The loop solves once per iteration and counts two separate things: how many Newton solves converged, and how many targets were actually met. Because the goals are random, some are unreachable - `solve_ik` returning `False` is an expected outcome here, not an error.

```python
if not physics.debugger.attach():
    return

rng = random.Random(RANDOM_SEED)

while physics.debugger.is_attached():
    ...
    if ik_solver.solve_ik():
        reached += 1
    solves += 1

    status = scene.get_solver_stats().convergence_status
    if status == physics.ConvergenceStatus.CONVERGED:
        converged += 1
```

`solve_ik` reports `True` only when every active target lands within `positionErrorThres` and `rotationErrorThres`. Those thresholds gate the reachability report, not the solve itself; see [Parameters Reference](../../concepts/inverse_kinematics.mdx#parameters-reference) for tuning them.

The solved pose stays in the IK actor, where [`get_articulated_pose()`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Actor.get_articulated_pose) would read it back for transfer to a visualization scene. On exit, destroying the solver destroys the scene it owns, so the scene is never destroyed separately:

```python
physics.experimental.destroy_ik_solver(ik_solver)
physics.shutdown()
```

## Running

```bash
uv run --no-project examples/example_ik.py
```

`debugger.attach()` launches or focuses the Debugger and waits for it to connect; the arm holds each solved pose for a second before jumping to the next target. Detaching the Debugger ends the loop and prints how many solves converged and how many of the sampled targets were reached. Targets come from a seeded generator, so repeated runs are identical. See [Inspecting Scenes](../../debugging_scenes.md) for debugger connection, navigation, and playback controls.
