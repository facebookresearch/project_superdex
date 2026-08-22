---
title: Constraints
sidebar_position: 6
---

# Constraints

This example builds a **double pendulum on a moving base** and uses it as a guided
tour of the generic [`Constraint`](../../concepts/constraints.mdx) interface: creating
constraints, enumerating and inspecting them, tuning their parameters, reading per-step
diagnostics, animating a target, and destroying them.

**Source**: `examples/example_constraints_double_pendulum.py`

The scene has two rigid links:

- **`Link1`** is pinned to a fixed world point by a **rigid pivot-position** constraint — a 3-DoF ball joint to a world anchor. Rotation is free, so the link swings; the anchor is animated over time to create a *moving base*.
- **`Link2`** hangs off `Link1`'s free end via a **rigid spherical joint** — a body-to-body ball joint.

:::tip Constraints vs. articulated actors
Modeling a double pendulum with constraints (as done here) is the preferred approach when joints are **added or removed at runtime**. For a **fixed** joint structure, prefer an articulated actor instead — it is more efficient and robust.
:::

## Implementation

### Creating the Constraints

Both constraints share the same spring-damper parameters. The pivot pins a local point on `Link1` to a world anchor; the spherical joint ties `Link1`'s far end to `Link2`'s near end. (The `0.0125` offsets place the pivots on each link's centerline — the box mesh is corner-anchored, so its centerline sits at half the cross-section thickness.)

```python
pivot = scene.create_rigid_pivot_position_constraint(
    physics.RigidPivotPositionConstraintParams(
        actor=link1.get_handle(),
        local_position=[0, 0.0125, 0.0125],
        target_position=[0, 0.5, 0],   # the world anchor
        stiffness=2.5e4,
        damping=3.5,
        saturation=-1.0,               # force cap disabled
    )
)

spherical = scene.create_rigid_spherical_joint_constraint(
    physics.RigidSphericalJointConstraintParams(
        actor_a=link1.get_handle(),
        actor_b=link2.get_handle(),
        local_pos_a=[0.25, 0.0125, 0.0125],   # Link1 far end
        local_pos_b=[0, 0.0125, 0.0125],      # Link2 near end
        stiffness=2.5e4,
        damping=3.5,
        saturation=-1.0,
    )
)
```

### Enumerating and Looking Up Constraints

Every constraint has a stable handle. Enumerate all constraints in a scene with [`for_each_constraint`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Scene.for_each_constraint), and look one up with [`get_constraint`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Scene.get_constraint):

```python
types = []
scene.for_each_constraint(lambda c: types.append(c.get_type()))

assert scene.get_constraint(pivot.get_handle()) == pivot
```

### Introspecting a Constraint

The generic interface reports how a constraint is wired up. [`get_num_actors`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Constraint.get_num_actors) is a handy discriminator — the pivot involves **one** actor, the spherical joint **two**:

```python
for i in range(constraint.get_num_actors()):
    actor = constraint.get_actor(actor_index=i)
    dofs = constraint.get_dof_indices_for_actor(actor_index=i)
    print(actor.get_name(), list(dofs))
```

### Tuning Parameters

Stiffness, damping, and saturation can be read and changed at runtime:

```python
k = constraint.get_stiffness()
constraint.set_stiffness(stiffness=k * 2.0)
constraint.set_damping(damping=constraint.get_damping())
constraint.set_saturation(saturation=constraint.get_saturation())
```

### Reading Diagnostics Each Step

[`get_deviation`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Constraint.get_deviation) returns the current constraint error and needs no setup. [`get_force`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Constraint.get_force) requires a `CONSTRAINT_FORCE` query registered **before** stepping:

```python
pivot.register_query(physics.QueryType.CONSTRAINT_FORCE)   # once, before stepping
# ... inside the loop, after scene.step(dt):
deviation = pivot.get_deviation()                        # position error [m]
force = pivot.get_force()                                # generalized force
```

The example attaches the debugger to visualize the scene and prints each constraint's
force and the pivot deviation to the console roughly once per second.

### Driving the Moving Base

Setting the pivot's target every step animates the anchor along a small circle,
turning the fixed pendulum base into a moving one:

```python
pivot.set_target_position(physics.Real3(x, y, z))
```

Only target-bearing constraints support this. Calling [`set_target_position`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Constraint.set_target_position) on the spherical joint (which has no position target) raises a graceful error — the interface is uniform, but capabilities are type-specific:

```python
try:
    spherical.set_target_position(physics.Real3(0, 0.5, 0))
except physics.Error:
    pass  # not supported for a spherical joint
```

### Destroying Constraints

Remove a constraint explicitly with [`destroy_constraint`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Scene.destroy_constraint); destroying an actor **auto-destroys** the constraints attached to it:

```python
scene.destroy_constraint(spherical)   # Link2 detaches and falls away
scene.destroy_actor(link1)            # also removes the pivot constraint
```

## Features

- **World-anchor constraint** (`RigidPivotPosition`) pins a rigid body to a world point with a free rotation.
- **Body-to-body joint** (`RigidSphericalJoint`) couples two rigid bodies at a shared pivot.
- **Full interface tour**: creation, [`for_each_constraint`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Scene.for_each_constraint) enumeration, [`get_constraint`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Scene.get_constraint) lookup, introspection ([`get_type`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Constraint.get_type) / [`get_num_actors`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Constraint.get_num_actors) / [`get_actor`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Constraint.get_actor) / [`get_dof_indices_for_actor`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Constraint.get_dof_indices_for_actor)), parameter tuning, per-step [`get_deviation`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Constraint.get_deviation) / [`get_force`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Constraint.get_force), target animation, and destruction.
- **Moving base**: a time-varying [`set_target_position`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Constraint.set_target_position) drives the anchor along a circle.
- **Live diagnostics**: each constraint's force and the pivot deviation are printed to the console as the simulation runs.
- **Scripted timeline**: the example weakens the pivot (via stiffness), removes the middle joint, then destroys `Link1` to show its constraint auto-destroy.

## Running

```bash
uv run --no-project examples/example_constraints_double_pendulum.py
```

This example launches or focuses the SuperDex Physics Debugger and runs while it remains connected. See [Inspecting Scenes](../../debugging_scenes.md) for connection, navigation, and playback controls.

## Also Available as a Prefab

The same scene ships as a declarative [prefab](../../concepts/prefabs.mdx) — the **static** version of this example. It builds the identical double pendulum, but without the code-only extras (the moving base, live diagnostics, and the scripted add/remove timeline). It's the source of truth for the scene's geometry and constraint parameters.

**Source**: `assets/samples/constraints_double_pendulum.mochi_scene`

Load it into a fresh scene (or use [`physics.prefab.add_to_scene(...)`](pathname:///generated/api/v1.0/python/api/prefab.html#superdex.physics.prefab.add_to_scene) / C++ `prefab::AddToScene(...)` to add it into an existing one):

```python
from superdex.physics.utils.scene_helpers import create_scene_from_prefab

scene = create_scene_from_prefab("samples/constraints_double_pendulum.mochi_scene")
```

The two constraints map directly onto prefab keys — [`rigidPivotPosition`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1prefab_1_1ConstraintLists.html) and [`rigidSphericalJoint`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1prefab_1_1ConstraintLists.html) — mirroring the programmatic parameters above (actors are referenced by name):

```json
"constraints": {
  "rigidPivotPosition": [
    {
      "actor": "Link1",
      "localPosition": [0, 0.0125, 0.0125],
      "targetPosition": [0, 0.5, 0],
      "stiffness": 25000,
      "damping": 3.5
    }
  ],
  "rigidSphericalJoint": [
    {
      "actorA": "Link1", "localPosA": [0.25, 0.0125, 0.0125],
      "actorB": "Link2", "localPosB": [0, 0.0125, 0.0125],
      "stiffness": 25000,
      "damping": 3.5
    }
  ]
}
```
