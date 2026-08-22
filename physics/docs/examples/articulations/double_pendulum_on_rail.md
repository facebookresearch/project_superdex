---
title: Plain
sidebar_position: 1
---

# Plain

This example builds a **double pendulum on a rail** entirely in code and uses it as
a guided tour of the SuperDex Physics
[articulated actor](../../concepts/actors/articulated_actors.mdx) API: building the
joint/link chain, introspecting its topology, reading forward-kinematics state,
manipulating the state directly, modeling joints live, actuating without a controller,
reading the end-effector Jacobian, and controlling contact.

**Source**: `examples/example_articulations_double_pendulum_on_rail.py`

The scene is a single articulated actor — a serial 4-joint / 4-link chain welded to the ceiling:

```text
ceiling -[Hard]-> RailHousing -[Prismatic]-> Cart -[Revolute]-> UpperArm -[Spherical]-> LowerArm
```

A cart slides on a horizontal prismatic rail; hanging from it is a double pendulum (a revolute upper hinge and a spherical lower joint) whose tip strikes a ball resting on the ground. It has **5 DoFs** (Prismatic 1 + Revolute 1 + Spherical 3; the `Hard` root contributes 0).

:::tip Articulated actors vs. constraints
An articulated actor has a **fixed** joint topology and is more efficient and robust than an equivalent set of constraints. Prefer it whenever the set of joints does not change at runtime. If you need to **add or remove joints at runtime**, model the joints as constraints instead — see the [Constraints example](../contact_constraints/constraints.md).
:::

## Implementation

### Building the Articulated Chain

An articulated actor is described by parallel `joints[]` and `links[]` arrays, where `joints[i]` is the inbound joint of `links[i]` and `parent_link = i - 1` forms a serial chain. `joint.parent_link_from_joint` places the joint frame in its parent link's frame; `link.parent_joint_from_link` places the link body relative to its inbound joint frame.

```python
params = physics.ArticulatedActorParams(name="DoublePendulumOnRail")
params.world_from_root = physics.TransformRT(translation=[0, 0.75, 0])
params.joints = [
    # Root weld: fixes the rail housing to the world (0 DoFs).
    physics.ArticulatedJointParams(name="CeilingWeld", type=physics.ArticulatedJointType.HARD),
    # Horizontal rail with soft limits, viscous friction, and armature inertia.
    physics.ArticulatedJointParams(
        name="Rail",
        type=physics.ArticulatedJointType.PRISMATIC,
        axis=[1, 0, 0],
        min_limit=[-0.2, 0, 0],  # scalar limit times axis
        max_limit=[0.2, 0, 0],
        limit_stiffness=250.0,
        limit_damping=8.8,
        friction=physics.ArticulatedJointFrictionParams(viscous=0.018),
        inertia=0.125,
    ),
    # Upper pendulum hinge (revolute about Z) and lower ball joint (spherical).
    physics.ArticulatedJointParams(name="UpperSwing", type=physics.ArticulatedJointType.REVOLUTE, axis=[0, 0, -1]),
    physics.ArticulatedJointParams(name="LowerSwing", type=physics.ArticulatedJointType.SPHERICAL),
]
params.links = [
    # RailHousing, Cart, UpperArm, LowerArm — each a Box collider with a
    # parent_link, a parent_joint_from_link offset, a layer, and a density.
    # (See the example source for the exact geometry.)
    # ...
]
articulation = scene.create_articulated_actor(params)

# Seed the swing: a small rail drift plus an upper-hinge kick.
articulation.set_articulated_joint_velocities(velocities=[0.3, 4.2, 0, 0, 0])
```

### Introspecting the Articulation

[`get_articulated_shape_info`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Actor.get_articulated_shape_info) is a one-stop dump of the topology. Each link is also a queryable rigid **sub-actor**, and joint limits are exposed as inspectable constraints.

```python
info = articulation.get_articulated_shape_info()
for i in range(len(info.link_names)):
    # parents[i] is the *parent link index* of link i (-1 for the root).
    print(info.link_names[i], info.parents[i], info.joint_types[i])

articulation.get_num_dofs()                          # 5
articulation.get_nested_link_actors()                # 4 rigid sub-actors
articulation.get_articulated_joint_limit_constraints()  # limits as constraints
```

### Reading the State (Forward Kinematics)

Read the joint-space pose, the world transforms of every link, and the joint velocities. These use pre-sized output containers.

```python
num_dofs = articulation.get_num_dofs()

pose = physics.DynamicArrayReal(num_dofs)
articulation.get_articulated_pose(pose)

transforms = physics.DynamicArrayTransformRT(len(articulation.get_nested_link_actors()))
articulation.get_articulated_link_transforms(transforms)

velocities = physics.DynamicArrayReal(num_dofs)
articulation.get_articulated_joint_velocities(velocities)
```

### Manipulating the State Directly

Set the pose from joint-space DoFs (e.g. slide the rail by hand), compose a joint-space delta in the tangent space (so spherical DoFs behave correctly), or set the pose from link transforms (IK-style).

```python
# Slide the rail + rotate the upper hinge, then set the actor to that pose.
articulation.set_articulated_pose_from_joints(pose=poked)

# Tangent-space add: out_pose = poked (+) delta.
articulation.add_articulated_delta_to_pose(pose=poked, delta_dofs=delta, out_pose=out_pose)
articulation.set_articulated_pose_from_joints(pose=out_pose)

# IK-style: read link transforms, nudge the end-effector, and write them back.
articulation.set_articulated_pose_from_links(world_from_links=transforms)

# Kick it swinging.
articulation.set_articulated_joint_velocities(velocities=[0.3, 4.2, 0, 0, 0])
```

### Modeling Joints Live

Per-joint friction (viscous/coulomb) and armature inertia can be read and changed mid-simulation — for example, to damp the pendulum by raising the swing joints' friction.

```python
friction = list(articulation.get_articulated_joint_friction_params())  # one per joint
friction[2] = physics.ArticulatedJointFrictionParams(viscous=0.022)   # damp the revolute
articulation.set_articulated_joint_friction_params(friction)

inertia = articulation.get_articulated_joint_inertia_params()     # one per joint
articulation.set_articulated_joint_inertia_params(list(inertia))
```

### Actuating without a Controller

Apply generalized forces to specific DoFs, or pin DoFs with a boundary condition (e.g. freeze the rail so only the pendulum swings).

```python
# Push the cart along the rail with a constant DoF force.
articulation.set_external_forces_on_dofs(dof_indices=[0], force_values=[1.0])
articulation.clear_external_forces()

# Freeze the rail DoF at its current value, then release it.
articulation.add_boundary_condition_dofs_world(dof_indices=[0], dof_values_world=[rail_pos])
articulation.clear_boundary_conditions()
```

### Mass, Root, and Center of Mass

Mass and the root transform are whole-articulation queries; center of mass and velocity are **per-rigid-body** queries, so read them from a nested link sub-actor rather than the top-level articulated actor. (The articulated equivalent of [`set_velocity`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Actor.set_velocity) is [`set_articulated_joint_velocities`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Actor.set_articulated_joint_velocities).)

```python
articulation.get_mass()                 # total mass of the chain
articulation.get_root_transform()       # whole-articulation pose
articulation.set_root_transform(t)      # teleports the whole articulation

links = articulation.get_nested_link_actors()
lower_arm = scene.get_actor(links[len(links) - 1])
lower_arm.get_center_of_mass_transform()   # per-link (rigid) query
```

### The End-Effector Jacobian

The Jacobian (joint motion → link motion) is read from a nested link sub-actor. It is a flattened `6 x num_dofs` (3 translation + 3 rotation rows per joint DoF).

```python
links = articulation.get_nested_link_actors()
end_effector = scene.get_actor(links[len(links) - 1])
jacobian = end_effector.get_articulated_jacobian()
```

### Controlling Contact

The scene uses string **layers** for coarse control and per-actor overrides for the finest control. Here only the end-effector tip collides with the ball (and the ball with the ground); everything else is disabled.

```python
scene.enable_layer_contact_symmetric("Pendulum", "Ball", enable=False)
scene.is_layer_contact_enabled("EndEffector", "Ball")    # left enabled
scene.get_num_contact_layers()
scene.enumerate_contact_layer_names(lambda name: ...)

# Finest-grained: toggle the specific LowerArm sub-actor against the ball.
scene.enable_actor_contact_symmetric(
    links[len(links) - 1],
    ball.get_handle(),
    enable=True,
    include_nested_actors=physics.IncludeNestedActors.NO,
)
```

## Features

- **Build-time modeling**: all joint types (`Hard` / `Prismatic` / `Revolute` / `Spherical`), joint limits, per-joint friction and armature inertia, and per-link shape / collider / layer / density.
- **Introspection**: [`get_articulated_shape_info`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Actor.get_articulated_shape_info), [`get_num_dofs`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Actor.get_num_dofs), [`get_nested_link_actors`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Actor.get_nested_link_actors), and joint limits as inspectable constraints.
- **Forward kinematics**: read pose, link transforms, and joint velocities.
- **Direct state manipulation**: set the pose from joints or from link transforms, add a tangent-space delta, and set joint velocities.
- **Live joint modeling**: read/write per-joint friction and armature inertia mid-simulation.
- **Actuation without a controller**: external DoF forces and DoF boundary conditions.
- **End-effector Jacobian** read from a nested link sub-actor.
- **Contact control**: layer-level and per-actor-pair enable/disable.
- **Live console output**: the example prints the rail position and the upper-hinge angle once per simulated second.
- **Scripted timeline**: freeze the rail → release → push the cart with an external force → damp the pendulum by raising joint friction.

## Running

```bash
uv run --no-project examples/example_articulations_double_pendulum_on_rail.py
```

This example launches or focuses the SuperDex Physics Debugger and runs while it remains connected. See [Inspecting Scenes](../../debugging_scenes.md) for connection, navigation, and playback controls.

## Also Available as a Prefab

The same scene ships as a declarative [prefab](../../concepts/prefabs.mdx) — the **static** version of this example. It builds the identical chain, but without the code-only extras (the console reporting and the scripted timeline). It's the source of truth for the scene's geometry and joint parameters.

**Source**: `assets/samples/articulations_double_pendulum_on_rail.mochi_scene`

Load it into a fresh scene (or use [`physics.prefab.add_to_scene(...)`](pathname:///generated/api/v1.0/python/api/prefab.html#superdex.physics.prefab.add_to_scene) / C++ `prefab::AddToScene(...)` to add it into an existing one):

```python
from superdex.physics.utils.scene_helpers import create_scene_from_prefab

scene = create_scene_from_prefab("samples/articulations_double_pendulum_on_rail.mochi_scene")
```

The joints and links map directly onto prefab keys (joint limits are encoded as `scalar · axis`):

```json
"actors": {
  "articulated": [
    {
      "name": "DoublePendulumOnRail",
      "translation": [0, 0.75, 0],
      "jointVelocities": [0.3, 4.2, 0, 0, 0],
      "joints": [
        { "name": "CeilingWeld", "type": "Hard" },
        {
          "name": "Rail", "type": "Prismatic", "axis": [1, 0, 0],
          "minLimit": [-0.2, 0, 0], "maxLimit": [0.2, 0, 0],
          "limitStiffness": 250, "limitDamping": 8.8,
          "friction": { "viscous": 0.018 }, "inertia": 0.125
        },
        { "name": "UpperSwing", "type": "Revolute", "axis": [0, 0, -1] },
        { "name": "LowerSwing", "type": "Spherical" }
      ],
      "links": [ /* RailHousing, Cart, UpperArm, LowerArm */ ]
    }
  ]
}
```
