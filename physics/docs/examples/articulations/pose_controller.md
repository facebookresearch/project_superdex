---
title: Pose Controller
sidebar_position: 4
---

# Pose Controller

This example extends the [Plain double pendulum on a rail](./double_pendulum_on_rail.md) with the SuperDex Physics [Pose Controller](../../concepts/pose_controller.mdx). The controller switches between joint-space and link-space tracking while the same four-link, five-DoF articulation kicks a ball and then traces a circle with its end effector.

**Source**: `examples/example_articulations_pose_controller.py`

The articulation has one link entry for each inbound joint. Its `Hard` root contributes no DoFs, the rail and upper hinge contribute one each, and the spherical lower joint contributes three:

```text
0: RailHousing / CeilingWeld (Hard)      -> no DoFs
1: Cart        / Rail        (Prismatic) -> DoF 0
2: UpperArm    / UpperSwing  (Revolute)  -> DoF 1
3: LowerArm    / LowerSwing  (Spherical) -> DoFs 2-4
```

## Controller Timeline

The example demonstrates three tracking configurations in one continuous simulation:

| Simulation time | Controller mode | Motion |
|---|---|---|
| 0-3 s | Hybrid | Hold the seeded rest pose with joint and end-effector tracking. |
| 3-7 s | Joint only | Pull the rail back, drive it forward, and swing the upper hinge to kick the ball. |
| 7 s onward | Link only | Move the `LowerArm` end effector around a horizontal circle. |

## Initializing the Actor

The example creates a scene, loads the Plain articulation prefab, and selects its articulated actor. The prefab also adds the ground plane and ball used by the example.

```python
scene = physics.create_scene("Articulations Pose Controller Scene")
result = physics.prefab.add_to_scene(
    prefab_path=str(
        resolve_asset("samples/articulations_double_pendulum_on_rail.mochi_scene")
    ),
    root_path=str(
        resolve_asset_root(
            "samples/articulations_double_pendulum_on_rail.mochi_scene"
        )
    ),
    scene=scene,
)
articulation = next(
    actor
    for actor in result.filter(physics.ActorType.ARTICULATED)
    if actor.get_name() == ARTICULATION_NAME
)
```

## Configuring Tracking

[`PoseControllerParams(num_links)`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.PoseControllerParams) creates three link-indexed arrays initialized with zero stiffness and damping. Entry `i` configures link `i` or its inbound joint.

```python
params = physics.PoseControllerParams(NUM_LINKS)

# The rail is prismatic (linear units) and the upper hinge revolute (angular
# units), so each needs its own gain pair.
params.joint_tracking[CART_LINK] = physics.PoseTrackingParams(
    stiffness=125.0,   # [N/m]
    damping=8.8,       # [N*s/m]
)
params.joint_tracking[UPPER_ARM_LINK] = physics.PoseTrackingParams(
    stiffness=31.25,   # [N*m/rad]
    damping=2.2,       # [N*m*s/rad]
)

params.link_pos_tracking[END_EFFECTOR_LINK] = physics.PoseTrackingParams(
    stiffness=75.0,
    damping=5.3,
)
params.link_rot_tracking[END_EFFECTOR_LINK] = physics.PoseTrackingParams(
    stiffness=1.56,
    damping=0.22,
)

articulation.add_articulated_pose_controller(params)
```

The [`joint_tracking`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.PoseControllerParams.joint_tracking) array is not DoF-indexed. Its `UpperArm` entry controls the revolute inbound joint, while the single `LowerArm` entry would configure all three DoFs of the spherical inbound joint. The root link also keeps an entry even though its `Hard` joint has no controllable DoFs.

A default [`PoseTrackingParams`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.PoseTrackingParams) has zero stiffness and damping, which disables that tracking entry. To switch modes, the example constructs a complete parameter set and replaces the controller parameters:

```python
def set_joint_only_controller(articulation: physics.Actor) -> None:
    params = physics.PoseControllerParams(NUM_LINKS)
    params.joint_tracking[CART_LINK] = physics.PoseTrackingParams(
        stiffness=125.0,
        damping=8.8,
    )
    params.joint_tracking[UPPER_ARM_LINK] = physics.PoseTrackingParams(
        stiffness=31.25,
        damping=2.2,
    )
    articulation.set_articulated_pose_controller_params(params)
```

The link-only configuration does the converse: it fills only the end-effector entries in [`link_pos_tracking`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.PoseControllerParams.link_pos_tracking) and [`link_rot_tracking`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.PoseControllerParams.link_rot_tracking). Replacing the full object turns tracking slices on or off without removing and re-adding the controller.

## Updating Targets

The target input representation is independent of what the controller tracks. The active [`joint_tracking`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.PoseControllerParams.joint_tracking), [`link_pos_tracking`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.PoseControllerParams.link_pos_tracking), and [`link_rot_tracking`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.PoseControllerParams.link_rot_tracking) entries determine which constraints apply; choosing a joint-space or link-space target API only determines how the desired pose is supplied.

SuperDex Physics maintains feasible joint-space and link-space representations of the same target. A joint-space target is converted through forward kinematics to link targets, including for active link tracking. A link-space target is converted to a feasible joint target, including for active joint tracking. Hybrid controllers can therefore use either target input representation without changing which tracking slices are enabled.

### Joint-Space Targets

Joint-space targets contain one value per articulation DoF. They are also converted internally to link targets for any active link position or rotation tracking. At startup, the example resets the actor and controller to a five-DoF rest target:

```python
rest_target = np.zeros(NUM_DOFS, dtype=np_real)
articulation.set_articulated_pose_from_joints(pose=rest_target)
articulation.set_articulated_joint_velocities(
    velocities=np.array([0.3, 4.2, 0.0, 0.0, 0.0], dtype=np_real)
)
articulation.reset_articulated_target_pose(pose=rest_target)
```

During the kick, a smooth interpolation first moves the prismatic rail backward. It then drives the rail forward while rotating the upper hinge. Each frame supplies the resulting target before stepping:

```python
joint_target = _joint_kick_target(sim_time, rest_target)
articulation.set_articulated_target_pose(pose=joint_target)
scene.step(TIME_STEP)
```

[`set_articulated_target_pose`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Actor.set_articulated_target_pose) infers target velocity from how the target changes between simulation steps. This lets damping respond to a smoothly moving trajectory rather than treating every update as a stationary target.

### Link-Space Targets

Link-space targets contain one world transform per link, even when only one link has non-zero tracking gains. They are converted internally to a feasible joint target for any active joint tracking; supplying link transforms does not imply that only link tracking is active. At the handoff to link-only mode, the example reads all current link transforms and preserves them as the circle center:

```python
circle_center_transforms = physics.DynamicArrayTransformRT(NUM_LINKS)
articulation.get_articulated_link_transforms(
    out_world_from_links=circle_center_transforms
)
```

Each subsequent target copies that complete array and replaces the end-effector transform. The X-Z translation follows a circle whose radius ramps smoothly from zero; the original end-effector rotation is retained.

The first link-space target is a discontinuous handoff from joint-space control, so it uses the reset API:

```python
articulation.reset_articulated_target_link_transforms(
    world_from_targets=link_targets
)
```

Resetting sets the supplied transforms and clears inferred target velocity, avoiding an artificial velocity kick at the mode boundary. Later frames use [`set_articulated_target_link_transforms`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Actor.set_articulated_target_link_transforms), allowing the controller to infer the velocity of the smooth circular trajectory.

## Reading Controller Generalized Force

Controller generalized force is computed on demand. Register the query before stepping, read it after a completed step, and cancel the registration when finished:

```python
query = articulation.register_query(
    physics.QueryType.ARTICULATED_CONTROLLER_FORCE
)

scene.step(TIME_STEP)
controller_forces = articulation.get_articulated_controller_force()

articulation.cancel_query(query)
```

The result has one generalized-force value per articulation DoF. Translational DoFs use force `[N]`; rotational DoFs use torque `[N·m]`. In this articulation, DoF 0 is the prismatic rail force and DoFs 1-4 are joint torques. Results are unavailable until a step has completed after registration.

## Features

- **Hybrid tracking**: combine joint targets with end-effector position and rotation targets.
- **Runtime mode changes**: replace complete controller parameter sets to enable joint-only or link-only tracking.
- **Smooth joint trajectory**: pull back and kick using interpolated rail and hinge targets.
- **Cartesian trajectory**: trace a ramped X-Z circle with the end effector.
- **Set and reset semantics**: infer velocity for continuous targets and clear it at discontinuous handoffs.
- **Generalized-force query**: report controller force or torque for every DoF.

## Running

```bash
uv run --no-project examples/example_articulations_pose_controller.py
```

The example runs through all three phases and keeps tracing the circular trajectory until the Debugger disconnects. See [Inspecting Scenes](../../debugging_scenes.md) for debugger connection, navigation, and playback controls.

## Also Available as a Prefab

The static hybrid setup is also available as a declarative [prefab](../../concepts/prefabs.mdx). It nests the Plain articulation prefab, resolves the actor through the nested path, and supplies positional tracking arrays:

**Source**: `assets/samples/articulations_pose_controller.mochi_scene`

```json
{
  "controllers": [
    {
      "articulatedActor": "Pendulum/DoublePendulumOnRail",
      "jointTracking": [
        {},
        { "stiffness": 125, "damping": 8.8 },
        { "stiffness": 31.25, "damping": 2.2 },
        {}
      ],
      "linkPosTracking": [
        {}, {}, {}, { "stiffness": 75, "damping": 5.3 }
      ],
      "linkRotTracking": [
        {}, {}, {}, { "stiffness": 1.56, "damping": 0.22 }
      ]
    }
  ],
  "prefabs": [
    {
      "name": "Pendulum",
      "path": "samples/articulations_double_pendulum_on_rail.mochi_scene"
    }
  ]
}
```

Array positions follow the same link and inbound-joint mapping as the Python parameters. The prefab defines the initial hybrid controller; the programmatic example adds the runtime mode changes, moving targets, and generalized-force readout.
