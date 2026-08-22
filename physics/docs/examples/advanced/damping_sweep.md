---
title: Damping Parameter Sweep
sidebar_position: 2
---

# Damping Parameter Sweep

In scenarios where quantitatively accurate dynamics are desired, it is recommended to use the `BDF2` time integrator. However, to use this integrator effectively, it is important to calibrate physical damping parameters of all actors in a scene, because this integrator adds much less numerical dissipation than the default `BackwardEuler`. Two of the most important phenomena that often require physical damping for realistic responses are elasticity in deformable actors and compliant normal contact between rigid actors.

This example qualitatively demonstrates both of these damping mechanisms by dropping ducks with different damping parameters on a plane. One row of soft ducks sweeps the stiffness-damping coefficient, which dissipates energy within the deforming material, while another row of rigid ducks sweeps the normal viscous contact damping coefficient, which dissipates energy during impact. Both rows are built programmatically from the same tetrahedral mesh asset.

**Source**: `examples/example_damping_sweep.py`

For the underlying mathematical models of soft-actor viscoelasticity and normal contact damping, see the pages on [Soft Actors](../../concepts/actors/soft/overview.mdx#continuous-model) and [Contact](../../concepts/contact.md#friction-and-damping).

## Implementation

### Select the Time Integrator

To illustrate the effects of physical damping parameters, this scene uses `BDF2` rather than the default `BackwardEuler`.

```python
scene = physics.create_scene("Damping Sweep Scene")

solver_params = scene.get_solver_params()
solver_params.integration_method = physics.IntegrationMethod.BDF2
scene.set_solver_params(solver_params)
```

`BackwardEuler` adds a large amount of numerical dissipation at any reasonable time step size, which would obscure the effects of physical damping. `BDF2` is second-order accurate, so numerical dissipation no longer dominates and the parameter sweeps here reflect the specified physical damping. See [Comparison and Recommendations](../../concepts/dynamics.md#comparison-and-recommendations) for the tradeoffs between the available integrators.

### Create the Ground Plane

The ground is a static rigid actor with an infinite half-space geometry. Unlike the penalty coefficient, dissipation coefficients have no static-collider exception: they are combined over a contact pair as a geometric mean, so a zero coefficient on either actor disables the effect entirely. The ground therefore requires a non-zero coefficient of its own.

```python
GROUND_NORMAL_DAMPING_COEFFICIENT = 1.0  # [s/m]

plane_shape = physics.create_plane_shape(normal=[0, 1, 0], distance=0)
scene.create_rigid_actor(
    name="ground",
    shape=plane_shape,
    is_static=True,
    contact=physics.ContactParams(
        normal_viscous_damping_coefficient=GROUND_NORMAL_DAMPING_COEFFICIENT
    ),
)
```

See [Contact Parameter Combination](../../concepts/contact.md#contact-parameter-combination) for the full set of combination rules.

The soft ducks leave [`normal_viscous_damping_coefficient`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.ContactParams.normal_viscous_damping_coefficient) at its default of zero (dissipating energy instead through their bulk viscoelasticity), so the ground's coefficient has no effect on them.

### Load the Shared Shape

Both the soft and rigid rows of ducks use the same tetrahedral mesh, which is loaded into a shape once and shared between actors.

```python
DUCK_SCALE = 0.5

duck_shape = physics.load_shape_from_file(
    file_path=str(resolve_asset("duck/duck_coarse.mochi.h5")),
    bake_scale=[DUCK_SCALE] * 3,
)
```

`DUCK_SCALE` is baked into the shape and also drives the drop height, the column spacing, and the separation between the two rows, so the whole scene resizes with one constant.

### Sweep the Soft Material Damping

The parameter [`stiffness_damping_coefficient`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.SoftMaterialParams.stiffness_damping_coefficient) is the coefficient $\beta_R$ of the strain-rate dissipation potential, with units of time. It is swept from 1 ms to 4 ms across the row.

```python
STIFFNESS_DAMPING_COEFFICIENTS = (0.001, 0.002, 0.003, 0.004)  # [s]

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
```

Because the stiffness damping potential depends only on strain rate, this dissipation is only active during changes of shape, and is insensitive to rigid motions.
Mass-proportional damping is also available, but it adds drag in world space, and is only suitable for artificial stabilization of scenes, or as a crude model of air resistance on extremely light deformable objects like foam or cloth.

### Sweep the Normal Contact Damping

The parameter [`normal_viscous_damping_coefficient`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.ContactParams.normal_viscous_damping_coefficient) is the coefficient $c_n$ of the Hunt–Crossley impact model, with units of inverse velocity. The sweep is specified in terms of the effective coefficient of each duck-ground pair; each duck stores the value that produces it under the geometric mean.

```python
EFFECTIVE_NORMAL_DAMPING_COEFFICIENTS = (0.1, 0.2, 0.3, 0.4)  # [s/m]

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
```

See [Restitution Behavior of Rigid Collisions](../../concepts/contact.md#restitution-behavior-of-rigid-collisions) for some discussion on the interpretation of the contact damping coefficient in terms of a velocity-dependent coefficient of restitution (CoR).

### Run Interactively

The simulation advances with a fixed 1/300 s time step, which is small enough that numerical dissipation of the `BDF2` integrator will not overwhelm physical dissipation. If larger time steps are desired while still minimizing numerical dissipation, a non-dissipative integrator such as `SymplecticDIRK12` (equivalent to the implicit midpoint method) may be used, but this requires that all stiffness terms be accompanied by appropriate physical damping, which may be difficult to ensure in more complex scenes, and the value of strictly avoiding numerical dissipation when there is a large truncation error from the time step size is questionable for most use cases.

```python
TIME_STEP = 1.0 / 300.0  # [s]

physics.initialize(num_worker_threads=-1)
scene, _, _ = create_damping_sweep_simulation()

if not physics.debugger.attach():
    physics.shutdown()
    return

while physics.debugger.is_attached():
    scene.step(TIME_STEP)

physics.shutdown()
```

The energy from the impact is transferred to vibrational modes of the soft actors, which are damped to differing degrees, with vibrations continuing longer in the less-damped actors.  Because the contact forces on rigid actors are not directly aligned with their centers of mass, they do not bounce back with a pure linear translation (as in the most elementary interpretation of coefficients of restitution in particle dynamics).  Some energy is instead transferred into rotational modes, leading to chattering with the ground, which is again damped out to differing degrees, persisting longer in actors with less normal viscous contact damping.

## Running

```bash
uv run --no-project examples/example_damping_sweep.py
```

This example launches or focuses the SuperDex Physics Debugger and runs while it remains connected. See [Inspecting Scenes](../../debugging_scenes.md) for connection, navigation, and playback controls.
