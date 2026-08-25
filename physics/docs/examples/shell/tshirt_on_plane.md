---
title: T-shirt on Plane
sidebar_position: 1
---

# T-shirt on Plane

:::caution Experimental
Shell actors and shell self-contact are part of the **experimental** API. Their APIs may change in future releases.
:::

This example simulates a triangle-mesh T-shirt falling onto a static ground plane. It demonstrates loading a garment mesh, deriving shell properties from familiar three-dimensional material parameters, and enabling point-cloud self-contact.

**Source**: `examples/example_tshirt_on_plane.py`

For details on the shell formulation and its parameters, see [Shell Actors](../../concepts/actors/shell.mdx).
The current page provides some additional discussion and interpretation of these parameters in the specific context of modeling very thin cloth-like shells at garment scales with interactive performance.

## Implementation

### Create the Scene and Ground

This example sets non-default solver parameters, which is often necessary to attain interactive framerates in simulations of cloth garments modeled as flexible shells.
The loss of accuracy from limiting nonlinear solver iterations can be partially compensated by using the strong Wolfe line search, which is empirically effective for improving solution quality in deformable actors when there is a limited iteration budget.

```python
scene = physics.create_scene("T-shirt on Plane Scene")

solver_params = scene.get_solver_params()
solver_params.non_linear_solver.max_iter = 2
solver_params.non_linear_solver.line_search_type = physics.LineSearchType.WOLFE_STRONG
solver_params.experimental_eval.implicit_normal_force_for_dissipation = True
scene.set_solver_params(solver_params)
```

The current normal load is used for dissipation because it improves the accuracy and stability of frictional point-cloud self-contact. See [Stage-Discrete Dissipation](../../concepts/contact.md#stage-discrete-dissipation) for details.
This is technically inconsistent with the mathematical assumptions behind the `WOLFE_STRONG` line search, because contact dissipation with an implicit normal force does not derive exactly from an incremental potential.
However, the potential-based line search nonetheless improves convergence in spite of this inconsistency, since most terms of the formulation do derive from a potential.

This example uses the default linear solver, but using the experimental [`physics.LinearSolverType.PARALLEL_CG`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.LinearSolverType) linear solver can improve performance even further in applications where strict bit-level determinism is not required and a suitable multi-core environment is available.

The ground plane that the T-shirt falls onto is modeled as a static rigid actor with an infinite half-space geometry.

```python
plane_shape = physics.create_plane_shape(normal=[0, 1, 0], distance=0)
rigid_plane_actor = scene.create_rigid_actor(
    name="ground", shape=plane_shape, is_static=True
)
```

### Load the Garment Mesh

Shell actors use surface triangle meshes rather than the tetrahedral volume meshes used by soft actors. This asset contains a triangular simulation mesh and a subdivided visual mesh for smoother rendering.

```python
shape_path = str(resolve_asset("garments/tshirt_visual_subdiv_2.mochi.h5"))
shape = physics.load_shape_from_file(shape_path)
```

This example uses a preprepared model asset for brevity. The [Authoring Assets](../../authoring_scenes/authoring_assets.mdx) workflow shows how to load an OBJ physics mesh into [`ModelData`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.ModelData), validate it, and save it in HDF5 format. (That path does not create the optional subdivided visual mesh included in this example's asset.)

### Define the Shell Material

`ShellMaterialParams` stores thickness-integrated membrane and bending stiffnesses. The conversion utility derives those values and the areal density from common three-dimensional material properties:

```python
YOUNGS_MODULUS = 1e5  # [Pa]
POISSON_RATIO = 0.25
DENSITY = 1e3  # [kg/m^3]
THICKNESS = 2e-3  # [m]

material = physics.experimental.shell_material_params_from3d_isotropic(
    YOUNGS_MODULUS, POISSON_RATIO, DENSITY, THICKNESS
)
```

The four arguments are, in order: Young's modulus [Pa], Poisson's ratio, volumetric density [kg/m³], and assumed thickness [m].
See [Parameter Selection](../../concepts/actors/shell.mdx#parameter-selection) for the conversion formulas.
The parameter conversion is based on homogeneity assumptions that do not directly apply to woven textiles, but the 3D parameters nonetheless provide more intuitive handles for tuning material behavior than the raw membrane and bending stiffnesses, and are more useful than 2D parameters for both qualitative tuning and parameterizing representative property distributions.

### Enable Self-Contact

The shirt is translated 50 cm along `-x` and 10 cm along `+y` from its authored position. Its point-cloud collider uses shell nodes as collider-side samples with a 1.5 cm interaction radius. Setting `self_contact` allows the shirt to collide with other parts of the same shell.

```python
INITIAL_X_TRANSLATION = -0.5  # [m]
INITIAL_HEIGHT = 0.1  # [m]
CONTACT_RADIUS = 1.5e-2  # [m]

shell_params = physics.experimental.ShellActorParams(
    name="T-shirt",
    shape=shape,
    material=material,
    world_from_local=physics.TransformRT(
        translation=[INITIAL_X_TRANSLATION, INITIAL_HEIGHT, 0]
    ),
)
shell_params.point_cloud_collider.radius = CONTACT_RADIUS
shell_params.point_cloud_collider.self_contact = True

shell_actor = physics.experimental.create_shell_actor(scene, shell_params)
```

The contact radius controls the interaction range. While it corresponds, in principle, to the thickness of the shell in the context of self-contact, it is typically chosen significantly larger than the shell's physical thickness.
For example, `CONTACT_RADIUS` is much larger than the `THICKNESS` used to estimate material properties above.
There are two reasons for using an artificially large radius:

- It should be on the order of the triangle size to avoid introducing gaps in the point-cloud collider's discrete representation of the geometry (i.e., quadrature of the [double-integral generalization](../../concepts/contact.md#double-integral-generalization) of penalty contact).
- Collision detection is performed at discrete time steps, so using the true shell thickness could lead to tunneling artifacts, especially with the large time steps needed for interactive simulations.

Note that, because `PointCloud` colliders only interact with each other, the point-cloud contact radius _does not_ influence contact between the shell actor and objects without point-cloud colliders (e.g., an articulated actor modeling a hand).
In those interactions, the shell only has a colliding-actor role. (See [Contact Roles](../../concepts/contact.md#contact-roles) for further discussion on this distinction.) This allows for much finer manipulations of shell actors than would be possible with the point-cloud collider's approximate geometry.
This is appropriate for many practical scenarios where high fidelity is needed between an end effector and a thin object, while self-contact only needs to prevent qualitative topological violations in the trajectory.

### Run Interactively

The example is initialized with a hardware-derived worker count to take advantage of the parallelism available on most modern CPUs. Multi-threaded parallelism is a necessity for attaining interactive performance for garment-scale shell actors with sufficient resolution to capture qualitative behaviors of cloth. It then attaches the debugger for interactive visualization and advances the simulation with fixed 60 Hz time stepping until the debugger detaches. This comparatively large time step is practical because SuperDex Physics uses stable implicit time integration:

```python
TIME_STEP = 1.0 / 60.0  # [s]

physics.initialize(num_worker_threads=-1)
scene, _, _ = create_tshirt_on_plane_simulation()

if not physics.debugger.attach():
    physics.shutdown()
    return

while physics.debugger.is_attached():
    scene.step(TIME_STEP)

physics.shutdown()
```

For simplicity, this example does not adjust step size dynamically to synchronize simulation time with wall-clock time. However, optimized builds can be expected to run within a moderate factor of real time on modern multi-core CPUs.

## Running

```bash
uv run --no-project examples/example_tshirt_on_plane.py
```

This example launches or focuses the SuperDex Physics Debugger and runs while it remains connected. See [Inspecting Scenes](../../debugging_scenes.md) for connection, navigation, and playback controls.
