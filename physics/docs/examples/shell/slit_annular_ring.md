---
title: Slit Annular Ring Benchmark
sidebar_position: 2
---

# Slit Annular Ring Benchmark

:::caution Experimental
Shell actors are part of the **experimental** API. Their API may change in future releases.
:::

This example solves the slit annular ring benchmark described by [Sze et al. (2004)](#references), a standard test for geometrically nonlinear thin-shell analysis from the engineering mechanics literature.

Using a shell actor to solve an engineering benchmark complements interactive cloth examples, such as the [T-shirt on Plane](./tshirt_on_plane.md) example, which often fall in a regime of extremely low bending stiffness and prioritize qualitative visual behavior and real-time performance over quantitatively accurate structural analysis.

Much of the engineering literature on shell analysis emphasizes the small-displacement limit, where a linearized theory can be used to model stiff structures. However, in the context of human-scale object manipulation, such structures (e.g., the roof of a building) would typically be modeled as static geometry or rigid actors, and SuperDex Physics does not support the corresponding linear shell theory. The slit annular ring specifically exercises large deformation of a shell structure while retaining significant bending stiffness, placing it outside the regime targeted by many specialized cloth simulators.

The example also demonstrates how to prescribe nodal boundary conditions, apply external forces to selected degrees of freedom (DoFs), and use damped dynamics to approach a static reference solution.

**Source**: `examples/example_slit_annular_ring.py`

For the shell formulation and parameter reference, see [Shell Actors](../../concepts/actors/shell.mdx).

## Benchmark Setup

The benchmark geometry and reference displacement are scaled down by a factor of 10 from the dimensions of the canonical benchmark setup in the literature. Other parameters, including stiffness and load, are scaled in a dimensionally covariant way, preserving the nondimensional equilibrium response while making the geometry a better fit for SuperDex Physics' assumed meter-kilogram-second (MKS) system of units. Following this scaling, the supplied mesh is an annulus with an inner radius of 0.6 m and an outer radius of 1.0 m, and the Python example assigns a thickness of 3 mm, a Young's modulus of 210 MPa, and a Poisson's ratio of zero.

The example loads a prepared asset consisting of a triangular physics mesh with 325 vertices, selected to approach the static solution within a few seconds of wall-clock time while maintaining an acceptable level of displacement error for manipulation problems:

```python
shape = physics.load_shape_from_file(
    str(resolve_asset("samples/slit_annular_ring.mochi.h5"))
)
```

The same physics mesh could instead originate as an OBJ file and follow the [Authoring Assets](../../authoring_scenes/authoring_assets.mdx) workflow for loading, validation, and HDF5 serialization, but that conversion is outside the scope of this example.
Similarly, the node sets used below are specified directly to produce a simple, self-contained example.
In a larger engineering workflow, a third-party mesh generator or preprocessing tool could identify and export the node sets for prescribed and loaded boundaries.

## Create the Shell Actor

The material helper converts three-dimensional isotropic properties into the thickness-integrated membrane and bending parameters used by the shell actor:

```python
material = physics.experimental.shell_material_params_from3d_isotropic(
    youngs_modulus3d=YOUNGS_MODULUS,
    poissons_ratio3d=POISSONS_RATIO,
    density3d=DENSITY,
    thickness=THICKNESS,
)
material.mass_damping_coefficient = MASS_DAMPING_COEFFICIENT

actor = physics.experimental.create_shell_actor(
    scene,
    physics.experimental.ShellActorParams(
        name="SlitAnnularRing",
        shape=shape,
        material=material,
        world_from_local=physics.TransformRT(),
        has_gravity=False,
    ),
)
```

The example uses a mass density of 1,000 kg/m³ to perform a dynamic solve and adds mass-proportional damping to dissipate transient motion.
In certain cases, especially for cloth simulation, mass damping may have a physical interpretation as a drag force from the surrounding air.
In this problem, however, mass and damping are artificial regularization parameters that drive the solution toward static equilibrium, since the original benchmark is static and has no inertial response.

## Prescribe Boundary Conditions

The first two azimuthal node columns on every radial ring are fixed at their reference positions. Fixing the nodes on the slit edge and the immediately adjacent column creates a discrete clamped boundary:

```python
coordinates = np.asarray(
    list(physics.get_shape_mesh(shape).coordinates), dtype=np_real
)
fixed_positions = coordinates.reshape(-1, 3)[FIXED_NODES].reshape(-1)
actor.add_boundary_condition_nodes_world(
    node_indices=FIXED_NODES,
    node_positions_world=fixed_positions,
)
```

[`add_boundary_condition_nodes_world`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Actor.add_boundary_condition_nodes_world) prescribes the world-space position of each selected node. Here, the target positions are copied from the undeformed mesh, so all three translational DoFs of every selected node remain fixed.

## Apply External Nodal Forces

A total transverse force of 0.32 N is distributed uniformly over the five nodes on the opposite slit edge. Shell nodes have three displacement DoFs in `x`, `y`, and `z`, so the `z` DoF for node `i` is `3 * i + 2`:

```python
loaded_dofs = 3 * LOADED_NODES + 2
force_values = np.full(
    len(LOADED_NODES),
    TOTAL_TRANSVERSE_FORCE / len(LOADED_NODES),
    dtype=np_real,
)
actor.set_external_forces_on_dofs(
    dof_indices=loaded_dofs,
    force_values=force_values,
)
```

This API applies forces directly to the selected DoFs. The example uses explicit node indices because they are stable for the supplied benchmark mesh.

## Approach the Static Solution with Damped Dynamics

As mentioned above, the original benchmark is a static problem, usually solved as a sequence of nonlinear static load steps that gradually continue the load to its maximum value. SuperDex Physics is designed around dynamic problems, so this example instead applies the maximum load immediately and advances an implicit dynamic simulation with an artificial mass density and a 30 Hz time-stepping rate in simulation time. Modern CPUs are able to step faster in wall-clock time. A mass-damping coefficient of 0.3 s⁻¹ removes kinetic energy so that the oscillatory transient asymptotically approaches static equilibrium with minimal overshoot.

```python
scene.set_gravity([0, 0, 0])

while physics.debugger.is_attached():
    scene.step(TIME_STEP)
```

Dynamic relaxation makes the benchmark directly runnable through the same stepping interface used by interactive simulations. It also means that the displacement should be interpreted after the transient has substantially decayed, rather than at a predetermined load step.

## Compare with the Reference Response

The response quantity is the `z` displacement of node 324, the outer-radius node on the loaded slit edge. The example reports it once per simulated second alongside the 1.75 m reference value from the literature:

```python
displacement_z = get_tracked_displacement(actor)[2]
print(
    f"t={scene.get_total_simulation_time():.1f} s: node {TRACKED_NODE} "
    f"u_z={displacement_z:.6g} m; steady reference="
    f"{REFERENCE_TRACKED_Z_DISPLACEMENT:.6g} m"
)
```

As the damped motion settles, the computed displacement approaches the published reference to within a few percent (about 4% for the supplied mesh). This is a reasonable discretization error for a coarse mesh of a few hundred linear triangles undergoing very large deformation. The example prints the comparison for inspection rather than imposing an automated convergence criterion.

## Running

Run the example:

```bash
uv run --no-project examples/example_slit_annular_ring.py
```

The debugger displays the deformation and controls how long the simulation runs. Displacement logging appears in the terminal once simulation playback begins. See [Inspecting Scenes](../../debugging_scenes.md) for debugger connection, navigation, and playback controls.

## References

- K. Y. Sze, X. H. Liu, and S. H. Lo, [Popular Benchmark Problems for Geometric Nonlinear Analysis of Shells](https://doi.org/10.1016/j.finel.2003.11.001), *Finite Elements in Analysis and Design*, 40, pp. 1551–1569, 2004.
