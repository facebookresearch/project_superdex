---
title: Soft Duck
sidebar_position: 3
---

# Soft Duck

This example simulates a deformable duck-shaped body falling onto a ground plane using the finite element method (FEM). It demonstrates creating a soft actor from a tetrahedral mesh and its contact with a rigid actor.

**Source**: `examples/example_soft_duck.py`

For the underlying deformation formulation and material models, see [Soft Actors](../../concepts/actors/soft/overview.mdx). For the compliant interaction between the duck and the ground, see [Contact](../../concepts/contact.md).

## Key Concepts

### Tetrahedral Simulation Mesh

A soft actor requires a volumetric mesh whose tetrahedra fill the object's interior. SuperDex Physics uses this mesh for FEM simulation and derives its triangular boundary surface for surface operations and default rendering.

A shape may additionally contain a distinct triangular visual mesh embedded in the simulation mesh, but the `duck_1899.mochi.h5` asset used here does not rely on one. The `1899` identifies the simulation mesh's node count, not its tetrahedron count.

```python
from superdex.physics.paths import resolve_asset

shape_path = str(resolve_asset("duck/duck_1899.mochi.h5"))
tet_mesh_shape = physics.load_shape_from_file(
    file_path=shape_path,
)
```

The [Soft Duck with Visual Mesh](../advanced/soft_duck_visual_mesh.md) example compares these two rendering representations directly.

### Creating a Soft Actor

Create a soft actor from the tetrahedral shape and place it above the ground:

```python
soft_duck_actor = scene.create_soft_actor(
    name="duck",
    shape=tet_mesh_shape,
    world_from_local=physics.TransformRT(translation=[-0.5, 0.5, -1.0]),
)
```

By default, SuperDex Physics uses a stable Neo-Hookean material with default density and elastic parameters. Applications can customize these through the actor's material parameters; see [Soft Materials](../../concepts/actors/soft/materials/overview.md).

### Rigid-Soft Contact

The ground is a static rigid actor with an implicit plane shape:

```python
plane_shape = physics.create_plane_shape(normal=[0, 1, 0], distance=-0.5)
rigid_plane_actor = scene.create_rigid_actor(
    name="ground",
    shape=plane_shape,
    is_static=True,
)
```

No example-specific contact setup is required. The duck supplies deformable surface samples and the plane supplies an analytic collider, as described under [Contact Roles](../../concepts/contact.md#contact-roles).

### Simulation and Lifecycle

The example attaches the remote debugger and advances the scene at 60 Hz. Its cleanup function explicitly demonstrates actor, scene, and global resource destruction. Destroying actors is unnecessary immediately before destroying their scene, and destroying the scene is unnecessary immediately before [`physics.shutdown()`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.shutdown).

```python
TIME_STEP = 1.0 / 60.0  # [s]
if physics.debugger.attach():
    while physics.debugger.is_attached():
        scene.step(TIME_STEP)

scene.destroy_actor(soft_duck_actor)
scene.destroy_actor(rigid_plane_actor)
physics.destroy_scene(scene)
physics.shutdown()
```

## Running

```bash
uv run --no-project examples/example_soft_duck.py
```

This example launches or focuses the SuperDex Physics Debugger and runs while it remains connected. See [Inspecting Scenes](../../debugging_scenes.md) for connection, navigation, and playback controls.
