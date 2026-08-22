---
title: Soft Duck with Visual Mesh
sidebar_position: 1
---

# Soft Duck with Visual Mesh

This example drops two identical soft ducks side by side. Both use the same coarse tetrahedral mesh for simulation, but only one uses the asset's embedded high-resolution visual mesh. The comparison shows how visual detail can be decoupled from simulation resolution.

**Source**: `examples/example_soft_duck_visual_mesh.py`

For the volumetric deformation formulation, see [Soft Actors](../../concepts/actors/soft/overview.mdx). For the interaction with the ground plane, see [Contact](../../concepts/contact.md).

## Simulation and Visual Meshes

The `duck_coarse.mochi.h5` asset contains two distinct representations:

- A coarse tetrahedral simulation mesh used for FEM.
- A triangular visual mesh whose vertices are embedded in the tetrahedra using element indices and barycentric weights.

As the simulation mesh deforms, SuperDex Physics maps its node positions through the embedding to update the visual mesh. The remote debugger automatically renders this visual mesh when it is available.

:::note
This example uses a prepared asset for simplicity.
For C++ asset-authoring workflows, `superdex::model_utils::GenerateVisualMeshEmbedding()` can generate the embedding for a [`ModelData`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1ModelData.html) containing a tetrahedral physics mesh and triangular visual mesh in the same coordinate frame. This utility is not currently exposed in Python.
:::

```python
import superdex.physics as physics
from superdex.physics.paths import resolve_asset

physics.initialize(num_worker_threads=0)
scene = physics.create_scene("Soft Body Visual Mesh Scene")

shape_with_visual_mesh = physics.load_shape_from_file(
    file_path=str(resolve_asset("duck/duck_coarse.mochi.h5")),
)
```

## Building the Comparison Shape

The example extracts the tetrahedral simulation mesh and creates a second shape without the embedded visual mesh:

```python
simulation_mesh = physics.get_shape_mesh(shape_with_visual_mesh)
shape_without_visual_mesh = physics.create_mesh_shape(simulation_mesh)
```

For an actor without a visual mesh, the debugger falls back to the triangular boundary surface derived from the tetrahedral simulation mesh. This is different from a separately authored visual mesh even though both are triangular surfaces.

## Placing the Actors Side by Side

The two actors differ only in their shapes and horizontal positions:

```python
visual_mesh_actor = scene.create_soft_actor(
    name="duck_with_visual_mesh",
    shape=shape_with_visual_mesh,
    world_from_local=physics.TransformRT(translation=[-1.0, 0.5, -0.5]),
)
simulation_mesh_actor = scene.create_soft_actor(
    name="duck_without_visual_mesh",
    shape=shape_without_visual_mesh,
    world_from_local=physics.TransformRT(translation=[0.0, 0.5, -0.5]),
)

plane_shape = physics.create_plane_shape(normal=[0, 1, 0], distance=0.0)
ground_actor = scene.create_rigid_actor(
    name="ground", shape=plane_shape, is_static=True
)
```

The left duck therefore renders the embedded visual mesh, while the right duck exposes the coarse simulation mesh's boundary. Both undergo the same FEM simulation and [rigid-soft contact](../../concepts/contact.md#contact-roles).

## Simulation and Lifecycle

The example attaches the remote debugger and advances both actors at 60 Hz. As in the basic examples, explicit actor and scene destruction demonstrates each lifecycle operation; destroying the scene also destroys its actors, and [`physics.shutdown()`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.shutdown) also destroys remaining scenes.

```python
TIME_STEP = 1.0 / 60.0  # [s]
if physics.debugger.attach():
    while physics.debugger.is_attached():
        scene.step(TIME_STEP)

scene.destroy_actor(visual_mesh_actor)
scene.destroy_actor(simulation_mesh_actor)
scene.destroy_actor(ground_actor)
physics.destroy_scene(scene)
physics.shutdown()
```

## Running

```bash
uv run --no-project examples/example_soft_duck_visual_mesh.py
```

This example launches or focuses the SuperDex Physics Debugger and runs while it remains connected. See [Inspecting Scenes](../../debugging_scenes.md) for connection, navigation, and playback controls.
