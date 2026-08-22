---
title: Rigid Bodies
sidebar_position: 1
---

# Rigid Bodies

This example simulates a sphere and cube falling onto a table under gravity. It demonstrates four ways to create rigid actors and their shapes in SuperDex Physics.

**Source**: `examples/example_rigid_bodies.py`

For details about rigid-body dynamics and collision representations, see [Rigid Actors](../../concepts/actors/rigid_actors.mdx). For the compliant collision model used when the objects land, see [Contact](../../concepts/contact.md).

## Key Concepts

### Initialization and Scene Creation

Every SuperDex Physics program starts by initializing the engine. A scene contains its actors, constraints, and simulation state.

```python
import superdex.physics as physics
from superdex.physics.paths import resolve_asset, resolve_asset_root

physics.initialize(num_worker_threads=0)
scene = physics.create_scene("Rigid Bodies Scene")

# The default gravity is (0, -9.8, 0); set it explicitly for illustration.
scene.set_gravity([0, -9.8, 0])
```

A worker count of `0` runs on the calling thread. Use `-1` to let SuperDex Physics select a worker count, or a positive value to choose it explicitly.

### Four Ways to Create Shapes and Actors

**1. Implicit shape** — An infinite plane needs no mesh data:

```python
plane_shape = physics.create_plane_shape(normal=[0, 1, 0], distance=-1.0)
ground_actor = scene.create_rigid_actor(
    name="ground",
    shape=plane_shape,
    is_static=True,
)
```

**2. Mesh loaded from a resolved asset** — Load a pre-built sphere and scale it to a radius of 0.2 m:

```python
sphere_shape = physics.load_shape_from_file(
    file_path=str(resolve_asset("sphere/icosphere_3subdiv.1.mochi.json")),
    bake_scale=[0.2, 0.2, 0.2],
)
```

**3. Mesh defined programmatically** — Supply triangle vertices and connectivity directly:

```python
coordinates = np.array(
    [
        [-half, -half, -half],
        [half, -half, -half],
        # ... 8 vertices total
    ],
    dtype=np.float32,
).flatten()
connectivity = np.array(
    [
        [0, 2, 1],
        [0, 3, 2],
        # ... 12 triangles total
    ],
    dtype=np.int32,
).flatten()

cube_shape = physics.create_tri_mesh_shape(
    coordinates=coordinates,
    connectivity=connectivity,
)
```

**4. Actor loaded from a prefab** — Resolve both the prefab and the asset root against which its nested paths were authored:

```python
table_prefab_path = str(resolve_asset("table/table.mochi_scene"))
physics.prefab.add_to_scene(
    prefab_path=table_prefab_path,
    root_path=str(resolve_asset_root("table/table.mochi_scene")),
    scene=scene,
    params=physics.prefab.PrefabParams(
        name="tablePrefab",
        rotation=physics.Quaternion.rotation_x(-90 * math.pi / 180),
        translation=[0, -1.0, 0],
    ),
)
table_actor = find_actor(scene, "tablePrefab/Table")
```

The prefab name prefixes its actor names, so the table is addressed as `"tablePrefab/Table"`.

### Dynamic and Static Rigid Actors

Dynamic actors respond to forces and contact, while static actors remain fixed unless their root transforms are prescribed explicitly.

```python
sphere_actor = scene.create_rigid_actor(
    name="sphere",
    shape=sphere_shape,
    is_static=False,
    density=1000.0,
    world_from_local=physics.TransformRT(translation=[-0.5, 0.2, 0]),
    collider_type=physics.ColliderType.SPHERE,
)
```

The [`collider_type`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.RigidActorParams.collider_type) selects the geometry queried by other actors during [contact](../../concepts/contact.md#collider-representations).

### Simulation and Visualization

The example attaches the remote debugger, then advances the scene at 60 Hz until the debugger detaches:

```python
TIME_STEP = 1.0 / 60.0  # [s]
if physics.debugger.attach():
    while physics.debugger.is_attached():
        scene.step(TIME_STEP)
```

### Lifecycle Cleanup

The example explicitly destroys actors, the scene, and global engine state to illustrate each lifecycle operation. Destroying individual actors is unnecessary immediately before destroying their scene, and destroying the scene is unnecessary immediately before [`physics.shutdown()`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.shutdown).

```python
scene.destroy_actor(sphere_actor)
scene.destroy_actor(cube_actor)
scene.destroy_actor(ground_actor)
scene.destroy_actor(table_actor)
physics.destroy_scene(scene)
physics.shutdown()
```

## Running

```bash
uv run --no-project examples/example_rigid_bodies.py
```

This example launches or focuses the SuperDex Physics Debugger and runs while it remains connected. See [Inspecting Scenes](../../debugging_scenes.md) for connection, navigation, and playback controls.
