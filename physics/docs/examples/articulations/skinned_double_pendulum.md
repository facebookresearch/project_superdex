---
title: Skinned
sidebar_position: 2
---

# Skinned

This example builds a **double pendulum wearing a skinned surface** entirely in code
and uses it as a guided tour of the SuperDex Physics
[skinned articulated actor][skinned-actor]
API: attaching a linear-blend-skinned triangle mesh to an articulation, introspecting
query support, reading the deformed surface, showing how the surface bends with the
joints, and using the skin as a **colliding-only** contact surface.

[skinned-actor]: ../../concepts/actors/articulated_actors.mdx#skinned-surface

**Source**: `examples/example_articulations_skinned_double_pendulum.py`

The scene is a single skinned articulated actor — a 2-joint / 2-link revolute chain anchored to the world:

```text
world -[Revolute]-> UpperArm -[Revolute]-> LowerArm
     \--- skin: one LBS tube bound to both links ---/
```

A thin rectangular tube along +X spans both arms in the root-local rest frame. Vertices near the upper arm are bound to link 0, vertices past the elbow to link 1, and a smooth weight ramp blends them across the joint so one continuous surface bends as the pendulum swings. A ball rests on the ground within the tip's reach and is struck by the swinging skin.

The skin is **colliding-only**: it probes other actors for contact, but it does not act as a collider. The links act as colliders.

:::tip Coordinate frames
[`world_from_root`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.ArticulatedActorParams.world_from_root) places the complete articulated actor, moving the links and skin together without changing their alignment. This example instead puts the 0.5 m pivot height in `joint_0.parent_link_from_joint` and leaves [`world_from_root`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.ArticulatedActorParams.world_from_root) at identity to match the reference `doublependulum` prefab.
:::

## Implementation

### Building the Articulated Chain with a Skin

An articulated actor is described by parallel `joints[]` and `links[]` arrays (see [Double Pendulum on Rail](./double_pendulum_on_rail.md) for the base pattern). The skin is an optional [`ArticulatedSkinParams`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.ArticulatedSkinParams) holding a triangle-mesh shape that **contains skinning weights and link indices**. Its rest coordinates are authored in the skeleton's rest frame so the tube overlays the arms at rest.

```python
ROOT_HEIGHT = 0.5  # [m]
ARM_SCALE = [0.25, 0.025, 0.025]  # [m]

arm_shape = physics.load_shape_from_file(
    file_path=str(resolve_asset("cube/cube_fine_mesh.mochi.json")),
    bake_scale=ARM_SCALE,  # corner-anchored cube [0,1]^3 -> [0, L] x [0, w] x [0, w]
)

# Shared skin asset: authored as tube along +X with per-node LBS weights to links 0/1.
skin_shape = physics.load_shape_from_file(
    file_path=str(resolve_asset("samples/articulations_parts/skin.mochi.json")),
    bake_scale=[1, 1, 1],  # bake only moves positions, never weights
)

params = physics.ArticulatedActorParams(name="SkinnedDoublePendulum")
params.joints = [
    physics.ArticulatedJointParams(
        name="joint_0", type=physics.ArticulatedJointType.REVOLUTE, axis=[0, 0, -1],
        parent_link_from_joint=physics.TransformRT(translation=[0, ROOT_HEIGHT, 0]),
    ),
    physics.ArticulatedJointParams(
        name="joint_1", type=physics.ArticulatedJointType.REVOLUTE, axis=[0, 0, -1],
        parent_link_from_joint=physics.TransformRT(translation=[0.25, 0.0125, 0.0125]),
    ),
]
params.links = [
    physics.ArticulatedLinkParams(
        name="UpperArm", parent_link=-1,
        parent_joint_from_link=physics.TransformRT(translation=[0, -0.0125, -0.0125]),
        shape=arm_shape, collider_type=physics.ColliderType.BOX, layer="Pendulum", density=1000.0,
    ),
    physics.ArticulatedLinkParams(
        name="LowerArm", parent_link=0,
        parent_joint_from_link=physics.TransformRT(translation=[0, -0.0125, -0.0125]),
        shape=arm_shape, collider_type=physics.ColliderType.BOX, layer="Pendulum", density=1000.0,
    ),
]

# Attach the skinned surface. The skin is a colliding contact surface. Two optional tuning
# knobs (left at defaults here) bound its per-step contact cost: pairing
# boundary_element_type=P1Q1 with boundary_subsampling reduces contact samples.
params.skin = physics.ArticulatedSkinParams(shape=skin_shape, layer="Skin")
articulation = scene.create_articulated_actor(params)

# Seed a chaotic swing so the skin sweeps into the ball.
articulation.set_articulated_joint_velocities(velocities=[4.2, 0.0])
```

### Introspecting the Articulation and Probing Query Support

The topology API is identical to plain articulations ([`get_num_dofs`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Actor.get_num_dofs), [`get_nested_link_actors`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Actor.get_nested_link_actors), [`get_articulated_shape_info`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Actor.get_articulated_shape_info)), but the skin surface itself lives on the top-level articulated actor and only supports surface / contact queries.

```python
print(f"num_dofs = {articulation.get_num_dofs()}")  # 2

link_handles = articulation.get_nested_link_actors()
info = articulation.get_articulated_shape_info()
for i in range(len(info.link_names)):
    print(info.link_names[i], info.parents[i], info.joint_types[i])

# The skin is a *colliding surface*: surface-node and contact queries apply,
# while volumetric / soft queries do not.
for q in [
    physics.QueryType.SURFACE_NODE_POSITIONS,
    physics.QueryType.SURFACE_NODE_NORMALS,
    physics.QueryType.CONTACT_POINTS,
    physics.QueryType.TOTAL_CONTACT_FORCE,
    physics.QueryType.NODE_POSITIONS,
    physics.QueryType.VISUAL_NODE_POSITIONS,
    physics.QueryType.ELEMENTS_DEFORMATION_GRADIENT,
]:
    print(f"is_query_supported({q}) = {articulation.is_query_supported(q)}")
# SURFACE_NODE_* and CONTACT_* -> True, others -> False
```

### Reading the Skin Surface

Query data is computed during [`scene.step`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Scene.step), so you register the query, step, then read back. The reference topology is always available via [`get_surface_mesh`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Actor.get_surface_mesh).

```python
mesh = articulation.get_surface_mesh()
print(mesh.get_num_nodes(), mesh.get_num_elements())  # 52 nodes, ~104 triangles

pos_query = articulation.register_query(physics.QueryType.SURFACE_NODE_POSITIONS)
nrm_query = articulation.register_query(physics.QueryType.SURFACE_NODE_NORMALS)
scene.step(TIME_STEP)

positions = np.array(articulation.get_surface_mesh_node_positions_local())
normals = np.array(articulation.get_surface_mesh_node_normals_local())
aabb = articulation.get_aabb_world()

# Spatial query on the deformed surface:
volume = physics.Aabb(min=[0.2, ROOT_HEIGHT-0.05, -0.05], max=[0.55, ROOT_HEIGHT+0.05, 0.05])
hits: list[int] = []
articulation.query_nodes_in_volume_local(volume, True, lambda node, _pos: hits.append(node))

articulation.cancel_query(pos_query)
articulation.cancel_query(nrm_query)
```

### Controlling Contact with a Skin in the Colliding Role

The scene uses string **layers** for coarse filtering. The per-link `Pendulum` colliders are disabled against everything, leaving only the `Skin` (colliding) vs `Ball` (collider) pair active. The ball still collides with the ground (`Environment`).

```python
scene.enable_layer_contact_symmetric("Pendulum", "Pendulum", enable=False)
scene.enable_layer_contact_symmetric("Pendulum", "Skin", enable=False)
scene.enable_layer_contact_symmetric("Pendulum", "Ball", enable=False)
scene.enable_layer_contact_symmetric("Pendulum", "Environment", enable=False)
scene.enable_layer_contact_symmetric("Skin", "Environment", enable=False)

print(scene.is_layer_contact_enabled("Skin", "Ball"))      # True
print(scene.is_layer_contact_enabled("Pendulum", "Ball"))  # False

scene.get_num_contact_layers()
scene.enumerate_contact_layer_names(lambda name: ...)
```

During the interactive run the example registers contact queries to report force from the ball:

```python
contact_points = articulation.register_query(physics.QueryType.CONTACT_POINTS)
contact_force = articulation.register_query(physics.QueryType.TOTAL_CONTACT_FORCE)
# ... after scene.step():
force = articulation.get_contact_force_from_actor_world(ball)
```

### Tuning Contact Cost (Optional)

[`ArticulatedSkinParams`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.ArticulatedSkinParams) exposes two knobs that bound per-step contact cost, left at defaults in this example but documented in the [API](../../concepts/actors/articulated_actors.mdx#articulatedskinparams-reference):

- [`boundary_element_type`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.ArticulatedSkinParams.boundary_element_type): e.g. `P1Q1`
- [`boundary_subsampling`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.ArticulatedSkinParams.boundary_subsampling): reduces integration samples on the surface — best paired with `P1Q1`.

### Validation — A Non-Skinned Shape Is Rejected

The skin **must** carry per-node skinning weights and link indices. In `mochi_physics/src/mochi_scene.cpp` `GetSkinShape` now checks:

```cpp
MOCHI_ERROR_IF_NOT(shape->GetMeshSkinning() != nullptr, ..., "Skin shape must contain skinning data...");
```

The example demonstrates graceful failure:

```python
plain_tri = physics.create_tri_mesh_shape(coordinates=[0,0,0, 1,0,0, 0,1,0], connectivity=[0,1,2])
params.skin = physics.ArticulatedSkinParams(shape=plain_tri, layer="Skin")
try:
    scene.create_articulated_actor(params)
except physics.Error:
    print("non-skinned skin: rejected (as expected)")
```

## Features

- **Skinned articulated actor**: [`ArticulatedSkinParams`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.ArticulatedSkinParams) with a triangle-mesh shape carrying LBS weights and link indices, sharing the same `skin.mochi.json` asset as the prefab.
- **Colliding-only surface**: skin detects contact against other actors (ball), while per-link boxes remain colliders.
- **Introspection**: [`get_articulated_shape_info`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Actor.get_articulated_shape_info), [`get_num_dofs`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Actor.get_num_dofs), [`get_nested_link_actors`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Actor.get_nested_link_actors), and [`is_query_supported`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Actor.is_query_supported) probe showing surface queries supported, volumetric queries not.
- **Surface read-back**: [`get_surface_mesh`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Actor.get_surface_mesh), [`get_surface_mesh_node_positions_local`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Actor.get_surface_mesh_node_positions_local) / [`get_surface_mesh_node_normals_local`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Actor.get_surface_mesh_node_normals_local), [`get_aabb_world`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Actor.get_aabb_world), [`query_nodes_in_volume_local`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Actor.query_nodes_in_volume_local).
- **Contact filtering**: layer-level enable/disable, enumeration of layers, [`get_contact_force_from_actor_world`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Actor.get_contact_force_from_actor_world) for ball impact.
- **Validation**: C++ guard `GetMeshSkinning() != nullptr` and taught error path for plain meshes.
- **Live console output**: topology, supported queries, surface extents, and once-per-second skin AABB, joint angles, contact counts and force.
- **Scripted timeline**: chaotic swing from seeded velocities, then joint friction damping at `T_DAMP=12s`.

## Running

```bash
uv run --no-project examples/example_articulations_skinned_double_pendulum.py
```

This example launches or focuses the SuperDex Physics Debugger and runs while it remains connected. See [Inspecting Scenes](../../debugging_scenes.md) for connection, navigation, and playback controls.

## Also Available as a Prefab

The same scene ships as a declarative [prefab](../../concepts/prefabs.mdx) — the **static** version of this example. It builds the identical 2-link chain with the shared skin asset, but without the code-only extras (console API tour, contact tests, and the damping timeline). It's the source of truth for the scene's geometry.

**Source**: `assets/samples/articulations_skinned_double_pendulum.mochi_scene`

Load it into a fresh scene (or use [`physics.prefab.add_to_scene(...)`](pathname:///generated/api/v1.0/python/api/prefab.html#superdex.physics.prefab.add_to_scene) / C++ `prefab::AddToScene(...)` to add it into an existing one):

```python
from superdex.physics.utils.scene_helpers import create_scene_from_prefab

scene = create_scene_from_prefab("samples/articulations_skinned_double_pendulum.mochi_scene")
```

The skin maps directly onto prefab keys — note `actors.articulated[].skin.shape` pointing to the shared asset and the contact filter disabling the per-link colliders:

```json
{
  "actors": {
    "articulated": [
      {
        "name": "SkinnedDoublePendulum",
        "jointVelocities": [4.2, 0.0],
        "joints": [
          { "name": "joint_0", "type": "Revolute", "axis": [0, 0, -1], "parentLinkFromJoint": { "translation": [0, 0.5, 0] } },
          { "name": "joint_1", "type": "Revolute", "axis": [0, 0, -1], "parentLinkFromJoint": { "translation": [0.25, 0.0125, 0.0125] } }
        ],
        "links": [
          { "name": "UpperArm", "parentLink": -1, "parentJointFromLink": { "translation": [0, -0.0125, -0.0125] }, "shape": "cube/cube_fine_mesh.mochi.json", "shapeScale": [0.25, 0.025, 0.025], "colliderType": "Box", "layer": "Pendulum", "density": 1000 },
          { "name": "LowerArm", "parentLink": 0, "parentJointFromLink": { "translation": [0, -0.0125, -0.0125] }, "shape": "cube/cube_fine_mesh.mochi.json", "shapeScale": [0.25, 0.025, 0.025], "colliderType": "Box", "layer": "Pendulum", "density": 1000 }
        ],
        "skin": { "shape": "samples/articulations_parts/skin.mochi.json", "layer": "Skin" }
      }
    ],
    "rigid": [ { "name": "Ball", "shape": "sphere/icosphere_4subdiv.1.mochi.json", "colliderType": "Sphere", "layer": "Ball", "scale": [0.05, 0.05, 0.05], "translation": [0.05, 0.05, 0], "density": 500 } ]
  },
  "contactFilter": {
    "layerContactSymmetric": [
      { "layers": ["Pendulum", "Pendulum"], "enable": false },
      { "layers": ["Pendulum", "Skin"], "enable": false },
      { "layers": ["Pendulum", "Ball"], "enable": false },
      { "layers": ["Pendulum", "Environment"], "enable": false },
      { "layers": ["Skin", "Environment"], "enable": false }
    ]
  }
}
```

The skin asset itself (`skin.mochi.json`) contains `mesh.coordinates`, `mesh.connectivity`, and `mesh.skinning { weightsPerNode, indices, weights }` — a 52-node tube with per-node LBS weights ramped `1.0/0.0 -> 0.5/0.5 -> 0.0/1.0` across the elbow.
