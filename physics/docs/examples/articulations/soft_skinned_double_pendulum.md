---
title: Soft-Skinned
sidebar_position: 3
---

# Soft-Skinned

This example builds a **double pendulum carrying a soft body** entirely in code and
uses it as a guided tour of the SuperDex Physics
[soft-skinned articulated actor](../../concepts/actors/soft_skinned_actors.mdx) API:
attaching a tetrahedral soft mesh via constrained nodes and
[`SoftSkinnedActorParams.soft_attach_links`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.SoftSkinnedActorParams.soft_attach_links), introspecting nested actors, reading the
deformed soft volume, and using the nested soft actor as a contact surface.

**Source**: `examples/example_articulations_soft_skinned_double_pendulum.py`

The scene is a soft-skinned actor — a 2-joint / 2-link revolute chain anchored to the world with a soft volume attached to the second link:

```text
world -[Revolute]-> UpperArm (0.25 m) -[Revolute]-> LowerArm (0.125 m) -[soft attached]-> SoftArm (0.1 m)
```

The nested soft actor uses a tetrahedral mesh spanning `X=0.375→0.475` in the skeleton rest frame (total rigid 0.375 + soft 0.1 = 0.475 m). Its first cross-section nodes (4 nodes at X=0.375) are listed as [`constrainedNodes`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1ModelData.html) in the shape JSON and are bound to `LowerArm` via [`SoftSkinnedActorParams.softAttachLinks`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1SoftSkinnedActorParams.html). Contact between the nested soft actor and its attachment link is automatically disabled. A ball rests on the ground within reach and is struck by the swinging nested soft actor.

:::tip Coordinate frames
For each nested soft actor, the corresponding [`SoftActorParams`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1SoftActorParams.html) entry's [`world_from_local`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.SoftActorParams.world_from_local) must be identity, and its mesh must be authored in the root link's local frame at the skeleton's reference pose. Use `skeleton_params.world_from_root` to place the complete soft-skinned actor; it moves the links and soft mesh together. This example puts the 0.5 m pivot height in `joint_0.parent_link_from_joint`, so [`world_from_root`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.ArticulatedActorParams.world_from_root) remains identity.
:::

## Implementation

### Building the Chain with a Soft Attach

An articulated skeleton is described by parallel `joints[]` and `links[]` arrays (see [Double Pendulum on Rail](./double_pendulum_on_rail.md) for base). The nested soft actor is an optional [`SoftActorParams`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.SoftActorParams) with a tet-mesh shape that contains [`constrainedNodes`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1ModelData.html). Its rest coordinates are authored in the skeleton's rest frame so the rod overlays the second arm tip at rest.

```python
ROOT_HEIGHT = 0.5  # [m]
ARM_SCALE = [0.25, 0.025, 0.025]  # [m]
SECOND_ARM_SCALE = [0.125, 0.025, 0.025]  # [m]

arm_shape = physics.load_shape_from_file(
    file_path=str(resolve_asset("cube/cube_fine_mesh.mochi.json")),
    bake_scale=ARM_SCALE,
)
second_arm_shape = physics.load_shape_from_file(
    file_path=str(resolve_asset("cube/cube_fine_mesh.mochi.json")),
    bake_scale=SECOND_ARM_SCALE,
)

# Soft volume X=0.375..0.475, Y/Z 0..0.025 (center 0.0125 = w/2) with constrainedNodes=[0,1,2,3] at attachment end
soft_shape = physics.load_shape_from_file(
    file_path=str(resolve_asset("samples/articulations_parts/soft.mochi.json")),
    bake_scale=[1, 1, 1],
)

skeleton_params = physics.ArticulatedActorParams(name="SoftSkinnedDoublePendulum")
skeleton_params.joints = _make_joints()  # joint_0 at [0,ROOT_HEIGHT,0], joint_1 at [0.25,0.0125,0.0125]
skeleton_params.links = _make_links(arm_shape, second_arm_shape)

soft_params = physics.SoftActorParams(
    name="SoftArm",
    shape=soft_shape,
    layer="Soft",
    has_gravity=False,   # must be False, use skeleton's gravity
    has_inertia=False,
    has_stress=True,     # unposed elasticity, accurate for rigid attachment
)
soft_params.material = physics.SoftMaterialParams()
soft_params.material.type = physics.SoftMaterialType.NEO_HOOKEAN
soft_params.material.neo_hookean.youngs_modulus = 1.5e4  # softer
soft_params.material.density = 500.0

ss_params = physics.SoftSkinnedActorParams(
    skeleton_params=skeleton_params,
    soft_params=[soft_params],
    soft_attach_links=["LowerArm"],
    has_gravity=True,
    has_inertia=True,
    has_stress=False,          # posed gravity+inertia, unposed elasticity
    enable_colliding_links=True, # links are not wrapped by the soft, so they collide
)
soft_skinned_actor = scene.create_soft_skinned_actor(ss_params)
soft_skinned_actor.set_articulated_joint_velocities(velocities=[4.2, 0.0])
```

The soft shape JSON carries the attachment information:

```json
{
  "mesh": {
    "nodesPerElement": 4,
    "coordinates": [0.375,0,0, 0.375,0.025,0, ... 0.475,0,0.025],
    "connectivity": [0,1,3,4, ...]
  },
  "constrainedNodes": [0,1,2,3]
}
```

[`constrainedNodes`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1ModelData.html) are the 4 corner nodes at `X=0.375` bound to `LowerArm` via [`soft_attach_links`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.SoftSkinnedActorParams.soft_attach_links).

### Introspecting the Soft-Skinned Actor

The top-level articulated actor exposes nested link actors and nested soft actors:

```python
print(f"num_dofs = {actor.get_num_dofs()}")  # 2
link_handles = actor.get_nested_link_actors()
soft_handles = actor.get_nested_soft_actors()
print([scene.get_actor(h).get_name() for h in link_handles])  # ['UpperArm','LowerArm']
print([scene.get_actor(h).get_name() for h in soft_handles])   # ['SoftArm']

info = actor.get_articulated_shape_info()
for i in range(len(info.link_names)):
    print(info.link_names[i], info.parents[i], info.joint_types[i])

print(f"softAttachLinks = ['LowerArm']")
```

Query support is split: [`is_query_supported`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Actor.is_query_supported) reports that the top-level articulated actor does not support contact queries directly, while the nested soft actor does:

```python
for q in [
    physics.QueryType.SURFACE_NODE_POSITIONS,
    physics.QueryType.CONTACT_POINTS,
    physics.QueryType.TOTAL_CONTACT_FORCE,
    physics.QueryType.NODE_POSITIONS,
]:
    print(f"top-level articulated actor is_query_supported({q}) = {actor.is_query_supported(q)}")
    soft_actor = scene.get_actor(soft_handles[0])
    print(f"nested soft is_query_supported({q}) = {soft_actor.is_query_supported(q)}")
# CONTACT_* and NODE_POSITIONS -> True on nested soft
```

### Reading the Soft Volume

The nested soft actor's world AABB is available immediately, before any step. The articulated actor itself has **no bounds** because it carries no skin geometry — so query the nested soft actor instead:

```python
soft_actor = scene.get_actor(actor.get_nested_soft_actors()[0])

# The nested soft actor's world AABB is available before stepping.
aabb = soft_actor.get_aabb_world()
print(aabb.min, aabb.max)
```

Node positions, by contrast, are query data computed during [`scene.step`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Scene.step), so register on the nested soft actor, step, then read back:

```python
pos_query = soft_actor.register_query(physics.QueryType.NODE_POSITIONS)
scene.step(TIME_STEP)

# Deformed node positions computed during the step (3 values per node).
node_positions = soft_actor.get_node_positions_local()

soft_actor.cancel_query(pos_query)
```

### Controlling Contact with a Nested Soft Actor in the Colliding Role

String layers filter contact. Per-link `Pendulum` colliders are disabled vs `Ball` and `Environment`, leaving only `Soft` vs `Ball` active. Contact between `LowerArm` and `SoftArm` is auto-disabled via [`softAttachLinks`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1SoftSkinnedActorParams.html).

```python
scene.enable_layer_contact_symmetric("Pendulum", "Pendulum", enable=False)
scene.enable_layer_contact_symmetric("Pendulum", "Ball", enable=False)
scene.enable_layer_contact_symmetric("Pendulum", "Environment", enable=False)
scene.enable_layer_contact_symmetric("Soft", "Environment", enable=False)

print(scene.is_layer_contact_enabled("Soft", "Ball"))      # True
print(scene.is_layer_contact_enabled("Pendulum", "Ball"))  # False
```

During interactive run the example registers contact queries on the nested soft actor to report force from the ball:

```python
soft_actor = scene.get_actor(actor.get_nested_soft_actors()[0])
contact_points = soft_actor.register_query(physics.QueryType.CONTACT_POINTS)
contact_force = soft_actor.register_query(physics.QueryType.TOTAL_CONTACT_FORCE)
# ... after scene.step():
force = soft_actor.get_contact_force_from_actor_world(ball)
```

## Features

- **Soft-skinned actor**: [`SoftSkinnedActorParams`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1SoftSkinnedActorParams.html) with [`skeleton_params`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.SoftSkinnedActorParams.skeleton_params) (2 revolute joints, 2 links 0.25 m + 0.125 m) and [`soft_params`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.SoftSkinnedActorParams.soft_params) tet mesh with [`constrainedNodes`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1ModelData.html), attached via `soft_attach_links ["LowerArm"]`.
- **Constrained nodes**: shape JSON carries `constrainedNodes: [0,1,2,3]` at attachment end, required for soft-skinned actors.
- **Contact roles**: the nested soft actor is a colliding actor, not a collider. The links are colliders, and `enableCollidingLinks=True` also enables them as colliding actors. Layer filtering disables Pendulum vs Ball/Environment, leaving Soft vs Ball. LowerArm↔Soft is disabled automatically.
- **Introspection**: [`get_nested_link_actors`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Actor.get_nested_link_actors), [`get_nested_soft_actors`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Actor.get_nested_soft_actors), [`get_articulated_shape_info`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Actor.get_articulated_shape_info), [`is_query_supported`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Actor.is_query_supported) split top-level articulated actor vs. nested actors.
- **Nested soft actor read-back**: [`get_aabb_world`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Actor.get_aabb_world) on the nested soft actor, `NODE_POSITIONS`, `CONTACT_POINTS`, `TOTAL_CONTACT_FORCE` queries on the nested soft actor.
- **Contact filtering**: layer enable/disable, enumeration, [`get_contact_force_from_actor_world`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Actor.get_contact_force_from_actor_world).
- **Live output**: topology, query support, world AABB, once-per-second joint angles, minimum Y of the nested soft actor, ball contacts and force.

## Running

```bash
uv run --no-project examples/example_articulations_soft_skinned_double_pendulum.py
```

This example launches or focuses the SuperDex Physics Debugger and runs while it remains connected. See [Inspecting Scenes](../../debugging_scenes.md) for connection, navigation, and playback controls.

## Also Available as a Prefab

Same scene ships as declarative [prefab](../../concepts/prefabs.mdx). It builds identical chain with shared soft asset.

**Source**: `assets/samples/articulations_soft_skinned_double_pendulum.mochi_scene`

```python
from superdex.physics.utils.scene_helpers import create_scene_from_prefab
scene = create_scene_from_prefab("samples/articulations_soft_skinned_double_pendulum.mochi_scene")
```

The nested soft actor configuration maps directly to prefab keys — note `actors.softSkinned[].softAttachLinks`, `softParams[].shape` pointing to shared tet asset with [`constrainedNodes`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1ModelData.html), and contact filter:

```json
{
  "actors": {
    "softSkinned": [
      {
        "skeletonParams": {
          "name": "SoftSkinnedDoublePendulum",
          "jointVelocities": [4.2, 0.0],
          "joints": [
            { "name": "joint_0", "type": "Revolute", "axis": [0,0,-1], "parentLinkFromJoint": {"translation": [0,0.5,0]} },
            { "name": "joint_1", "type": "Revolute", "axis": [0,0,-1], "parentLinkFromJoint": {"translation": [0.25,0.0125,0.0125]} }
          ],
          "links": [
            { "name": "UpperArm", "parentLink": -1, "parentJointFromLink": {"translation": [0,-0.0125,-0.0125]}, "shape": "cube/cube_fine_mesh.mochi.json", "shapeScale": [0.25,0.025,0.025], "colliderType": "Box", "layer": "Pendulum", "density": 1000 },
            { "name": "LowerArm", "parentLink": 0, "parentJointFromLink": {"translation": [0,-0.0125,-0.0125]}, "shape": "cube/cube_fine_mesh.mochi.json", "shapeScale": [0.125,0.025,0.025], "colliderType": "Box", "layer": "Pendulum", "density": 1000 }
          ]
        },
        "softParams": [
          {
            "name": "SoftArm",
            "shape": "samples/articulations_parts/soft.mochi.json",
            "layer": "Soft",
            "hasGravity": false,
            "hasInertia": false,
            "hasStress": true,
            "material": { "type": "NeoHookean", "density": 500.0, "neoHookean": {"youngsModulus": 15000.0} }
          }
        ],
        "softAttachLinks": ["LowerArm"],
        "hasGravity": true,
        "hasInertia": true,
        "hasStress": false,
        "enableCollidingLinks": true
      }
    ],
    "rigid": [{ "name": "Ball", "shape": "sphere/icosphere_4subdiv.1.mochi.json", "colliderType": "Sphere", "layer": "Ball", "scale": [0.05,0.05,0.05], "translation": [0.05,0.05,0], "density": 500 }]
  },
  "contactFilter": {
    "layerContactSymmetric": [
      { "layers": ["Pendulum","Pendulum"], "enable": false },
      { "layers": ["Pendulum","Ball"], "enable": false },
      { "layers": ["Pendulum","Environment"], "enable": false },
      { "layers": ["Soft","Environment"], "enable": false }
    ]
  }
}
```

The soft asset `soft.mochi.json` contains `mesh.coordinates`, `mesh.connectivity` (16 nodes, 15 tets from 5-tet split per cube segment) and `constrainedNodes [0,1,2,3]` at `X=0.375` end, authored at `X 0.375→0.475, Y/Z 0→0.025` so it overlays the second arm tip exactly with width 0.025.
