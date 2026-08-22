---
title: Contact Filtering
sidebar_position: 5
---

# Contact Filtering

This example demonstrates how to selectively enable or disable contact between groups of objects using **layers** and between specific actor pairs using **per-actor filtering**.

**Source**: `examples/example_contact_filtering.py`

## Key Concepts

### Default Behavior

By default, contact is **enabled** between all layers and between all actor pairs, except between adjacent links in articulated and soft-skinned actors. Contact only occurs if both the layer-level and actor-level checks allow it.

### Layer-Based Filtering

Actors are assigned to named layers at creation time:

```python
platform_1 = scene.create_rigid_actor(
    name="platform_1",
    layer="layer_1",
    shape=platform_shape,
    is_static=True,
    world_from_local=physics.TransformRT(translation=[0, 0.5, 0]),
)

platform_2 = scene.create_rigid_actor(
    name="platform_2",
    layer="layer_2",
    shape=platform_shape,
    is_static=True,
    world_from_local=physics.TransformRT(translation=[0, 0, 0]),
)
```

Contact between entire layers can then be disabled:

```python
scene.enable_layer_contact_symmetric("layer_1", "layer_2", enable=False)
```

This means objects in `layer_1` will pass through objects in `layer_2` (and vice versa). In this example, the stack of cubes in `layer_2` passes through the higher `platform_1` and lands on the lower `platform_2`.

**Symmetric vs asymmetric**: [`enable_layer_contact_symmetric`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Scene.enable_layer_contact_symmetric) disables contact in both directions. For one-directional filtering, use [`enable_layer_contact_asymmetric(layer_a, layer_b, enable=False)`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Scene.enable_layer_contact_asymmetric).

### Actor-Based Filtering

For finer control, contact can be toggled between specific actor pairs:

```python
scene.enable_actor_contact_symmetric(
    cubes["bottom_cube_2"].get_handle(),
    cubes["middle_cube_2"].get_handle(),
    enable=False,
    include_nested_actors=physics.IncludeNestedActors.NO,
)
```

This disables contact only between `bottom_cube_2` and `middle_cube_2`. Other cubes in the same layer still collide normally — `top_cube_2` still lands on `middle_cube_2`.

### Combining Both Levels

Contact requires **both** checks to pass:

| Layer contact | Actor contact | Result |
|---|---|---|
| Enabled | Enabled | Contact occurs |
| Enabled | Disabled | No contact |
| Disabled | Enabled | No contact |
| Disabled | Disabled | No contact |

This two-level system provides coarse-grained control (layers) with fine-grained overrides (actors).

## Running

```bash
uv run --no-project examples/example_contact_filtering.py
```

This example launches or focuses the SuperDex Physics Debugger and runs while it remains connected. See [Inspecting Scenes](../../debugging_scenes.md) for connection, navigation, and playback controls.
