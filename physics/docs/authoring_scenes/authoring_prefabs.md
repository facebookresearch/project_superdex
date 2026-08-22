---
sidebar_position: 3
title: "Authoring Prefabs"
---

# Authoring Prefabs

SuperDex Physics prefabs are JSON documents, commonly named `.mochi_scene` or `.mochi_prefab`, that describe complete scenes or reusable scene fragments. This guide focuses on the authoring workflow. See [Prefabs](../concepts/prefabs.mdx) for composition and contact-filter behavior.

For authoritative fields, defaults, units, and supported constraint types, see the generated [C++ Prefabs API reference](pathname:///generated/api/v1.0/cpp/group__prefabs.html).

## Start with the Top-Level Structure

Every section is optional:

```json
{
  "actors": {
    "rigid": [],
    "soft": [],
    "articulated": [],
    "softSkinned": []
  },
  "constraints": {},
  "controllers": [],
  "prefabs": [],
  "scene": {},
  "contactFilter": {}
}
```

The four actor arrays contain file-backed variants of the corresponding C++ actor parameters:

- `rigid` contains `RigidActorPrefab` entries ([C++](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1prefab_1_1RigidActorPrefab.html), [Python](pathname:///generated/api/v1.0/python/api/prefab.html#superdex.physics.prefab.RigidActorPrefab)).
- `soft` contains `SoftActorPrefab` entries ([C++](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1prefab_1_1SoftActorPrefab.html), [Python](pathname:///generated/api/v1.0/python/api/prefab.html#superdex.physics.prefab.SoftActorPrefab)).
- `articulated` contains `ArticulatedActorPrefab` entries ([C++](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1prefab_1_1ArticulatedActorPrefab.html), [Python](pathname:///generated/api/v1.0/python/api/prefab.html#superdex.physics.prefab.ArticulatedActorPrefab)).
- `softSkinned` contains `SoftSkinnedActorPrefab` entries ([C++](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1prefab_1_1SoftSkinnedActorPrefab.html), [Python](pathname:///generated/api/v1.0/python/api/prefab.html#superdex.physics.prefab.SoftSkinnedActorPrefab)).

## Name Actors for References

Constraints, pose controllers, and actor contact filters resolve actors by name. A local name such as `Link1` refers to an actor in the current prefab. A hierarchy path such as `Pendulum/DoublePendulumOnRail` refers to an actor inside a nested instance.

Any name or hierarchy path used by a constraint, controller, or contact filter must identify exactly one actor in that prefab scope. To reference an articulated link or nested soft actor, give its parent actor or an enclosing prefab instance a non-empty name. Instance names may otherwise be empty or repeated.

Test leaf prefabs independently before composing them. This catches ambiguous paths and missing assets close to their source.

## Compose with Nested Prefabs

A prefab reference creates an instance of another prefab with a hierarchy prefix and a transform:

```json
{
  "prefabs": [
    {
      "name": "Pendulum",
      "path": "samples/articulations_double_pendulum_on_rail.mochi_scene",
      "scale": 1.0,
      "rotation": [0, 0, 0, 1],
      "translation": [0, 0, 0]
    }
  ]
}
```

[`scale`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1prefab_1_1PrefabReference.html) is uniform. Rotation uses quaternion order `[x, y, z, w]`. These values transform the complete nested instance and compose through deeper nesting. A reference does not provide arbitrary overrides for child actor properties; edit the child prefab, create another variant, or configure the instantiated actors at runtime.

Only the top-level prefab applies [`scene`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1prefab_1_1ScenePrefab.html) settings. Nested [`scene`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1prefab_1_1ScenePrefab.html) objects are ignored.

## Choose Paths Deliberately

Paths inside a prefab resolve as follows:

1. Absolute paths are used unchanged.
2. `./...` paths are relative to the file that contains the reference.
3. Other relative paths are relative to the asset root supplied when loading.

Each file-backed nested prefab gets its own base for `./` paths while sharing the caller-provided asset root for ordinary relative paths. For a top-level prefab loaded from a JSON string, `./` falls back to the supplied root because there is no containing file.

The root does not locate the top-level prefab. Pass the top-level file path separately, then pass the asset root against which its ordinary relative references were authored:

```python
import superdex.physics as physics
from superdex.physics.paths import resolve_asset, resolve_asset_root

relative_path = "samples/articulations_pose_controller.mochi_scene"
physics.prefab.add_to_scene(
    prefab_path=str(resolve_asset(relative_path)),
    root_path=str(resolve_asset_root(relative_path)),
    scene=scene,
)
```

[`resolve_asset()`](pathname:///generated/api/v1.0/python/api/scene_setup.html#superdex.physics.paths.resolve_asset) and [`resolve_asset_root()`](pathname:///generated/api/v1.0/python/api/scene_setup.html#superdex.physics.paths.resolve_asset_root) may select among multiple packaged roots. Do not assume the correct root is the prefab's parent directory. Downstream applications should ship the complete transitive prefab and asset tree, or set `SUPERDEX_ASSETS_PATH`, and preserve directory relationships used by `./` references.

Use ordinary root-relative paths when several prefabs share an asset tree. Use `./` for assets intentionally packaged beside a prefab, including exported generated assets. Avoid absolute paths in distributable content.

## Add Constraints and Controllers

The [`constraints`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1prefab_1_1ScenePrefab.html) object groups supported constraints by type. Actor fields use local names or hierarchy paths and must resolve unambiguously. Pose controllers are entries in [`controllers`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1prefab_1_1ScenePrefab.html); their [`articulatedActor`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1prefab_1_1PoseControllerPrefab.html) field follows the same rule.

A controller in a parent prefab can target a nested articulation by setting [`articulatedActor`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1prefab_1_1PoseControllerPrefab.html) to a hierarchy path such as `Pendulum/DoublePendulumOnRail`. See [`PoseControllerPrefab`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1prefab_1_1PoseControllerPrefab.html) for controller fields and the [Pose Controller example](../examples/articulations/pose_controller.md) for a complete form.

## Configure Contact Filters

[`contactFilter`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1prefab_1_1ScenePrefab.html) has four optional arrays: [`layerContactSymmetric`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1prefab_1_1ContactFilter.html), [`layerContactAsymmetric`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1prefab_1_1ContactFilter.html), [`actorContactSymmetric`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1prefab_1_1ContactFilter.html), and [`actorContactAsymmetric`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1prefab_1_1ContactFilter.html). Layer entries identify a [`layers`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1prefab_1_1LayerContactEntry.html) pair; actor entries identify an [`actors`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1prefab_1_1ActorContactEntry.html) pair and may set [`includeNestedActors`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1prefab_1_1ActorContactEntry.html). Each entry sets `enable`.

Symmetric entries update both directions; asymmetric entries update one ordered direction. Actor and layer tables are independent, and either can prevent contact. See [Contact Filters](../concepts/prefabs.mdx#contact-filters) for expansion, ordering, overrides, and adjacent-link behavior.

## Load and Test

Use the public API shown in [Loading](../concepts/prefabs.mdx#loading). [`PrefabParams`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1prefab_1_1PrefabParams.html) ([C++](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1prefab_1_1PrefabParams.html), [Python](pathname:///generated/api/v1.0/python/api/prefab.html#superdex.physics.prefab.PrefabParams)) can add an instance name and transform or disable top-level scene settings.

C++ `AddToScene` and Python [`physics.prefab.add_to_scene()`](pathname:///generated/api/v1.0/python/api/prefab.html#superdex.physics.prefab.add_to_scene) are not transactional: an error can leave earlier scene-setting changes and created objects in place. Validate prefabs before adding them to long-lived scenes, or discard and recreate the destination scene after an error.

## Export Scenes

Export writes a prefab and generated mesh assets:

```cpp
superdex::Error error;
superdex::prefab::ExportScene(scene, "MyScene", outputDir, error);
```

```python
import superdex.physics as physics

physics.prefab.export_scene(
    scene=scene,
    export_name="MyScene",
    output_dir=output_dir,
)
```

The output layout is:

```text
<outputDir>/MyScene/
+-- MyScene.mochi_scene
+-- generated_assets/
    +-- *.mochi.h5
```

Generated mesh references use `./generated_assets/...`, so the export directory can be moved as a unit. Export may leave a partial directory on failure, so use a fresh output name and use the directory only after export succeeds.

### Export Limitations

Export reconstructs selected scene and creation data; it is not a lossless snapshot or authoring round trip.

C++ `ExportScene` and Python [`physics.prefab.export_scene()`](pathname:///generated/api/v1.0/python/api/prefab.html#superdex.physics.prefab.export_scene) support rigid, soft, articulated, and soft-skinned actors, write the current gravity and solver settings, and record the effective disabled actor/layer contact state. They write current actor root transforms, but articulated links use their rest configuration and deformable actors use reference geometry rather than current deformation.

They do not export:

- Generic scene constraints or articulated pose controllers.
- Shell actors, rod actors or rod-specific relationships, or experimental transmissions.
- Implicit non-mesh shapes.
- Current articulated joint pose or joint velocities.
- Runtime state, including rigid velocities, soft deformation and nodal velocities, controller targets, runtime per-element material changes such as animated shape-target tensors, callbacks, queries, and recordings.
- Original authoring structure and metadata, including the nested-prefab hierarchy, paths, comments, render metadata, authoring transforms, and specialized shape-construction settings such as SDF authoring parameters.

Generated HDF5 files contain serializable runtime model data, but do not make specialized shape configurations losslessly round-trippable. Duplicate actor names may be made unique, and export may canonicalize the contact-filter JSON rather than reproduce the original entries.

Scene export records effective disabled contact pairs, not explicit entries that enable contact. If an explicit entry enabled contact between adjacent articulation links, reloading the export applies automatic adjacent-link filtering and disables that pair again. Add the equivalent `enable: true` entry to the exported prefab or re-enable the pair after loading.

C++ `ExportActor` and Python [`physics.prefab.export_actor()`](pathname:///generated/api/v1.0/python/api/prefab.html#superdex.physics.prefab.export_actor) are narrower than their scene counterparts: they support standalone rigid, soft, and articulated actors, but not soft-skinned parents or nested actors. They do not export scene contact filters.

## Examples

- [Rigid Bodies](../examples/basic/rigid_bodies.md) shows direct prefab loading, an instance name, and an instance rotation.
- [Pose Controller](../examples/articulations/pose_controller.md) shows nested composition and a controller targeting `Pendulum/DoublePendulumOnRail`.
- [Constraints](../examples/contact_constraints/constraints.md) shows rigid constraints whose actor fields resolve by name.
- [Soft Skinned](../examples/articulations/soft_skinned_double_pendulum.md) shows the [`softSkinned`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1prefab_1_1ActorLists.html) actor structure and attached soft geometry.

## Best Practices

1. Build small leaf prefabs and compose them into scenes.
2. Give every referenced actor and nested instance a clear, unambiguous name.
3. Choose root-relative or `./` paths according to how the asset tree will be packaged.
4. Test leaf prefabs before testing their parents.
5. Package the complete transitive dependency tree, including nested prefabs and referenced assets.
6. Treat exported prefabs as editable reconstructions and review the limitations before relying on a round trip.
