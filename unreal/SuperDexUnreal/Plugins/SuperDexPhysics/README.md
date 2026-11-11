# SuperDex Physics for Unreal

[SuperDex Physics](https://facebookresearch.github.io/project_superdex/physics/) is a
contact-first physics engine purpose-built for tactile manipulation. This plugin runs it in
Unreal Engine 5.5.

- `SuperDexPhysics` (runtime) simulates the SuperDex Physics scene of the game world: rigid,
  soft, shell, and rod meshes, implicit shapes, articulations, and constraints as components,
  with their models, prefabs, and materials as assets.
- `SuperDexPhysicsEditor` (editor) imports SuperDex Physics prefabs and models, and edits
  models and articulations.

The plugin carries, in `ThirdParty/`, the SuperDex Physics native library (`mochi_physics`) and
the third-party Marl library (`marl`) that it links, with their headers in
`Source/SuperDexPhysics/ThirdParty/`. In a source checkout, `hydrate_plugins.py` in the SuperDex
Unreal project builds and stages them; see the
[SuperDex Unreal README](https://github.com/facebookresearch/project_superdex/tree/main/unreal/SuperDexUnreal#readme).

## License

First-party code in this plugin is Apache-2.0 licensed; see [LICENSE](LICENSE). Its first-party
content, the assets in `Content/` and the icons in `Resources/`, is licensed under CC-BY-4.0; see
[Resources/LICENSE](Resources/LICENSE). Third-party code and dependencies retain their own terms;
the ones this plugin carries are listed in
[thirdparty_licenses/NOTICE.md](thirdparty_licenses/NOTICE.md), beside their license texts.
