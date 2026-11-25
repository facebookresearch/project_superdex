# SuperDex Examples for Unreal

## Overview

This content-only plugin holds example levels for the SuperDexPhysics and SuperDexRobotics
plugins. It needs those plugins and SuperDexAssets.

| Level | Contents |
| --- | --- |
| `Content/Sampler/L_Sampler` | A workbench with the reusable `BP_PrefabSpawner` cycling task prefabs and `BP_RoboperatorSpawner` switching proxy hands or virtual FR3/DG-5F and Tianji/Wuji combinations |
| `Content/Sampler/L_Sampler_Android` | The standalone Android sampler |
| `Content/RuntimeSceneLoad/L_RuntimeSceneLoad` | `L_Sampler`'s lighting around the [runtime_scene_load SuperDex scene](https://github.com/facebookresearch/project_superdex/tree/main/assets/scenes/runtime_scene_load), teleoperated through `ASuperDexRuntimeSceneLoadExample`; the scene's README says how to run it |
| `Content/Kitchen/L_Kitchen` | A kitchen manipulation scene with objects, articulated mechanisms, and a soft body |

The reusable spawners live under `/SuperDexAssets/utilities/`. The levels also include keyboard
shortcut overlays and an FPS/SPS counter. For setup and controls, see the
[Unreal guides](https://projectsuperdex.com/robotics/docs/latest/unreal/getting-started/).

The levels and Blueprints are the original work of Meta Platforms, Inc. and affiliates. The levels
also use content that ships with Unreal Engine (basic shapes, sky, and template materials); that
content is not part of this plugin and is covered by your Unreal Engine license.

## License

The Meta-authored content in this plugin is distributed under the
[Creative Commons Attribution 4.0 International License (CC-BY-4.0)](LICENSE). You must give
appropriate credit to Meta Platforms, Inc. and affiliates when you use or redistribute it.
The SuperDexAssets prefabs it spawns carry their own IP notes; see that plugin's README.
