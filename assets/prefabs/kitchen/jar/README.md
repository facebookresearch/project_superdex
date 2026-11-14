# Jar Prefab

## Overview

This package contains a Meta-owned glass jar with a screw-on lid for use in SuperDex simulations, part of the [kitchen](../README.md) scene. The CAD geometry and all derived digital assets in this directory are the original work of Meta Platforms, Inc. and affiliates. They were modeled in-house; no third-party digital models or geometry were used.

The jar (78 mm across and 129 mm tall with its lid) and the lid are two rigid actors that meet only through contact: the lid unscrews along a coarse thread (9.75 mm pitch) in about one turn.

## Contents

- `cad/`: the SolidWorks assembly `jar.SLDASM`, its parts (`jar_body`, `jar_lid`), and one STEP interchange model per part.
- `collision/`: SuperDex Physics collision models generated in Studio from the STEP models (SDF).
- `intermediates/`: the Studio processing configuration for each collision model.
- `render/`: render models (GLB) with PBR materials.
- `tuning/`: `superdex_asset_tuner` scenarios used to tune the prefab.
- `jar.mochi_prefab`: the prefab.

## Modifications and Processing

- Modeled the body and the lid in SolidWorks, threads included, scaled both to 78% about their origins, and assembled them in `cad/jar.SLDASM` with the lid screwed on.
- Exported one STEP AP214 interchange model per part, with its face colors.
- Generated the collision models in Studio (`superdex_studio --process`) from the STEP models, meshed only as finely as contact needs: surfaces that get touched are finer, and the SDF resolution is set separately from the mesh. The CAD is Z-up like SuperDex, so no rotation was applied.
- Generated the render models from the SolidWorks tessellation of each part, with PBR materials assigned per part and face. The body is clear glass (`KHR_materials_transmission`, index of refraction 1.5) and the lid is red plastic.
- The collision models are fine enough to keep the threads and the 4 mm glass wall: their SDFs have 1 mm (body) and 0.8 mm (lid) voxels.

## Physics Tuning

The scenarios are in `tuning/`; run them with `superdex_asset_tuner run tuning/<name>.superdex_tuner` or open them in Studio. `unscrew` twists the lid off while the jar is held.

- The lid is finger tight: it starts to turn at about 14 mN m and climbs off the thread in about one turn.
- The jar weighs about 0.18 kg (1500 kg/m^3 for its thick glass walls) and the lid 0.03 kg.
- The threads are meshed at 1.6 mm and the rest at 3 to 5 mm: the thread faces carry their own colors in the CAD, which drive Studio's color-based face sizing.
- The jar and lid are stiffer than the rest of the kitchen (penalty 1e12 Pa/m) and smooth their contact force over 1 mm rather than Mochi's default 5 mm. Fingers meet the jar's contact parameters, and with the default smoothing the force builds up so slowly over the first millimeters that robot fingers on default contact parameters pushed through the glass wall. Pushing with 45 N, they now stop about 1 mm into it.
- The jar and lid start contact 0.1 mm apart rather than at Mochi's default 1 mm: the threads' clearance is smaller than that, and at the jar's contact stiffness the lid would lock.

## Third-Party Material

None. All textures and materials in this directory are Meta-authored.

## License

The Meta-authored digital assets in this directory are distributed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](./LICENSE). This license applies only to the Meta-authored digital assets in this directory. You must give appropriate credit to Meta Platforms, Inc. and affiliates when you use or redistribute these assets.

**You are responsible for ensuring your use is compatible with all applicable licenses.**
