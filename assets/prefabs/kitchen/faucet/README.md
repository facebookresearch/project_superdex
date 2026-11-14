# Kitchen Faucet Prefab

## Overview

This package contains a Meta-owned single-lever kitchen faucet prefab for use in SuperDex simulations, part of the [kitchen](../README.md) scene. The CAD geometry and all derived digital assets in this directory are the original work of Meta Platforms, Inc. and affiliates. They were modeled in-house; no third-party digital models or geometry were used.

The faucet is one articulation: the base is fixed in the world (on a Hard root joint), the gooseneck spout swivels about the vertical axis, and the lever sits on a spherical joint on the side of the base.

## Contents

- `cad/`: the SolidWorks assembly `faucet.SLDASM`, its parts (`faucet_base`, `faucet_spout`, `faucet_handle`), and one STEP interchange model per part.
- `collision/`: SuperDex Physics collision models generated in Studio from the STEP models (SDF).
- `intermediates/`: the Studio processing configuration for each collision model.
- `render/`: render models (GLB) with PBR materials.
- `tuning/`: `superdex_asset_tuner` scenarios used to tune the prefab.
- `faucet.mochi_prefab`: the prefab.

## Modifications and Processing

- Modeled the base, spout and lever in SolidWorks and assembled them in `cad/faucet.SLDASM`, the lever on the base's 45 deg side boss.
- Exported one STEP AP214 interchange model per part, with its face colors.
- Generated the collision models in Studio (`superdex_studio --process`) from the STEP models, meshed only as finely as contact needs: surfaces that get touched are finer, and the SDF resolution is set separately from the mesh. The CAD is Z-up like SuperDex, so no rotation was applied.
- Generated the render models from the SolidWorks tessellation of each part, with PBR materials assigned per part and face.
- Assembled the articulation from the CAD layout: the faucet frame is the base's, with Z up and the spout reaching over +X.

## Physics Tuning

The scenarios are in `tuning/`; run them with `superdex_asset_tuner run tuning/<name>.superdex_tuner` or open them in Studio. `turn` turns the spout and lifts the lever.

- **Spout** swivels +-57 deg and starts to turn at about 0.15 N m; joint friction holds it wherever it is left.
- **Lever** stands upright, tilts 29 deg outward and swings +-29 deg on a spherical joint (it cannot twist). It starts to tilt at about 0.1 N m, and friction holds it where it is left.
- The faucet has no water.

## Third-Party Material

None. All textures and materials in this directory are Meta-authored.

## License

The Meta-authored digital assets in this directory are distributed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](./LICENSE). This license applies only to the Meta-authored digital assets in this directory. You must give appropriate credit to Meta Platforms, Inc. and affiliates when you use or redistribute these assets.

**You are responsible for ensuring your use is compatible with all applicable licenses.**
