# Soap Dispenser Prefab

## Overview

This package contains a Meta-owned countertop soap dispenser prefab for use in SuperDex simulations, part of the [kitchen](../README.md) scene. The CAD geometry and all derived digital assets in this directory are the original work of Meta Platforms, Inc. and affiliates. They were modeled in-house; no third-party digital models or geometry were used.

The dispenser is one articulation: the base is fixed in the world and the pump head presses down on a prismatic joint against a return spring.

## Contents

- `cad/`: the SolidWorks assembly `dispenser.SLDASM`, its parts (`dispenser_base`, `dispenser_pump`), and one STEP interchange model per part.
- `collision/`: SuperDex Physics collision models generated in Studio from the STEP models (SDF).
- `intermediates/`: the Studio processing configuration for each collision model.
- `render/`: render models (GLB) with PBR materials.
- `tuning/`: `superdex_asset_tuner` scenarios used to tune the prefab.
- `dispenser.mochi_prefab`: the prefab.

## Modifications and Processing

- Modeled the base and the pump head in SolidWorks and assembled them in `cad/dispenser.SLDASM`.
- Exported one STEP AP214 interchange model per part, with its face colors.
- Generated the collision models in Studio (`superdex_studio --process`) from the STEP models, meshed only as finely as contact needs: surfaces that get touched are finer, and the SDF resolution is set separately from the mesh. The CAD is Z-up like SuperDex, so no rotation was applied.
- Generated the render models from the SolidWorks tessellation of each part, with PBR materials assigned per part and face.
- Assembled the articulation from the CAD layout: the base's bottom is at Z = 0 and the spout points -Y.

## Physics Tuning

The scenarios are in `tuning/`; run them with `superdex_asset_tuner run tuning/<name>.superdex_tuner` or open them in Studio. `press` presses the pump head and lets go.

- The head travels 25 mm. A preloaded return spring (an `articulatedSingleDofTarget` constraint) holds it up: it starts to move at about 3 N and bottoms out at about 9 N, and springs back when released.
- A loader that ignores prefab constraints loses the return spring.
- The dispenser dispenses nothing.

## Third-Party Material

None. All textures and materials in this directory are Meta-authored.

## License

The Meta-authored digital assets in this directory are distributed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](./LICENSE). This license applies only to the Meta-authored digital assets in this directory. You must give appropriate credit to Meta Platforms, Inc. and affiliates when you use or redistribute these assets.

**You are responsible for ensuring your use is compatible with all applicable licenses.**
