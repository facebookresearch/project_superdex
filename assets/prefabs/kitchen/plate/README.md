# Dinner Plate Prefab

## Overview

This package contains a Meta-owned ceramic dinner plate prefab for use in SuperDex simulations, part of the [kitchen](../README.md) scene. The CAD geometry and all derived digital assets in this directory are the original work of Meta Platforms, Inc. and affiliates. They were modeled in-house; no third-party digital models or geometry were used.

The plate (240 mm across, 22 mm tall, about 3.5 mm thick, on a 72 mm foot ring) is a rigid actor.

## Contents

- `cad/`: the SolidWorks part `plate.SLDPRT` and its STEP interchange model.
- `collision/`: SuperDex Physics collision models generated in Studio from the STEP models (SDF).
- `intermediates/`: the Studio processing configuration for each collision model.
- `render/`: render models (GLB) with PBR materials.
- `plate.mochi_prefab`: the prefab.

## Modifications and Processing

- Modeled the plate in SolidWorks as a revolved profile.
- Exported one STEP AP214 interchange model per part, with its face colors.
- Generated the collision models in Studio (`superdex_studio --process`) from the STEP models, meshed only as finely as contact needs: surfaces that get touched are finer, and the SDF resolution is set separately from the mesh. The CAD is Z-up like SuperDex, so no rotation was applied.
- Generated the render models from the SolidWorks tessellation of each part, with PBR materials assigned per part and face.
- Its collision model is fine (1.5 mm) to keep the thin rim, so plates stack.

## Physics Tuning

- The plate is ceramic (2400 kg/m^3), about 0.52 kg.

## Third-Party Material

None. All textures and materials in this directory are Meta-authored.

## License

The Meta-authored digital assets in this directory are distributed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](./LICENSE). This license applies only to the Meta-authored digital assets in this directory. You must give appropriate credit to Meta Platforms, Inc. and affiliates when you use or redistribute these assets.

**You are responsible for ensuring your use is compatible with all applicable licenses.**
