# Kitchen Sink Prefab

## Overview

This package contains a Meta-owned kitchen sink prefab for use in SuperDex simulations, part of the [kitchen](../README.md) scene. The CAD geometry and all derived digital assets in this directory are the original work of Meta Platforms, Inc. and affiliates. They were modeled in-house; no third-party digital models or geometry were used.

The sink is a counter section with an inset stainless basin, 1 m wide, 600 mm deep and 1 m tall: it stands on the floor with the countertop 1 m up. It is one articulation fixed in the world: the counter, a strainer sleeve in the drain and a disposal ring are hard-jointed, and the disposal blade under the drain spins freely.

## Contents

- `cad/`: the SolidWorks assembly `sink.SLDASM`, its parts (`sink_counter`, `sink_drain`, `sink_disposal_base`, `sink_disposal_blade`), and one STEP interchange model per part.
- `collision/`: SuperDex Physics collision models generated in Studio from the STEP models (SDF).
- `intermediates/`: the Studio processing configuration for each collision model.
- `render/`: render models (GLB) with PBR materials.
- `sink.mochi_prefab`: the prefab.

## Modifications and Processing

- Modeled the counter and the disposal in SolidWorks, and modeled the strainer sleeve for this package. Removed the counter's hose hole and edge notches, which served features this package does not have, and resized it to 1 m wide, 600 mm deep and 1 m tall. Assembled them in `cad/sink.SLDASM`: the countertop is at Z = 0, the basin is centered on the origin (325 mm from the back and 275 mm from the front), and the wall side is -X.
- Exported one STEP AP214 interchange model per part, with its face colors.
- Generated the collision models in Studio (`superdex_studio --process`) from the STEP models, meshed only as finely as contact needs: surfaces that get touched are finer, and the SDF resolution is set separately from the mesh. The CAD is Z-up like SuperDex, so no rotation was applied.
- Generated the render models from the SolidWorks tessellation of each part, with PBR materials assigned per part and face.

## Physics Tuning

- The disposal blade spins on a revolute joint with a little friction and has no motor.
- The counter's collision mesh is 10 mm on the countertop and in the basin and 50 mm on its sides, bottom and drain bore: those faces carry their own colors in the CAD, which drive Studio's color-based face sizing. Its SDF has 12 mm voxels.

## Third-Party Material

None. All textures and materials in this directory are Meta-authored.

## License

The Meta-authored digital assets in this directory are distributed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](./LICENSE). This license applies only to the Meta-authored digital assets in this directory. You must give appropriate credit to Meta Platforms, Inc. and affiliates when you use or redistribute these assets.

**You are responsible for ensuring your use is compatible with all applicable licenses.**
