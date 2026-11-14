# Wall Cabinet Prefab

## Overview

This package contains a Meta-owned two-door wall cabinet prefab for use in SuperDex simulations, part of the [kitchen](../README.md) scene. The CAD geometry and all derived digital assets in this directory are the original work of Meta Platforms, Inc. and affiliates. They were modeled in-house; no third-party digital models or geometry were used.

The cabinet is one articulation: the box (1000 x 300 x 500 mm, as wide as the [sink](../sink/README.md) counter, with one shelf) is fixed in the world with its back at X = 0 and its front facing +X, and each door swings on its outer hinge.

## Contents

- `cad/`: the SolidWorks assembly `cabinet.SLDASM`, its parts (`cabinet_box`, `cabinet_door_left`, `cabinet_door_right`), and one STEP interchange model per part.
- `collision/`: SuperDex Physics collision models generated in Studio from the STEP models (SDF).
- `intermediates/`: the Studio processing configuration for each collision model.
- `render/`: render models (GLB) with PBR materials.
- `tuning/`: `superdex_asset_tuner` scenarios used to tune the prefab.
- `cabinet.mochi_prefab`: the prefab.

## Modifications and Processing

- Modeled the box and the left door in SolidWorks; the right door is rebuilt from the left one as its mirror image. Assembled them in `cad/cabinet.SLDASM`, the 479 mm doors hinged 20 mm in from the sides.
- Exported one STEP AP214 interchange model per part, with its face colors.
- Generated the collision models in Studio (`superdex_studio --process`) from the STEP models, meshed only as finely as contact needs: surfaces that get touched are finer, and the SDF resolution is set separately from the mesh. The CAD is Z-up like SuperDex, so no rotation was applied.
- Generated the render models from the SolidWorks tessellation of each part, with PBR materials assigned per part and face. The box is painted navy outside and white inside, with an oak-veneered back panel in a satin finish, and the doors are navy with brass handles.
- Assembled the articulation from the CAD layout.

## Physics Tuning

The scenarios are in `tuning/`; run them with `superdex_asset_tuner run tuning/<name>.superdex_tuner` or open them in Studio. `door` pulls the left door open by its handle and pushes it shut.

- Each door opens 110 deg. Joint friction holds a door wherever it is left; there is no catch holding it shut.
- The box and doors are wood (600 kg/m^3): about 26 kg for the box and 1.6 kg per door.
- The door handles are meshed at 3 mm and the rest of the cabinet at 25 to 30 mm: the handle faces carry their own color in the CAD, which drives Studio's color-based face sizing. The doors' SDFs have 2 mm voxels: the door panels are only 10 mm thick, and on a coarser grid a hard push goes through them.

## Third-Party Material

The wood texture embedded in `render/cabinet_box.glb` is [Oak Veneer 01](https://polyhaven.com/a/oak_veneer_01) by Jenelle van Heerden from Poly Haven, released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).

## License

The Meta-authored digital assets in this directory are distributed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](./LICENSE). This license applies only to the Meta-authored digital assets in this directory. You must give appropriate credit to Meta Platforms, Inc. and affiliates when you use or redistribute these assets.

**You are responsible for ensuring your use is compatible with all applicable licenses.**
