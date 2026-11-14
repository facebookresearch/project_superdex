# Double Light Switch Prefab

## Overview

This package contains a Meta-owned double toggle light switch prefab for use in SuperDex simulations, part of the [kitchen](../README.md) scene. The CAD geometry and all derived digital assets in this directory are the original work of Meta Platforms, Inc. and affiliates. They were modeled in-house; no third-party digital models or geometry were used.

The switch is one articulation: the wall plate is fixed in the world, facing +X with its back face at X = -25 mm, and each toggle flips between two stops on a revolute joint.

## Contents

- `cad/`: the SolidWorks assembly `switch.SLDASM`, its parts (`switch_plate`, `switch_toggle`), and one STEP interchange model per part.
- `collision/`: SuperDex Physics collision models generated in Studio from the STEP models (SDF).
- `intermediates/`: the Studio processing configuration for each collision model.
- `render/`: render models (GLB) with PBR materials.
- `tuning/`: `superdex_asset_tuner` scenarios used to tune the prefab.
- `switch.mochi_prefab`: the prefab.

## Modifications and Processing

- Modeled the plate and the toggle in SolidWorks and assembled the plate with two toggles in `cad/switch.SLDASM`.
- Exported one STEP AP214 interchange model per part, with its face colors.
- Generated the collision models in Studio (`superdex_studio --process`) from the STEP models, meshed only as finely as contact needs: surfaces that get touched are finer, and the SDF resolution is set separately from the mesh. The CAD is Z-up like SuperDex, so no rotation was applied.
- Generated the render models from the SolidWorks tessellation of each part, with PBR materials assigned per part and face.
- Assembled the articulation from the CAD layout. The toggles rest on their lower (off) stop, 20 deg below horizontal, and flip up to 20 deg above it.

## Physics Tuning

The scenarios are in `tuning/`; run them with `superdex_asset_tuner run tuning/<name>.superdex_tuner` or open them in Studio. `flip` flips each toggle with a slowly growing force at its tip.

- **Bistable toggles**: an over-center spring per toggle, a `rigidPivotPosition` constraint, pulls a point at the toggle's tip toward a fixed point 15 mm behind its pivot. The spring is longest with the toggle level, so it pushes the toggle to whichever stop is nearer and holds it there. A toggle snaps over at about 1.6 N at its tip.
- A loader that ignores prefab constraints loses the springs, and the toggles then rest wherever they are left.
- The switch switches nothing.

## Third-Party Material

None. All textures and materials in this directory are Meta-authored.

## License

The Meta-authored digital assets in this directory are distributed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](./LICENSE). This license applies only to the Meta-authored digital assets in this directory. You must give appropriate credit to Meta Platforms, Inc. and affiliates when you use or redistribute these assets.

**You are responsible for ensuring your use is compatible with all applicable licenses.**
