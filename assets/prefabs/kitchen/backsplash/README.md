# Kitchen Backsplash Prefab

## Overview

This package contains a Meta-owned tiled wall section for use in SuperDex simulations, part of the [kitchen](../README.md) scene. The CAD geometry and all derived digital assets in this directory are the original work of Meta Platforms, Inc. and affiliates. They were modeled in-house; no third-party digital models or geometry were used.

The backsplash is a static rigid actor, 1 m wide (as wide as the [sink](../sink/README.md) counter) and 100 mm thick, running from the floor (1 m below the countertop, the part's Z = 0) to 1 m above the countertop, with its front face at X = +50 mm and a 100 mm square cut-out for the [switch](../switch/README.md), 100 mm in from its +Y side.

## Contents

- `cad/`: the SolidWorks assembly `backsplash.SLDASM`, its parts (`backsplash`), and one STEP interchange model per part.
- `collision/`: SuperDex Physics collision models generated in Studio from the STEP models (SDF).
- `intermediates/`: the Studio processing configuration for each collision model.
- `render/`: render models (GLB) with PBR materials.
- `backsplash.mochi_prefab`: the prefab.
- `textures/`: the source textures of the wall tiles (base color, metallic-roughness and normal maps for one 100 mm tile).

## Modifications and Processing

- Modeled the wall section in SolidWorks.
- Exported one STEP AP214 interchange model per part, with its face colors.
- Generated the collision models in Studio (`superdex_studio --process`) from the STEP models, meshed only as finely as contact needs: surfaces that get touched are finer, and the SDF resolution is set separately from the mesh. The CAD is Z-up like SuperDex, so no rotation was applied.
- Generated the render models from the SolidWorks tessellation of each part, with PBR materials assigned per part and face. The front is covered in white octagonal ceramic tiles with blue corner inserts, a procedural texture generated for this package (`textures/`), projected at 100 mm per tile.

## Third-Party Material

None. All textures and materials in this directory are Meta-authored.

## License

The Meta-authored digital assets in this directory are distributed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](./LICENSE). This license applies only to the Meta-authored digital assets in this directory. You must give appropriate credit to Meta Platforms, Inc. and affiliates when you use or redistribute these assets.

**You are responsible for ensuring your use is compatible with all applicable licenses.**
