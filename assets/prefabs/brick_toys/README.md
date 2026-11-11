# Large Brick Toy Prefab

## Overview

This package contains a Meta-owned 2x2 large interlocking brick toy prefab for use in SuperDex simulations. The render and collision models are the original work of Meta Platforms, Inc. and affiliates. They were modeled in-house; no third-party digital source files were used.

## Modifications and Processing

- Manually authored the brick geometry in SolidWorks: a render model, an exact-tolerance model, and a separate behavior-optimized collision model.
- Exported the collision model as a STEP interchange model.
- Generated GLB render geometry from the source CAD (`large_brick_toy_2x2_render.glb` and `large_brick_toy_2x2_exact_tolerance.glb`).
- Generated SDF collision geometry for SuperDex Physics in Studio (`intermediates/large_brick_toy_2x2_collision.StudioProcessing.json`).
- Assembled the asset with contact settings as `large_brick_toy_2x2.mochi_prefab`.

## License

The Meta-authored digital assets in this directory are distributed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](./LICENSE). This license applies only to the Meta-authored digital assets in this directory. You must give appropriate credit to Meta Platforms, Inc. and affiliates when you use or redistribute these assets.

The license does not grant patent, design-patent, trademark, trade-dress, or other rights in any physical product. Third-party product names and trademarks are not used in or licensed with these assets.

**You are responsible for ensuring your use is compatible with all applicable licenses.**
