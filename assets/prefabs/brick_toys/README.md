# Large Brick Toy Prefab

## Overview

This package contains a Meta-owned 2x2 large interlocking brick toy prefab for use in SuperDex simulations. The render and collision models are the original work of Meta Platforms, Inc. and affiliates. They were modeled in-house; no third-party digital source files were used.

## Modifications and Processing

- Manually authored the brick geometry in SolidWorks: a render model, an exact-tolerance model, and a separate behavior-optimized collision model.
- Exported the collision model as a STEP interchange model.
- Generated GLB render geometry from the source CAD (`large_brick_toy_2x2_render.glb` and `large_brick_toy_2x2_exact_tolerance.glb`).
- Generated SDF collision geometry for SuperDex Physics in Studio (`intermediates/large_brick_toy_2x2_collision.StudioProcessing.json`).
- Assembled the asset with contact settings and a 6.5 g mass as `large_brick_toy_2x2.mochi_prefab`.
- Added red, green, blue, yellow, white, black, teal and orange variants (`large_brick_toy_2x2_<color>.superdex_prefab`), whose render models differ only in base color; `brick_toys.mochi_prefab` shows them as four connected pairs in a 2 x 2 grid.

## Physics Tuning

- `tuning/insert_force` and `tuning/pull_force` push an aligned brick onto another and pull a connected one off, with rising forces; `tuning/push_pull` drives one onto another and back off at 10 mm/s.
- `tuning/pinch_assemble` puts one brick on another with an OpenXR index fingertip and thumb tip; `tuning/column_lift` lifts a column of 24 by its top brick.
- `tuning/rest` (a brick and a connected pair) and `tuning/finger_press` (a fingertip pressed onto a stud).

## License

The Meta-authored digital assets in this directory are distributed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](./LICENSE). This license applies only to the Meta-authored digital assets in this directory. You must give appropriate credit to Meta Platforms, Inc. and affiliates when you use or redistribute these assets.

The license does not grant patent, design-patent, trademark, trade-dress, or other rights in any physical product. Third-party product names and trademarks are not used in or licensed with these assets.

**You are responsible for ensuring your use is compatible with all applicable licenses.**
