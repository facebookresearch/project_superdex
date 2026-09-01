# Puzzle Cube Prefab

## Overview

This package contains a Meta-owned puzzle cube prefab for use in SuperDex simulations. The digital geometry is the original work of Meta Platforms, Inc. and affiliates. It was modeled in-house; no third-party digital source files were used. The mechanism is inspired but not identical to that of a real 2x2 puzzle cube; it is not modeled after any particular physical toy that can be purchased.

## Modifications and Processing

- Manually authored the puzzle cube geometry in SolidWorks and exported a STEP interchange model.
- Generated render geometry GLB from the source CAD.
- Generated collision geometry for SuperDex Physics in Studio.
- Assembled the cube as `puzzle_cube.mochi_prefab` as aggregate of `puzzle_cube_center.mochi_prefab` and `puzzle_cube_corner.mochi_prefab`.

## License

The Meta-authored digital assets in this directory are distributed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](./LICENSE). This license applies only to the Meta-authored digital assets in this directory. You must give appropriate credit to Meta Platforms, Inc. and affiliates when you use or redistribute these assets.

**You are responsible for ensuring your use is compatible with all applicable licenses.**
