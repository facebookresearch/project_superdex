# Puzzle Cube Prefab

## Overview

This package contains a Meta-owned 2x2 puzzle cube prefab for use in SuperDex simulations, built as a single articulation: the corners turn about the center on spherical joints. The digital geometry is the original work of Meta Platforms, Inc. and affiliates. It was modeled in-house; no third-party digital source files were used. The mechanism is inspired but not identical to that of a real 2x2 puzzle cube; it is not modeled after any particular physical toy that can be purchased.

## Modifications and Processing

- Manually authored the puzzle cube geometry in SolidWorks and exported a STEP interchange model.
- Generated render geometry GLB from the source CAD.
- Generated collision geometry for SuperDex Physics in Studio.
- Assembled the cube in Studio as a single articulation (a bot), scaled to 85% of the CAD model, and exported it as `puzzle_cube.mochi_prefab` with Studio's Bot > Export Prefab.
- Gave each corner shell its own render model, `render/puzzle_cube_corner<N>_shell.glb`, colored for the corner's place in the solved cube: white +Z, yellow -Z, red +X, orange -X, blue +Y, green -Y.

## Physics Tuning

`puzzle_cube.mochi_prefab` is tuned with `superdex_asset_tuner`; the scenarios are in `tuning/` (run them with `superdex_asset_tuner run tuning/<name>.superdex_tuner` or open them in Studio). `tuning/solve.superdex_tuner` starts from a scrambled cube and solves it with PosePaths.

- The corners hang off the center on spherical joints; contact alone keeps them in their layers. Two layers split the pieces: `puzzle_cube_core` (the center and the inner corner pieces) and `puzzle_cube_shell` (the outer shells). Core and shells never touch. The core also ignores the `LeftHand`, `RightHand` and `GenericObject` layers: hands and props on those layers only need to touch the shells.
- Mochi disables contact between adjacent links, so the center <-> inner corner contacts that guide the corners are re-enabled with actor contact filters. A loader that ignores actor contact filters loses these contacts, and without them the corners drift out of place within a few turns.
- The core contacts start 0.3 mm inside the surface to absorb the overlap left by tessellating the zero-clearance CAD, then stiffen within 1 mm, so the layers turn freely with little play.
- The shells' small touching corners at each slice plane give the detent: shell contact starts 0.275 mm out, so the aligned corners press on each other and their friction (1.6, which also grips the cube when it is picked up) holds the layer. Away from alignment the shells clear each other.
- The corners' spherical joints have viscous friction (1e-3 N m s/rad), which damps every corner motion relative to the center (turning, wobble and rattle), and dry friction (0.75 mN m, smoothed below 0.01 rad/s), which holds the corners still under small loads and evens out the detent between the three axes. Together with the shells, an aligned layer holds until about 33 mN m in total, and the layer needs about 6 mN m to turn slowly, 12 mN m at 90 deg/s and 39 mN m at 360 deg/s. A flicked layer stops instead of coasting. The dry friction costs about 0.7 ms more solver time per step while the cube is at rest.
- The shell and inner corner collision is meshed finer than the rest (`targetEdgeLengthFraction` 0.03 and 0.04): the shell corners only touch over a few tenths of a millimeter, which coarser meshes turn into uneven interference that jams the layers.

## License

The Meta-authored digital assets in this directory are distributed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](./LICENSE). This license applies only to the Meta-authored digital assets in this directory. You must give appropriate credit to Meta Platforms, Inc. and affiliates when you use or redistribute these assets.

**You are responsible for ensuring your use is compatible with all applicable licenses.**
