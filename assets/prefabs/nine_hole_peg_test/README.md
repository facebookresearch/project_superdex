# Nine-Hole Peg Test Asset

## Overview

This package contains a nine-hole peg-test simulation asset that was created in-house by modeling after the [Baseline 453602 9-Hole Wooden Pegboard](https://www.amazon.com/dp/B00ORZAWDE), a physical object purchased from [Amazon.com](https://www.amazon.com/). No third-party digital source files were used; the geometry in this directory is Meta's own work.

## Reference Object

| Field | Value |
| --- | --- |
| Product name | Baseline 453602 9-Hole Wooden Pegboard |
| Source / purchase link | [Amazon product B00ORZAWDE](https://www.amazon.com/dp/B00ORZAWDE) |
| Vendor | [Amazon.com](https://www.amazon.com/) |
| Brand | Baseline |
| Manufacturer | Fabrication Enterprises, Inc. |

## Contents

- `cad/`: SolidWorks parts for the board (`base_nine_hole`) and a peg (`peg_nine_hole`), and a STEP interchange model of each.
- `collision/`: SuperDex Physics collision models generated in Studio from the STEP models (SDF).
- `intermediates/`: the Studio processing configuration for each collision model.
- `render/`: render models (GLB) with PBR materials.
- `tuning/`: `superdex_asset_tuner` scenarios used to tune the prefab.
- `nine_hole_peg_test.mochi_prefab`: the test: the board and nine nested pegs.
- `nine_hole_peg.superdex_prefab`: one peg.

## Derivations

The following work was done to produce this simulation-ready asset:

- Hand-modeled the pegboard base and peg geometry in SolidWorks using the physical object as a reference; the geometry was not 3D-scanned. The board is 125 mm square and 39.8 mm tall, with nine 6.84 mm holes 22 mm deep on a 29 mm grid; the pegs are 5.68 mm across and 40.2 mm long, with rounded ends.
- Exported a STEP interchange model per part.
- Generated the collision models in Studio (`superdex_studio --process`) from the STEP models. The board's hole walls, hole floors and rounded edges are colored in the CAD so Studio sizes them separately: 2.75 mm on the hole walls, 6 mm on the rounded edges and 40 mm elsewhere, for 3,888 triangles; a peg is a uniform 2 mm mesh (428), about 7,700 for the whole test. Every collision mesh is closed and manifold. Their SDFs have 1.2 mm (board) and 0.4 mm (peg) voxels; through them, a hole's radius is within 0.27 mm of the CAD's.
- Generated the render models from Studio's tessellation of the STEP models, with crease-split normals and the wood grain along the board and along each peg. They match the product: light rubberwood under a clear finish for the board and walnut-stained wood for the pegs, both glossy as from a polyurethane finish. Their textures are Poly Haven wood scans (see Third-Party Material), recolored to the product's woods, with the roughness lowered to a finish's.
- Assembled the test from the board, centered on the origin, and nine copies of the peg prefab, each seated on the floor of its hole and centered on the hole's axis.

## Physics Tuning

The scenarios are in `tuning/`; run them with `superdex_asset_tuner run tuning/<name>.superdex_tuner` or open them in Studio.

- `rest`: no peg moves when the scene starts (under 1 um over 2 s).
- `extract_insert`: pulling the center peg straight out takes about its own weight (6 mN for a 0.6 g peg; 650 kg/m^3 hardwood), and pushing it back in, 0.34 mm off the hole's axis and tilted 3 degrees, takes no more.
- `pinch_lift_highpoly` and `pinch_lift_lowpoly`: an OpenXR index fingertip and thumb tip, gripping about 3 N each, lift the center peg out, carry it and put it back, then pick it up again and drop it into its hole from 2 mm inside the rim. It slips less than 0.3 mm down in the fingers, and it doesn't stick to them when they let go: the fingers are clear of it 35 ms after they start to open (50 ms at 10 ms steps) without lifting it or flicking it sideways, and it lands on the floor of its hole about as fast as it would free-fall (0.07 s for 20 mm; at 10 ms steps it rattles down the hole in 0.33 s). With the lowpoly fingertips, whose SDFs are coarse, the peg tilts about 5 degrees in the pinch but holds still, and the scenario reaches the drop as with the highpoly fingertips.
- Contact: the pegs are grippy (Coulomb friction 1.0 and viscous friction 20 s/m, so an OpenXR fingertip holds them at 0.71 plus the hand's viscous friction) and the board slippery (0.12, so a peg slides in its hole at 0.35). The pegs' friction falloff velocity is 1 mm/s rather than Mochi's default 10 mm/s; with the default, a peg crept sideways out of a fingertip pinch. Contact starts 0.1 mm out and smooths over 0.1 mm, well inside the 0.57 mm clearance between a peg and its hole, at a penalty of 1e10 Pa/m, which keeps the light pegs stable at 10 ms time steps.
- The pegs sample contact at one point per triangle (P1Q1): with Mochi's default three, their coarse bottom caps supported them unevenly and they rocked when the scene started.
- The board is a free body (density 1,200 kg/m^3, 0.68 kg, a little heavier than solid maple so it stays put when a peg is pinched in its hole). Contact between a peg and the board is checked one way, the peg's surface against the board's SDF (the board's contact layer `PegBoard` doesn't collide with the pegs' `Peg`): the board's coarse hole floors checked against a peg's SDF tipped the corner pegs over. The tuners rest it on a ground plane.

## Third-Party Material

The wood textures embedded in `render/` are derived from [Oak Veneer 04](https://polyhaven.com/a/oak_veneer_04) (board) and [Black Walnut Veneer 02](https://polyhaven.com/a/black_walnut_veneer_02) (pegs) by Jenelle van Heerden from Poly Haven, released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).



## License & IP Notes

The Meta-authored digital assets in this directory are distributed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](./LICENSE). You must give appropriate credit to Meta Platforms, Inc. when you use or redistribute them.

The license applies only to Meta-authored CAD, render meshes, collision data, processing metadata, and prefab manifests in this directory. It does not grant rights in the underlying physical product design, trade dress, product names, logos, or trademarks.

These assets were modeled after a commercially available product (the Baseline 9-Hole Wooden Pegboard by Fabrication Enterprises, Inc.). While the 3D geometry is Meta's own work, the underlying product design may be protected by Fabrication Enterprises' copyright, design-patent, or trade-dress rights. Accordingly:

- These assets were created for simulation, research, visualization, and software development. CC-BY-4.0 applies only to copyrightable material Meta owns and may license; it does not grant patent, design-patent, trademark, trade-dress, or other rights in the referenced physical product. Manufacturing physical hardware may require separate permission from the applicable rightsholders.
- The Fabrication Enterprises and Baseline names, logos, and product names may not be used to imply endorsement, certification, sponsorship, or an official partnership without prior written permission.

**You are responsible for ensuring your use is compatible with all applicable third-party rights.**
