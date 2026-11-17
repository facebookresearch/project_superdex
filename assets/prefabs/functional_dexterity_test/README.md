# Functional Dexterity Test Asset

## Overview

This package contains a Functional Dexterity Test simulation asset that was created in-house by modeling after the [North Coast Functional Dexterity Test, model NC32152](https://www.amazon.com/dp/B0052ZX5NW), a physical object purchased from [Amazon.com](https://www.amazon.com/). No third-party digital source files were used; the geometry in this directory is Meta's own work.

## Reference Object

| Field | Value |
| --- | --- |
| Product name | North Coast Functional Dexterity Test, model NC32152 |
| Source / purchase link | [Amazon product B0052ZX5NW](https://www.amazon.com/dp/B0052ZX5NW) |
| Vendor | [Amazon.com](https://www.amazon.com/) |
| Brand / manufacturer | North Coast Medical |

## Contents

- `cad/`: SolidWorks parts for the board (`pegboard`) and a peg (`peg`), and a STEP interchange model of each.
- `collision/`: SuperDex Physics collision models generated in Studio from the STEP models (SDF).
- `intermediates/`: the Studio processing configuration for each collision model.
- `render/`: render models (GLB) with PBR materials.
- `tuning/`: `superdex_asset_tuner` scenarios used to tune the prefab.
- `functional_dexterity_test.mochi_prefab`: the test: the board and sixteen nested pegs.
- `fdt_peg.mochi_prefab`: one peg.

## Derivations

The following work was done to produce this simulation-ready asset:

- Hand-modeled the pegboard and peg geometry in SolidWorks using the physical object as a reference; the geometry was not 3D-scanned. The board is 206 mm square and 37 mm tall, with sixteen 24.8 mm holes 30 mm deep on a 45 mm grid; the pegs are 22 mm across and 40 mm long.
- Exported a STEP interchange model per part.
- Generated the collision models in Studio (`superdex_studio --process`) from the STEP models. The board's hole walls, hole floors and rounded edges are colored in the CAD so Studio sizes them separately: 6.5 mm on the hole walls, 10 mm on the rounded edges and 60 mm elsewhere, for 3,602 triangles; a peg is a uniform 4.5 mm mesh (396), about 9,900 for the whole test. Every collision mesh is closed and manifold. Their SDFs have 1.6 mm (board) and 0.75 mm (peg) voxels; through them, a hole's radius is within 0.42 mm of the CAD's (0.25 mm on average), well inside the 1.4 mm peg-to-hole clearance.
- Generated the render models from Studio's tessellation of the STEP models, with crease-split normals and the wood grain along the board and along each peg. They match the product: a warm maple board and light birch pegs, both glossy as from a polyurethane finish, each peg painted black on one end and white on the other. The wood textures are Poly Haven wood scans (see Third-Party Material), recolored to the product's woods, with the roughness lowered to a finish's.
- Assembled the test from the board, centered on the origin, and sixteen copies of the peg prefab, each seated on the floor of its hole, centered on the hole's axis, white end up.

## Physics Tuning

The scenarios are in `tuning/`; run them with `superdex_asset_tuner run tuning/<name>.superdex_tuner` or open them in Studio.

- `rest`: no peg moves when the scene starts (under 1 um over 2 s).
- `extract_insert`: pulling a peg straight out takes about its own weight (0.1 N for a 9.6 g peg; 650 kg/m^3 hardwood), and pushing it back in, 0.84 mm off the hole's axis and tilted 3 degrees, takes no more.
- `pinch_lift_highpoly` and `pinch_lift_lowpoly`: an OpenXR index fingertip and thumb tip, gripping about 3 N each, lift a peg out, carry it and put it back in its hole, then pick it up again and drop it into its hole from 2 mm inside the rim. It slips less than 0.4 mm (lowpoly) and 1.4 mm (highpoly) down in the fingers, and it doesn't stick to them when they let go: the fingers are clear of it 35 to 50 ms after they start to open, without lifting it or flicking it sideways, and it lands on the floor of its hole as fast as it would free-fall (0.07 s for 28 mm), at 5 and 10 ms steps alike. The pegs stand only 10 mm proud of the board, so the fingertips brush the board as they grip, as real ones do.
- Contact: the pegs are grippy (Coulomb friction 1.0 and viscous friction 20 s/m, so an OpenXR fingertip holds them at 0.71 plus the hand's viscous friction) and the board slippery (0.12, so a peg slides in its hole at 0.35). The pegs' friction falloff velocity is 1 mm/s rather than Mochi's default 10 mm/s, so they creep less in a pinch. Contact starts 0.1 mm out and smooths over 0.1 mm, well inside the 1.4 mm clearance between a peg and its hole, at a penalty of 1e10 Pa/m.
- The pegs sample contact at one point per triangle (P1Q1), as the nine-hole test's do.
- The board is a free body (density 1,200 kg/m^3, 1.58 kg), with one-way peg-board contact, as in the nine-hole test. The tuners rest it on a ground plane.

## Third-Party Material

The wood textures embedded in `render/` are derived from [Cherry Veneer](https://polyhaven.com/a/cherry_veneer) (board) and [Oak Veneer 03](https://polyhaven.com/a/oak_veneer_03) (pegs) by Jenelle van Heerden from Poly Haven, released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).



## License & IP Notes

The Meta-authored digital assets in this directory are distributed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](./LICENSE). You must give appropriate credit to Meta Platforms, Inc. when you use or redistribute them.

The license applies only to Meta-authored CAD, render meshes, collision data, processing metadata, and prefab manifests in this directory. It does not grant rights in the underlying physical product design, trade dress, product names, logos, or trademarks.

These assets were modeled after a commercially available product (the Functional Dexterity Test by North Coast Medical). While the 3D geometry is Meta's own work, the underlying product design may be protected by North Coast Medical's copyright, design-patent, or trade-dress rights. Accordingly:

- These assets were created for simulation, research, visualization, and software development. CC-BY-4.0 applies only to copyrightable material Meta owns and may license; it does not grant patent, design-patent, trademark, trade-dress, or other rights in the referenced physical product. Manufacturing physical hardware may require separate permission from the applicable rightsholders.
- The North Coast Medical name, logos, and product names may not be used to imply endorsement, certification, sponsorship, or an official partnership without prior written permission.

**You are responsible for ensuring your use is compatible with all applicable third-party rights.**
