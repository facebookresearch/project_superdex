# Paper Cup Asset

## Overview

This package contains a paper-cup simulation asset that was created in-house by modeling after the [8 oz ecotainer Hot Cup, Carte Blanc, SKU SMRE8-CB](https://greenpaperproducts.com/products/disposable-compostable-8ounce-hot-cups-smre8-cb), a physical object purchased from [Green Paper Products](https://greenpaperproducts.com/). No third-party digital source files were used; the geometry in this directory is Meta's own work.

## Reference Object

| Field | Value |
| --- | --- |
| Product name | 8 oz ecotainer Hot Cup, Carte Blanc, SKU SMRE8-CB |
| Source / purchase link | [Green Paper Products listing](https://greenpaperproducts.com/products/disposable-compostable-8ounce-hot-cups-smre8-cb) |
| Vendor | [Green Paper Products](https://greenpaperproducts.com/) |
| Brand | ecotainer / Carte Blanc |
| Manufacturer | Graphic Packaging International |

## Derivations

The following work was done to produce this simulation-ready asset:

- Physically scanned the real cup on a 2D scanner and modeled it in SolidWorks (`cad/paper_cup.SLDPRT`) as a revolve: a 2 mm wall, a 2 mm floor over a bottom skirt, and a rolled rim. The `Collision` configuration makes the rim a polygon in section; its profile is revolved into `cad/paper_cup_collision.stl`, even rings of triangles around the cup, which the collision model is generated from.
- Generated the collision model in Studio (about 2,000 triangles) with a 1.5 mm SDF, fine enough for the 2 mm wall.
- Generated the render model from the SolidWorks tessellation of the `Default` configuration, with a paper texture printed with the SuperDex logo under a clear coat.
- Created the Meta-authored processing metadata for this asset in [`intermediates/paper_cup.StudioProcessing.json`](./intermediates/paper_cup.StudioProcessing.json).
- Created reusable single-cup and ten-cup pyramid prefabs with contact settings and placement transforms; the pyramid's cups start at rest. A cup weighs 40 g.
- Made `paper_cup_pyramid_half_built.superdex_prefab`: the pyramid's bottom two rows, with its other three cups nested upright in a stack beside it.

## Physics Tuning

- `tuning/pinch_lift` pinches the wall between an OpenXR index fingertip outside and thumb tip inside, lifts and carries the cup and sets it down; `tuning/pinch_squeeze` squeezes the wall twice as hard. Neither fingertip passes into the wall.
- `tuning/stack` drops four cups into each other (they nest about 18 mm apart) and pulls the top one out again; `tuning/drop` drops a cup tilted onto the ground and one into another from 100 mm.
- `tuning/rest` (a standing, an upside-down and a nested cup stack) and `tuning/rest_pyramid` start at rest: nothing moves more than about 0.1 mm.

## License & IP Notes

The Meta-authored digital assets in this directory are distributed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](./LICENSE). You must give appropriate credit to Meta Platforms, Inc. when you use or redistribute them.

The license applies only to the Meta-authored CAD model, render mesh, collision data, prefab manifests and tuners in this directory, together with this asset's Meta-authored processing metadata at [`intermediates/paper_cup.StudioProcessing.json`](./intermediates/paper_cup.StudioProcessing.json). The license does not grant rights in the underlying physical product design, trade dress, product names, logos, or trademarks, including the SuperDex logo printed on the cup.

These assets were modeled after a commercially available product (the ecotainer Carte Blanc hot cup by Graphic Packaging International). While the 3D geometry is Meta's own work, the underlying product design may be protected by Graphic Packaging International's copyright, design-patent, or trade-dress rights. Accordingly:

- These assets were created for simulation, research, visualization, and software development. CC-BY-4.0 applies only to copyrightable material Meta owns and may license; it does not grant patent, design-patent, trademark, trade-dress, or other rights in the referenced physical product. Manufacturing physical hardware may require separate permission from the applicable rightsholders.
- The Graphic Packaging International, ecotainer, and Carte Blanc names, logos, and product names may not be used to imply endorsement, certification, sponsorship, or an official partnership without prior written permission.

**You are responsible for ensuring your use is compatible with all applicable third-party rights.**
