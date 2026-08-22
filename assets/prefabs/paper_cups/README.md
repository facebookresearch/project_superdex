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

- Physically scanned the real cup on a 2D scanner, then hand-modeled the cup geometry in SolidWorks using the scan and physical object as references.
- Converted the CAD work into a GLB render mesh and a Mochi collision model for the released package.
- Created the Meta-authored processing metadata for this asset in [`../intermediates/paper_cup.StudioProcessing.json`](../intermediates/paper_cup.StudioProcessing.json).
- Adjusted the visual appearance for rendering, including a reference-inspired cup label; associated product names and marks remain the property of their respective owners.
- Created reusable single-cup and ten-cup pyramid prefabs with contact settings and placement transforms.

## License & IP Notes

The Meta-authored digital assets in this directory are distributed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](./LICENSE). You must give appropriate credit to Meta Platforms, Inc. when you use or redistribute them.

The license applies only to the Meta-authored render mesh, collision data, and prefab manifests in this directory, together with this asset's Meta-authored processing metadata at [`../intermediates/paper_cup.StudioProcessing.json`](../intermediates/paper_cup.StudioProcessing.json). Its inclusion does not extend this package's license to any other content in `../intermediates`. The license does not grant rights in the underlying physical product design, trade dress, product label, product names, logos, or trademarks.

These assets were modeled after a commercially available product (the ecotainer Carte Blanc hot cup by Graphic Packaging International). While the 3D geometry is Meta's own work, the underlying product design may be protected by Graphic Packaging International's copyright, design-patent, or trade-dress rights. Accordingly:

- These assets were created for simulation, research, visualization, and software development. CC-BY-4.0 applies only to copyrightable material Meta owns and may license; it does not grant patent, design-patent, trademark, trade-dress, or other rights in the referenced physical product. Manufacturing physical hardware may require separate permission from the applicable rightsholders.
- The Graphic Packaging International, ecotainer, and Carte Blanc names, logos, and product names may not be used to imply endorsement, certification, sponsorship, or an official partnership without prior written permission.

**You are responsible for ensuring your use is compatible with all applicable third-party rights.**
