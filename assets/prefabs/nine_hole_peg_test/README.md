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

## Derivations

The following work was done to produce this simulation-ready asset:

- Hand-modeled the pegboard base and peg geometry in SolidWorks using the physical object as a reference; the geometry was not 3D-scanned.
- Exported STEP CAD interchange files and generated GLB render meshes with wood-like visual materials.
- Generated SDF-based Mochi collision models from the CAD-derived meshes using per-part SuperDex Studio processing configurations.
- Assembled a static board and movable peg with contact settings, mass, and placement metadata.

## License & IP Notes

The Meta-authored digital assets in this directory are distributed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](./LICENSE). You must give appropriate credit to Meta Platforms, Inc. when you use or redistribute them.

The license applies only to Meta-authored CAD, render meshes, collision data, processing metadata, and prefab manifests in this directory. It does not grant rights in the underlying physical product design, trade dress, product names, logos, or trademarks.

These assets were modeled after a commercially available product (the Baseline 9-Hole Wooden Pegboard by Fabrication Enterprises, Inc.). While the 3D geometry is Meta's own work, the underlying product design may be protected by Fabrication Enterprises' copyright, design-patent, or trade-dress rights. Accordingly:

- These assets were created for simulation, research, visualization, and software development. CC-BY-4.0 applies only to copyrightable material Meta owns and may license; it does not grant patent, design-patent, trademark, trade-dress, or other rights in the referenced physical product. Manufacturing physical hardware may require separate permission from the applicable rightsholders.
- The Fabrication Enterprises and Baseline names, logos, and product names may not be used to imply endorsement, certification, sponsorship, or an official partnership without prior written permission.

**You are responsible for ensuring your use is compatible with all applicable third-party rights.**
