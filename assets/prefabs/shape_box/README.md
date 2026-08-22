# Shape Sorting Box Asset

## Overview

This package contains a simulation-ready shape-sorting-box asset that was created in-house by modeling after the [Melissa & Doug Shape Sorting Cube — Classic Wooden Toy With 12 Shapes](https://www.amazon.com/dp/B00005RF5G), a physical object purchased from [Amazon.com](https://www.amazon.com/). No third-party digital source files were used; the geometry in this directory is Meta's own work.

## Reference Object

| Field | Value |
| --- | --- |
| Product name | Melissa & Doug Shape Sorting Cube — Classic Wooden Toy With 12 Shapes |
| Source / purchase link | [Amazon product B00005RF5G](https://www.amazon.com/dp/B00005RF5G) |
| Vendor | [Amazon.com](https://www.amazon.com/) |
| Brand / manufacturer | Melissa & Doug |

## Derivations

The following work was done to produce this simulation-ready asset:

- Hand-modeled the cube body, lid, and 12 shape pieces in SolidWorks using the physical object as a reference; the geometry was not 3D-scanned.
- Exported STEP CAD interchange files and generated GLB render meshes with differentiated visual materials.
- Generated SDF-based Mochi collision models from the CAD-derived meshes using per-part SuperDex Studio processing configurations.
- Assembled the body, lid, and 12 movable shape pieces into a prefab with placement transforms, contact settings, density values, and simulation metadata.

## License & IP Notes

The Meta-authored digital assets in this directory are distributed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](./LICENSE). You must give appropriate credit to Meta Platforms, Inc. when you use or redistribute them.

The license applies only to Meta-authored CAD, render meshes, collision data, processing metadata, and prefab manifest in this directory. It does not grant rights in the underlying physical product design, trade dress, product name, logos, or trademarks.

These assets were modeled after a commercially available product (the Shape Sorting Cube by Melissa & Doug). While the 3D geometry is Meta's own work, the underlying product design may be protected by Melissa & Doug's copyright, design-patent, or trade-dress rights. Accordingly:

- These assets were created for simulation, research, visualization, and software development. CC-BY-4.0 applies only to copyrightable material Meta owns and may license; it does not grant patent, design-patent, trademark, trade-dress, or other rights in the referenced physical product. Manufacturing physical hardware may require separate permission from the applicable rightsholders.
- The Melissa & Doug name, logos, and product names may not be used to imply endorsement, certification, sponsorship, or an official partnership without prior written permission.

**You are responsible for ensuring your use is compatible with all applicable third-party rights.**
