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

## Derivations

The following work was done to produce this simulation-ready asset:

- Hand-modeled the pegboard and peg geometry in SolidWorks using the physical object as a reference; the geometry was not 3D-scanned.
- Exported STEP CAD interchange files and generated GLB render meshes with a wood-like visual appearance.
- Generated SDF-based Mochi collision models from the CAD-derived meshes using per-part SuperDex Studio processing configurations.
- Assembled a static pegboard and reusable peg prefab with placement transforms, contact parameters, and simulation metadata.

## License & IP Notes

The Meta-authored digital assets in this directory are distributed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](./LICENSE). You must give appropriate credit to Meta Platforms, Inc. when you use or redistribute them.

The license applies only to Meta-authored CAD, render meshes, collision data, processing metadata, and prefab manifests in this directory. It does not grant rights in the underlying physical product design, trade dress, product names, logos, or trademarks.

These assets were modeled after a commercially available product (the Functional Dexterity Test by North Coast Medical). While the 3D geometry is Meta's own work, the underlying product design may be protected by North Coast Medical's copyright, design-patent, or trade-dress rights. Accordingly:

- These assets were created for simulation, research, visualization, and software development. CC-BY-4.0 applies only to copyrightable material Meta owns and may license; it does not grant patent, design-patent, trademark, trade-dress, or other rights in the referenced physical product. Manufacturing physical hardware may require separate permission from the applicable rightsholders.
- The North Coast Medical name, logos, and product names may not be used to imply endorsement, certification, sponsorship, or an official partnership without prior written permission.

**You are responsible for ensuring your use is compatible with all applicable third-party rights.**
