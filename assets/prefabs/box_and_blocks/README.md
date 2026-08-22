# Box and Block Test Asset

## Overview

This package contains a Box and Block Test simulation asset that was created in-house by modeling after a [commercially available Sammons Preston Box and Block Test kit](https://www.amazon.com/dp/B07BHXTVWC), a physical object purchased from [Amazon.com](https://www.amazon.com/). No third-party digital source files were used; the geometry in this directory is Meta's own work.

The physical product is sold under the Sammons Preston brand. Its manufacturer was not identified in the available provenance materials.

## Reference Object

| Field | Value |
| --- | --- |
| Product name | Sammons Preston Box and Block Test kit |
| Source / purchase link | [Amazon product B07BHXTVWC](https://www.amazon.com/dp/B07BHXTVWC) |
| Vendor | [Amazon.com](https://www.amazon.com/) |
| Brand | Sammons Preston |
| Manufacturer | Not identified in the source materials |

## Derivations

The following work was done to produce this simulation-ready asset:

- Hand-modeled the divided box and block geometry in SolidWorks using the physical object as a reference; the geometry was not 3D-scanned.
- Exported STEP CAD interchange files and generated GLB render meshes, including separate colored block variants.
- Generated SDF-based Mochi collision models from the CAD-derived meshes using the recorded SuperDex Studio processing configurations.
- Assembled the static box and colored block prefabs with simulation transforms, contact settings, and visual materials.

## License & IP Notes

The Meta-authored digital assets in this directory are distributed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](./LICENSE). You must give appropriate credit to Meta Platforms, Inc. when you use or redistribute them.

The license applies only to Meta-authored CAD, render meshes, collision data, processing metadata, and prefab manifests in this directory. It does not grant rights in the underlying physical product design, trade dress, product names, logos, or trademarks.

These assets were modeled after a commercially available Box and Block Test kit. While the 3D geometry is Meta's own work, the underlying product design may be protected by copyright, design-patent, or trade-dress rights. Accordingly:

- These assets were created for simulation, research, visualization, and software development. CC-BY-4.0 applies only to copyrightable material Meta owns and may license; it does not grant patent, design-patent, trademark, trade-dress, or other rights in the referenced physical product. Manufacturing physical hardware may require separate permission from the applicable rightsholders.
- Any manufacturer, vendor, brand, logo, or product name associated with the reference object may not be used to imply endorsement, certification, sponsorship, or an official partnership without prior written permission from the applicable rightsholder.

**You are responsible for ensuring your use is compatible with all applicable third-party rights.**
