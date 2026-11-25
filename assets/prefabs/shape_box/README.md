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
- The holes were drawn as their pieces offset outward by 1 mm (their corners are 1 mm rounds). Each piece is its hole offset back inward by exactly 1 mm, so every piece has a uniform 1 mm clearance in its hole.
- Exported STEP CAD interchange files and generated GLB render meshes. The pieces' paints are tinted to the product's colors; the plywood has clear-coated faces and shows its plies, darker and matte, on the cut edges, the hole walls and the box-joint fingers at the corners.
- Generated SDF-based Mochi collision models from the STEP files with per-part SuperDex Studio processing configurations: about 12k triangles in all, the pieces 470 to 830 each. The hole walls and the lid slot are colored in the CAD so Studio meshes them finer than the rest of the box, and the body's collision model comes from a `Collision` configuration without its 0.5 mm edge rounds, which Studio could not mesh watertight.
- Assembled the body, lid, and 12 movable shape pieces into a prefab, the pieces resting in their holes or on the floor of the box.
- Made `shape_box_four_out.superdex_prefab`: the same box with the easiest piece for each face (ellipse, square, octagon, hexagon) standing on the table in front of it.

## Physics Tuning

`tuning/` holds `superdex_asset_tuner` scenarios: `rest` (the prefab settles), `insert_*` (each piece pushed through its hole, one face at a time), `lid` (the lid slid out of its slot and back), and `walls_press` / `walls_throw` (pieces pressed with 300 N and thrown at the walls and the lid). Pieces slide through their holes without catching and nothing passes through a wall or the lid. Thin parts have a speed limit, as there is no continuous collision detection: at 5 ms steps a piece thrown at the 4.9 mm lid faster than about 3 m/s starts to sink into it, and at 10 ms steps a piece at 10 m/s can pass through a 10 mm wall.

## Third-Party Material

The plywood edge texture embedded in `render/shape_box_body.glb` and `render/shape_box_lid.glb` is derived from [Plywood](https://polyhaven.com/a/plywood) by Rob Tuytel from Poly Haven, released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).

## License & IP Notes

The Meta-authored digital assets in this directory are distributed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](./LICENSE). You must give appropriate credit to Meta Platforms, Inc. when you use or redistribute them.

The license applies only to Meta-authored CAD, render meshes, collision data, processing metadata, and prefab manifest in this directory. It does not grant rights in the underlying physical product design, trade dress, product name, logos, or trademarks.

These assets were modeled after a commercially available product (the Shape Sorting Cube by Melissa & Doug). While the 3D geometry is Meta's own work, the underlying product design may be protected by Melissa & Doug's copyright, design-patent, or trade-dress rights. Accordingly:

- These assets were created for simulation, research, visualization, and software development. CC-BY-4.0 applies only to copyrightable material Meta owns and may license; it does not grant patent, design-patent, trademark, trade-dress, or other rights in the referenced physical product. Manufacturing physical hardware may require separate permission from the applicable rightsholders.
- The Melissa & Doug name, logos, and product names may not be used to imply endorsement, certification, sponsorship, or an official partnership without prior written permission.

**You are responsible for ensuring your use is compatible with all applicable third-party rights.**
