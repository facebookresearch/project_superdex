# Kitchen Sponge Prefab

## Overview

This package contains a Meta-owned kitchen sponge prefab for use in SuperDex simulations, part of the [kitchen](../README.md) scene. The CAD geometry and all derived digital assets in this directory are the original work of Meta Platforms, Inc. and affiliates. They were modeled in-house; no third-party digital models or geometry were used.

The sponge is a rigid plastic grip (114 x 66 mm, the 4.5 in length of a standard kitchen sponge, with finger grooves along its sides) on top of a soft foam pad, 23 mm thick. The pad hangs from the grip by node-to-rigid constraints.

## Contents

- `cad/`: the SolidWorks assembly `sponge.SLDASM`, its parts (`sponge_grip`, `sponge_pad`), and one STEP interchange model per part.
- `collision/`: SuperDex Physics collision models generated in Studio from the STEP models (a tetrahedral mesh for the soft actor, SDFs for the rest).
- `intermediates/`: the Studio processing configuration for each collision model.
- `render/`: render models (GLB) with PBR materials.
- `tuning/`: `superdex_asset_tuner` scenarios used to tune the prefab.
- `sponge.mochi_prefab`: the prefab.
- `textures/`: the source textures of the pad's foam (base color, metallic-roughness and normal maps for one 40 mm tile).

## Modifications and Processing

- Modeled the grip and the pad in SolidWorks (the pad from the grip's footprint), scaled both to 76% about the center of their bottom face, and assembled them in `cad/sponge.SLDASM`, the grip on top.
- Exported one STEP AP214 interchange model per part.
- Generated the grip's collision model (SDF) and the pad's tetrahedral mesh in Studio (`superdex_studio --process`). For the pad, an alpha wrap of its STEP model is remeshed to 10 mm edges, and Constrained Delaunay tetrahedralization fills that surface, keeping it as the mesh boundary: about 450 nodes, with no dihedral angle under 20 deg.
- Generated the grip's render model from the SolidWorks tessellation.
- Made the pad's render model from Studio's 4 mm remesh of its STEP model: one primitive, with box-projected UVs and a procedural open-cell foam texture generated for this package (`textures/`), 40 mm per tile. Studio's processing embeds that render model in the tetrahedral mesh as the pad's visual mesh, so a renderer that supports it (Studio) draws the textured pad deformed by the simulation; others draw the simulation mesh. Regenerate the collision model whenever the render model changes.

## Physics Tuning

The scenarios are in `tuning/`; run them with `superdex_asset_tuner run tuning/<name>.superdex_tuner` or open them in Studio. `lift` lifts the sponge by its grip, waves it, and presses it down. `pinch` pinches the pad between two OpenXR index fingertips and drags it while the grip is held (`pinch.superdex_prefab` adds the fingertips).

- **Attachment**: every node of the pad within 5 mm of its top face, interior ones included, is tied to the grip where it is (`deformableNodeToRigid` by node index, `fixToDeformablePos`), about 115 constraints sharing 200 kN/m and 200 N s/m. The pad hangs from a solid slab, so no single tie point shows when it is pulled. The node indices are those of `collision/sponge_pad.mochi.h5`: remeshing the pad means regenerating them. The grip floats 1 mm above the pad so the two never collide. A loader that ignores prefab constraints drops the pad.
- The pad is polyurethane foam: Young's modulus 25 kPa (as in the original Unreal scene), Poisson ratio 0.2 and 60 kg/m^3 (about 9 g). Softer foam lets fingers crush and invert its elements. A 30 N press on the grip (about 24 g) squashes the pad about 4 mm, and it springs back. Two fingertips pinching it with 5 N each dent it about 15 mm without splintering; a few elements under the fingertips briefly invert at the deepest pinch and recover.

## Third-Party Material

None. All textures and materials in this directory are Meta-authored.

## License

The Meta-authored digital assets in this directory are distributed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](./LICENSE). This license applies only to the Meta-authored digital assets in this directory. You must give appropriate credit to Meta Platforms, Inc. and affiliates when you use or redistribute these assets.

**You are responsible for ensuring your use is compatible with all applicable licenses.**
