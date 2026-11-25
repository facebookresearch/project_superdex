# Headphones Prefab

## Overview

This package contains Meta-owned over-ear studio headphones for use in SuperDex simulations, part of the [headphones](../README.md) set. The CAD geometry and all derived digital assets in this directory are the original work of Meta Platforms, Inc. and affiliates. They were modeled in-house; no third-party digital models or geometry were used. The design is inspired by, but not identical to, typical closed-back studio headphones; it is not modeled after any particular product, and carries no product logos (the cups' caps carry an embossed SuperDex hand).

The headphones are one articulation: the band is split at its apex into two links on a sprung pivot, and each cup swivels on its yoke. The padded underside of the band is an articulation skin, and the ear pads are soft actors tied to the cups. The left cup has a 1/4 in jack for the [cable](../cable/README.md).

## Contents

- `cad/`: the SolidWorks parts `hp_band` (configurations `Left` and `Right`), `hp_band_pad`, `hp_cup` (`Default`, the left cup with the jack, and `Right` without it; `LeftCollision` and `RightCollision`), `hp_pad` and `head_proxy` (a head for the tuners, and the display head of the set one folder up), and their STEP interchange models.
- `collision/`: SuperDex Physics collision models generated in Studio (SDFs, the pads' tetrahedral mesh and the band's skin).
- `intermediates/`: the Studio processing configuration for each collision model, and the band padding's surface before skinning.
- `render/`: render models (GLB) with PBR materials.
- `tuning/`: `superdex_asset_tuner` scenarios used to tune the prefab.
- `headphones.superdex_prefab`: the headphones, lying on their back.

## Modifications and Processing

- Modeled every part in SolidWorks in the frame of the link it belongs to. The cups' `Collision` configurations leave out the cap ring and disc and the speaker grille, and widen the jack hole to a clearance hole the jack's bore fills (the jack's bore and detent come from the cable's `jack_socket`).
- Generated the collision models in Studio from the STEP models: 3 mm edges and 1 mm SDFs for the band and cups. The pad's alpha wrap is remeshed to 13 mm edges and filled with 421 tetrahedra; its render model is embedded as its visual mesh.
- Skinned the band padding's 4 mm Studio surface to the two band links for the skin's collision model, blending smoothly across its middle 120 mm so the padding bends evenly as the band opens; its render model is skinned the same way (joints `band_l`, `band_r`).
- Generated the render models from the SolidWorks tessellation: black leatherette (a grain normal map generated for this package), matte and satin black plastic, brushed steel sliders and a silver cap ring. The cap disc's embossed hand is a normal and roughness map made from the SuperDex logo. The jack's socket is gold: the left cup's render model has the socket's clearance hole in place of its bore (cut into the tessellation, as the CAD's `JackClearance` cuts it), and the `jack_bore` link draws the socket (`../cable/render/jack_socket.glb`).
- Generated the head's render model from Studio's fine tessellation of its STEP model, its normals smoothed across the stacked cones of its dome: teak under a gloss polyurethane clear coat, a satin base of Poly Haven's Teak Veneer, graded darker and richer as a film finish leaves wood, under a `KHR_materials_clearcoat` coat. The grain is mapped as if the head were turned from a solid blank, with no seams: straight vertical grain front and back, flame figure on the sides and end grain across the crown.

## Physics Tuning

The scenarios are in `tuning/`; run them with `superdex_asset_tuner run tuning/<name>.superdex_tuner` or open them in Studio.

- **Band**: the right half pivots on the left at the apex. A spring (`articulatedSingleDofTarget`) holds it shut on a hard stop with about 1.6 N between the cups, so it pulls apart to go over a head but doesn't compress; about 3 N opens it 30 mm, a 150 mm head (`band_spring`).
- **Swivels**: each cup swivels +-90 deg from the worn pose to hard stops, with enough friction and damping that it turns only when pushed (`swivel`).
- **Pads**: 30 kPa foam, 12 g each, their 32 base nodes tied to the cup's backplate (`deformableNodeToRigid`, 5 kN/m each); they don't collide with their own cup. `pads_on_head` lowers the headphones onto a head: the pads push the band open about 25 mm and squash, and the headphones stay on when let go. Worn on this head, the cups reach below its base, so the tuner has no ground. A loader that ignores prefab constraints drops the pads and the band spring.
- **Jack**: the plug pushes in at about 14 N and pulls out at about 21 N with the headphones held by a cup (`plug_push_in`, `plug_pull_out`). `lift_by_cable` lifts the headphones off the table by the far plug: they hang and swing from the cable, and the plug stays in.
- Every link collides except the jack's two, which only push the plug out (`nonCollidingLinks` on the band's skin; unset, only the skin would collide). Links that touch where they're joined don't collide with each other. The headphones weigh about 280 g.

## Third-Party Material

The wood texture embedded in `render/head_proxy.glb` is derived from [Teak Veneer](https://polyhaven.com/a/teak_veneer) by Jenelle van Heerden from Poly Haven, released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). All other textures and materials in this directory are Meta-authored.

## License

The Meta-authored digital assets in this directory are distributed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](./LICENSE). This license applies only to the Meta-authored digital assets in this directory. You must give appropriate credit to Meta Platforms, Inc. and affiliates when you use or redistribute these assets.

**You are responsible for ensuring your use is compatible with all applicable licenses.**
