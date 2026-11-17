# Rubber Duck Prefab

## Overview

This package contains a soft rubber duck toy prefab for use in SuperDex simulations. Its render model is [Rubber Duck Toy](https://polyhaven.com/a/rubber_duck_toy) by Plat251 from Poly Haven, released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/); everything else in this directory was derived from it by Meta. Poly Haven's duck is a 30 cm giant; it is scaled to 30% here, 64 x 90 x 86 mm, the size of a standard bath duck.

## Contents

- `collision/`: the SuperDex Physics tetrahedral mesh, with the render model embedded as its visual mesh and a finer contact skin.
- `intermediates/`: the Studio processing configuration for the collision model.
- `render/`: the render model (GLB) with PBR materials.
- `tuning/`: `superdex_asset_tuner` scenarios used to tune the prefab.
- `rubber_duck.superdex_prefab`: the prefab.

## Modifications and Processing

- Packed Poly Haven's glTF into one GLB, scaled to 30%, with its textures resized to 2048 px and the material's `KHR_materials_specular` and `KHR_materials_ior` extensions dropped.
- Generated the tetrahedral mesh in Studio (`superdex_studio --process`) from the render model. An alpha wrap of it is remeshed to 13.5 mm edges, and Constrained Delaunay tetrahedralization fills that surface, keeping it as the mesh boundary: 171 nodes, 492 tets, with no dihedral angle under 22 deg. The coarse boundary rounds off the beak and tail.
- Embedded the render model as the visual mesh, and a 3 mm remesh of the alpha wrap as the contact skin (`useContactSkin`), so contact follows the duck's real shape and small objects such as fingertips can't slip between the boundary's nodes.

## Physics Tuning

The scenario is in `tuning/`; run it with `superdex_asset_tuner run tuning/drop_press.superdex_tuner` or open it in Studio. `drop_press` drops the duck from 10 cm, then pushes a 30 mm sphere (the [sphere](../sphere/README.md) prefab) down into its head with at most 5 N.

- The duck is a soft vinyl toy, modeled solid: Young's modulus 50 kPa, Poisson ratio 0.3 and 200 kg/m^3 (about 37 g), with friction 0.6. It lands upright, squashes about 5 mm under a 4.6 N press without inverting elements, rocking back on its tail, and springs back.

## Third-Party Material

The render model, its geometry and textures, is [Rubber Duck Toy](https://polyhaven.com/a/rubber_duck_toy) by Plat251 from Poly Haven, released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).

## License

The Meta-authored digital assets in this directory are distributed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](./LICENSE). This license applies only to the Meta-authored digital assets in this directory. You must give appropriate credit to Meta Platforms, Inc. and affiliates when you use or redistribute these assets. The third-party material above remains under CC0 1.0.

**You are responsible for ensuring your use is compatible with all applicable licenses.**
