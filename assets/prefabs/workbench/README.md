# Workbench Prefab

## Overview

This package contains a Meta-owned workbench prefab for use in SuperDex simulations: a light oak butcher-block top on a white powder-coated steel frame, 1.5 m wide (Y), 0.6 m deep (X) and 0.9 m tall. The digital geometry is the original work of Meta Platforms, Inc. and affiliates. It was modeled in-house; no third-party digital models or geometry were used.

## Modifications and Processing

- Modeled the bench in SolidWorks (`cad/workbench.SLDPRT`). The apron under the top is solid, so the top's collision model is 100 mm thick. The `Collision` configuration drops the rounded edges and is exported as the STEP model the collision model is generated from, so the top's collision edges are square.
- Generated the collision model in Studio: about 1,000 triangles, sized by face color (finer on the top), with a 10 mm SDF whose grid lines up with the top's faces so its edges stay sharp.
- Generated the render model from the SolidWorks tessellation of the `Default` configuration.
- Assembled the asset as `workbench.superdex_prefab`, a static actor with default contact parameters and its origin on the floor under the center of the top.

## Physics Tuning

`tuning/drop`, `tuning/throw` and `tuning/press` drop, throw (10 m/s) and press (100 to 500 N) objects from other SuperDex prefabs onto the top. None passes through it; the deepest sinks about 3 mm into it.

## Third-Party Material

The oak texture in `render/workbench.glb` is derived from [Oak Veneer 05](https://polyhaven.com/a/oak_veneer_05) by Jenelle van Heerden from Poly Haven, released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).

## License

The Meta-authored digital assets in this directory are distributed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](./LICENSE). This license applies only to the Meta-authored digital assets in this directory. You must give appropriate credit to Meta Platforms, Inc. and affiliates when you use or redistribute these assets.

**You are responsible for ensuring your use is compatible with all applicable licenses.**
