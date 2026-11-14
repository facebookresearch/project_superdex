# Kitchen Prefab

## Overview

This package contains a Meta-owned kitchen sink scene for use in SuperDex simulations, built from the objects in its subfolders: a counter with an inset sink, a faucet and a soap dispenser in front of a tiled wall with a double light switch, a wall cabinet above, and loose props (a screw-top jar, a sponge and plates). The CAD geometry and all derived digital assets in this directory are the original work of Meta Platforms, Inc. and affiliates. They were modeled in-house; no third-party digital models or geometry were used.

The scene stands on the floor: its origin is on the floor, centered on the back of the backsplash, with +X toward the front. The unit is 1 m wide (Y) and 0.7 m deep (X), the countertop is 1 m above the floor, and the cabinet's top is 2 m up. The fixtures are fixed in the world.

## Contents

- `kitchen.mochi_prefab`: the scene, nesting the object prefabs.
- `cad/kitchen.SLDASM`: the SolidWorks assembly of the scene, made of the objects' assemblies and parts.
- `tuning/`: a `superdex_asset_tuner` scenario that checks the whole scene at rest.
- One folder per object, each a standalone prefab with its own CAD, collision and render models, tuning scenarios, readme and license:
  - [`sink/`](./sink/README.md): the counter, basin, strainer and disposal.
  - [`faucet/`](./faucet/README.md): a single-lever faucet with a swiveling spout.
  - [`dispenser/`](./dispenser/README.md): a soap dispenser with a spring-loaded pump.
  - [`backsplash/`](./backsplash/README.md): the tiled wall section.
  - [`switch/`](./switch/README.md): a double light switch with bistable toggles.
  - [`cabinet/`](./cabinet/README.md): a two-door wall cabinet with a shelf.
  - [`jar/`](./jar/README.md): a glass jar with a screw-on lid.
  - [`sponge/`](./sponge/README.md): a sponge, a rigid grip on a soft pad.
  - [`plate/`](./plate/README.md): a ceramic dinner plate.

## Modifications and Processing

- Modeled the objects in SolidWorks (see each object's readme), assembled each one, and assembled the scene in `cad/kitchen.SLDASM`.
- Each object prefab keeps its CAD frame; `kitchen.mochi_prefab` places them by the same offsets as the assembly. The jar and the sponge stand on the counter, one plate lies in the basin, and three more sit in the cabinet, two stacked on its shelf and one on its floor.

## Physics Tuning

The object prefabs are tuned on their own; see their readmes. `tuning/rest.superdex_tuner` leaves the whole scene alone to check that the props settle where they are placed and no joint drifts.

- Interactions only need contact, joints and prefab constraints: the dispenser's return spring, the switch's over-center springs and the ties between the sponge's grip and pad are constraints. A loader that ignores prefab constraints loses those.
- Every actor's contact penalty coefficient is 1e11 Pa/m, like the other SuperDex prefabs. With Mochi's default (1e9), hands push through the thin moving parts.
- There are no controllers, water, soap or electrics: the faucet, dispenser, switch and disposal move but do nothing.
- The sponge's pad is a soft actor with a textured render model, drawn deformed where soft render models are supported (Studio) and from its simulation mesh elsewhere.

## Third-Party Material

The wood texture in the cabinet is [Oak Veneer 01](https://polyhaven.com/a/oak_veneer_01) by Jenelle van Heerden from Poly Haven, released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). All other textures and materials are Meta-authored.

## License

The Meta-authored digital assets in this directory are distributed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](./LICENSE). This license applies only to the Meta-authored digital assets in this directory. You must give appropriate credit to Meta Platforms, Inc. and affiliates when you use or redistribute these assets.

**You are responsible for ensuring your use is compatible with all applicable licenses.**
