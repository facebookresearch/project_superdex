# Fidget Box Description

## Overview

This package contains the Fidget Box for SuperDex Robotics. The CAD geometry and all derived digital assets in this directory are the original work of Meta Platforms, Inc. and affiliates. No third-party digital model or geometry was used.

## Derivations

The following work was done to produce the simulation-ready asset:

- Authored the source CAD geometry in `cad/`.
- Generated the GLB render meshes in `render/` from the CAD geometry.
- Used SuperDex Studio to import the CAD geometry, refine the derived meshes, and generate the HDF5 collision models in `collision/`; `intermediates/` records the Studio processing configuration.
- Assembled `fidget_box.superdex_bot` from the render and collision assets, then exported as a prefab.

## License

The CAD geometry, render meshes, collision data, Studio processing metadata, and bot description in this directory are the original work of Meta Platforms, Inc. and affiliates and are distributed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](./LICENSE). This license applies only to those asset files.

Suggested attribution: “Fidget Bot assets © Meta Platforms, Inc. and affiliates, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); [source](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/fun/fidget_bot).” Indicate whether you modified the assets.

**You are responsible for ensuring your use is compatible with all applicable licenses.**
