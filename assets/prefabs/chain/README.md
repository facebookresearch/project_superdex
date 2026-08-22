# Chain Prefab

## Overview

This package contains a Meta-owned chain prefab for use in SuperDex simulations. The digital geometry is the original work of Meta Platforms, Inc. and affiliates. It was modeled in-house; no third-party digital source files were used.

## Modifications and Processing

- Manually authored the chain-link geometry in SolidWorks and exported a STEP interchange model.
- Generated render geometry from the source CAD. The two checked-in GLB files are byte-identical, and `chain.mochi_prefab` uses `chain_link_10_5_2_metalic.glb` as its render model.
- Generated collision geometry for SuperDex Physics.
- Assembled repeated chain links into `chain.mochi_prefab`.

## License

The Meta-authored digital assets in this directory are distributed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](./LICENSE). This license applies only to the Meta-authored digital assets in this directory. You must give appropriate credit to Meta Platforms, Inc. and affiliates when you use or redistribute these assets.

**You are responsible for ensuring your use is compatible with all applicable licenses.**
