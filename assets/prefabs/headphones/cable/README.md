# Audio Cable Prefab

## Overview

This package contains a Meta-owned 1 m audio cable with 1/4 in (6.35 mm) TRS plugs at both ends, and a test block with a matching jack, for use in SuperDex simulations, part of the [headphones](../README.md) set. The CAD geometry and all derived digital assets in this directory are the original work of Meta Platforms, Inc. and affiliates. They were modeled in-house; no third-party digital models or geometry were used. The plug's pin follows the standard 1/4 in TRS profile; the cable is not modeled after any particular product.

The cable is a rod (`cable`, 855 mm, 6 mm thick) with a rigid plug actor tied to each end (`plug_a`, `plug_b`). A plug held in a jack is held by contact alone, as in a real jack: a ridge in the jack sits in the groove behind the plug's tip.

## Contents

- `cad/`: the SolidWorks parts `cable_plug`, `jack_socket` and `jack_test_block`, and their STEP interchange models.
- `collision/`: SuperDex Physics collision models generated in Studio (SDFs, and the cable's polyline with its tubes).
- `intermediates/`: the Studio processing configuration for each collision model, and the cable's bare centerline.
- `render/`: render models (GLB) with PBR materials.
- `tuning/`: `superdex_asset_tuner` scenarios used to tune the prefabs.
- `cable.superdex_prefab`: the cable, lying straight on the ground.
- `cable_plug.superdex_prefab`: one plug, nested twice in the cable.
- `jack_test.superdex_prefab`: the jack test block and a loose plug, for tuning the plug and jack on their own.

## Modifications and Processing

- Modeled the plug in SolidWorks, its pin profile measured off a manufacturer's 1/4 in TRS plug drawing (6.29 mm sleeve, 30.3 mm from the tip to the seating shoulder) on a 12.7 mm chrome barrel. Its `Collision` configuration leaves out the insulator rings, the knurled bands and the cable bore, which are for the render model only. The pin's faces are colored so Studio meshes them at 0.35 mm, the barrel at 1.5 mm.
- Modeled the jack as `jack_socket`: configurations `Bore` (a 6.8 mm bore, 33 mm deep, with a chamfered mouth) and `Detent` (the retaining ridge), one collision model each. `jack_test_block` is a steel block holding the jack's bore, with a brass insert around its mouth; its `Collision` configuration has a clearance hole the `Bore` fills.
- Generated the cable's tubes in Studio from its centerline (`intermediates/cable_centerline.mochi.h5`, 86 elements of 10 mm): a 6 mm contact skin and visual mesh. Its render model is a tube on the visual mesh's rings, textured with orange heat-shrink ends printed "SuperDex" and a black jacket printed "SuperDex  MADE IN THE USA". Studio draws rods with their visual mesh, without the render model's texture, and the Unreal importer leaves rods out for now.
- Generated the other render models from the SolidWorks tessellation, the knurl a normal map generated for this package. `render/jack_socket.glb`, the socket's bore and detent in gold for the headphones' jack (and the MIDI keyboard's PHONES jack), is Studio's tessellation of their STEP models; `jack_socket_white.glb` and `jack_socket_red.glb` are the same in white and red, for the keyboard's L and R jacks.

## Physics Tuning

The scenarios are in `tuning/`; run them with `superdex_asset_tuner run tuning/<name>.superdex_tuner` or open them in Studio.

- **Plug and jack**: the jack's bore and detent are links hard-jointed to the block (the headphones hang the same ones off their left cup), so the detent gets its own contact: a 0.3 mm threshold and 0.2 mm smoothing let it give like a jack's tip spring, and its friction is low (0.02), which sets how hard the plug pulls out. Its steep entry flank and long exit flank make the plug easier to push in than to pull out: about 13 N in and 22 N out pushed and pulled slowly (`jack_test_push_in`, `jack_test_pull_out`), 16 N and 22 N at 15 mm/s (`jack_test_insert_extract`), within the 10 to 30 N of typical 1/4 in plug specifications. `jack_test_hang` hangs 3 N from a plug in a face-down jack: it stays in. The jack's parts are on the `jack_socket` layer, which does not check for contact against plugs (`audio_plug`), so a plug only meets them through its own samples; every penalty is the same, so the plug behaves the same in a static or a moving jack. Their bore and detent also have no contact points of their own (`boundarySubsampling` density 0), so only the plug's points touch them even where layer filters are dropped, as in Unreal's Blueprint import.
- **Cable**: 40 g/m, flexural stiffness 2e-3 N m^2 (a 20 cm end droops about 4 cm), its contact skin rubbery (friction 0.6). Each end node is tied to its plug where it starts (`deformableNodeToRigid`) and each end element turns with it (`rodElementRotationToRigid`). `cable_lift` lifts it by one plug; `cable_rest` leaves it on the ground. A loader that ignores prefab constraints drops the plugs.
- Each plug weighs 30 g.

## Third-Party Material

The cable's jacket text is set in DejaVu Sans Bold ([DejaVu Fonts License](https://dejavu-fonts.github.io/License.html)). All other textures and materials in this directory are Meta-authored.

## License

The Meta-authored digital assets in this directory are distributed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](./LICENSE). This license applies only to the Meta-authored digital assets in this directory. You must give appropriate credit to Meta Platforms, Inc. and affiliates when you use or redistribute these assets.

**You are responsible for ensuring your use is compatible with all applicable licenses.**
