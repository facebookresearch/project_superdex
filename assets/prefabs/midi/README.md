# MIDI Keyboard Prefab

## Overview

This package contains a Meta-owned 37-key MIDI keyboard controller prefab for use in SuperDex simulations, built as a single articulation: the housing is the root link on a free joint, and every key and control hangs off it on its own joint. The CAD geometry and all derived digital assets in this directory are the original work of Meta Platforms, Inc. and affiliates. They were modeled in-house; no third-party digital models or geometry were used. The design is inspired by, but not identical to, typical desktop MIDI controllers; it is not modeled after any particular product that can be purchased.

The keyboard has 22 white and 15 black keys (three octaves from C3 to C6), a pitch wheel and a mod wheel, a joystick, an endless main knob, four buttons and five sliders. A jack plate on its left end has three 1/4 in jacks, L, R and PHONES, for the [headphones' cable](../headphones/cable/README.md).

## Contents

- `cad/`: the SolidWorks assembly `midi.SLDASM`, its parts, and one STEP interchange model per part.
- `collision/`: SuperDex Physics collision models (SDF) generated in Studio from the STEP models.
- `intermediates/`: the Studio processing configuration for each collision model.
- `render/`: render models (GLB) with PBR materials.
- `tuning/`: `superdex_asset_tuner` scenarios used to tune the prefab.
- `midi.mochi_prefab`: the prefab.

## Modifications and Processing

- Manually authored the keyboard geometry in SolidWorks and assembled it in `cad/midi.SLDASM`. Each white key shape (C, D, E, F, G, A, B) is its own part, plus a full-width top C that ends the keyboard, and the black keys sit in the notches between them. Four rubber feet sit under the housing. The jack plate (`midi_jack_plate`) sits in the bottom back corner of the left end, flush with the end's wood frame: a brushed aluminum plate with a block behind it that fills a pocket in the housing and holds the jacks' 5.5 mm clearance holes.
- Exported one STEP interchange model per part. The housing, the main knob and the mod wheel have a `Collision` configuration, which their STEP models come from: it leaves out the housing's edge fillets and the knob's top fillet, which are for the render models only, and the mod wheel's knurling, too fine to matter for contact.
- Generated the collision models in Studio (`superdex_studio --process`) from the STEP models, rotated from the CAD's Y-up frame into SuperDex's Z-up frame: remeshed to 10 mm edges for the keys, 12 mm for the housing, 6 mm for the other controls and the feet and 3 mm for the jack plate, about 36k triangles for the whole keyboard. The jack plate is its own link so its holes get a 1 mm SDF; the housing's 4.76 mm SDF is far too coarse for them.
- Generated the render models from the SolidWorks tessellation of each part, with materials taken from the CAD appearances, then a near-black top panel and side inlays with horizontal ribs (a normal map generated for this package, 4 mm pitch), the same black without ribs in the panel's wells and slider tracks and on the key bed, and brushed metal wheels, joystick and knob. The jack plate's labels are printed into its texture (generated for this package). The housing's render model has 1 mm fillets on its edges, and the main knob's a 0.75 mm fillet around its top. The four buttons share one shape and have one render model per color.
- Assembled the articulation from the CAD layout. The part meshes keep the CAD layout (keys along X, players' side at -Y). The prefab turns the articulated actor 90 deg about Z and centers its footprint, so in the prefab frame the keyboard sits on the XY plane with Z up, the keys run along Y, and the players' side faces +X.

## Physics Tuning

`midi.mochi_prefab` is tuned with `superdex_asset_tuner`; the scenarios are in `tuning/` (run them with `superdex_asset_tuner run tuning/<name>.superdex_tuner` or open them in Studio). `feel` ramps a load onto one control of each kind and lets go; `slam` hits the controls one after another with about ten times their working load; `rest` leaves the keyboard alone on the ground. `jacks_insert_extract` pushes the cable's plug into each jack and pulls it back out, `jacks_rest` leaves a plug in each, and `phones_cable` drops the cable plugged into PHONES.

- **Keys** hinge at their back bottom edge. A preloaded return spring holds each key against its up stop: a white key starts to move at about 0.54 N at its front edge and bottoms out after 9.5 mm (3.3 deg), and a black key starts at about 0.6 N and bottoms out after 4.3 deg. Keys come back in about 30 ms without bouncing.
- **Buttons** travel 3.5 mm and need about 1 N to start moving.
- **Sliders** travel +-55 mm and have no spring: about 0.4 N of dry friction plus viscous damping holds them wherever they are left, even on the 15 deg panel.
- **Main knob** turns endlessly and needs about 6 mN m to turn.
- **Pitch wheel** turns +-40 deg and springs back to center. **Mod wheel** turns +-45 deg and its friction (about 12 mN m) holds it where it is left.
- **Joystick** tilts +-25 deg in both directions on a spherical joint, springs back to center, and cannot twist.
- The return springs are `articulatedSingleDofTarget` and `articulated3dRotationTarget` constraints; joint friction and joint limits do the rest. A loader that ignores prefab constraints (the Unreal prefab importer does today) loses the return springs.
- Keys and controls are on the `midi_controls` layer, which does not collide with itself: the controls only move along their joints, so contact between them would only cost solver time. Hands and props still touch them, and Mochi already disables contact between each control and the housing.
- **Jacks** are the cable's jack: its socket's collision models (bore and detent ridge) and render model (white for L, red for R, gold for PHONES), with the headphones' jack's contact settings, hard-jointed to the jack plate. The plug pushes in at about 14 N and pulls out at about 21 N in each (`jacks_insert_extract`, 15 mm/s), as in the headphones. A seated plug sags about 0.6 mm in the jack's play, and a cable plugged into PHONES stays in when dropped (`jacks_rest`, `phones_cable`). The sockets have no contact points of their own (`boundarySubsampling` density 0), so only a plug's points touch them, also in Unreal, whose Blueprint import drops layer filters.
- The keyboard weighs about 6 kg and stands on four high-friction rubber feet, hard-jointed to the housing.

## Third-Party Material

The wood texture embedded in `render/midi_housing.glb` is [Teak Veneer](https://polyhaven.com/a/teak_veneer) by Jenelle van Heerden from Poly Haven, released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). The jack plate's labels are set in DejaVu Sans Bold ([DejaVu Fonts License](https://dejavu-fonts.github.io/License.html)).

## License

The Meta-authored digital assets in this directory are distributed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](./LICENSE). This license applies only to the Meta-authored digital assets in this directory. You must give appropriate credit to Meta Platforms, Inc. and affiliates when you use or redistribute these assets.

**You are responsible for ensuring your use is compatible with all applicable licenses.**
