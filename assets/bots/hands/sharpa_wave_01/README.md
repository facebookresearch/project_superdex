# Sharpa Wave (wave_01) Description

These bots are a port of the upstream MJCF. They keep the upstream values exactly wherever SuperDex
can represent them. The values that it cannot represent are listed under [Deviations](#deviations).
Task labs that use them: `baoding/baoding_sharpa_tilt_fixed` (left) and
`inhand_rotation/inhand_rotation_sharpa_wmap` (right), both under `tasks/internal/`.

## Overview

This package contains the robot description of the Sharpa Wave (wave_01) dexterous hand, right and
left, with 22 actuated finger joints per side. It is converted from the MJCF and STL assets of the
upstream [sharpa-urdf-usd-xml](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml) repository
at commit
[`d323b371d02f4fc04a0873c19c5234cae6d803dd`](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml/tree/d323b371d02f4fc04a0873c19c5234cae6d803dd):
`wave_01/right_sharpa_wave/right_sharpa_wave.xml` and `wave_01/left_sharpa_wave/left_sharpa_wave.xml`
with their `meshes/`.

| Path | Contents |
| --- | --- |
| `<side>/sharpa_wave_01_<side>.superdex_bot` | The bot |
| `<side>/upstream/<side>_sharpa_wave.xml` | The upstream MJCF, byte for byte (the meshes are not included) |
| `<side>/render/*.glb` | One render model per body: the body's visual geoms merged |
| `<side>/collision/*.mochi.h5` | One collision model per collider (see the elastomer pads below) |
| `<side>/cad/*.stl` | The upstream collision STL of each collider, byte for byte, renamed after the collider |
| `<side>/intermediates/*.StudioProcessing.json` | The recipe that builds each collision model from its `cad/` STL |

## Provenance and hash check

The assets were taken from a copy of mjlab_hand (`src/mjlab_hand/robot/sharpa_wave/`). Its
`asset_manifest.json` pins every asset to the upstream repository and commit above with a sha256.
Before any file was used, all 52 files in the manifest were checked against it: 52 matched and none
was missing or unlisted. The files kept here match these manifest entries:

| File here | Upstream file | sha256 |
| --- | --- | --- |
| `right/upstream/right_sharpa_wave.xml` | `wave_01/right_sharpa_wave/right_sharpa_wave.xml` | `43d9cb63d724889b69574a5e0981aee4a2f30d825c85f3098988e3a7a3bb9980` |
| `left/upstream/left_sharpa_wave.xml` | `wave_01/left_sharpa_wave/left_sharpa_wave.xml` | `3cbeb46259d4ba63cbdb83085255d1a8f8031c51e0101a6622f6e7e81a64dc11` |
| first six lines of `LICENSE` | `LICENSE.txt` | `f6a348fca3d77ddf60e66a3288b060e09b12afdebeabc6fc490b272ecfa10e43` (upstream, CRLF line endings; kept here with LF endings, which hash to `ae93e4fe3d2ea28dfcd49cee9307cf2d8b7e9f7467133ec55d92d76c0a519720`) |
| `<side>/cad/*.stl` | the matching `wave_01/<side>_sharpa_wave/meshes/*.STL` | each matches its manifest entry |

The hashes were not checked against GitHub itself, because the host's proxy refused connections
to github.com (HTTP 403).

## How the bots were made

1. SuperDex Studio's MJCF importer, driven headless over its MCP bridge (the `import` app
   command), with `world_joint=free`, `render_source=visual`, `render_operation=convert`,
   `collision_source=collision`, `collision_operation=separate`, `remesh=true`, `bake_sdf=true`
   and `keep_recipes=true`.
2. Every collision recipe was rebuilt with `superdex_studio --process` at SDF `resolutionDelta`
   1.0, the Wuji Hand 2 bots' setting, in place of the importer's default of 0.25. At the default
   setting each model is 3-7 MB (118 MB per side); at 1.0 each is 100-180 kB.
3. The importer names each pad link `<side>_<finger>_DP_collision_1`. These links were renamed to
   the upstream geom names, `<side>_<finger>_elastomer` (joint `<side>_<finger>_elastomer_fixed`),
   by text replacement. Nothing else in the bot files was edited.

## Modifications

- Each body became a link, named after the body. Each hinge kept its name, axis and range.
- Collision geometry was re-authored: each upstream collision mesh was wrapped, remeshed and
  baked into a signed distance field for SuperDex Physics.
- Visual geometry was converted: each body's visual STLs were merged into one GLB, keeping the
  upstream `rgba`.
- Each fingertip's elastomer pad got a link of its own, Hard-jointed to the distal phalanx, so that
  the pads keep their own collision geometry. Upstream, the pads are task keypoints, matched by
  `(left|right)_(thumb|index|middle|ring|pinky)_elastomer`.
- A Free root joint, `world_joint`, was added, as on the other SuperDex hand bots.
- The MJCF's actuators are not carried over: SuperDex labs define the actuators.

## Deviations

These are the upstream values that the bots do not represent exactly. "Lab" means that a task lab
has to set the value.

| # | Upstream value | In these bots |
| --- | --- | --- |
| D1 | `elastomer_geom` class: `solref="0.06 0.9"` on the pad collision geoms (right hand). On the left hand's index, middle, ring and pinky, upstream puts `solref` on the visual pad geoms instead, so in MuJoCo those collision pads use the default `0.02 1`. | Dropped. A bot has no per-link contact stiffness or damping. Lab. |
| D2 | Geom `friction`: not set, so MuJoCo's default `1 0.005 0.0001` applies | Dropped. A bot has no per-link friction. Lab. |
| D3 | Joint `armature`, `frictionloss` and `damping` (classes `CMC_joint`, `PCMC_joint`, `MCP_joint`, `PIP_joint`, `DIP_joint`) | Mapped to the joint's `inertia`, `friction.coulomb` and `friction.viscous`. The values are equal up to float32 rounding (worst relative difference 5e-8). In SuperDex, `coulomb` is a Coulomb model with no stiction, whereas MuJoCo's `frictionloss` is a dry-friction constraint. |
| D4 | Actuators: one `<position>` per joint with `kp` (thumb CMC_FE 6.95, thumb CMC_AA 13.2, MCP_FE 4.76, MCP_AA 6.62, thumb IP / PIP / DIP 0.9, pinky CMC 1.38), `dampratio="0.9"` and `inheritrange="1.0"` (control range = joint range) | Not in the bot (the importer reports "Skipped 22 x actuator gains and control ranges"). Lab. MuJoCo derives `kv` from `dampratio` and the joint-space inertia, so it has no single value to copy. |
| D5 | Actuator `forcerange`: none, so actuator force is unlimited | No `effortLimit` on any joint. This matches upstream, but a lab may still set a limit. |
| D6 | `contype`/`conaffinity`: 1/1 on the collision geoms (group 3), 0/0 on the visual geoms (group 1), plus 28 `<exclude>` pairs | Represented. SuperDex disables contact between adjacent links by default, and 9 `contactOverrides` make up the rest of the MJCF's rules. An independent check of all 325 pairs of collision links found the same contact policy as MuJoCo's on both sides, with no mismatches. |
| D7 | Collision meshes: MuJoCo collides each mesh geom as its convex hull | Each collision model is an SDF of the wrapped and remeshed mesh, which keeps concave regions that the convex hull fills. Contact geometry therefore differs from MuJoCo's. |
| D8 | Primitive geoms | None upstream, so nothing was lost. |
| D9 | The root body `<side>_hand_C_MC` has no joint, so MuJoCo welds it to the world (fixed base) | The root has a Free `world_joint`. A fixed-hand lab needs a copy of the bot with this joint set to `Hard`, as `wuji_reorient/bots/` does for Wuji. |
| D10 | The pads (D1) carry no mass of their own: each body's mass comes from its `<inertial>` | Each pad link has the importer's minimum mass, 1e-6 x the distal link's mass (4.7e-9 kg; thumb 8.0e-9 kg right, 1.0e-8 kg left). Its center of mass and inertia are the ones the engine computes from the pad's collision model at that mass, written into the bot (SuperDex Studio's Bake Mass & Inertial Properties, `BakeMassProperties`), so the bot round-trips through URDF; evaluations with and without them are byte-identical. Bodies match upstream within float32 rounding: mass to 6e-8 relative, COM to 1.3e-9 m and inertia to 5e-7 of each tensor's largest entry. The totals are 1.245685 kg right and 1.247685 kg left. The fixed-hand copies in `baoding/bots/` and `inhand_rotation/bots/` were generated before the bake and keep mass-only pads, so `link_pseudo_inertia` DR still cannot target the pads there. |
| D11 | DOF order thumb, index, middle, ring, pinky (body and actuator order) | The importer orders the root's children by name, so the DOF order is index, middle, pinky, ring, thumb. Joint names are unchanged. Labs that list joints by name are not affected. |
| D12 | Home pose and `soft_joint_pos_limit_factor=0.9` (in mjlab_hand's `sharpa_constants.py`, not the MJCF) | Not in the bot, whose `defaultPose` is all zeros (MJCF `qpos0`). Lab. |
| D13 | No sensors | No `joint_encoder` sensor. The Wuji Hand 2 and DG5F bots have one, so labs that measure `joint_encoder` need it added. |

Upstream quirks are kept as they are:
- The left thumb's distal mass is 0.009981 kg, against 0.007981 kg on the right.
- On the left hand, `solref` sits on the visual pad geoms (D1).
- The left MJCF declares two meshes that no geom uses: `elastomer_surface.STL` and
  `thumb_elastomer_surface.STL`.

## License

The upstream Sharpa Wave assets are provided under the Apache License, Version 2.0 (Copyright 2025
Sharpa Group). `LICENSE` holds the upstream `LICENSE.txt` (its text verbatim, with Unix line endings), followed by the full text of
the Apache License, Version 2.0. `NOTICE` gives the attribution.

The render and collision models, recipes and bot files were derived from the upstream assets by
Meta Platforms, Inc. and affiliates, and are provided under the same license.

**You are responsible for ensuring your use is compatible with all third-party licenses.**
