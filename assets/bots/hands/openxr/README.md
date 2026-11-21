# Meta XR Hand Description (OpenXR skeleton)

## Overview

This package contains the Meta XR Hand (high- and low-poly variants, left and right) rigged to the OpenXR hand skeleton of [`XR_EXT_hand_tracking`](https://registry.khronos.org/OpenXR/specs/1.1/html/xrspec.html#XR_EXT_hand_tracking), for use as robot descriptions in SuperDex Robotics. The geometry is the original work of Meta and is the same geometry as the [`oculus_xr`](../oculus_xr) hands.

## Skeleton

The bot has exactly the 26 OpenXR hand joints. Link `xr_NN_<joint>` is OpenXR joint `NN` (`XrHandJointEXT`), and its frame is the unperturbed OpenXR joint frame:

- +Z runs along the bone, away from the finger tip.
- +Y points out of the back of the hand.
- X = Y × Z (right-handed).

Raw OpenXR joint poses therefore drive the bot with no per-bone correction: a link's local rotation is the OpenXR parent-from-joint rotation, and the root link is the wrist joint.

Links appear in `XrHandJointEXT` order, except that the wrist (`xr_01_wrist`) is the root and comes before the palm (`xr_00_palm`).

| Link | Joint | Geometry |
|---|---|---|
| `xr_01_wrist` | Free (root) | Palm, wrist, and thenar (trapezium) region |
| `xr_00_palm` | Hard | none |
| `xr_02_thumb_metacarpal` | Spherical | yes |
| `xr_03_thumb_proximal`, `xr_04_thumb_distal` | Revolute | yes |
| `xr_06/11/16_*_metacarpal` | Hard | none (part of the wrist geometry) |
| `xr_07/12/17_*_proximal` | Spherical | yes |
| `xr_08/13/18_*_intermediate`, `xr_09/14/19_*_distal` | Revolute | yes |
| `xr_21_little_metacarpal` | Revolute | yes |
| `xr_22_little_proximal` | Spherical | yes |
| `xr_23_little_intermediate`, `xr_24_little_distal` | Revolute | yes |
| `xr_*_tip` | Hard | none |

The rest pose is Meta's OpenXR hand bind skeleton. The palm and the index, middle, and ring metacarpals carry the wrist rotation, and the tips carry the distal rotation.

## Relation to `oculus_xr`

Each reused link frame is the `oculus_xr` link frame rotated by Rx(+90°) for the right hand and Rx(-90°) for the left. Meshes, joint axes, and joint limits are re-expressed by that rotation, so relative to the wrist every link sits exactly where it sits in `oculus_xr`.

The bot's world placement (`worldFromRoot`) is the identity for both hands, so the wrist frame is the world frame: the fingers point down (-Z) and the back of the hand faces +Y. The `oculus_xr` hands are placed differently, so the two do not overlap in Studio.

Differences:

- The `oculus_xr` forearm stub and trapezium (`bone_02_thumb_metacarpal`) have no OpenXR joint, so their geometry is part of the wrist link. The trapezium is held at its rest pose. The OpenXR thumb metacarpal rotates about the carpometacarpal joint, and its spherical limits bound the combined range of the `oculus_xr` trapezium and carpometacarpal joints (a per-axis rotation-vector box, from sampling both ranges).
- The OpenXR index, middle, and ring metacarpals are shapeless Hard links. They are rigid with respect to the wrist.
- Self-contact differs. Mochi disables contact between adjacent links, looking through Hard joints and shapeless links. Through the shapeless metacarpals, the index, middle, and ring proximal phalanges are adjacent to the wrist and everything attached to it, so contact that `oculus_xr` allowed among them, the thumb metacarpal, and the little metacarpal is off.
- The wrist's collision mesh is the boolean union of the merged meshes, and its SDF is baked at the finest SDF resolution of the source links. The `oculus_xr` low-poly wrist SDF is much coarser.

## Proxy-hand meshes

`render/openxr_proxy_hand_<side>.glb` is a skinned mesh of the user's hand for showing tracked OpenXR poses, such as the hand a Roboperator mirrors. Its joints are the bot's links, in link order, and its bind pose is the bot's rest pose, so it shares the bones and reference pose of the bot's skeletal mesh in Unreal.

Each bot uses it as a render-only skin: it is drawn in place of the links' render meshes, and the links' colliders are unchanged.

It is the TeleOperator hand mesh (`intermediates/source_hand_<side>.glb`, exported from Unreal) re-skinned and scaled by 0.74 onto the bot by [`openxr_proxy_hand_from_nimble.py`](../../../../superdex_robotics/tools/internal/openxr_proxy_hand_from_nimble.py):

```
fbpython ../../../../superdex_robotics/tools/internal/openxr_proxy_hand_from_nimble.py <side>/intermediates/source_hand_<side>.glb --bot <side>/openxr_hand_highpoly_<side>.superdex_bot --out <side>/render/openxr_proxy_hand_<side>.glb
```

## License

These assets are the original work of Meta Platforms, Inc. and affiliates and are distributed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](./LICENSE). You must give appropriate credit to Meta Platforms, Inc. when you use or redistribute these assets.

**You are responsible for ensuring your use is compatible with all applicable licenses.**
