---
title: Bot Assets
sidebar_label: Bot Assets
sidebar_position: 5
---

# Bot Assets

SuperDex Robotics ships a library of ready-to-load bots under
`assets/bots/`. Each is a `.superdex_bot` file you can load
directly with `load_bot_prefab_from_file`. Bots are grouped by the category
folder they live in; left/right variants are listed separately. All bot asset
visuals are GLB and have been given custom PBR materials to better represent the
real product, compared to the visual assets provided by upstream sources
(COLLADA, etc.).

Each entry carries three preparation flags:

- **Collision** — how the collision mesh was remeshed to SuperDex standards:
  **Remeshed Visual** (rebuilt from the source visual mesh) or **Remeshed CAD**
  (rebuilt from source CAD geometry).
- **Link Dynamics** — provenance of the link inertial values.
- **Joint Dynamics** — provenance of the joint dynamic values.

Link and Joint Dynamics each display one of: **Mfgr.** (from the manufacturer's
source URDF), **CAD** (derived from CAD), **Tuned** (hand-tuned), or **SysID**
(parameterized by system identification).

## Arms

| Thumbnail | Robot Name | Asset Link | Collision | Link Dynamics | Joint Dynamics |
|-----------|------------|------------|-----------|---------------|----------------|
| <img src="../../img/bot_assets/fr3.png" alt="FR3" width="120" /> | Franka Research 3 (FR3) | [`fr3`](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/arms/fr3) | Remeshed Visual | Mfgr. | SysID |
| <img src="../../img/bot_assets/fr3_v2.png" alt="FR3 v2" width="120" /> | Franka Research 3 (FR3 v2) | [`fr3_v2`](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/arms/fr3_v2) | Remeshed Visual | Mfgr. | SysID |
| <img src="../../img/bot_assets/openarm_v20_left_arm.png" alt="OpenArm v2.0 left arm" width="120" /> | OpenArm v2.0 Arm (Left) | [`openarm_v20_left_arm`](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/arms/openarm_v20/left) | Remeshed CAD | Mfgr. | Mfgr. |
| <img src="../../img/bot_assets/openarm_v20_right_arm.png" alt="OpenArm v2.0 right arm" width="120" /> | OpenArm v2.0 Arm (Right) | [`openarm_v20_right_arm`](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/arms/openarm_v20/right) | Remeshed CAD | Mfgr. | Mfgr. |

## Grippers

| Thumbnail | Robot Name | Asset Link | Collision | Link Dynamics | Joint Dynamics |
|-----------|------------|------------|-----------|---------------|----------------|
| <img src="../../img/bot_assets/2f_85.png" alt="Robotiq 2F-85" width="120" /> | Robotiq 2F-85 | [`2f_85`](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/grippers/2f_85) | Remeshed Visual | Mfgr. | Mfgr. |
| <img src="../../img/bot_assets/openarm_v20_left_gripper.png" alt="OpenArm v2.0 left gripper" width="120" /> | OpenArm v2.0 Gripper (Left) | [`openarm_v20_left_gripper`](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/grippers/openarm_v20/left) | Remeshed CAD | Mfgr. | Mfgr. |
| <img src="../../img/bot_assets/openarm_v20_right_gripper.png" alt="OpenArm v2.0 right gripper" width="120" /> | OpenArm v2.0 Gripper (Right) | [`openarm_v20_right_gripper`](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/grippers/openarm_v20/right) | Remeshed CAD | Mfgr. | Mfgr. |

## Hands

| Thumbnail | Robot Name | Asset Link | Collision | Link Dynamics | Joint Dynamics |
|-----------|------------|------------|-----------|---------------|----------------|
| <img src="../../img/bot_assets/allegro_v5_left.png" alt="Allegro Hand v5 left" width="120" /> | Allegro Hand v5 (Left) | [`allegro_v5_left`](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/hands/allegro_v5/left) | Remeshed Visual | Mfgr. | Tuned |
| <img src="../../img/bot_assets/allegro_v5_right.png" alt="Allegro Hand v5 right" width="120" /> | Allegro Hand v5 (Right) | [`allegro_v5_right`](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/hands/allegro_v5/right) | Remeshed Visual | Mfgr. | Tuned |
| <img src="../../img/bot_assets/dg5f_short_left.png" alt="DG-5F short left" width="120" /> | Tesollo DG-5F Short (Left) | [`dg5f_short_left`](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/hands/dg5f_short/left) | Remeshed CAD | Mfgr. | SysID |
| <img src="../../img/bot_assets/dg5f_short_right.png" alt="DG-5F short right" width="120" /> | Tesollo DG-5F Short (Right) | [`dg5f_short_right`](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/hands/dg5f_short/right) | Remeshed CAD | Mfgr. | SysID |
| <img src="../../img/bot_assets/dg5f_short_seed_left.png" alt="DG-5F short seed left" width="120" /> | Tesollo DG-5F Short, SEED (Left) | [`dg5f_short_seed_left`](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/hands/dg5f_short_seed/left) | Remeshed CAD | Mfgr. | SysID |
| <img src="../../img/bot_assets/dg5f_short_seed_right.png" alt="DG-5F short seed right" width="120" /> | Tesollo DG-5F Short, SEED (Right) | [`dg5f_short_seed_right`](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/hands/dg5f_short_seed/right) | Remeshed CAD | Mfgr. | SysID |
| <img src="../../img/bot_assets/dg5f_long_left.png" alt="DG-5F long left" width="120" /> | Tesollo DG-5F Long (Left) | [`dg5f_long_left`](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/hands/dg5f_long/left) | Remeshed CAD | Mfgr. | SysID |
| <img src="../../img/bot_assets/dg5f_long_right.png" alt="DG-5F long right" width="120" /> | Tesollo DG-5F Long (Right) | [`dg5f_long_right`](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/hands/dg5f_long/right) | Remeshed CAD | Mfgr. | SysID |
| <img src="../../img/bot_assets/dg5f_long_seed_left.png" alt="DG-5F long seed left" width="120" /> | Tesollo DG-5F Long, SEED (Left) | [`dg5f_long_seed_left`](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/hands/dg5f_long_seed/left) | Remeshed CAD | Mfgr. | SysID |
| <img src="../../img/bot_assets/dg5f_long_seed_right.png" alt="DG-5F long seed right" width="120" /> | Tesollo DG-5F Long, SEED (Right) | [`dg5f_long_seed_right`](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/hands/dg5f_long_seed/right) | Remeshed CAD | Mfgr. | SysID |
| <img src="../../img/bot_assets/wuji_hand2_beta1_left.png" alt="Wuji Hand 2 Beta 1 left" width="120" /> | Wuji Hand 2, Beta 1 (Left) | [`wuji_hand2_beta1_left`](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/hands/wuji_hand2_beta1/left) | Remeshed CAD | Mfgr. | Mfgr. |
| <img src="../../img/bot_assets/wuji_hand2_beta1_right.png" alt="Wuji Hand 2 Beta 1 right" width="120" /> | Wuji Hand 2, Beta 1 (Right) | [`wuji_hand2_beta1_right`](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/hands/wuji_hand2_beta1/right) | Remeshed CAD | Mfgr. | Mfgr. |

## Avatar Hands

Not production robots — these are human hands built with the bot definition
format, for avatar and hand-tracking use.

| Thumbnail | Robot Name | Asset Link | Collision | Link Dynamics | Joint Dynamics |
|-----------|------------|------------|-----------|---------------|----------------|
| <img src="../../img/bot_assets/oculus_xr_hand_highpoly_left.png" alt="Meta XR hand high-poly left" width="120" /> | Meta XR Hand, High-poly (Left) | [`oculus_xr_hand_highpoly_left`](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/hands/oculus_xr/left) | Remeshed Visual | Mfgr. | Mfgr. |
| <img src="../../img/bot_assets/oculus_xr_hand_highpoly_right.png" alt="Meta XR hand high-poly right" width="120" /> | Meta XR Hand, High-poly (Right) | [`oculus_xr_hand_highpoly_right`](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/hands/oculus_xr/right) | Remeshed Visual | Mfgr. | Mfgr. |
| <img src="../../img/bot_assets/oculus_xr_hand_lowpoly_left.png" alt="Meta XR hand low-poly left" width="120" /> | Meta XR Hand, Low-poly (Left) | [`oculus_xr_hand_lowpoly_left`](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/hands/oculus_xr/left) | Remeshed Visual | Mfgr. | Mfgr. |
| <img src="../../img/bot_assets/oculus_xr_hand_lowpoly_right.png" alt="Meta XR hand low-poly right" width="120" /> | Meta XR Hand, Low-poly (Right) | [`oculus_xr_hand_lowpoly_right`](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/hands/oculus_xr/right) | Remeshed Visual | Mfgr. | Mfgr. |

## Torsos

| Thumbnail | Robot Name | Asset Link | Collision | Link Dynamics | Joint Dynamics |
|-----------|------------|------------|-----------|---------------|----------------|
| <img src="../../img/bot_assets/openarm_v20_torso.png" alt="OpenArm v2.0 torso" width="120" /> | OpenArm v2.0 Torso | [`openarm_v20_torso`](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/torsos/openarm_v20) | Remeshed CAD | Mfgr. | Mfgr. |

## Sensors

| Thumbnail | Robot Name | Asset Link | Collision | Link Dynamics | Joint Dynamics |
|-----------|------------|------------|-----------|---------------|----------------|
| <img src="../../img/bot_assets/dg5f_seed.png" alt="DG-5F SEED fingertip" width="120" /> | DG-5F SEED Fingertip | [`dg5f_seed`](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/sensors/dg5f_seed) | Remeshed CAD | Mfgr. | Mfgr. |

## Arm + Hand Combos

Combos are [`ModBotPrefab`](./modifying_bots.mdx) recipes — a base arm (or torso)
with grippers/hands attached. Their preparation flags are **N/A** because each
combo is a composite of the standalone robots above; see those entries for the
per-component Collision, Link Dynamics, and Joint Dynamics ratings.

| Thumbnail | Robot Name | Asset Link | Collision | Link Dynamics | Joint Dynamics |
|-----------|------------|------------|-----------|---------------|----------------|
| <img src="../../img/bot_assets/openarm_v20.png" alt="OpenArm v2.0 bimanual" width="120" /> | OpenArm v2.0 (Bimanual) | [`openarm_v20`](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/arm_hand_combos/openarm_v20) | N/A | N/A | N/A |
| <img src="../../img/bot_assets/fr3_dg5f_short_left.png" alt="FR3 + DG-5F short left" width="120" /> | FR3 + DG-5F Short (Left) | [`fr3_dg5f_short_left`](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/arm_hand_combos/fr3_dg5f_short/left) | N/A | N/A | N/A |
| <img src="../../img/bot_assets/fr3_dg5f_short_right.png" alt="FR3 + DG-5F short right" width="120" /> | FR3 + DG-5F Short (Right) | [`fr3_dg5f_short_right`](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/arm_hand_combos/fr3_dg5f_short/right) | N/A | N/A | N/A |
| <img src="../../img/bot_assets/fr3_dg5f_short_seed_right.png" alt="FR3 + DG-5F short seed right" width="120" /> | FR3 + DG-5F Short, SEED (Right) | [`fr3_dg5f_short_seed_right`](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/arm_hand_combos/fr3_dg5f_short_seed/right) | N/A | N/A | N/A |
| <img src="../../img/bot_assets/fr3_v2_2f_85.png" alt="FR3 v2 + Robotiq 2F-85" width="120" /> | FR3 v2 + Robotiq 2F-85 | [`fr3_v2_2f_85`](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/arm_hand_combos/fr3_v2_2f_85) | N/A | N/A | N/A |
| <img src="../../img/bot_assets/fr3_v2_allegro_v5_right.png" alt="FR3 v2 + Allegro Hand v5 right" width="120" /> | FR3 v2 + Allegro Hand v5 (Right) | [`fr3_v2_allegro_v5_right`](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/arm_hand_combos/fr3_v2_allegro_v5/right) | N/A | N/A | N/A |
| <img src="../../img/bot_assets/openarm_v20_wuji.png" alt="OpenArm v2.0 + Wuji Hand 2" width="120" /> | OpenArm v2.0 + Wuji Hand 2 (Bimanual) | [`openarm_v20_wuji`](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/arm_hand_combos/openarm_v20) | N/A | N/A | N/A |

## Fun

Playful, non-production bots for demos and fun.

| Thumbnail | Robot Name | Asset Link | Collision | Link Dynamics | Joint Dynamics |
|-----------|------------|------------|-----------|---------------|----------------|
| <img src="../../img/bot_assets/fr3_v2_with_eyes.png" alt="FR3 v2 with googly eyes" width="120" /> | FR3 v2 with Googly Eyes | [`fr3_v2_with_eyes`](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/fun/arm_eyes_combos) | N/A | N/A | N/A |
| <img src="../../img/bot_assets/googly_eyes.png" alt="Googly eyes" width="120" /> | Googly Eyes | [`googly_eyes`](https://github.com/facebookresearch/project_superdex/tree/main/assets/bots/fun/googly_eyes) | N/A | N/A | N/A |
