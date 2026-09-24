# Changelog – Project SuperDex

All notable changes to this repository will be documented here.

## [Unreleased]

- Added the BrainCo Revo2 hand (left and right, `assets/bots/hands/revo2`) with a
  Revo2 Touch-style `TACTILE_PAD` sensor on each fingertip (3D force and proximity),
  generated from BrainCo's URDF by `tools/build_revo2_assets.py`.
- Added FR3 V2 + Revo2 assemblies (`assets/bots/arm_hand_combos/fr3_v2_revo2`).
- Added `superdex.lab.sensors.tactile`, a Python fingertip tactile sensor built on
  SuperDex contact points: single-element or taxel-array force sensing, saturation,
  resolution, noise, lag, and SDF-based proximity.
- Added the `superdex_gym/Fr3Revo2-v0` environment (FR3 + Revo2 on a desk, lifting a
  paper cup) with fingertip tactile observations, and a scripted grasp demo with
  tactile force control (`superdex_lab/apps/envs/run_fr3_revo2_grasp.py`).
- Added `superdex.lab.gym.utils.render_materials`, which paints render-model material
  colors in the default viewer.
- Fixed `AttachBot` dropping the attached bot's linear transmissions and spatial tendons
  (e.g. a hand's coupled finger joints when mounted on an arm).

## [2026-08-24]

- Initial release.
