# Changelog – Project SuperDex

All notable changes to this repository will be documented here.

## [Unreleased]

- Added the BrainCo Revo2 hand (left and right, `assets/bots/hands/revo2`) with a
  `TACTILE_PAD` taxel-array sensor on each fingertip pad, generated from BrainCo's URDF
  by `tools/build_revo2_assets.py`.
- Added FR3 V2 + Revo2 assemblies (`assets/bots/arm_hand_combos/fr3_v2_revo2`).
- Added `superdex.lab.sensors.tactile`, a Python taxel-array tactile sensor built on
  SuperDex contact points.
- Added the `superdex_gym/Fr3Revo2-v0` grasp-and-lift environment with tactile
  observations, and a scripted grasp demo (`superdex_lab/apps/envs/run_fr3_revo2_grasp.py`).
- Fixed `AttachBot` dropping the attached bot's linear transmissions and spatial tendons
  (e.g. a hand's coupled finger joints when mounted on an arm).

## [2026-08-24]

- Initial release.
