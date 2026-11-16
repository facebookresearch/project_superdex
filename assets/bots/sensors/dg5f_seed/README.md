# Seed Robotics SINGLEX-3 Sensor Package

This directory contains the Seed Robotics SINGLEX-3 files identified in [`LICENSE`](LICENSE). Seed Robotics approved those files for release under Apache-2.0.

SINGLEX-3 tactile sensor geometry based on hardware by Seed Robotics — [https://www.seedrobotics.com](https://www.seedrobotics.com)

The SuperDex assembly in this directory also references Tesollo DG-5F-M short-wrist fingertip geometry.

`seed.superdex_sensor` configures the public Mochi-native SEED MLP response with
`models/v0.1/forcenet_weights.json`. PyTorch artifacts remain under `internal/`
and are not part of the open-source package.

- Seed Robotics license scope and attribution: see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
- Tesollo hand terms and attribution: [`../../hands/dg5f_short`](../../hands/dg5f_short) — see its README, LICENSE, and NOTICE.

**You are responsible for ensuring your use is compatible with all third-party licenses of the referenced sub-assets.**

## Mounting adapter CAD

[`step/DG5F_Seed_Fingertip_Mount.STEP`](step/DG5F_Seed_Fingertip_Mount.STEP) contains the mounting adapter CAD matching [`render/dg5f_link_tip_render.glb`](render/dg5f_link_tip_render.glb).

The mounting adapter STEP is distributed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](step/LICENSE), with attribution to Meta Platforms, Inc. and affiliates. This license applies only to the STEP file.

The STEP coordinates are in millimeters. To align them with the physical mount-link frame, use `(x, y, z) = (STEP_z, STEP_x, STEP_y - 11)`, then convert to meters for SuperDex.
