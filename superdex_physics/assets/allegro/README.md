# Allegro Hand Description

## Overview

This package contains a simplified robot description of the [Allegro Hand](https://www.allegrohand.com) developed by [Wonik Robotics](https://www.wonikrobotics.com). It is derived from [the MuJoCo Menagerie (wonik_allegro)](https://github.com/google-deepmind/mujoco_menagerie/tree/main/wonik_allegro).

## Modifications

One or more of the following modifications were made to adapt the original assets:

- Converted the original model into the SuperDex Physics asset format.
- Simplified and/or re-authored collision geometry (convex decomposition) for the physics engine.
- Adjusted dynamics parameters (mass, inertia, friction, damping) for realistic simulated behavior.
- Decimated and/or re-baked visual meshes for rendering.

## License

The Allegro Hand assets are provided by Wonik Robotics under the **Wonik Robotics Asset License** (see the [LICENSE](./LICENSE) file). This is a custom license that permits use, copying, modification, merging, and distribution **solely for non-commercial research, educational, evaluation, and internal-development purposes**; commercial use requires Wonik Robotics' prior written permission.

These assets have been modified by Meta. As required by the license, the copyright and permission notice (LICENSE) and a notice of modification ([NOTICE](./NOTICE)) must be retained in all copies or substantial portions of the assets.

**You are responsible for ensuring your use is compatible with all third-party licenses.**
