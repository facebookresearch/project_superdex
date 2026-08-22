# Franka Research 3 (FR3) V2 Description

## Overview

This package contains a simplified robot description of the [Franka Research 3 (FR3) V2](https://franka.de/products) developed by [Franka Robotics](https://franka.de). It is derived from [the franka_description package](https://github.com/frankarobotics/franka_description).

## Modifications

One or more of the following modifications were made to adapt the original assets:

- Simplified and/or re-authored collision geometry for compatibility with SuperDex Physics.
- Adjusted kinematic parameters (relative joint/link transforms, axes, and limits) for SuperDex Physics conventions.
- Adjusted dynamics parameters (mass, inertia, friction, damping) for realistic simulated behavior.
- Adjusted visual geometry and/or materials to improve rendering fidelity.
- Added or modified actuator descriptions.

## License

The original Franka Research 3 (FR3) V2 assets are provided by Franka Robotics under the [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0). The derived assets in this directory are distributed under the same [Apache-2.0](./LICENSE).

This package includes a [NOTICE](./NOTICE) file carrying the upstream attribution required by the license; retain it in redistributions.

**You are responsible for ensuring your use is compatible with all third-party licenses.**
