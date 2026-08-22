# Robotiq 2F-85 Adaptive Gripper Description

## Overview

This package contains a simplified robot description of the [Robotiq 2F-85 Adaptive Gripper](https://robotiq.com/products/adaptive-grippers) developed by [Robotiq](https://robotiq.com). It is derived from [the ROS-Industrial robotiq package](https://github.com/ros-industrial-attic/robotiq).

## Modifications

One or more of the following modifications were made to adapt the original assets:

- Simplified and/or re-authored collision geometry for compatibility with SuperDex Physics.
- Adjusted kinematic parameters (relative joint/link transforms, axes, and limits) for SuperDex Physics conventions.
- Adjusted dynamics parameters (mass, inertia, friction, damping) for realistic simulated behavior.
- Adjusted visual geometry and/or materials to improve rendering fidelity.
- Added or modified actuator descriptions.

## License

The original Robotiq 2F-85 Adaptive Gripper assets are provided by Robotiq under the [BSD-2-Clause](https://opensource.org/licenses/BSD-2-Clause). The derived assets in this directory are distributed under the same [BSD-2-Clause](./LICENSE).

**You are responsible for ensuring your use is compatible with all third-party licenses.**
