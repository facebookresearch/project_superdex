# Tianji Marvin M6-S-CCS-696-V4.0 Arms

## Overview

This package contains left and right SuperDex descriptions of the Tianji Marvin M6-S-CCS-696-V4.0 arm by Tianji Intelligent System Co., LTD., derived from CAD supplied by Tianji:

- Left: [`left/Marvin_M6_S_L_CCS_696_V40.superdex_bot`](left/Marvin_M6_S_L_CCS_696_V40.superdex_bot)
- Right: [`right/Marvin_M6_S_R_CCS_696_V40.superdex_bot`](right/Marvin_M6_S_R_CCS_696_V40.superdex_bot)

## Modifications

Kinematics, meshes, link masses and centres of mass are the vendor CAD/URDF. The following were adapted for SuperDex:

- Collision geometry re-authored from simplified CAD (`left/cad/`, `right/cad/`) for SuperDex Physics; the mesh-processing recipes are under each side's `intermediates/`.
- Visual materials authored for rendering.
- Joint dynamics and link rotational inertia identified by torque-level system identification on a physical right arm:
  - right: link `momentOfInertia`, joint `inertia` (armature) and `friction` identified; mass and COM held at the URDF values, so gravity matches the URDF gravity the controller compensates with;
  - left: not measured yet; carries the right arm's identified model. Joint `inertia` and `friction` are copied joint by joint (same drives). The arms are mirror builds of the same parts (Link1-4 and Link7 identical, Link5 turned 180 deg about z, Link6 about x), so each link's identified `momentOfInertia` is copied through that turn; mass and COM stay at the left URDF values.

Link rotational inertia is weakly identified (on most links the data pushes it to the physical bound); joint torques are carried mainly by the armature. Held-out joint-torque error is 1.2-1.7%, vs 10-13% for the CAD values. Viscous friction is ~0 (friction is flat or falls with speed on this arm).

## License

Tianji arm simulation assets derived from CAD supplied by Tianji Intelligent System Co., LTD. Adapted by Meta for Project SuperDex. Licensed under CC BY 4.0. See [LICENSE](./LICENSE) for the full license text.

This package includes a [NOTICE](./NOTICE) file carrying this attribution; retain it in redistributions.

**You are responsible for ensuring your use is compatible with all third-party licenses.**
