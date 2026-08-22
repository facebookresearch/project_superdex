# Benchmarks Description

## Overview

This package contains simulation-ready descriptions of three classic reinforcement-learning
benchmark environments - ant, cart-pole, and half-cheetah - built from simple geometric
primitives.

The ant and half-cheetah models are derived from the MuJoCo reference models distributed with
OpenAI Gym:

- `ant/` from [`gym/envs/mujoco/assets/ant.xml`](https://github.com/openai/gym/blob/master/gym/envs/mujoco/assets/ant.xml)
- `half_cheetah/` from [`gym/envs/mujoco/assets/half_cheetah.xml`](https://github.com/openai/gym/blob/master/gym/envs/mujoco/assets/half_cheetah.xml)

Their kinematic structure, joint limits, and link proportions reproduce those models, and the
assets say so themselves: `half_cheetah/half_cheetah.mochi_scene` describes its scene as a
"replica of the Gymnasium example built with MuJoCo", and `ant/ant.mochi_prefab` reproduces the
reference ant's parameters numerically - leg offsets of 0.2828 (0.2 * sqrt(2)), hip limits of
+/-0.5235988 rad (+/-30 deg), and ankle limits of -1.2217305 .. -0.5235988 rad
(-70 deg .. -30 deg).

The cart-pole model has no upstream asset. OpenAI Gym's cart-pole environment is defined in
Python and ships no model file, so the geometry, kinematics, and dynamics in `cart_pole/` were
authored at Meta to reproduce the behavior of that environment. The scene description in
`cart_pole/cart_pole.mochi_scene` calls it a replica of the Gymnasium example; that refers to the
environment's behavior, not to a transcribed model file.

## Modifications

One or more of the following modifications were made to adapt the reference models:

- Re-authored the link geometry as simple primitives in the SuperDex Physics format.
- Generated collision and render geometry for compatibility with SuperDex Physics.
- Adjusted kinematic parameters (relative joint/link transforms, axes, and limits) for SuperDex
  Physics conventions.
- Adjusted dynamics parameters (mass, inertia, friction, damping, joint stiffness) for stable
  simulated behavior.
- Assembled the bodies into scene and prefab manifests, with contact filtering and, for the
  ant, a controller configuration.

## Source terms and license scope

Two sets of terms apply to this directory, each reproduced in full in its own file:
[LICENSE](./LICENSE) for CC-BY-4.0 and [LICENSE-MIT](./LICENSE-MIT) for the MIT License.
[NOTICE](./NOTICE) records the upstream attribution.

**`ant/` and `half_cheetah/`** - the upstream OpenAI Gym models are licensed under the MIT
License, Copyright (c) 2016 OpenAI
([full terms](https://github.com/openai/gym/tree/master?tab=License-1-ov-file), also reproduced
in [LICENSE-MIT](./LICENSE-MIT)). The MIT copyright notice and permission notice must be
retained when you redistribute these files.

The Meta-authored content layered on top - the primitive geometry, the collision and render
meshes, the tuned dynamics parameters, and the scene, prefab, and controller manifests - is
additionally licensed under CC-BY-4.0 ([LICENSE](./LICENSE)).

**`cart_pole/`** - original work of Meta Platforms, Inc. and affiliates, licensed under
CC-BY-4.0 ([LICENSE](./LICENSE)) only. No upstream asset is incorporated.

You must give appropriate credit to Meta Platforms, Inc. when you use or redistribute the
Meta-authored content.

**You are responsible for ensuring your use is compatible with all applicable licenses.**
