---
title: Overview
sidebar_position: 1
---

# SuperDex Physics Overview

SuperDex Physics is a contact-first physics engine purpose-built for tactile manipulation and applications where stable contact and accurate sensing matter. As the simulation backbone of Project SuperDex, it delivers stable, high-fidelity contact physics for demanding dexterous interactions such as multi-finger grasps, in-hand reorientation, and non-convex contact.

## Key Capabilities

**Multi-physics simulation** — A unified solver for rigid bodies, soft bodies, rods and tendons, and shells and cloth, with more physics on the way.

**Arbitrary rigid & soft articulations** — Articulated bodies with varying joint types support both rigid links and deformable elements in a single model.

**Non-convex collision** — Accurate contact force distributions for arbitrary geometries, including non-convex and deforming objects.

**Spatially dense contact forces** — Rather than reporting a single resultant contact force, SuperDex Physics computes a full 3D force distribution over contact surfaces. This enables realistic tactile sensor modeling and provides richer observation signals for RL policies.

**Inverse Kinematics** — Constraint-aware inverse-kinematics, built on the same nonlinear optimization core as the forward dynamics, solve physically accurate poses under collision, end-effector, and trajectory constraints.

**Numerical stability** — Robust simulation without the restrictive time-step stability limits of explicit or semi-implicit methods.

**Proven dexterity** — User studies demonstrate near real-world manipulation performance in VR scenarios, validating that SuperDex Physics' contact and deformable body simulation is sufficient for natural dexterous interactions.

## Integrations

**SuperDex Lab** — SuperDex Lab builds on SuperDex Physics with [Gymnasium](https://gymnasium.farama.org/)-compatible environments and an off-the-shelf Ray/RLlib integration for reinforcement-learning workflows.

## Supported Platforms and APIs

- **Platforms**: Linux (x86-64), Windows (x86-64), macOS (arm64)
- **APIs**: C++, Python

## Getting Started

Explore our documentation to begin building with SuperDex Physics:

- **[Getting Started](./getting_started.mdx)** - Set up and build SuperDex Physics on your platform
- **[C++ API Reference](./api_reference/cpp.mdx)** - Complete C++ API documentation
- **[Python API Reference](./api_reference/python.mdx)** - Complete Python API documentation
- **[Examples](./examples/0_getting_started.md)** - Sample code and examples
