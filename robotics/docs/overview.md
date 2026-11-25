---
title: Overview
sidebar_position: 1
slug: /overview
---

# SuperDex Robotics

Welcome to the SuperDex Robotics guides. These pages introduce the core concepts
and capabilities of SuperDex Robotics — defining, composing, and simulating
robots on the SuperDex Physics engine — and walk through the common workflows.
They are meant to familiarize you with the library, not to be comprehensive; for
the full, exhaustive reference, see the [C++](./api_reference/cpp.mdx) and
[Python](./api_reference/python.mdx) API references.

## Key Capabilities

The following capabilities are available in R2 of SuperDex Robotics.

**Declarative robot definitions** — Robots are described declaratively in a serialized robot format: links, joints, default pose, and other parameters. In scope it is closer to URDF than to MJCF or USD — it strictly describes a robot.

**Robot composition** — Build complex robots from a base with modification recipes, such as attaching a gripper to an arm, without editing the original definition.

**Controllers, Sensors, Actuators** — A uniform, extensible framework for all three component kinds, with built-in joint-space PD and operational-space (OSC) controllers, and register-your-own support for the rest. The same controller code drives a simulated or a real robot.

**URDF and MJCF import** — Load robot descriptions at runtime for prototyping, or prepare native collision and render assets in Studio.

**Inverse kinematics** — Solve link-frame goals or retarget one bot onto another with reusable IK assets; controllers drive the resulting joint pose.

**Skinned contact surfaces** — Attach a continuous collision and render skin to an articulated bot without replacing its links or DOFs.

**Bot scenes and tasks** — Combine physics assets, bots, controllers and sensors in a reusable workcell; swap episode reset state and termination criteria independently.

**Numeric sensor graphs** — Acquire contact and joint-wrench observations after physics, transform them into policy tensors, and read timestamped samples independently of acquisition.

Bots also support authored linear transmissions, spatial tendons and cycle-closing joints. These are distinct from per-link actuator components; see [Bots](./bots.mdx#mechanical-coupling).

## Supported Platforms and APIs

- **Platforms**: Linux (x86-64), Windows (x86-64), macOS (arm64)
- **APIs**: C++, Python

## Getting Started

Explore our documentation to begin building with SuperDex Robotics:

- **[Getting Started](./getting_started.mdx)** - Set up and build SuperDex Robotics on your platform
- **[C++ API Reference](./api_reference/cpp.mdx)** - Complete C++ API documentation
- **[Python API Reference](./api_reference/python.mdx)** - Complete Python API documentation
- **[Examples](./examples/overview.mdx)** - Walk through the shipped Python examples end to end

## What's Here

| Guide | What it covers |
|-------|----------------|
| [Getting Started](./getting_started.mdx) | Installing SuperDex and building it from source. |
| [Bots](./bots.mdx) | The `BotPrefab` schema and the `.superdex_bot` file format. |
| [Modifying Bots](./modifying_bots.mdx) | Composing bots from a base with `ModBotPrefab` modification recipes. |
| [Bot Context & Lifetime](./bot_context_lifetime.mdx) | How `RoboticsContext` owns bot and controller memory. |
| [Controllers, Sensors & Actuators](./bot_components.mdx) | Creating, driving and writing the three component kinds. |
| [Bot Assets](./bot_assets.mdx) | The bots we ship in the `assets/bots` folder. |
| [Inverse Kinematics](./inverse_kinematics.mdx) | IK assets, link-frame goals, retargeting and solve limits. |
| [Bot Skinning](./bot_skinning.mdx) | Collision and render skins, composition and coverage. |
| [Bot Scenes](./bot_scenes.mdx) | Combined workcell configuration and runtime lookup. |
| [Bot Tasks](./bot_tasks.mdx) | Episode reset state, randomization and outcomes. |
| [Sensors](./sensors.mdx) | Sensor sources and transforms, sensor graphs and timestamped reads. |
