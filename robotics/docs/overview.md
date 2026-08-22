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

The following capabilities are available in the initial release of SuperDex
Robotics.

**Declarative robot definitions** — Robots are described declaratively in a serialized robot format: links, joints, default pose, and other parameters. In scope it is closer to URDF than to MJCF or USD — it strictly describes a robot.

**Robot composition** — Build complex robots from a base with modification recipes, such as attaching a gripper to an arm, without editing the original definition.

**Controllers, Sensors, Actuators** — A uniform, extensible framework for all three component kinds, with built-in joint-space PD and operational-space (OSC) controllers, and register-your-own support for the rest. The same controller code drives a simulated or a real robot.

**URDF import** — Bring existing robots into SuperDex Robotics by importing from URDF.

We have planned and are actively developing additional features for future
releases, including tendon actuation, soft robot linkages and skins, and
frameworks for describing robotics scenes and tasks with domain randomization.
Look for these in upcoming releases.

## Supported Platforms and APIs

- **Platforms**: Linux (x86-64), Windows (x86-64), macOS (arm64)
- **APIs**: C++, Python

## Getting Started

Explore our documentation to begin building with SuperDex Robotics:

- **[Getting Started](./getting_started.md)** - Set up and build SuperDex Robotics on your platform
- **[C++ API Reference](./api_reference/cpp.mdx)** - Complete C++ API documentation
- **[Python API Reference](./api_reference/python.mdx)** - Complete Python API documentation
- **[Examples](./examples/overview.mdx)** - Walk through the shipped Python examples end to end

## What's Here

| Guide | What it covers |
|-------|----------------|
| [Getting Started](./getting_started.md) | Installing SuperDex and building it from source. |
| [Bots](./bots.mdx) | The `BotPrefab` schema and the `.superdex_bot` file format. |
| [Modifying Bots](./modifying_bots.mdx) | Composing bots from a base with `ModBotPrefab` modification recipes. |
| [Bot Context & Lifetime](./bot_context_lifetime.mdx) | How `RoboticsContext` owns bot and controller memory. |
| [Controllers, Sensors & Actuators](./bot_components.mdx) | Creating, driving and writing the three component kinds. |
| [Bot Assets](./bot_assets.md) | The bots we ship in the `assets/bots` folder. |
