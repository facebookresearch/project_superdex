---
title: Getting Started
sidebar_label: Getting Started
sidebar_position: 2
slug: /getting_started
---

# Getting Started

This page covers installing SuperDex and building it from source. For a tour of
the library and its concepts, see the [Overview](./overview.md); to jump straight
into worked examples, see the [examples](./examples/overview.mdx).

## Quick Start

SuperDex Robotics can be installed as part of Project SuperDex. Follow the
[quick-start instructions (Python)](https://github.com/facebookresearch/project_superdex#quick-start-python) or [build it from source (C++, Python)](https://github.com/facebookresearch/project_superdex#building-from-source).

## Example Code

Once installed, here is a minimal SuperDex Robotics Python script that loads a bot
into a scene and steps the simulation:

```python
import superdex.physics as physics
import superdex.robotics as robotics
from superdex.physics.paths import resolve_asset

physics.initialize(num_worker_threads=0)

scene = physics.create_scene("Quick Start")
scene.set_gravity([0, 0, -9.81])

# Load a bot and instantiate it in the scene.
bot_prefab = robotics.load_bot_prefab_from_file(
    str(resolve_asset("bots/arms/fr3/fr3.superdex_bot"))
)
bots_context = robotics.create_context()
bot = robotics.create_bot(scene, bot_prefab, bots_context)

for _ in range(120):
    scene.step(1.0 / 60.0)

robotics.destroy_bot(scene, bot)
physics.shutdown()
```

Running the OSC control example opens the SuperDex Physics debugger, where you can
watch the bot simulate in real time:

<img src="../../img/getting_started/osc_control_example.webp" alt="SuperDex Physics debugger showing the FR3 arm from the OSC control example" />

Explore more [SuperDex Robotics examples](./examples/overview.mdx).
