---
title: Loading a Bot
sidebar_label: Loading a Bot
sidebar_position: 1
---

# Loading a Bot

Loads a robot from a `.superdex_bot` file using the SuperDex bindings and
simulates it in a physics scene.

**Source**: `examples/basic/example_bot_loading.py`

## Key Concepts

### The robot file

A `.superdex_bot` is a self-contained robot description. The example resolves the
bundled FR3 arm by its asset path, but any `.superdex_bot` path works:

```python
from superdex.physics.paths import resolve_asset

bot_path = str(resolve_asset("bots/arms/fr3/fr3.superdex_bot"))
```

### Initializing the engine and scene

Initialize the physics engine before creating scenes or actors.
`num_worker_threads=0` runs single-threaded; pass `-1` to auto-select:

```python
physics.initialize(num_worker_threads=0)
```

SuperDex robots use a Z-up convention, so gravity points down the -Z axis:

```python
scene = physics.create_scene("Bot Loading Example")
scene.set_gravity([0, 0, -9.81])
```

### Loading and instantiating the bot

Loading a `.superdex_bot` returns a *prefab*: a reusable template describing the
robot's links and joints. Instantiating it as a live `Bot` builds the robot's
articulated actor and seeds its default pose. The robotics context tracks every
bot and controller you create:

```python
bot_prefab = robotics.load_bot_prefab_from_file(bot_path)

robotics_context = robotics.create_context()
bot = robotics.create_bot(scene, bot_prefab, robotics_context)
bot_actor = bot.get_articulated_actor()
```

Add a static ground plane for the robot to rest on (normal points up, +Z):

```python
plane_shape = physics.create_plane_shape(normal=[0, 0, 1], distance=0)
scene.create_rigid_actor(name="ground", shape=plane_shape, is_static=True)
```

### Inspecting the robot

The prefab exposes the robot's links and joints, and the actor reports its
degrees of freedom:

```python
print(f"Robot: {bot_prefab.name}")
print(f"  Links: {len(bot_prefab.links)}")
print(f"  Joints: {len(bot_prefab.joints)}")
print(f"  DOFs: {bot_actor.get_num_dofs()}")
```

### Simulating with the debugger

Simulate at 60 Hz (each step advances 1/60 of a second):

```python
time_step = 1.0 / 60.0
```

Declare the scene's coordinate convention so the debugger renders it the right
way up: SuperDex is X-forward, Y-left, Z-up (FLU). This must come before
`attach()`, which starts the server:

```python
physics.get_debug_server().set_coordinate_space(
    physics.CoordinateSpace(axes=physics.CoordinateSpaceAxes.FLU)
)
```

The SuperDex Physics Debugger is a separate desktop app for viewing and
interacting with the simulation. The loop runs until you close the debugger;
`attach()` returns `False` if it can't connect:

```python
if physics.debugger.attach():
    while physics.debugger.is_attached():
        scene.step(time_step)
```

### Teardown

Destroy the bot, then shut the engine down cleanly:

```python
robotics.destroy_bot(scene, bot)
physics.shutdown()
```

## Running

```bash
uv run python superdex_robotics/examples/basic/example_bot_loading.py
```

To load your own robot, pass a path to a `.superdex_bot` file:

```bash
uv run python superdex_robotics/examples/basic/example_bot_loading.py /path/to/my_robot.superdex_bot
```
