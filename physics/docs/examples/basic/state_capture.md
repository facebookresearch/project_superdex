---
title: State Capture
sidebar_position: 8
---

# State Capture

This example lets a rigid duck fall onto a plane, restoring a checkpoint of the initial state every three seconds, so that the simulation loops repeatedly.

**Source**: `examples/example_state_capture_restore.py`

For the complete capture/restore API and ownership rules, see the [State Capture concept page](../../concepts/state_capture.mdx). State capture and restore is not limited to the simple same-scene case illustrated here.

## Scene Setup

The example creates one static ground plane and one dynamic rigid duck. Complete the scene topology before capturing state because restoration updates existing simulation state; it does not create actors or shapes.

```python
scene, _ = create_state_capture_simulation()
initial_state = scene.capture_state()
```

The returned handle is owned by, and can be restored only into, the scene that captured it.

## Restoring Every Three Seconds

The simulation steps at 60 Hz while the Debugger is attached. When scene time reaches three seconds, restoring the initial checkpoint resets the duck and the captured scene time:

```python
while physics.debugger.is_attached():
    scene.step(TIME_STEP)
    if scene.get_total_simulation_time() >= RESTORE_INTERVAL:
        scene.restore_state(initial_state, release_immediately=False)
```

Because scene time returns to `0.0 s`, the same condition naturally starts the next three-second interval. `release_immediately=False` keeps the handle valid for every repetition.

## Releasing the Handle

A reusable state handle keeps scene-owned state alive until the handle is released or the scene is destroyed. Release the handle explicitly when the scene will continue running:

```python
scene.release_state(initial_state)
```

This example shuts down the physics context immediately on exit, which destroys the scene and its captured state:

```python
try:
    ...
finally:
    physics.shutdown()
```

## What to Expect

The duck starts high above the plane and rotated onto its side, so that it impacts on a convex wingtip and tumbles in a way that is sensitive to small perturbations in its initial configuration. Every three seconds, the duck should return to exactly the same initial pose and repeat the same subsequent motion. The example runs single-threaded with a fixed time step, so the duck should tumble in exactly the same way after each state restore.

## Running

```bash
uv run --no-project examples/example_state_capture_restore.py
```

`debugger.attach()` launches or focuses the Debugger and waits for it to connect. The scene starts paused; press **Play** in the Debugger to advance the simulation. Detaching the Debugger ends the simulation loop. See [Inspecting Scenes](../../debugging_scenes.md) for debugger connection, navigation, and playback controls.
