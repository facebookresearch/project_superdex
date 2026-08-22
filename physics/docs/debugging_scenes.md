---
title: "Inspecting Scenes"
---

# Inspecting Scenes

The SuperDex Physics Debugger is a separate application that connects to your simulation over the network and lets you view a running scene, control its playback, and inspect its actors. Your application owns and steps its scenes, while the debugger receives scene data and sends playback commands back. This separation keeps the debugger optional and allows it to connect to simulations running in another process or on another computer.

Most documented Python [examples](./examples/0_getting_started.md) open the debugger automatically and wait for you to start the simulation.

## Connecting from Python

Call [`physics.debugger.attach()`](pathname:///generated/api/v1.0/python/api/debugger.html#superdex.physics.debugger.attach) after creating the scene, then step the scene while the debugger remains connected:

```python
import superdex.physics as physics


def main() -> None:
    physics.initialize(num_worker_threads=-1)
    try:
        scene = physics.create_scene("My Scene")
        # Add actors and constraints to the scene.

        if not physics.debugger.attach():
            return

        while physics.debugger.is_attached():
            scene.step(1.0 / 60.0)
    finally:
        physics.shutdown()


if __name__ == "__main__":
    main()
```

[`attach()`](pathname:///generated/api/v1.0/python/api/debugger.html#superdex.physics.debugger.attach) launches or focuses the debugger and waits briefly for it to connect. It returns `False` if no connection is established. Closing or disconnecting the debugger makes [`is_attached()`](pathname:///generated/api/v1.0/python/api/debugger.html#superdex.physics.debugger.is_attached) return `False`, allowing the application to leave its simulation loop and clean up.

## Moving the Camera

Click the viewport before using the camera controls.

| Input | Action |
| --- | --- |
| `W`, `A`, `S`, `D` | Move forward, left, backward, and right |
| Hold the right mouse button and move the mouse | Look around |
| `Q`, `E` | Move down and up |
| Hold `Shift` while moving | Move faster |
| `F` | Frame the current scene |

## Controlling Playback

Scenes start paused unless you choose otherwise when connecting. The application still calls [`scene.step()`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Scene.step), as shown above; the debugger controls when each call proceeds.

| Control | Behavior |
| --- | --- |
| Play | Advances all scenes at up to real-time speed. It does not speed up a simulation that is already slower than real time. |
| Pause | Pauses all scenes before their next simulation step. |
| Restore initial state | Restores the selected scene to the state captured before its first step after the debugger connects. It does not recreate the scene or change the current playback mode. |
| Single step | Pauses all scenes, advances the selected scene by one application-defined step, and remains paused. |
| Fast forward | Advances all scenes without real-time throttling, as quickly as the application can simulate them. Select it again to return to real-time playback. |

The initial state is captured just before the scene's first step after the debugger connects. Before that step, the restore control reports that no initial state is available. If you restore while playing, the scene continues playing from the restored state. See [State Capture](./concepts/state_capture.mdx) for capture and restore details.

## Working with Multiple Scenes

The debugger displays one scene at a time. Choose a scene from the **Scene** dropdown to inspect it; selecting another scene changes what is displayed but does not change playback.

Play, Pause, and Fast Forward apply to every scene in the connected application, including scenes that are not currently displayed. Restore Initial State and Single Step apply only to the selected scene. This lets several scenes advance together while you switch between them to compare their results.
