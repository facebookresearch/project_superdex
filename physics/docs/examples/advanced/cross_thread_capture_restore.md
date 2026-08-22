---
title: Cross-Thread Capture/Restore
sidebar_position: 3
---

# Cross-Thread Capture/Restore

Policy training and planning often evaluate several trajectories from one common simulation checkpoint. A parent can capture that checkpoint once, then give each worker a candidate control sequence and the same state bytes. Every worker restores those bytes into its own compatible scene and evaluates an independent simulated trajectory, or *rollout*.

This example applies that pattern to the controlled articulation from the [Pose Controller example](../articulations/pose_controller.md). The parent owns one source scene and fans its immutable checkpoint bytes out to two worker-owned destination scenes. Each worker repeatedly runs the same five-second double-pendulum kick — one at the nominal speed, one at double speed — restores the checkpoint, and starts again.

**Source**: `examples/example_cross_thread_capture_restore.py`

Start with the [State Capture example](../basic/state_capture.md) for the same-scene handle lifecycle. For the complete API comparison and compatibility rules, see the [State Capture concept page](../../concepts/state_capture.mdx).

## Capture One Parent Checkpoint

The parent creates the source scene before attaching the Debugger. It captures into a fresh mutable [`DynamicArrayUint8`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.DynamicArrayUint8), then converts the result to immutable built-in `bytes` before starting either worker:

```python
source_scene, _ = create_simulation(
    "Cross-Thread Capture/Restore Source"
)
state_buffer = physics.DynamicArrayUint8()
source_scene.capture_state_to_bytes(state_buffer)
state_bytes = bytes(state_buffer)
```

Capture appends to the supplied array. Prefer a fresh, empty array for each snapshot, as above. If a buffer must be reused, slice out the newly appended capture before passing it to restore:

```python
capture_start = len(state_buffer)
scene.capture_state_to_bytes(state_buffer)
state_bytes = bytes(state_buffer)[capture_start:]
```

Tracking the offset alone is not sufficient because restore reads from byte zero. The byte representation is intended for short-lived exchange between compatible scenes, not long-term storage or asset interchange.

The source remains parent-owned and alive while the Debugger is attached. The parent attaches while it is the only scene, so the stationary source is selected initially. It calls `source_scene.update_debugger()` periodically so the scene remains inspectable even though it is not stepped.

## Start Peer Rollout Workers

After attachment, the same `bytes` object is passed to two independent tasks along with different candidate controls. Both workers are submitted as peers; neither depends on the other being submitted or creating its scene first:

```python
workers = (
    executor.submit(
        _repeat_rollout,
        "Cross-Thread Capture/Restore Rollout 1",
        state_bytes,
        NOMINAL_TIME_SCALE,
        stop,
    ),
    executor.submit(
        _repeat_rollout,
        "Cross-Thread Capture/Restore Rollout 2",
        state_bytes,
        FAST_TIME_SCALE,
        stop,
    ),
)
```

The workers do not exchange physics objects or communicate with one another. The shared stop event coordinates only application shutdown; immutable built-in `bytes` is the only simulation-state payload transferred from the parent.

## Restore and Repeat Each Branch

On normal completion, each worker creates, restores, steps, and destroys its destination scene on the same worker thread. It repeats one fixed five-second rollout so users can inspect it at their own pace in the Debugger:

```python
def _repeat_rollout(name, state_bytes, time_scale, stop):
    scene, articulation = create_simulation(name)
    scene.restore_state_from_bytes(state_bytes)
    while not stop.is_set() and physics.debugger.is_attached():
        for _ in range(ROLLOUT_STEPS):
            joint_target = joint_kick_target(
                scene.get_total_simulation_time(), time_scale
            )
            articulation.set_articulated_target_pose(pose=joint_target)
            scene.step(TIME_STEP)
        scene.restore_state_from_bytes(state_bytes)
    physics.destroy_scene(scene)
```

`ROLLOUT_STEPS` is derived from the five-second motion duration and the simulation time step. Both branches share it, so the double-speed kick finishes early and holds its final pose until the shared reset. Restore resets both the articulation state and scene time, so the same trajectory begins again naturally; each repetition is not a new candidate rollout.

Play, Pause, and Fast Forward apply to every scene. Because the stationary source is selected initially, pressing Play appears to do nothing in that view. Switch to Rollout 1 or Rollout 2 to see motion; both rollout scenes remain alive and follow the same global playback mode. Single Step and the Debugger's restore button apply only to the selected scene and are useful after selecting a rollout, because the source is never stepped.

This is a minimal ownership and state-transfer pattern. Production policy-training systems may use process pools, vectorized simulation, accelerators, or dedicated batching infrastructure. Python threads do not guarantee a speedup for every workload.

## Compatibility and Failure Behavior

Byte restoration requires compatible scene composition and creation order, component layouts, numeric precision, and SuperDex Physics versions. It does not construct missing actors or replace scene and material configuration. A successful structural restore does not prove that persistent configuration is semantically equivalent.

Capture and restore are not transactional:

- If capture fails, the buffer may contain a partial snapshot. Discard it or truncate it to its original size before reuse.
- If restore fails, the destination scene may be partially modified. Restore a known-good checkpoint or recreate the scene before continuing.

The parent signals the persistent workers to stop and joins them inside the executor block. It destroys the source scene only after leaving that block:

```python
with ThreadPoolExecutor(max_workers=2) as executor:
    # Submit and monitor workers.
    # ...
    stop.set()
    for worker in workers:
        worker.result()

physics.destroy_scene(source_scene)
```

On normal completion, each worker destroys its destination scene after its loop ends. Futures propagate worker exceptions to the parent. If attachment fails, the example raises before creating an executor or starting workers. If one persistent worker fails after attachment, the parent sets the shared stop event and stops the Debugger server before joining. Stopping the server is what releases a sibling that is parked inside a paused [`scene.step()`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Scene.step): that wait has no timeout, and the stop event alone is only checked between steps, so the join would otherwise hang.

For clarity, this example does not guard every partially completed setup or rollout operation with cleanup handlers. Production code should use `finally` blocks to destroy worker-owned scenes on their owning threads and to call [`physics.shutdown()`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.shutdown) when setup or worker execution raises; context shutdown or normal interpreter exit otherwise reclaims remaining scenes.

## Features

- **Deterministic source view**: attach while the stationary source is the only scene, then select either rollout to see motion.
- **Common checkpoint**: capture one parent-owned source state for several branches.
- **Caller-owned transfer**: convert the checkpoint to immutable `bytes` before crossing thread boundaries.
- **Independent rollout branches**: restore and repeat two bounded control sequences without worker-to-worker physics communication.
- **Owner-thread lifecycle on normal completion**: create, restore, step, and destroy each destination scene on one worker thread.
- **Peer worker lifecycle**: submit both rollout workers as peers and propagate errors through futures.

## Running

```bash
uv run --no-project examples/example_cross_thread_capture_restore.py
```

The script captures the source scene, then calls [`physics.debugger.attach()`](pathname:///generated/api/v1.0/python/api/debugger.html#superdex.physics.debugger.attach), which launches or focuses the SuperDex Physics Debugger and waits for a connection. The stationary source is selected initially, and the two rollout scenes start after attachment. Press Play once, then switch to either rollout to see both trajectories advance; each resets every five seconds. Closing or disconnecting the Debugger normally ends the loops and cleans up all three scenes. See [Inspecting Scenes](../../debugging_scenes.md) for debugger connection, navigation, and playback controls.
