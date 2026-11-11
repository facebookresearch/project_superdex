# SuperDex Recorder

Records [SuperDex Physics](https://facebookresearch.github.io/project_superdex/physics/)
simulations as episodes in the [Rerun](https://rerun.io/) RRD format.

The `superdex.recorder` package writes transactional episodes with rigid and articulated state,
robot controllers, step durations, body parameters, and optional contact forces.
`EpisodeReader` reads selected frames back, and `EpisodeRestorer` restores a scene to a
recorded frame. Robot and keypoint shadows add visualization without changing the simulation.

## Recording an episode

Create a `Recorder`, begin an episode, and attach a `PhysicsRecorder` to the scene and a
`RoboticsRecorder` to each robot. Then record every step of your simulation loop and commit the
episode:

```python
import numpy as np
import superdex.physics as sdp
import superdex.recorder as sdrec
import superdex.robotics as sdr

sdp.initialize(num_worker_threads=0)
scene = sdp.create_scene("Recorded example")
robotics_context = sdr.create_context()
bot = sdr.create_bot(scene, sdr.load_bot_prefab_from_file(bot_path), robotics_context)
bot_actor = bot.get_articulated_actor()
all_dofs = np.arange(bot_actor.get_num_dofs(), dtype=np.int32)
time_step = 1.0 / 200.0

recorder = sdrec.Recorder(sdrec.RecorderConfig("recordings"))
metadata = sdrec.EpisodeMetadata(properties={"task": "my_task"})
with recorder.begin_episode(metadata) as episode:
    physics = sdrec.PhysicsRecorder(episode, scene)
    controls = sdrec.RoboticsRecorder(episode, bot)
    physics.record(0, scene.get_total_simulation_time())

    for frame in range(num_steps):
        t = scene.get_total_simulation_time()
        effort = ...  # This step's controller output, one value per DOF.
        bot_actor.set_external_forces_on_dofs(dof_indices=all_dofs, force_values=effort)
        controls.record(frame, t)
        scene.step(time_step)
        physics.record(frame + 1, scene.get_total_simulation_time())

    episode.commit()
```

- Create the recorders after you add every actor to the scene.
- Record frame 0 before the first step, and frame `N + 1` right after step `N`.
- Record each robot at the frame its step starts from: after you apply the step's effort and
  before you call `step`. `EpisodeRestorer` uses this pairing when it resumes from a frame.
- Call `record` from your simulation loop, or from a scene post-step callback in C++.
- Keep the episode, the scene and the robotics context alive while the recorders exist.
- To show a commanded pose or tracked hand keypoints, add a `RobotShadow` or a `SkeletonShadow`
  and record it each step too.

[`example_osc_jsc_recording.py`](../superdex_robotics/examples/control/example_osc_jsc_recording.py)
is the complete version: it drives an arm with OSC and a hand with JSC, and records a shadow of
the commanded pose.

## Building

Enable the recorder in a CMake source build with `-DSUPERDEX_BUILD_RECORDER=ON`. The Python
bindings also need `MOCHI_USE_PYBIND` and `SUPERDEX_BUILD_ROBOTICS`. Both are on by default,
but the lean configure command in the
[C++ build instructions](https://github.com/facebookresearch/project_superdex/blob/main/README.md#building-from-source-c)
turns `MOCHI_USE_PYBIND` off. The build downloads [LZ4](https://github.com/lz4/lz4) 1.9.4 and the
[Rerun C++ SDK](https://github.com/rerun-io/rerun) 0.33.0, which in turn builds
[Apache Arrow](https://github.com/apache/arrow) 18.0.0, and links all three into the recorder.

The CMake build installs only the `superdex_recorder` native extension, not the
`superdex.recorder` Python package, and the `uv sync` source build does not include the recorder.
Neither source build gives you an importable `superdex.recorder` yet.

## License

The first-party code in this directory is licensed Apache-2.0; see
[LICENSE](https://github.com/facebookresearch/project_superdex/blob/main/LICENSE).
The recorder incorporates Rerun, Apache Arrow, LZ4, and components they bundle; see
[`thirdparty_licenses`](thirdparty_licenses/NOTICE.md) for their licenses and license texts.
All other third-party code and dependencies keep their own licenses.
