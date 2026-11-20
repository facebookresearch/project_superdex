# SuperDex Recorder

[SuperDex Recorder](https://github.com/facebookresearch/project_superdex/tree/main/superdex_recorder)
records SuperDex simulations as episodes in the [Rerun](https://rerun.io/) RRD format: rigid,
articulated, soft and rod state, robot controllers, step durations, body parameters, and optional
contact forces. Open an episode in the Rerun viewer, read it back with `EpisodeReader`, or restore
a scene to a recorded frame with `EpisodeRestorer`.

Provides the `superdex.recorder` package, built on top of `superdex-physics` and
`superdex-robotics`.

```bash
pip install superdex-recorder
```

This wheel carries the FP32 native extension. For FP64, install the `fp64` extra -- which
pulls in `superdex-recorder-fp64` -- and select it at import time:

```bash
pip install 'superdex-recorder[fp64]'
export SUPERDEX_PRECISION=fp64
```

See the [repository README](https://github.com/facebookresearch/project_superdex#readme)
for the full list of SuperDex distributions.

## License

First-party code in this distribution is Apache-2.0 licensed; see
[LICENSE](https://github.com/facebookresearch/project_superdex/blob/main/LICENSE).
The recorder incorporates Rerun, Apache Arrow, LZ4, and components they bundle; their license
texts ship with this wheel. Third-party code and dependencies retain their own terms.
