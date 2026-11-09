# SuperDex Recorder

Records [SuperDex Physics](https://facebookresearch.github.io/project_superdex/physics/)
simulations as episodes in the [Rerun](https://rerun.io/) RRD format.

The `superdex.recorder` package writes transactional episodes with rigid and articulated state,
robot controllers, step durations, body parameters, and optional contact forces.
`EpisodeReader` reads selected frames back, and `EpisodeRestorer` restores a scene to a
recorded frame. Robot and keypoint shadows add visualization without changing the simulation.

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
