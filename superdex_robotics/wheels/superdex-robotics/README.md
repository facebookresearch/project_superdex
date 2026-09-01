# SuperDex Robotics

[SuperDex Robotics](https://facebookresearch.github.io/project_superdex/robotics/) is a
robotics SDK that provides robot definitions and composition, controllers, sensors,
actuators, and the framework that aggregates them into complete simulation configs.

Provides the `superdex.robotics` package: robot loading, compositing, control, actuation, and sensing built on top of `superdex-physics`.

```bash
pip install superdex-robotics
```

This wheel carries the single-precision native extension. For double precision, install the
`double` extra -- which pulls in `superdex-robotics-fp64` -- and select it at import time:

```bash
pip install 'superdex-robotics[double]'
export SUPERDEX_PRECISION=double
```

See the [repository README](https://github.com/facebookresearch/project_superdex#readme)
for the full list of SuperDex distributions.

## License

First-party code in this distribution is Apache-2.0 licensed; see
[LICENSE](https://github.com/facebookresearch/project_superdex/blob/main/LICENSE).
Third-party code and dependencies retain their own terms.
