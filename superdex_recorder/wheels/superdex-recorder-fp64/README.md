# SuperDex Recorder FP64

FP64 extension for
[SuperDex Recorder](https://github.com/facebookresearch/project_superdex/tree/main/superdex_recorder).

Payload only: this distribution carries the FP64 build of the native recorder
extension. The `superdex.recorder` API lives in `superdex-recorder`, and the FP64
physics and robotics libraries it links against live in `superdex-physics-fp64` and
`superdex-robotics-fp64`; all three are dependencies of this distribution.

You do not normally install this by name. Ask for it through the extra:

```bash
pip install 'superdex-recorder[fp64]'
export SUPERDEX_PRECISION=fp64
```

Precision is process-wide and resolved at import time, so set `SUPERDEX_PRECISION` before
the first `import superdex.recorder`.

See the [repository README](https://github.com/facebookresearch/project_superdex#readme)
for the full list of SuperDex distributions.

## License

First-party code in this distribution is Apache-2.0 licensed; see
[LICENSE](https://github.com/facebookresearch/project_superdex/blob/main/LICENSE).
The recorder incorporates Rerun, Apache Arrow, LZ4, and components they bundle; their license
texts ship with this wheel. Third-party code and dependencies retain their own terms.
