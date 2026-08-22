# SuperDex Physics FP64

Double-precision extension for
[SuperDex Physics](https://facebookresearch.github.io/project_superdex/physics/).

Payload only: this distribution carries the `float64` build of the native physics extension
and its shared libraries. The `superdex.physics` API lives in `superdex-physics`, which this
distribution depends on.

You do not normally install this by name. Ask for it through the extra:

```bash
pip install 'superdex-physics[double]'
export SUPERDEX_PRECISION=double
```

Precision is process-wide and resolved at import time, so set `SUPERDEX_PRECISION` before
the first `import superdex.physics`.

See the [repository README](https://github.com/facebookresearch/project_superdex#readme)
for the full list of SuperDex distributions.

## License

First-party code in this distribution is Apache-2.0 licensed; see
[LICENSE](https://github.com/facebookresearch/project_superdex/blob/main/LICENSE).
Third-party code and dependencies retain their own terms.
