# SuperDex Physics Debugger

A standalone debugger UI for inspecting
[SuperDex Physics](https://facebookresearch.github.io/project_superdex/physics/) scenes.

## Usage

Run it without installing anything:

```bash
uvx superdex-physics-debugger
```

Or install it so it stays around:

```bash
pip install superdex-physics-debugger
superdex-physics-debugger
```

## Why it ships on its own

The debugger connects to an already-running simulation over TCP instead of importing the Python
API, so it has no dependencies of its own, not even `superdex-physics`. That is what makes the
`uvx` command above work: uv can build a throwaway environment that contains only this wheel.

Nothing depends on the debugger either. `superdex-physics` does not require it, so you can debug
a simulation from a machine that has no SuperDex installed at all.

## Running the executable directly

The `superdex-physics-debugger` command is a thin launcher around `mochi_debugger`, which ships
inside the package:

```
<site-packages>/superdex_physics_debugger/_native/mochi_debugger
```

Its shared libraries sit next to it and are found through its RPATH, so you can launch this
binary directly instead of going through the command.

## License

The first-party code in this package is licensed Apache-2.0; see
[LICENSE](https://github.com/facebookresearch/project_superdex/blob/main/LICENSE).
All third-party code and dependencies keep their own licenses.
