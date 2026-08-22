# SuperDex Studio

[SuperDex Studio](https://facebookresearch.github.io/project_superdex/studio/) is a lightweight
desktop app for authoring and visualizing simulation assets. Use it to create, edit, and
validate robots, meshes, task prefabs, and scenes.

## Usage

```bash
pip install superdex-studio
superdex-studio
```

You need a GPU and a display to run it.

## What it ships

Everything Studio needs lives inside the package: the application itself, its shared libraries,
and its runtime resources (`assets/` and `processing_presets/`).

```
<site-packages>/superdex_studio/_native/
    superdex_studio
    libmochi_*.so  libsuperdex_robotics.so  libmarl.so*
    assets/  processing_presets/
```

The libraries are found through the executable's RPATH and the resources sit right beside it, so
you can also launch the `superdex_studio` binary directly (or from a pinned desktop shortcut)
instead of using the `superdex-studio` command.

## The mesh helper

Heavy geometry work (remeshing, SDF reconstruction, and STEP import) runs in a separate process,
`superdex-mesh-cli`, which Studio depends on and installs for you. That separation exists for
licensing reasons: the helper links CGAL and is GPL-3.0-or-later, whereas Studio's own code is
Apache-2.0. The two only ever talk over a pipe and are never linked together.

Studio finds the helper on its own, whether you start Studio from the command or run its binary
directly. See the [`superdex-mesh-cli`
README](https://github.com/facebookresearch/project_superdex/blob/main/superdex_mesh_cli/README.md)
for the full search order, or set `SUPERDEX_MESH_CLI_PATH` to point at a specific copy.

## License

Studio's own code is licensed Apache-2.0; see
[LICENSE](https://github.com/facebookresearch/project_superdex/blob/main/LICENSE).
The [SuperDex CAD Exporter](https://github.com/facebookresearch/project_superdex/tree/main/superdex_studio/superdex_cad_exporter)
is MIT-licensed; see its
[LICENSE](https://github.com/facebookresearch/project_superdex/blob/main/superdex_studio/superdex_cad_exporter/LICENSE).
All other third-party code keeps its own license, bundled in the wheel's `thirdparty_licenses`
payload. `superdex-mesh-cli`, which installs alongside Studio, is GPL-3.0-or-later.
