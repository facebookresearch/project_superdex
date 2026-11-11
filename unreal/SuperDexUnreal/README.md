# SuperDex Unreal

SuperDex Unreal brings [SuperDex Physics](https://facebookresearch.github.io/project_superdex/physics/)
and [SuperDex Robotics](https://facebookresearch.github.io/project_superdex/robotics/) to Unreal
Engine 5.5. This is a ready-to-open Unreal project that carries the SuperDex plugins.

| Plugin | Contents |
| --- | --- |
| [SuperDexPhysics](Plugins/SuperDexPhysics) | The SuperDex Physics simulation in Unreal: models, articulations, constraints, and the editor tools that import and edit them |
| [SuperDexRobotics](Plugins/SuperDexRobotics) | SuperDex bots on top of SuperDex Physics: loading, IK, control, pose sources, teleoperation, recording, and user studies |
| [SuperDexOpenXRPoseSource](Plugins/PoseSourcePlugins/SuperDexOpenXRPoseSource) | Tracked hand poses from OpenXR hand tracking for SuperDexRobotics |
| [SuperDexAssets](Plugins/SuperDexAssets) | Content only: task prefabs |
| [SuperDexExamples](Plugins/SuperDexExamples) | Content only: example levels |

## Usage

The plugins are source plugins. Before Unreal can compile them, build the native SuperDex
libraries they link and stage them into the plugins:

```bash
python hydrate_plugins.py
```

Then open `SuperDexUnreal.uproject` in Unreal Engine 5.5, which compiles the plugin modules the
first time the project opens.

Building needs CMake 3.25+ and Ninja, plus clang-cl 17+ on Windows. On Linux the libraries are
built with the engine's own toolchain, so pass `--ue-root` or set `UE_ROOT`. Windows (Win64) is the
supported platform; Linux is for development. See `hydrate_plugins.py` for the details.

To use the plugins in another project, `../package_plugins.py` packages each plugin as a zip that
unzips into a project's `Plugins/` directory.

See the [repository README](https://github.com/facebookresearch/project_superdex#readme)
for the full list of SuperDex distributions.

## License

First-party code in this project is Apache-2.0 licensed; see [LICENSE](LICENSE). First-party
content (Unreal assets, levels, and icons) is licensed under CC-BY-4.0: the content-only
SuperDexAssets and SuperDexExamples plugins, and the `Content/` and `Resources/` directories of the
other plugins. Each plugin's README names its license files, and the SuperDexAssets README adds IP
notes for its prefabs. Third-party code and dependencies retain their own terms; SuperDexPhysics,
SuperDexRobotics, and TeleopNodeZenoh list the ones they use in their
`thirdparty_licenses/NOTICE.md`.
