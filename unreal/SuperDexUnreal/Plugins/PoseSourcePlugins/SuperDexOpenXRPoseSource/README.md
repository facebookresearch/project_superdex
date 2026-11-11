# SuperDex OpenXR Pose Source

Tracked hand poses from OpenXR hand tracking (`XR_EXT_hand_tracking`) for the SuperDexRobotics
plugin, on the OpenXR hand skeleton. Enable this plugin to make `USuperDexOpenXRPoseSource`
available to the SuperDex pose subsystem.

It needs Epic's OpenXR and OpenXRHandTracking plugins. Both are optional dependencies: without
them the plugin still builds and loads, and its pose source is not created. See
[PoseSourcePlugins](https://github.com/facebookresearch/project_superdex/blob/main/unreal/SuperDexUnreal/Plugins/PoseSourcePlugins/README.md) for how pose source plugins work.

## License

First-party code in this plugin is Apache-2.0 licensed; see [LICENSE](LICENSE). Its plugin icon in
`Resources/` is licensed under CC-BY-4.0; see [Resources/LICENSE](Resources/LICENSE). Third-party
code and dependencies retain their own terms.
