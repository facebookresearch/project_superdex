# SuperDex Robotics for Unreal

[SuperDex Robotics](https://facebookresearch.github.io/project_superdex/robotics/) is a robotics
SDK that provides robot definitions and composition, controllers, sensors, and actuators. This
plugin brings SuperDex bots into Unreal Engine 5.5, on top of the SuperDexPhysics plugin.

- `SuperDexRobotics` (runtime) loads SuperDex bots and scenes into the SuperDex Physics scene,
  and drives them with controllers and IK.
- `SuperDexPose` (runtime) supplies tracked hand and body poses from pose sources, and the
  skeletons of the supported hands and grippers. Device-specific pose sources live in their own
  plugins; see [PoseSourcePlugins](https://github.com/facebookresearch/project_superdex/blob/main/unreal/SuperDexUnreal/Plugins/PoseSourcePlugins/README.md).
- `SuperDexTeleop` (runtime; Win64, Android) retargets tracked poses onto bots for teleoperation.
- `SuperDexRecorder` (runtime; Win64) records scenes and their bots to Rerun recordings; see its
  [README](Source/SuperDexRecorder/README.md).
- `SuperDexStudy` (runtime) presents questionnaire prompts for user studies.
- `SuperDexRoboticsEditor` (editor) imports bots and scenes.

The plugin carries the SuperDex Robotics native libraries (`superdex_robotics`, and
`superdex_recorder_shared` on Win64) in `ThirdParty/` and their headers in
`Source/<Module>/ThirdParty/`. In a source checkout, `hydrate_plugins.py` in the SuperDex Unreal
project builds and stages them; see the
[SuperDex Unreal README](https://github.com/facebookresearch/project_superdex/tree/main/unreal/SuperDexUnreal#readme).

## License

First-party code in this plugin is Apache-2.0 licensed; see [LICENSE](LICENSE). Its first-party
content, the assets in `Content/` and the icons in `Resources/`, is licensed under CC-BY-4.0; see
[Resources/LICENSE](Resources/LICENSE). Third-party code and dependencies retain their own terms;
the ones this plugin carries are listed in
[thirdparty_licenses/NOTICE.md](thirdparty_licenses/NOTICE.md), beside their license texts.
