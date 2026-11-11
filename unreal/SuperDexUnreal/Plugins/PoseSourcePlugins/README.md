# Pose source plugins

Optional plugins that feed tracked hand and body poses into SuperDexRobotics. Each source lives in
its own plugin so that SuperDexRobotics does not depend on any device SDK: enable only the sources
your project uses.

| Plugin | Pose source | Needs |
| --- | --- | --- |
| SuperDexOpenXRPoseSource | `USuperDexOpenXRPoseSource`: OpenXR hand tracking (XR_EXT_hand_tracking), on the OpenXR skeleton | Epic's OpenXR and OpenXRHandTracking plugins |

## How a pose source plugin works

- **Discovery.** The pose subsystem (`USuperDexPoseSubsystem`, SuperDexPose) creates one instance
  of every non-abstract `USuperDexPoseSource` class in the loaded modules. Enabling a pose source
  plugin is all it takes to make its sources available; nothing registers them.
- **Missing SDKs.** A plugin's SDK plugins are optional dependencies (`"Optional": true` in its
  `.uplugin`), and its module only uses APIs that every Unreal Engine has (or compiles the SDK code
  out), so it builds and loads without them. Its pose source then overrides
  `ShouldCreatePoseSource` to return false, so the subsystem does not create it: it does not exist
  in the world.
- **Settings.** A source that needs configuration (for example a bot skeleton and its behavior)
  keeps it in its own plugin, typically as a `UDeveloperSettings` class under Project Settings, and
  can also return false from `ShouldCreatePoseSource` until it is configured.
- **Redirects.** If you rename or move a pose source class, add CoreRedirects for its older names to
  the plugin's `Config/Default<Plugin>.ini`, so content saved against them still loads where the
  plugin is enabled.
