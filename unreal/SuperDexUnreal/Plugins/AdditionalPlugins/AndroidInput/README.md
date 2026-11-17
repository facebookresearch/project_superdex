# Android Input

On Android, Unreal's OpenXR integration takes over the app's native input queue, so gamepad and keyboard events (a Bluetooth controller or keyboard paired with a Meta Quest, for example) never reach the game. This plugin hooks that queue and forwards those events to every player controller as regular Unreal keys:

- Gamepad buttons, thumbsticks and triggers as the `Gamepad_*` keys and axes.
- Keyboard keys as their `EKeys` equivalents, plus `AndroidInput_PrintScreen` for SysRq.

The engine's own input handling still receives every event. On Win64 the module only registers its key, so projects that also target desktop can keep it enabled.

Enable the plugin in your project (it is not enabled by default) and bind the keys as usual, with Enhanced Input or legacy input mappings.

## License

First-party code in this plugin is Apache-2.0 licensed; see [LICENSE](LICENSE).
