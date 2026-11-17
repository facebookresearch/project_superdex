# SuperDex Unreal

SuperDex Unreal brings [SuperDex Physics](https://facebookresearch.github.io/project_superdex/physics/)
and [SuperDex Robotics](https://facebookresearch.github.io/project_superdex/robotics/) to Unreal
Engine 5.5 and 5.8. This is a ready-to-open Unreal project that carries the SuperDex plugins.

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
built with the engine's own toolchain: pass `--ue-root` or set `UE_ROOT`. Windows (Win64) is the
supported platform; Linux is for development. See `hydrate_plugins.py` for the details.

To use the plugins in another project, `../package_plugins.py` packages each plugin as a zip that
unzips into a project's `Plugins/` directory.

See the [repository README](https://github.com/facebookresearch/project_superdex#readme)
for the full list of SuperDex distributions.

## Experimental: Packaging for Meta Quest standalone (Android)

> [!WARNING]
> Running SuperDexUnreal standalone on a Meta Quest headset is currently experimental

This section packages SuperDexUnreal as a standalone Meta Quest app with Unreal Engine 5.8 on
Windows, installs it on a headset over USB, and starts it. The project is already set up for
Quest: it renders through Epic's OpenXR plugin and does **not** require the Meta XR Plugin.

### Prerequisites

1. **Unreal Engine 5.8 with Android support.** Install Unreal Engine 5.8 from the Epic Games
   Launcher. Then add the Android target platform: in the Launcher, open **Unreal Engine**, click the
   arrow next to Launch, choose **Options**, check **Android** under **Target Platforms**, and click
   **Apply**.
2. **Android Studio and the Android SDK.** Install the recommended Android Studio version according
   to Epic's [documentation](https://dev.epicgames.com/documentation/unreal-engine/android-development-requirements-for-unreal-engine?application_version=5.8)
   and start it once. In **Settings > Languages & Frameworks > Android SDK > SDK Tools**, install
   **Android SDK Command-line Tools (latest)**. Then run the setup script that ships with the
   engine. It installs the exact SDK, NDK (r27c, 27.2.12479018), and build tools Unreal 5.8
   expects, and sets the `ANDROID_HOME`, `JAVA_HOME`, and `NDKROOT` environment variables:

   ```bat
   "C:\Program Files\Epic Games\UE_5.8\Engine\Extras\Android\SetupAndroid.bat"
   ```

   Sign out of Windows and back in afterwards, for the new environment variables to take effect.
3. **The native build tools** from [Usage](#usage): Python 3.10 or newer, CMake 3.25 or newer, and
   clang-cl 17 or newer (the **C++ Clang tools for Windows** component of Visual Studio 2022). For
   the Android build, Ninja comes with the Android SDK; the host build needs Ninja on `PATH`.
4. **A Meta Quest in developer mode.** Turn on developer mode for the headset in the Meta Horizon
   phone app. Connect the headset to the PC with a USB cable, put it on, and allow USB debugging
   when it asks. `adb devices` should list the headset as `device`, not `unauthorized`.

### Steps

Run these commands in a Command Prompt (`cmd.exe`), from the `SuperDexUnreal` folder (the folder
with this README).

1. Build the native SuperDex libraries for Windows and for Android, and stage them into the
   plugins. The editor that cooks the Android build uses the Windows libraries, and the app on the
   headset uses the Android ones (run both):

   ```bat
   py -3 hydrate_plugins.py
   py -3 hydrate_plugins.py --android
   ```

2. Close the Unreal Editor if it is open.

3. Build, cook, and package the app:

   ```bat
   set "UE_ROOT=C:\Program Files\Epic Games\UE_5.8"
   "%UE_ROOT%\Engine\Build\BatchFiles\RunUAT.bat" BuildCookRun -project="%CD%\SuperDexUnreal.uproject" -nop4 -utf8output -target=SuperDexUnreal -platform=Android -cookflavor=ASTC -clientconfig=Development -build -cook -stage -package -pak -iostore -compressed -archive -archivedirectory="%CD%\Saved\PackagedBuild"
   ```

   The first run takes a while. The app ends up in `Saved\PackagedBuild\Android_ASTC`: an `.apk`
   file and an `.obb` file with the content.

4. Install the app and its content on the headset:

   ```bat
   adb install -r -g Saved\PackagedBuild\Android_ASTC\SuperDexUnreal-arm64.apk
   adb shell mkdir -p /sdcard/Android/obb/com.meta.superdex.SuperDexUnreal
   adb push Saved\PackagedBuild\Android_ASTC\main.1.com.meta.superdex.SuperDexUnreal.obb /sdcard/Android/obb/com.meta.superdex.SuperDexUnreal/
   ```

5. Put on the headset and start the app:

   ```bat
   adb shell am start -n com.meta.superdex.SuperDexUnreal/com.epicgames.unreal.GameActivity
   ```

   The app also shows up in the headset's library, under **Unknown Sources**.

After a change to the project, repeat steps 3 to 5. Repeat step 1 after a change to the native
SuperDex libraries.

### Troubleshooting

| Problem | Fix |
| --- | --- |
| `Platform Android is not a valid platform to build. SDK validation failed` | Unreal Engine 5.8 needs NDK r27c. Run `SetupAndroid.bat` again, then open a new Command Prompt. `echo %NDKROOT%` should end in `ndk\27.2.12479018`. |
| `hydrate_plugins.py --android` reports TeleopNodeZenoh's sources missing | Run `py -3 hydrate_plugins.py` without `--android` first. |
| `INSTALL_FAILED_UPDATE_INCOMPATIBLE` | A build signed with a different key is installed. Uninstall it with `adb uninstall com.meta.superdex.SuperDexUnreal`, then repeat step 4. |
| `adb devices` lists the headset as `unauthorized` | Put on the headset and allow USB debugging from the pop-up prompt |

## License

First-party code in this project is Apache-2.0 licensed; see [LICENSE](LICENSE). First-party
content (Unreal assets, levels, and icons) is licensed under CC-BY-4.0: the content-only
SuperDexAssets and SuperDexExamples plugins, and the `Content/` and `Resources/` directories of the
other plugins. Each plugin's README names its license files, and the SuperDexAssets README adds IP
notes for its prefabs. Third-party code and dependencies retain their own terms; SuperDexPhysics,
SuperDexRobotics, and TeleopNodeZenoh list the ones they use in their
`thirdparty_licenses/NOTICE.md`.
