/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "app/app.h"
#include "assets/asset_manager.h"
#include "core/common.h"
#include "rendering/render_target.h"
#include "rendering/viewport.h"
#include "simulation/mochi_async_scene.h"
#include "ui/imgui_widgets.h"

#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/path.h>
#include <mochi_mesh/mesh_cli_control.h>
#include <mochi_renderer/image_io.h>
#include <mochi_renderer/type_conversions.h>
#include <superdex_robotics/utils/archive_utils.h>
#include <superdex_robotics/utils/file_utils.h>

#include <ImGuizmo.h>
#include <imgui_internal.h> // ImGuiContext / dock node + tab bar internals for MaybeRaiseLogConsole
#include <imguios/common.h>

#include <tinyfiledialogs.h>

#ifdef MOCHI_MCP_ENABLED
#include <imgui_mcp_bridge.h>
#include "app/mcp_port.h"
#endif

#include <algorithm>
#include <array>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <system_error>

#include "editors/bot_scene_editor.h"
#include "editors/mochi_prefab_editor.h"
#include "editors/model_editor.h"
#include "io/urdf_importer.h"

#if MOCHI_PLATFORM_WINDOWS
#ifndef GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#endif
#ifndef GLFW_EXPOSE_NATIVE_WGL
#define GLFW_EXPOSE_NATIVE_WGL
#endif
#include <GLFW/glfw3native.h>
#elif MOCHI_PLATFORM_LINUX
#define GLFW_EXPOSE_NATIVE_GLX
#include <GLFW/glfw3native.h>
#endif

// Platform headers for resolving the executable's own path (see SuperDexStudio::GetExecutableDir).
// On Windows, <Windows.h> is already available via the force-included windows_compat.h.
#if MOCHI_PLATFORM_MACOS
#include <mach-o/dyld.h> // _NSGetExecutablePath
#elif !MOCHI_PLATFORM_WINDOWS
#include <unistd.h> // readlink
#endif

namespace superdex::studio {

//------------------------------------------------------------------------------------------------
// Helpers
//------------------------------------------------------------------------------------------------

#ifdef MOCHI_MCP_ENABLED
static int GetMcpPort() {
  char const* value = std::getenv("SUPERDEX_MCP_PORT");
  McpPortSelection const selection = SelectMcpPort(value);
  if (value != nullptr && !selection.usedOverride) {
    MOCHI_LOG_WARNING(
        "Ignoring invalid SUPERDEX_MCP_PORT='%s'; using default port %d (expected 1..65535).",
        value,
        kDefaultMcpPort);
  }
  return selection.port;
}
#endif

static mochi::Path const& GetIniFilePath() {
  namespace fs = std::filesystem;
  static mochi::Path const path = [] {
    auto tryBuild = [](mochi::Path const& root) -> mochi::Path {
      try {
        auto const dir = root / "superdex";
        fs::create_directories(dir.ToString());
        return dir / "SuperDexStudio.ini";
      } catch (fs::filesystem_error const& e) {
        MOCHI_LOG_ERROR("Config dir creation failed: %s", e.what());
        return mochi::Path{};
      }
    };
#if MOCHI_PLATFORM_WINDOWS
    if (char const* appdata = std::getenv("APPDATA")) {
      if (auto p = tryBuild(appdata); !p.IsEmpty()) {
        return p;
      }
    }
#else
    if (char const* xdg = std::getenv("XDG_CONFIG_HOME"); xdg && xdg[0] != '\0') {
      if (auto p = tryBuild(xdg); !p.IsEmpty()) {
        return p;
      }
    }
    if (char const* home = std::getenv("HOME")) {
      if (auto p = tryBuild(mochi::Path{home} / ".config"); !p.IsEmpty()) {
        return p;
      }
    }
#endif
    return mochi::Path{"SuperDexStudio.ini"};
  }();
  return path;
}

static mochi::Path const& GetSettingsFilePath() {
  // Store next to the INI file (same directory).
  static mochi::Path settingsFilePath = GetIniFilePath().GetParentPath() / "SuperDexStudio.json";
  return settingsFilePath;
}

constexpr char const* k_dock_space_name = "SuperDexStudioDockSpace";

// Names of all app-managed windows. These double as the keys into
// AppSettings::windowVisibility. AssetEditor-reported window names flow through
// WindowDeclaration::name instead and are not listed here.
constexpr char const* kWindowAssetsName = "Asset Browser";
constexpr char const* kWindowLogConsoleName = "Log Console";
constexpr char const* kWindowSettingsName = "Settings";
constexpr char const* kWindowImGuiDemoName = "ImGui Demo";
constexpr char const* kWindowImPlotDemoName = "ImPlot Demo";
constexpr char const* kWindowEditorName = "Editor"; // always-on host window

// Minimum time between automatic "raise the Log Console tab" actions (see MaybeRaiseLogConsole).
// Short enough to feel responsive, but a gap the user can use to grab/tear out sibling tabs even
// while errors log every frame; the first error after a quiet period surfaces immediately.
constexpr std::chrono::milliseconds kLogConsoleRaiseInterval{500};

// Default visibility for each app-managed window, used to seed windowVisibility
// on first run, for missing keys from older settings files, and on layout reset.
struct AppWindowDefault {
  char const* name;
  bool showByDefault;
};
constexpr std::array<AppWindowDefault, 5> kAppWindowDefaults = {{
    {kWindowAssetsName, true},
    {kWindowLogConsoleName, true},
    {kWindowSettingsName, false},
    {kWindowImGuiDemoName, false},
    {kWindowImPlotDemoName, false},
}};

//--------------------------------------------------------------------------------------------------
// ImGuios::Application
//--------------------------------------------------------------------------------------------------

SuperDexStudio::SuperDexStudio()
    : ImGuios::Application(
          std::make_unique<ImGuios::Window>(
              kDefaultWinWidth,
              kDefaultWinHeight,
              "SuperDex Studio",
              ImGuios::WindowFlags_MSAA | ImGuios::WindowFlags_Maximized |
                  ImGuios::WindowFlags_ScaleToMonitor),
#ifdef MOCHI_MCP_ENABLED
          ImGuios::McpConfig{.port = GetMcpPort(), .enabled = true}
#else
          ImGuios::McpConfig{}
#endif
      ) {
  _editorToRendererConverter = mochi::CoordinateSpaceConverter(_editorSpace, _renderSpace);
  _renderToEditorConverter = mochi::CoordinateSpaceConverter(_renderSpace, _editorSpace);

  LoadFont(
      "Roboto Regular Small",
      14,
      ImGuios::Fonts::Roboto_Regular_ttf,
      ImGuios::Fonts::Roboto_Regular_ttf_len);
  LoadFont(
      "Roboto Bold Small",
      14,
      ImGuios::Fonts::Roboto_Bold_ttf,
      ImGuios::Fonts::Roboto_Bold_ttf_len);
  LoadFont(
      "Roboto Regular Large",
      20,
      ImGuios::Fonts::Roboto_Regular_ttf,
      ImGuios::Fonts::Roboto_Regular_ttf_len);
  LoadFont(
      "Roboto Bold Large",
      20,
      ImGuios::Fonts::Roboto_Bold_ttf,
      ImGuios::Fonts::Roboto_Bold_ttf_len);
  LoadFont(
      "Asset Tile Icons", 50, ImGuios::Fonts::Roboto_Bold_ttf, ImGuios::Fonts::Roboto_Bold_ttf_len);
  LoadFont(
      "Folder Tile Icons",
      80,
      ImGuios::Fonts::Roboto_Bold_ttf,
      ImGuios::Fonts::Roboto_Bold_ttf_len);
  // Monospace font for the read-only mesh-statistics blocks, so their space-padded columns align.
  // A touch larger than the 14px UI font for readability in the stats pane.
  LoadFont(
      "Roboto Mono",
      17,
      ImGuios::Fonts::RobotoMono_Regular_ttf,
      ImGuios::Fonts::RobotoMono_Regular_ttf_len);

  // ImGui's IniFilename is a non-owning `char const*`, so we need to keep the
  // string stable for the lifetime of ImGui's use of it. The Path itself is a
  // function-local static; cache its narrow string the same way.
  static std::string const iniFileStr = GetIniFilePath().ToString();
  ImGui::GetIO().IniFilename = iniFileStr.c_str();

  if (!std::filesystem::exists(iniFileStr)) {
    MOCHI_LOG("%s layout not found.", iniFileStr.c_str());
    _needDefaultDockLayout = true;
  }

  // Configure ImGui config and style.
  ImGui::GetIO().ConfigFlags &= ~ImGuiConfigFlags_NavEnableKeyboard;
  ImGui::StyleColorsSuperdex();
  ImGui::GetStyle().CircleTessellationMaxError = 0.1f;

  // Configure ImGuizmo style.
  ImGuizmo::Style& gizmoStyle = ImGuizmo::GetStyle();
  constexpr float kPlaneAlpha = 0.380392f; // Matches ImGuizmo's default plane alpha.
  for (int axis = 0; axis < 3; ++axis) {
    ImVec4 const color = ImGui::ColorConvertU32ToFloat4(kAxes[axis].normal);
    gizmoStyle.Colors[ImGuizmo::DIRECTION_X + axis] = color;
    gizmoStyle.Colors[ImGuizmo::PLANE_X + axis] = ImVec4{color.x, color.y, color.z, kPlaneAlpha};
  }
  gizmoStyle.Colors[ImGuizmo::SELECTION] = ImVec4{1.0f, 0.6f, 0.0f, 1.0f};
}

void SuperDexStudio::OnInitialize() {
  using namespace mochi_renderer;
#if MOCHI_PLATFORM_MACOS
  // Metal backend for macOS - share the Metal device from ImGuiOS
  // so that textures can be shared between Filament and ImGui
  MochiRenderer::Config config;
  config.backend = filament::Engine::Backend::METAL;
  config.sharedContext = ImGuios::GetMetalDevice();
  config.featureLevel = filament::backend::FeatureLevel::FEATURE_LEVEL_3;
  _mochiRenderer = MochiRenderer::Create(config);
#else
  // OpenGL backend for Windows/Linux
  GLFWwindow* backup_current_context = glfwGetCurrentContext();
  void* main_opengl_context = nullptr;
#if MOCHI_PLATFORM_WINDOWS
  main_opengl_context = (void*)glfwGetWGLContext(GetMainWindow()->GetGLFW());
#elif MOCHI_PLATFORM_LINUX
  main_opengl_context = (void*)glfwGetGLXContext(GetMainWindow()->GetGLFW());
#endif
  glfwMakeContextCurrent(nullptr);

  // Create MochiRenderer with OpenGL backend and shared context
  MochiRenderer::Config config;
  config.backend = filament::Engine::Backend::OPENGL;
  config.sharedContext = main_opengl_context;
  config.featureLevel = filament::backend::FeatureLevel::FEATURE_LEVEL_3;
  _mochiRenderer = MochiRenderer::Create(config);

  glfwMakeContextCurrent(backup_current_context);
#endif

  // Load persistent app settings
  LoadSettings();
  // Seed any missing app-managed window keys with their defaults (does not
  // overwrite values loaded from disk), so app windows exist with correct
  // defaults even when loading older settings files.
  for (auto const& w : kAppWindowDefaults) {
    _appSettings.windowVisibility.try_emplace(w.name, w.showByDefault);
  }

  _defaultIbl = GetResourceManager().LoadIbl(GetDefaultIblPath());
  if (!_defaultIbl) {
    MOCHI_LOG_ERROR("Failed to load default IBL: %s", GetDefaultIblPath().c_str());
  }

  _assetManager = AssetManager::Create(this);
  if (!_assetManager) {
    MOCHI_LOG_ERROR("Failed to create AssetManager.");
    std::abort();
  }

  _assetBrowser = AssetBrowser::Create(this, _assetManager.get());
  if (!_assetBrowser) {
    MOCHI_LOG_ERROR("Failed to create AssetBrowser.");
    std::abort();
  }

  _assetBrowser->SetCurrentPath(_appSettings.assetBrowser.lastPath);
  _botLoader = std::make_unique<SuperDexStudioBotLoader>(&GetAssetManager());
  _renderer = Renderer::Create(
      GetEngine(), _mochiRenderer->GetFilamentRenderer(), kDefaultWinWidth, kDefaultWinHeight);

  // Setup file drop event. Folders are accepted too -- OpenPath opens one as a workspace root.
  GetMainWindow()->OnFileDropCallback = [this](std::vector<std::string> const& paths) {
    for (auto const& path : paths) {
      OpenPath(path);
    }
  };

  // Guard the window close (X) button: veto the close and open the confirmation modal when there
  // are unsaved changes (returning false re-arms GLFW's should-close flag via imguios).
  GetMainWindow()->OnCloseCallback = [this]() -> bool { return RequestAppClose(); };

  _logConsole = std::make_unique<LogConsole>();
  // Verbose capture is gated at the source, so a restored "show verbose" must be re-applied.
  LogConsole::ApplyVerboseCapture(_appSettings.logConsole);
  // Push the restored clear color into the renderer (editors pick up the view settings on create).
  ApplyGraphicsSettings();

  // Deliberately after the Log Console exists, and through the log rather than stderr: this is the
  // only report a windowed launch ever sees. Started from a taskbar shortcut there is no parent
  // console for `AttachParentConsole` to attach to, so anything written to stdout or stderr is
  // discarded. Absence of the helper is a supported configuration -- it is GPL-licensed and ships
  // separately -- so this warns and carries on.
  if (mochi::mesh::ResolveMeshCliPath().empty()) {
    MOCHI_LOG_WARNING(
        "superdex_mesh_cli helper not found, so mesh operations are unavailable. It ships "
        "separately because it is GPL-licensed: install the superdex-mesh-cli distribution, or "
        "set SUPERDEX_MESH_CLI_PATH to point at the executable.");
  }

  _mochiContext = mochi::CreateContext();

  // Enable remote debugger support
  // TODO: Enable later via app settings menu opt-in
  // auto& debugServer = _mochiContext->GetDebugServer();
  // debugServer.SetCoordinateSpace(_editorSpace);
  // debugServer.Start();

  _mochiContext->EnableFileCache(true);

  _botsContext = superdex::robotics::CreateRoboticsContext();

  // Register importers
  RegisterImporter(std::make_unique<UrdfImporter>(this));

  // Auto-load file from environment variable (for headless/MCP automation)
  if (char const* autoLoadPath = std::getenv("MOCHI_AUTO_LOAD")) {
    MOCHI_LOG("Auto-loading: %s", autoLoadPath);
    if (!OpenPath(autoLoadPath)) {
      MOCHI_LOG("Failed to load: %s", autoLoadPath);
    }
  }
}

// If a new error was logged (and "Focus Log Console on Error" is on), brings the Log Console's
// tab to the front of its dock group. When the console is in a different group than the focused
// window this is a soft tab-select (keyboard focus is untouched); when it shares the focused
// window's group it takes real focus (a clean switch, avoiding a soft-select flicker), but only
// between the user's input/selection actions. Throttled so error spam can't pin it or block
// sibling tabs.
void SuperDexStudio::MaybeRaiseLogConsole() {
  // Throttle: bring the console forward at most once per interval so a per-frame error spam can't
  // keep re-pinning its tab (which would make selecting / tearing out sibling tabs in the same dock
  // group impossible). The pending flag stays set until we actually raise, so errors during the
  // quiet window are not skipped. First error after a gap raises immediately (last-raise starts at
  // the epoch).
  auto const now = std::chrono::steady_clock::now();
  if (now - _lastLogConsoleRaise < kLogConsoleRaiseInterval) {
    return;
  }
  // Never fight an in-progress window/tab drag (e.g. the user tearing a tab out of the group).
  ImGuiContext& g = *ImGui::GetCurrentContext();
  if (g.MovingWindow != nullptr) {
    return;
  }
  ImGuiWindow* const logWindow = ImGui::FindWindowByName(kWindowLogConsoleName);
  if (logWindow == nullptr || logWindow->DockNode == nullptr ||
      logWindow->DockNode->TabBar == nullptr) {
    // Not docked in a tab group (floating / alone): nothing to bring forward; treat as handled.
    _lastLogConsoleRaise = now;
    _logConsolePendingRaise = false;
    return;
  }
  // The window the user is interacting with is often a CHILD (e.g. Bot Details renders its contents
  // inside a BeginChild, and a combo's active id lives in that child). Child windows have no
  // DockNode of their own, so walk up ParentWindow to the actually-docked ancestor before comparing
  // groups -- otherwise a text field / combo in the console's own group is misread as "elsewhere".
  auto const dockedAncestor = [](ImGuiWindow* w) -> ImGuiWindow* {
    for (; w != nullptr; w = w->ParentWindow) {
      if (w->DockNode != nullptr) {
        return w;
      }
    }
    return nullptr;
  };
  ImGuiWindow* const focusedDocked = dockedAncestor(g.NavWindow);
  ImGuiWindow* const activeDocked = dockedAncestor(g.ActiveIdWindow);
  auto const sharesConsoleGroup = [&](ImGuiWindow* docked) {
    return docked != nullptr && docked != logWindow && docked->DockNode == logWindow->DockNode;
  };
  // When the user's focused/active window is a DIFFERENT tab in the console's own group, a soft
  // tab-select just flickers (that window re-wins its tab next frame, clearing its widget's active
  // id
  // -- e.g. the text cursor -- for a frame), so give the console real focus instead: a clean switch
  // that interrupts the user. Everywhere else a soft select brings the console forward WITHOUT
  // changing keyboard/nav focus, so a text field or open combo in another group is untouched.
  if (sharesConsoleGroup(focusedDocked) || sharesConsoleGroup(activeDocked)) {
    // Only interrupt BETWEEN the user's input/selection actions -- never while they are typing in a
    // text field, dragging a widget, or have a combo/menu popup open. Stay pending so the console
    // still surfaces the moment the interaction ends.
    bool const userMidInteraction =
        ImGui::IsAnyItemActive() || ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopup);
    if (userMidInteraction) {
      return;
    }
    ImGui::SetWindowFocus(kWindowLogConsoleName);
  } else {
    logWindow->DockNode->TabBar->NextSelectedTabId = logWindow->TabId;
  }
  _lastLogConsoleRaise = now;
  _logConsolePendingRaise = false;
}

void SuperDexStudio::OnUpdate() {
  ImGuizmo::BeginFrame();
  // pump async resource loading
  GetResourceManager().PumpAsyncLoad();
  // render asset thumbnails, gated on mouse interaction (e.g. when interacting with slider)
  if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
    _assetManager->RenderAssetThumbnails(*_renderer);
  }
  // make dock space
  ImGuiID dockSpaceId = ImGui::GetID(k_dock_space_name);
  ImGui::DockSpaceOverViewport(
      dockSpaceId,
      ImGui::GetMainViewport(),
      ImGuiDockNodeFlags_PassthruCentralNode | ImGuiDockNodeFlags_NoDockingInCentralNode);
  // Only build default layout once
  if (_needDefaultDockLayout) {
    MOCHI_LOG("Initializing default layout.");
    BuildDefaultDockLayout();
    _needDefaultDockLayout = false;
    _needDefaultWindowFocus = true;
  }
  // Toggle Focus Mode (must come before windows for needDefaultFocus to work)
  if (ImGui::IsKeyChordPressed(ImGuiKey_F10)) {
    _focusMode = !_focusMode;
    if (!_focusMode) {
      _needDefaultWindowFocus = true;
    }
  }
  // Toggle borderless fullscreen of the OS window.
  if (ImGui::IsKeyChordPressed(ImGuiKey_F11)) {
    auto* mainWindow = GetMainWindow();
    mainWindow->SetBorderlessFullScreen(!mainWindow->IsBorderlessFullScreen());
  }
  // The main menu bar stays visible even in focus mode, so its toggle (and the File actions) remain
  // reachable.
  ShowMainMenu();
  // Runs whether or not the Settings window is open, so closing it mid-edit does not strand the
  // deferred save.
  CommitPendingSettings();
  // show windows
  if (!_focusMode) {
    bool& showImGuiDemo = GetWindowVisible(kWindowImGuiDemoName);
    if (showImGuiDemo) {
      ImGui::ShowDemoWindow(&showImGuiDemo);
    }
    bool& showImPlotDemo = GetWindowVisible(kWindowImPlotDemoName);
    if (showImPlotDemo) {
      ImPlot::ShowDemoWindow(&showImPlotDemo);
    }
    bool& showSettings = GetWindowVisible(kWindowSettingsName);
    if (showSettings) {
      ShowSettingsWindow(&showSettings);
    }
    bool& showAssets = GetWindowVisible(kWindowAssetsName);
    if (showAssets) {
      _assetBrowser->ShowWindow(&showAssets);
    }
    if (auto* editor = GetActiveAssetEditor()) {
      editor->ShowAuxiliaryWindows();
    }
    bool& showLogConsole = GetWindowVisible(kWindowLogConsoleName);
    // Always consume the flag (this clears it) so toggling the setting back on later doesn't fire
    // on a stale error; only act on it when "Focus Log Console on Error" is enabled.
    bool const newErrors = _logConsole->HasNewErrors();
    if (newErrors && _appSettings.logConsole.focusOnError) {
      // Log Console focus-on-error throttle: a new error sets _logConsolePendingRaise, and
      // MaybeRaiseLogConsole brings the console tab forward at most once per interval and clears
      // it, so a per-frame error spam can't keep re-pinning the tab (which would block
      // selecting/tearing out sibling tabs). Starts at the epoch so the first error surfaces
      // immediately.
      _logConsolePendingRaise = true;
      showLogConsole = true;
    }
    // Bring the console tab forward (throttled / guarded). Runs before ShowWindow so the tab-bar
    // selection is applied when the dock node is submitted below.
    if (_logConsolePendingRaise) {
      MaybeRaiseLogConsole();
    }
    if (showLogConsole) {
      if (_logConsole->ShowWindow(&showLogConsole, _appSettings.logConsole)) {
        SaveSettings();
      }
    }
  }
  ShowAssetEditorWindow();
  // Drive the active import modal (if any); clear it once the modal closes.
  if (_activeImporter != nullptr && !_activeImporter->ShowModalWindow()) {
    _activeImporter = nullptr;
  }
  // Drive the async-task progress modal (if a background batch is running).
  _asyncTasks.ShowModalWindow();
  // Drive the unsaved-changes close-confirmation modal (if a close was requested while dirty).
  ShowUnsavedChangesModal();
  // Handle common inputs
  if (ImGui::IsKeyChordPressed(ImGuiKey_S | ImGuiMod_Ctrl | ImGuiMod_Shift)) {
    SaveAllAssetEditors();
  } else if (ImGui::IsKeyChordPressed(ImGuiKey_S | ImGuiMod_Ctrl)) {
    SaveActiveAssetEditor();
  }
  if (ImGui::IsKeyChordPressed(ImGuiKey_W | ImGuiMod_Ctrl)) {
    if (!_assetEditors.empty() && _activeAssetEditorIdx >= 0) {
      CloseAssetEditor(_activeAssetEditorIdx);
    }
  }
  if (!_assetEditors.empty() && ImGui::IsKeyChordPressed(ImGuiKey_PageUp)) {
    auto next =
        std::clamp(_activeAssetEditorIdx + 1, 0, static_cast<int>(_assetEditors.size() - 1));
    SelectAssetEditor(next);
  }
  if (!_assetEditors.empty() && ImGui::IsKeyChordPressed(ImGuiKey_PageDown)) {
    auto prev =
        std::clamp(_activeAssetEditorIdx - 1, 0, static_cast<int>(_assetEditors.size() - 1));
    SelectAssetEditor(prev);
  }
  // Handle editor specific inputs.
  if (auto* editor = GetActiveAssetEditor()) {
    auto& undoStack = editor->GetUndoStack();
    // Undo/Redo
    if (editor->CanUndoRedo() && undoStack.IsInitialized()) {
      if (ImGui::IsKeyChordPressed(ImGuiKey_Z | ImGuiMod_Ctrl | ImGuiMod_Shift)) {
        undoStack.Redo();
      } else if (ImGui::IsKeyChordPressed(ImGuiKey_Z | ImGuiMod_Ctrl)) {
        undoStack.Undo();
      } else if (ImGui::IsKeyChordPressed(
                     ImGuiKey_Y | ImGuiMod_Ctrl)) { // NOLINT(bugprone-branch-clone)
        undoStack.Redo();
      }
    }
    // Undo stack: push snapshot if edited and mouse released
    if (undoStack.IsInitialized()) {
      undoStack.MaybePushSnapshot(ImGui::IsMouseDown(ImGuiMouseButton_Left));
    }
    // Handle editor inputs if ImGui doesn't want key capture (e.g. text input)
    if (!ImGui::GetIO().WantCaptureKeyboard) {
      editor->OnHandleInputs();
    }
  }
  // Allow editors to render stuff if the need to.
  if (auto* editor = GetActiveAssetEditor()) {
    editor->OnRender(_renderer.get());
  }
  // Focus windows post update (work around ImGui docking limitations)
  static bool firstFrame = true;
  if (firstFrame || _needDefaultWindowFocus) {
    // focus primary windows
    ImGui::SetWindowFocus(kWindowAssetsName);
    // focus editor windows in reverse order so first declared are prioritized if in same node
    if (auto* activeEditor = GetActiveAssetEditor()) {
      auto const& auxWindows = activeEditor->GetAuxiliaryWindows();
      for (auto it = auxWindows.rbegin(); it != auxWindows.rend(); ++it) {
        auto const& w = *it;
        _appSettings.windowVisibility.try_emplace(w.name, w.showByDefault);
        if (GetWindowVisible(w.name)) {
          ImGui::SetWindowFocus(w.name);
        }
      }
    }
    firstFrame = false;
    _needDefaultWindowFocus = false;
  }
}

void SuperDexStudio::OnShutdown() {
  SaveSettings();
  _logConsole.reset();
  while (!_assetEditors.empty()) {
    CloseAssetEditor(static_cast<int>(_assetEditors.size()) - 1);
  }
  if (_botsContext) {
    superdex::robotics::DestroyRoboticsContext(_botsContext);
    _botsContext = nullptr;
  }
  if (_mochiContext) {
    mochi::DestroyContext(_mochiContext);
    _mochiContext = nullptr;
  }
  _assetBrowser.reset();
  _assetManager.reset();
  _renderer.reset();
  _mochiRenderer.reset();
}

//--------------------------------------------------------------------------------------------------
// Accessors
//--------------------------------------------------------------------------------------------------

filament::Engine* SuperDexStudio::GetEngine() const {
  return _mochiRenderer->GetEngine();
}

mochi::Context* SuperDexStudio::GetMochiContext() const {
  return _mochiContext;
}

superdex::robotics::RoboticsContext* SuperDexStudio::GetRoboticsContext() const {
  return _botsContext;
}

AssetEditor* SuperDexStudio::GetActiveAssetEditor() const {
  if (_activeAssetEditorIdx >= 0 &&
      _activeAssetEditorIdx < static_cast<int>(_assetEditors.size())) {
    return _assetEditors[_activeAssetEditorIdx].get();
  }
  return nullptr;
}

bool& SuperDexStudio::GetWindowVisible(std::string const& name) {
  return _appSettings.windowVisibility[name];
}

std::filesystem::path SuperDexStudio::GetExecutableDir() {
#if MOCHI_PLATFORM_WINDOWS
  wchar_t buffer[MAX_PATH];
  DWORD const length = GetModuleFileNameW(nullptr, buffer, MAX_PATH);
  if (length == 0 || length == MAX_PATH) {
    return {};
  }
  return std::filesystem::path(std::wstring(buffer, length)).parent_path();
#elif MOCHI_PLATFORM_MACOS
  uint32_t size = 0;
  _NSGetExecutablePath(nullptr, &size);
  std::string buffer(size, '\0');
  if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
    return {};
  }
  return std::filesystem::path(buffer.c_str()).parent_path();
#else
  char buffer[4096];
  ssize_t const length = readlink("/proc/self/exe", buffer, sizeof(buffer));
  if (length <= 0) {
    return {};
  }
  return std::filesystem::path(std::string(buffer, static_cast<std::size_t>(length))).parent_path();
#endif
}

std::string const& SuperDexStudio::GetDefaultIblPath() {
  static std::string const path =
      (GetExecutableDir() / "assets" / "ibl" / "studio_small_08_1k.hdr").string();
  return path;
}

mochi_renderer::IBL* SuperDexStudio::GetDefaultIbl() const {
  return _defaultIbl;
}

mochi_renderer::SceneViewSettings const& SuperDexStudio::GetViewSettings() const {
  return _appSettings.graphics.view;
}

AssetManager& SuperDexStudio::GetAssetManager() const {
  return *_assetManager;
}

mochi_renderer::ResourceManager& SuperDexStudio::GetResourceManager() const {
  return *_mochiRenderer->GetResourceManager();
}

AssetBrowser& SuperDexStudio::GetAssetBrowser() const {
  return *_assetBrowser;
}

AppSettings& SuperDexStudio::GetAppSettings() {
  return _appSettings;
}

mochi::CoordinateSpaceConverter const& SuperDexStudio::GetEditorToRendererSpaceConverter() const {
  return _editorToRendererConverter;
}

mochi::CoordinateSpaceConverter const& SuperDexStudio::GetRendererToEditorSpaceConverter() const {
  return _renderToEditorConverter;
}

SuperDexStudioBotLoader const& SuperDexStudio::GetBotLoader() const {
  return *_botLoader;
}

//--------------------------------------------------------------------------------------------------
// Asset Editors
//--------------------------------------------------------------------------------------------------

void SuperDexStudio::OpenAssetEditor(Asset* asset) {
  if (asset == nullptr) {
    MOCHI_LOG_ERROR("Cannot open Asset for editing because it was null.");
    return;
  }
  auto existing = FindAssetEditorIndex(asset);
  if (existing != -1) {
    SelectAssetEditor(existing);
    // The editor already represents this asset; let it focus on the just-opened one (e.g. the Model
    // Editor shows that slot and hides the others, matching a fresh open).
    _assetEditors[static_cast<std::size_t>(existing)]->OnReopenedFor(asset);
  } else if (auto editor = asset->CreateEditor(this)) {
    editor->_needsSelect = true;
    editor->Initialize();
    _assetManager->RegisterReferencer(editor.get());
    _assetEditors.push_back(std::move(editor));
    ActivateAssetEditor(static_cast<int>(_assetEditors.size()) - 1);
  } else {
    MOCHI_LOG_ERROR(
        "Cannot open Asset %s (%s) because it provides no AssetEditor class.",
        asset->GetName().c_str(),
        asset->GetTypeLabel());
  }
}

void SuperDexStudio::RefreshEditors(std::vector<IAssetReferencer*> const& referencers) {
  if (referencers.empty()) {
    return;
  }
  // Affected assets (the Asset subset of `referencers`); used to refresh editors that represent
  // them but are not themselves referencers (e.g. the bot editor for its bot asset).
  std::vector<Asset*> affectedAssets;
  affectedAssets.reserve(referencers.size());
  for (IAssetReferencer* referencer : referencers) {
    if (auto* asset = dynamic_cast<Asset*>(referencer)) {
      affectedAssets.push_back(asset);
    }
  }
  for (auto const& editorPtr : _assetEditors) {
    AssetEditor* editor = editorPtr.get();
    // Refresh the editor if it directly referenced a rewritten path, or if it represents an asset
    // whose references changed.
    bool const refresh =
        std::find(referencers.begin(), referencers.end(), static_cast<IAssetReferencer*>(editor)) !=
            referencers.end() ||
        std::any_of(affectedAssets.begin(), affectedAssets.end(), [&](Asset* asset) {
          return editor->RepresentsAsset(asset);
        });
    if (refresh) {
      editor->Refresh();
    }
  }
}

void SuperDexStudio::ActivateAssetEditor(int index) {
  if (index < 0 || index >= static_cast<int>(_assetEditors.size())) {
    _activeAssetEditorIdx = -1;
    return;
  }
  AssetEditor* oldEditor = GetActiveAssetEditor();
  if (oldEditor) {
    oldEditor->OnDeactivate();
  }
  _activeAssetEditorIdx = index;
  if (AssetEditor* newEditor = _assetEditors[index].get()) {
    for (auto const& w : newEditor->GetAuxiliaryWindows()) {
      _appSettings.windowVisibility.try_emplace(w.name, w.showByDefault);
    }
    // focus aux windows if the previous editor was null or wasn't the same type of editor
    if (oldEditor == nullptr || typeid(*oldEditor) != typeid(*newEditor)) {
      _needDefaultWindowFocus = true;
    }
    newEditor->OnActivate();
  }
}

void SuperDexStudio::CloseAssetEditor(int index) {
  if (index < 0 || index >= static_cast<int>(_assetEditors.size())) {
    return;
  }
  _assetManager->UnregisterReferencer(_assetEditors[index].get());
  _assetEditors[index]->Shutdown();
  _assetEditors.erase(_assetEditors.begin() + index);
  int newActive = _activeAssetEditorIdx;
  if (newActive >= static_cast<int>(_assetEditors.size())) {
    newActive = static_cast<int>(_assetEditors.size()) - 1;
  }
  _activeAssetEditorIdx = -1;
  if (newActive >= 0) {
    _assetEditors[newActive]->_needsSelect = true;
    ActivateAssetEditor(newActive);
  }
}

int SuperDexStudio::FindAssetEditorIndex(Asset const* asset) const {
  if (asset == nullptr) {
    return -1;
  }
  for (int i = 0; i < static_cast<int>(_assetEditors.size()); ++i) {
    // Match any asset the editor represents (e.g. the Model Editor's CAD/render/mochi slots), so
    // opening a sibling that is already slotted focuses that editor instead of opening a new one.
    if (_assetEditors[i]->RepresentsAsset(asset)) {
      return i;
    }
  }
  return -1;
}

void SuperDexStudio::SelectAssetEditor(int index) {
  if (index >= 0 && index < static_cast<int>(_assetEditors.size())) {
    _assetEditors[index]->_needsSelect = true;
    ActivateAssetEditor(index);
  }
}

void SuperDexStudio::SaveActiveAssetEditor() {
  if (auto* editor = GetActiveAssetEditor()) {
    editor->Save();
  }
}

void SuperDexStudio::SaveAllAssetEditors() {
  for (auto& editor : _assetEditors) {
    if (editor->GetAsset()->IsDirty()) {
      editor->Save();
    }
  }
}

bool SuperDexStudio::GuardUnsavedAssets(char const* prompt, std::function<void()> onProceed) {
  // Snapshot every dirty asset (not just those with an open editor), each defaulted to "save"
  // (checked). The modal is app-modal and blocks input, so the asset set stays stable while open.
  _unsavedEntries.clear();
  GetAssetManager().ForEachAsset([this](Asset* asset, mochi::Path const&) {
    if (asset->IsDirty()) {
      _unsavedEntries.push_back({asset, true});
    }
  });
  if (_unsavedEntries.empty()) {
    return true; // Nothing unsaved: caller may proceed immediately.
  }
  _onUnsavedProceed = std::move(onProceed);
  _unsavedPrompt = prompt;
  _openUnsavedChangesModal = true; // Open the confirmation modal next frame.
  return false;
}

bool SuperDexStudio::RequestAppClose() {
  return GuardUnsavedAssets(
      "Would you like to save your unsaved work before closing?", [this] { Stop(); });
}

void SuperDexStudio::ShowUnsavedChangesModal() {
  if (_openUnsavedChangesModal) {
    ImGui::OpenPopup("Save Assets");
    _openUnsavedChangesModal = false;
  }
  // Center on the main viewport every frame (handles window resize / dpi changes).
  ImVec2 const center = ImGui::GetMainViewport()->GetCenter();
  ImGui::SetNextWindowPos(center, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
  // Fixed default width, auto height (0 height component auto-fits to content each frame).
  ImGui::SetNextWindowSize(ImVec2(640, 0), ImGuiCond_Always);
  if (ImGui::BeginPopupModal(
          "Save Assets",
          nullptr,
          ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings)) {
    ImGui::TextUnformatted(_unsavedPrompt);
    ImGui::Spacing();

    // Table of unsaved assets, styled like the URDF importer / prefab collision matrix: bordered
    // rows with a frozen header and blended (frameless) per-row widgets. The first three columns
    // auto-size to content (SizingFixedFit); Path stretches to fill the remaining width.
    constexpr ImGuiTableFlags kTableFlags = ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders |
        ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_ScrollY;
    // Bound the height so a long list scrolls instead of growing the modal off-screen.
    float const tableHeight = ImGui::GetTextLineHeightWithSpacing() * 10.0f;
    if (ImGui::BeginTable("##UnsavedAssets", 4, kTableFlags, ImVec2(0, tableHeight))) {
      ImGui::TableSetupColumn("Save");
      ImGui::TableSetupColumn("Asset");
      ImGui::TableSetupColumn("Type");
      ImGui::TableSetupColumn("Path", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupScrollFreeze(0, 1);
      ImGui::TableHeadersRow();

      // Blend the per-row checkbox into the table (matching the URDF importer / collision matrix).
      ImGui::PushFramelessWidgetStyle();
      for (size_t iRow = 0; iRow < _unsavedEntries.size(); ++iRow) {
        auto& entry = _unsavedEntries[iRow];
        Asset const* asset = entry.asset;
        ImGui::PushID(static_cast<int>(iRow));
        ImGui::TableNextRow();

        // Save checkbox (centered in its column).
        ImGui::TableNextColumn();
        float const cellWidth = ImGui::GetContentRegionAvail().x;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (cellWidth - ImGui::GetFrameHeight()) * 0.5f);
        ImGui::Checkbox("##save", &entry.save);

        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(asset->GetName().c_str());

        // Type label, tinted with the asset type's color.
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(
            ImGui::ColorConvertU32ToFloat4(asset->GetColor()), "%s", asset->GetTypeLabel());

        // Read-only path field: lets the user select and scroll to the end of long paths.
        ImGui::TableNextColumn();
        std::string pathStr = asset->GetPath().ToString();
        ImGui::SetNextItemWidth(-FLT_MIN);
        ImGui::InputText(
            "##path", pathStr.data(), pathStr.size() + 1, ImGuiInputTextFlags_ReadOnly);

        ImGui::PopID();
      }
      ImGui::PopFramelessWidgetStyle();
      ImGui::EndTable();
    }

    ImGui::Separator();
    ImGui::Spacing();

    // Shared modal teardown: optionally run the deferred proceed action, then clear state and
    // close the popup. Cancel passes runProceed=false so the pending action is discarded.
    auto closeModal = [this](bool runProceed) {
      std::function<void()> proceed = runProceed ? std::move(_onUnsavedProceed) : nullptr;
      _onUnsavedProceed = nullptr;
      _unsavedEntries.clear();
      ImGui::CloseCurrentPopup();
      if (proceed) {
        proceed();
      }
    };

    if (ImGui::Button("Save Selected", ImVec2(140, 0))) {
      // Save the checked assets; unchecked assets are left dirty (discarded). Use the open editor's
      // Save (updates its undo "saved" marker) only when it owns this exact asset; otherwise save
      // the asset directly, since an editor may merely represent (not own) a sibling slot asset.
      bool allSaved = true;
      for (auto& entry : _unsavedEntries) {
        if (!entry.save) {
          continue;
        }
        int const editorIdx = FindAssetEditorIndex(entry.asset);
        bool saved = false;
        if (editorIdx >= 0 && _assetEditors[editorIdx]->GetAsset() == entry.asset) {
          saved = _assetEditors[editorIdx]->Save();
        } else if (entry.asset->Save()) {
          entry.asset->SetDirty(false);
          saved = true;
        }
        if (saved) {
          // Uncheck what succeeded so a retry after a partial failure only re-saves what failed.
          entry.save = false;
        } else {
          MOCHI_LOG_WARNING("Failed to save asset '%s'.", entry.asset->GetName().c_str());
          allSaved = false;
        }
      }
      // Only proceed (which may close the app or unload assets) once every selected asset saved;
      // otherwise keep the modal open so the user can retry or cancel without losing changes.
      if (allSaved) {
        closeModal(true);
      }
    }
    ImGui::SameLine();
    if (ImGui::Button("Don't Save", ImVec2(140, 0))) {
      closeModal(true);
    }
    ImGui::SameLine();
    // Abandons the pending close/reset entirely: the app stays open with every asset untouched,
    // exactly as if the close button (or menu item) had never been pressed.
    if (ImGui::Button("Cancel", ImVec2(140, 0))) {
      closeModal(false);
    }
    ImGui::EndPopup();
  }
}

//--------------------------------------------------------------------------------------------------
// Export
//--------------------------------------------------------------------------------------------------

void SuperDexStudio::SaveAssetThumbnail(
    Asset& asset,
    int sizePx,
    mochi::Path const& outFile,
    mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF_NOT(asset.RendersThumbnail(), error, "Asset does not render a thumbnail.");
  MOCHI_ERROR_IF(sizePx <= 0, error, "Thumbnail export size must be positive.");
  MOCHI_ERROR_RETURN(error);

  // Render the asset into a temporary target at the requested resolution using the same path as
  // the browser tiles, so framing/lighting match. The transparent clear is applied inside
  // RenderAssetToTarget.
  std::unique_ptr<RenderTarget> target = RenderTarget::Create(GetEngine(), sizePx, sizePx);
  MOCHI_ERROR_IF(target == nullptr, error, "Failed to create render target for thumbnail export.");
  MOCHI_ERROR_RETURN(error);

  GetAssetManager().RenderAssetToTarget(asset, *target, *_renderer);

  std::vector<uint8_t> pixels;
  _renderer->ReadPixels(*target, pixels, error);
  MOCHI_ERROR_RETURN(error);

  mochi_renderer::WritePng(outFile, sizePx, sizePx, 4, pixels, error);
}

void SuperDexStudio::SaveViewportScreenshot(
    Viewport& viewport,
    mochi::Path const& outFile,
    mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);

  RenderTarget const& target = *viewport.GetRenderTarget();
  int width = 0;
  int height = 0;
  target.GetSize(width, height);
  MOCHI_ERROR_IF(width <= 0 || height <= 0, error, "Viewport has no rendered pixels to save.");
  MOCHI_ERROR_RETURN(error);

  std::vector<uint8_t> pixels;
  _renderer->ReadPixels(target, pixels, error);
  MOCHI_ERROR_RETURN(error);

  mochi_renderer::WritePng(outFile, width, height, 4, pixels, error);
}

//--------------------------------------------------------------------------------------------------
// File
//--------------------------------------------------------------------------------------------------

mochi::Path SuperDexStudio::GetFileDialogPath(
    char const* title,
    char const* const* filters,
    int numFilters,
    char const* filterDesc,
    bool isSaveDialog,
    mochi::Path const& defaultPathAndFile) {
#ifdef MOCHI_MCP_ENABLED
  std::string intercepted = ImGuiMcpBridge_ConsumeFileDialogResult();
  if (!intercepted.empty()) {
    return mochi::Path{intercepted};
  }
#endif
  // tinyfd derives the dialog's initial folder from the default path's directory, but on Windows
  // the native dialog only parses it when the separators are native ('\'); forward slashes make it
  // fall back to the last-used folder. Normalize to the platform-preferred separator first.
  std::filesystem::path defaultNative(defaultPathAndFile.ToString());
  defaultNative.make_preferred();
  std::string const defaultStr = defaultNative.string();
  char const* result = isSaveDialog
      ? tinyfd_saveFileDialog(title, defaultStr.c_str(), numFilters, filters, filterDesc)
      : tinyfd_openFileDialog(title, defaultStr.c_str(), numFilters, filters, filterDesc, 0);
  return result ? mochi::Path{std::string{result}} : mochi::Path{};
}

mochi::Path SuperDexStudio::GetFolderDialogPath(char const* title, mochi::Path const& defaultPath) {
#ifdef MOCHI_MCP_ENABLED
  std::string intercepted = ImGuiMcpBridge_ConsumeFileDialogResult();
  if (!intercepted.empty()) {
    return mochi::Path{intercepted};
  }
#endif
  std::string const defaultStr = defaultPath.ToString();
  char const* result = tinyfd_selectFolderDialog(title, defaultStr.c_str());
  return result ? mochi::Path{std::string{result}} : mochi::Path{};
}

bool SuperDexStudio::OpenPath(mochi::Path const& path) {
  // A folder is not an asset. Resolved before anything else so it can never reach the asset
  // classifier and come back as "unsupported asset type" -- not even a folder whose name carries
  // an asset extension (e.g. a `foo.superdex_bot/` directory).
  std::error_code ec;
  if (std::filesystem::is_directory(path.AsFilesystemPath(), ec)) {
    return AddFolderToWorkspace(path);
  }

  // A root marker file stands for the folder containing it.
  if (path.GetFilename() == superdex::robotics::kRootMarkerFile
#if MOCHI_INTERNAL
      || path.GetFilename() == superdex::robotics::kRootMarkerFileLegacy
#endif
  ) {
    return AddFolderToWorkspace(path.GetParentPath());
  }

  if (!OpenFile(path)) {
    return false;
  }
  AddRecentFile(path);
  return true;
}

bool SuperDexStudio::AddFolderToWorkspace(mochi::Path const& path) {
  if (!_assetBrowser->AddRootPath(path)) {
    return false;
  }
  // AddRootPath only navigates when it actually adds a root, and it skips a folder already covered
  // by an existing one. Navigating here as well means opening a folder that is already in the
  // workspace reveals it instead of doing nothing visible. (SetCurrentPath ignores a no-op
  // re-navigation, so this does not duplicate a history entry when the root was just added.)
  _assetBrowser->SetCurrentPath(path);
  AddRecentFolder(path);
  // Successful open: bring the Asset Browser forward. (A refusal logs an error, which the general
  // error handler already surfaces the Log Console for when focusOnError is set.)
  ImGui::SetWindowFocus(kWindowAssetsName);
  return true;
}

bool SuperDexStudio::OpenFile(mochi::Path const& path) {
  int existing = FindAssetEditorIndex(_assetManager->FindAssetByPath(path));
  if (existing >= 0) {
    _assetEditors[existing]->_needsSelect = true;
    ActivateAssetEditor(existing);
    return true;
  }

  if (Importer* importer = FindImporterForPath(path)) {
    BeginImport(importer, path);
    return true;
  }

  Asset* asset = _assetManager->LoadAsset(path);
  if (!asset) {
    MOCHI_LOG_ERROR("Failed to load: %s", path.ToString().c_str());
    return false;
  }
  if (superdex::robotics::IsBotPath(path.ToString()) ||
      superdex::robotics::IsBotArchivePath(path.ToString())) {
    auto botsRoot = superdex::robotics::FindBotsRoot(path.ToString());
    _assetBrowser->AddRootPath(botsRoot ? mochi::Path{*botsRoot} : path.GetParentPath());
  } else {
    _assetBrowser->AddRootPath(path.GetParentPath());
  }
  _assetBrowser->SelectAsset(asset);
  OpenAssetEditor(asset);
  return true;
}

void SuperDexStudio::RegisterImporter(std::unique_ptr<Importer> importer) {
  _importers.push_back(std::move(importer));
}

Importer* SuperDexStudio::FindImporterForPath(mochi::Path const& path) const {
  auto ext = path.GetExtension();
  std::transform(
      ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
  for (auto const& importer : _importers) {
    for (auto const& candidate : importer->GetExtensions()) {
      if (ext == candidate) {
        return importer.get();
      }
    }
  }
  return nullptr;
}

bool SuperDexStudio::CanImport() const {
  // Imports write generated assets into the asset browser's current directory,
  // so importing requires an active directory. Block while a background import
  // batch is still running so its writes can't race a new import.
  return !_assetBrowser->GetCurrentPath().IsEmpty() && !_asyncTasks.IsRunning();
}

void SuperDexStudio::BeginImport(Importer* importer, mochi::Path const& path) {
  if (importer == nullptr) {
    return;
  }
  if (!CanImport()) {
    MOCHI_LOG_WARNING(
        "Cannot import without an active asset browser directory: %s", path.ToString().c_str());
    return;
  }
  if (_activeImporter != nullptr) {
    MOCHI_LOG_WARNING("An import is already in progress; ignoring: %s", path.ToString().c_str());
    return;
  }
  if (_asyncTasks.IsRunning()) {
    MOCHI_LOG_WARNING(
        "A background import is still running; ignoring: %s", path.ToString().c_str());
    return;
  }
  _activeImporter = importer;
  importer->BeginImport(path, _assetBrowser->GetCurrentPath());
}

void SuperDexStudio::CreateRootMarker(mochi::Path const& dir, mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);
  // Don't overwrite an existing marker: it may hold a tag table we must not clobber.
  if (superdex::robotics::FindRootMarker(dir.AsFilesystemPath()).has_value()) {
    return;
  }
  auto const markerPath = dir.AsFilesystemPath() / superdex::robotics::kRootMarkerFile;
  mochi::WriteFile(markerPath, mochi::Span<char const>{}, error);
}

bool SuperDexStudio::BeginAsyncTasks(
    std::string title,
    std::vector<AsyncTask> tasks,
    std::function<void(bool allSucceeded)> onComplete,
    bool serial) {
  // Only one batch may run at a time (a single runner drives a single modal).
  // Reject re-entry defensively so a second caller can't clobber the in-flight
  // batch. Arguments are left untouched (not moved) so the caller can retry.
  if (_asyncTasks.IsRunning()) {
    MOCHI_LOG_WARNING("An async task batch is already running; ignoring '%s'.", title.c_str());
    return false;
  }
  AsyncTaskRunner::Config config;
  config.title = std::move(title);
  config.tasks = std::move(tasks);
  config.onComplete = std::move(onComplete);
  config.context = _mochiContext;
  config.serial = serial;
  // All studio background work runs in the superdex_mesh_cli helper, so a Cancel terminates the
  // in-flight helper subprocess(es); the blocked worker then unblocks and the batch winds down.
  config.onCancel = []() { mochi::mesh::CancelInFlightMeshCli(); };
  _asyncTasks.Begin(std::move(config));
  return true;
}

bool SuperDexStudio::IsAsyncTasksRunning() const {
  return _asyncTasks.IsRunning();
}

// Adds `path` to the front of `list`, deduplicating any prior entry that
// refers to the same file (mochi::Path normalizes case/separators/lexical form).
static void
AddRecentEntry(std::vector<std::string>& list, mochi::Path const& path, int maxEntries) {
  std::erase_if(list, [&](std::string const& existing) { return mochi::Path{existing} == path; });
  list.insert(list.begin(), path.ToString());
  if (static_cast<int>(list.size()) > maxEntries) {
    list.resize(maxEntries);
  }
}

void SuperDexStudio::AddRecentFile(mochi::Path const& path) {
  AddRecentEntry(_appSettings.recentEntries.files, path, kMaxRecentFiles);
  SaveSettings();
}

void SuperDexStudio::AddRecentFolder(mochi::Path const& path) {
  AddRecentEntry(_appSettings.recentEntries.folders, path, kMaxRecentFiles);
  SaveSettings();
}

void SuperDexStudio::LoadSettings() {
  auto const& path = GetSettingsFilePath();
  if (!std::filesystem::exists(path.ToString())) {
    return;
  }
  mochi::ErrorLog error;
  _appSettings = superdex::robotics::LoadParamsFromFile<AppSettings>(path.ToString(), error, true);
  if (!error.IsOK()) {
    return; // Don't set _settingsLoadedOK — prevents SaveSettings from overwriting the file.
  }
  // Prune recent entries that no longer exist on disk
  std::erase_if(_appSettings.recentEntries.files, [](std::string const& p) {
    return !std::filesystem::exists(p);
  });
  std::erase_if(_appSettings.recentEntries.folders, [](std::string const& p) {
    return !std::filesystem::exists(p);
  });
}

void SuperDexStudio::SaveSettings() {
  mochi::ErrorLog error;
  superdex::robotics::SaveParamsToFile(_appSettings, GetSettingsFilePath().ToString(), error);
}

//--------------------------------------------------------------------------------------------------
// ImGui
//--------------------------------------------------------------------------------------------------

void SuperDexStudio::ShowMainMenu() {
  mochi::ErrorLog error;

  auto* editor = GetActiveAssetEditor();
  if (ImGui::BeginMainMenuBar()) {
    // MenuBar -> File
    if (ImGui::BeginMenu("File")) {
      // Generic open (.superdex_bot / .mochi_scene / .mochi_prefab / .mochi_bot_scene / .urdf;
      // legacy .mochi_bot / .mochi_bot_archive still accepted)
      if (ImGui::MenuItem("Open...", "Ctrl+O")) {
        char const* filters[] = {
            "*.superdex_bot",
            "*.superdex_bot_archive",
            "*.mochi_bot",
            "*.mochi_bot_archive",
            "*.mochi_scene",
            "*.mochi_prefab",
#if MOCHI_INTERNAL
            "*.mochi_bot_scene",
            "*.mochi_bot_scene_archive",
#endif
            "*.urdf",
            "*.h5",
            "*.glb",
            "*.gltf",
            "*.step",
            "*.stp",
            "*.stl"};
        auto path =
            GetFileDialogPath("Open File", filters, IM_ARRAYSIZE(filters), "Mochi Files", false);
        if (!path.IsEmpty()) {
          OpenPath(path);
        }
      }
      if (ImGui::MenuItem("Add Folder to Workspace...")) {
        auto folder = GetFolderDialogPath("Add Folder to Workspace");
        if (!folder.IsEmpty()) {
          AddFolderToWorkspace(folder);
        }
      }
      auto const& recentFiles = _appSettings.recentEntries.files;
      auto const& recentFolders = _appSettings.recentEntries.folders;
      bool hasRecent = !recentFiles.empty() || !recentFolders.empty();
      if (ImGui::BeginMenu("Open Recent", hasRecent)) {
        // Recent files
        int removeFileIndex = -1;
        // Opening is deferred until after the loop: OpenPath re-records what it opened, which
        // reorders `recentFiles` and would leave `recentPath` dangling mid-iteration.
        int openFileIndex = -1;
        for (int i = 0; i < static_cast<int>(recentFiles.size()); ++i) {
          auto const& recentPath = recentFiles[i];
          auto label = ICON_FA_FILE "  " + mochi::Path(recentPath).GetFilename();
          ImGui::PushID(i);
          if (ImGui::MenuItem(label.c_str())) {
            openFileIndex = i;
          }
          if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", recentPath.c_str());
          }
          ImGui::PopID();
        }
        // An entry that no longer opens (deleted, or no longer a supported type) drops off the
        // list.
        if (openFileIndex >= 0) {
          mochi::Path const file{recentFiles[openFileIndex]};
          if (!OpenPath(file)) {
            removeFileIndex = openFileIndex;
          }
        }
        if (removeFileIndex >= 0) {
          _appSettings.recentEntries.files.erase(
              _appSettings.recentEntries.files.begin() + removeFileIndex);
          SaveSettings();
        }
        // Separator and recent folders
        if (!recentFiles.empty() && !recentFolders.empty()) {
          ImGui::Separator(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
        }
        int removeFolderIndex = -1;
        // Deferred for the same reason as the files loop above.
        int openFolderIndex = -1;
        for (int i = 0; i < static_cast<int>(recentFolders.size()); ++i) {
          auto const& folderPath = recentFolders[i];
          auto label = ICON_FA_FOLDER "  " + mochi::Path(folderPath).ToString();
          ImGui::PushID(static_cast<int>(recentFiles.size()) + i);
          if (ImGui::MenuItem(label.c_str())) {
            openFolderIndex = i;
          }
          ImGui::PopID();
        }
        if (openFolderIndex >= 0) {
          // Identical to picking the folder from File -> Add Folder to Workspace, or dropping it on
          // the window. The staleness check stays here so a folder deleted since it was recorded
          // drops off the list quietly, rather than being reported as a failed asset load.
          mochi::Path const folder{recentFolders[openFolderIndex]};
          std::error_code ec;
          if (!std::filesystem::is_directory(folder.AsFilesystemPath(), ec) ||
              !AddFolderToWorkspace(folder)) {
            removeFolderIndex = openFolderIndex;
          }
        }
        if (removeFolderIndex >= 0) {
          _appSettings.recentEntries.folders.erase(
              _appSettings.recentEntries.folders.begin() + removeFolderIndex);
          SaveSettings();
        }
        ImGui::Separator(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
        if (ImGui::MenuItem("Clear Recently Opened...")) {
          _appSettings.recentEntries.files.clear();
          _appSettings.recentEntries.folders.clear();
          SaveSettings();
        }
        ImGui::EndMenu();
      }
      if (ImGui::BeginMenu("Import", !_importers.empty() && CanImport())) {
        for (auto& importer : _importers) {
          auto const label = importer->GetDisplayName() + "...";
          if (ImGui::MenuItem(label.c_str())) {
            std::vector<std::string> patterns;
            patterns.reserve(importer->GetExtensions().size());
            for (auto const& ext : importer->GetExtensions()) {
              patterns.push_back("*" + ext);
            }
            std::vector<char const*> filters;
            filters.reserve(patterns.size());
            for (auto const& pattern : patterns) {
              filters.push_back(pattern.c_str());
            }
            auto const title = "Import " + importer->GetDisplayName();
            // macOS: tinyfiledialogs' open panel filters by Uniform Type Identifier, not by glob,
            // so a custom extension with no registered type (e.g. .urdf) greys out every matching
            // file. Disable the filter there so files stay selectable -- the importer, and thus the
            // expected extension, is already chosen from this menu.
            int numFilters = static_cast<int>(filters.size());
#if MOCHI_PLATFORM_MACOS
            numFilters = 0;
#endif
            auto path = GetFileDialogPath(
                title.c_str(),
                filters.data(),
                numFilters,
                importer->GetDisplayName().c_str(),
                false);
            if (!path.IsEmpty()) {
              BeginImport(importer.get(), path);
              AddRecentFile(path);
            }
          }
        }
        ImGui::EndMenu();
      }
      // Import lands in the asset browser's current directory, so it's disabled without one.
      // Point the user at the fix on hover, even while the menu is disabled.
      if (_assetBrowser->GetCurrentPath().IsEmpty() &&
          ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("Open a folder or workspace to import files.");
      }
      ImGui::Separator(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
      // Save
      ImGui::BeginDisabled(!editor || !editor->GetAsset()->IsDirty());
      if (ImGui::MenuItem("Save", "Ctrl+S")) {
        SaveActiveAssetEditor();
      }
      ImGui::EndDisabled();
      if (ImGui::MenuItem("Save All", "Ctrl+Shift+S")) {
        SaveAllAssetEditors();
      }
      // Save Viewport Screenshot: writes the active editor's rendered viewport (at its native pixel
      // size) to a PNG chosen via a save dialog, defaulting to <asset name>_screenshot.png next to
      // the asset. Only available when the active editor has a viewport.
      ImGui::Separator(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
      ImGui::BeginDisabled(!editor || !editor->GetViewport());
      if (ImGui::MenuItem("Save Screenshot...")) {
        // The enclosing BeginDisabled() guarantees a disabled MenuItem returns false, so reaching
        // here means both editor and its viewport are non-null.
        Viewport* viewport = editor->GetViewport();
        auto* const asset = editor->GetAsset();
        mochi::Path const defaultPath =
            asset->GetPath().GetParentPath() / (asset->GetName() + "_screenshot.png");
        char const* filters[] = {"*.png"};
        mochi::Path const chosen = GetFileDialogPath(
            "Save Viewport Screenshot",
            filters,
            1,
            "PNG Image (*.png)",
            /*isSaveDialog=*/true,
            defaultPath);
        if (!chosen.IsEmpty()) {
          SaveViewportScreenshot(*viewport, chosen, mochi::ErrorLog{});
        }
      }
      ImGui::EndDisabled();
      ImGui::Separator(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
      // Show-only (not a visibility toggle) so re-picking this while the window is buried behind
      // others brings it forward instead of closing it.
      if (ImGui::MenuItem("Settings...")) {
        GetWindowVisible(kWindowSettingsName) = true;
        ImGui::SetWindowFocus(kWindowSettingsName);
      }
      ImGui::Separator(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
      if (ImGui::MenuItem("Reset Studio")) {
        auto doReset = [this] {
          while (!_assetEditors.empty()) {
            CloseAssetEditor(static_cast<int>(_assetEditors.size()) - 1);
          }
          _assetBrowser->SelectAsset(nullptr);
          _assetBrowser->ClearRootPaths();
          GetAssetManager().UnloadAllAssets();
        };
        if (GuardUnsavedAssets(
                "Would you like to save your unsaved work before resetting?", doReset)) {
          doReset();
        }
      }
      ImGui::Separator(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));

      // Quit SuperDexStudio
      if (ImGui::MenuItem("Exit")) {
        if (RequestAppClose()) {
          Stop();
        }
      }
      ImGui::EndMenu();
    }

    // MenuBar -> Edit
    if (ImGui::BeginMenu("Edit")) {
      bool const canUndo =
          editor ? editor->CanUndoRedo() && editor->GetUndoStack().CanUndo() : false;
      bool const canRedo =
          editor ? editor->CanUndoRedo() && editor->GetUndoStack().CanRedo() : false;
      if (ImGui::MenuItem("Undo", "Ctrl+Z", false, canUndo)) {
        editor->GetUndoStack().Undo();
      }
      if (ImGui::MenuItem("Redo", "Ctrl+Y", false, canRedo)) {
        editor->GetUndoStack().Redo();
      }
      ImGui::EndMenu();
    }

    // MenuBar -> Window
    if (ImGui::BeginMenu("Window")) {
      ImGui::BeginDisabled(_focusMode);
      ImGui::MenuItem("Asset Browser", nullptr, &GetWindowVisible(kWindowAssetsName));
      ImGui::MenuItem(kWindowLogConsoleName, nullptr, &GetWindowVisible(kWindowLogConsoleName));
      ImGui::Separator(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
      auto* activeEditor = GetActiveAssetEditor();
      if (activeEditor) {
        bool auxWinShown = false;
        for (auto const& w : activeEditor->GetAuxiliaryWindows()) {
          if (!w.debug) {
            ImGui::MenuItem(w.name, nullptr, &GetWindowVisible(w.name));
            auxWinShown = true;
          }
        }
        if (auxWinShown) {
          ImGui::Separator(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
        }
      }
      if (ImGui::BeginMenu("Debug")) {
        if (activeEditor) {
          bool auxWinShown = false;
          for (auto const& w : activeEditor->GetAuxiliaryWindows()) {
            if (w.debug) {
              ImGui::MenuItem(w.name, nullptr, &GetWindowVisible(w.name));
              auxWinShown = true;
            }
          }
          if (auxWinShown) {
            ImGui::Separator(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
          }
        }
        ImGui::MenuItem(kWindowImGuiDemoName, nullptr, &GetWindowVisible(kWindowImGuiDemoName));
        ImGui::MenuItem(kWindowImPlotDemoName, nullptr, &GetWindowVisible(kWindowImPlotDemoName));
        ImGui::EndMenu();
      }
      ImGui::EndDisabled(); // _focusMode
      ImGui::Separator(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
      ImGui::MenuItem("Focus Mode", "F10", &_focusMode);
      bool borderless = GetMainWindow()->IsBorderlessFullScreen();
      if (ImGui::MenuItem("Borderless Fullscreen", "F11", &borderless)) {
        GetMainWindow()->SetBorderlessFullScreen(borderless);
      }
      ImGui::Separator(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
      if (ImGui::MenuItem("Reset Layout")) {
        _needDefaultDockLayout = true;
      }
      ImGui::EndMenu();
      ImGui::MarkIniSettingsDirty();
    }

    // Re-query the active editor: earlier menu actions (e.g. "Reset Studio")
    // may have destroyed the editor cached at the top of this frame, leaving
    // the local `editor` pointer dangling.
    if (auto* activeEditor = GetActiveAssetEditor()) {
      activeEditor->ShowMainMenuItems();
    }

    ImGui::EndMainMenuBar();
  }
}

bool SuperDexStudio::ShowGraphicsSettings() {
  auto& view = _appSettings.graphics.view;
  bool changed = false;
  changed |= ImGui::BoolCombo("MSAA", &view.msaaEnabled);
  ImGui::BeginDisabled(!view.msaaEnabled);
  {
    constexpr int msaaValues[] = {1, 2, 4, 8, 16};
    constexpr char const* msaaLabels[] = {"1", "2", "4", "8", "16"};
    int msaaIndex = 0;
    for (int i = 0; i < IM_ARRAYSIZE(msaaValues); ++i) {
      if (msaaValues[i] == view.msaaSampleCount) {
        msaaIndex = i;
        break;
      }
    }
    if (ImGui::Combo("MSAA Samples", &msaaIndex, msaaLabels, IM_ARRAYSIZE(msaaLabels))) {
      view.msaaSampleCount = msaaValues[msaaIndex];
      changed = true;
    }
  }
  ImGui::EndDisabled();

  changed |= ImGui::BoolCombo("Shadows", &view.shadowsEnabled);
  changed |= ImGui::BoolCombo("SSAO", &view.ssaoEnabled);

  changed |= ImGui::BoolCombo("Post Processing", &view.postProcessingEnabled);
  ImGui::BeginDisabled(!view.postProcessingEnabled);
  changed |= ImGui::BoolCombo("Bloom", &view.bloomEnabled);
  ImGui::BeginDisabled(!view.bloomEnabled);
  changed |= ImGui::SliderFloat("Bloom Strength", &view.bloomStrength, 0.0f, 1.0f);
  ImGui::EndDisabled();
  changed |= ImGui::BoolCombo("Vignette", &view.vignetteEnabled);
  ImGui::EndDisabled();
  ImGui::BeginDisabled(!view.postProcessingEnabled);
  changed |= ImGui::SliderFloat("Exposure", &view.exposure, -10.0f, 10.0f);
  {
    // NB: if we add new ToneMappers, this will need to be updated
    static char const* ToneMapperSettingNames[] = {
        "Linear", "ACES", "Filmic", "PBR Neutral", "GT7"};
    int toneMapperIndex = static_cast<int>(view.toneMapper);
    if (ImGui::Combo(
            "Tone Mapper",
            &toneMapperIndex,
            ToneMapperSettingNames,
            IM_ARRAYSIZE(ToneMapperSettingNames))) {
      view.toneMapper = static_cast<mochi_renderer::ToneMapperSetting>(toneMapperIndex);
      changed = true;
    }
  }
  ImGui::EndDisabled();
  changed |= ImGui::BoolCombo("Skybox", &view.showSkybox);
  ImGui::BeginDisabled(view.showSkybox);
  changed |= ImGui::ColorEdit4(
      "Clear Color", _appSettings.graphics.clearColor.data(), ImGuiColorEditFlags_AlphaPreview);
  ImGui::EndDisabled();
  return changed;
}

void SuperDexStudio::ApplyGraphicsSettings() {
  for (auto& editor : _assetEditors) {
    editor->ApplySceneViewSettings(_appSettings.graphics.view);
  }
  _renderer->SetClearColor(mochi_renderer::ToFilament(_appSettings.graphics.clearColor));
}

bool SuperDexStudio::ShowPhysicsSettings() {
  bool changed = false;
  if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Indent();
    changed |= ImGui::SimpleReflectionStruct(_appSettings.physics.scene);
    ImGui::Unindent();
  }
  if (ImGui::CollapsingHeader("Solver")) {
    ImGui::Indent();
    changed |= ImGui::SimpleReflectionStruct(_appSettings.physics.solver);
    ImGui::Unindent();
  }
  if (ImGui::CollapsingHeader("Debug Draw", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Indent();
    changed |= ShowDebugDrawSettings(GetMochiContext(), _appSettings.physics.debug).AnyChange();
    ImGui::Unindent();
  }
  if (ImGui::CollapsingHeader("Studio")) {
    ImGui::Indent();
    changed |= ImGui::SimpleReflectionStruct(_appSettings.physics.studio);
    // Drag tuning is studio-only too, and unlike the rest of the physics settings it is never
    // seeded into an editor, so it lives here rather than in the per-editor window.
    if (ImGui::CollapsingHeader("Physics Drag")) {
      ImGui::Indent();
      changed |= ImGui::SimpleReflectionStruct(_appSettings.physicsDrag);
      ImGui::Unindent();
    }
    ImGui::Unindent();
  }
  return changed;
}

// One entry per Settings-window category, in display order. `applyLive` runs on the frame a value
// changes (cheap live preview); `applyOnCommit` runs once the edit is released (expensive work).
// Either may be null.
struct SettingsCategory {
  char const* label;
  bool (*draw)(SuperDexStudio&);
  void (*reset)(SuperDexStudio&);
  void (*applyLive)(SuperDexStudio&);
  void (*applyOnCommit)(SuperDexStudio&);

  void ApplyLive(SuperDexStudio& app) const {
    if (applyLive != nullptr) {
      applyLive(app);
    }
  }
  void ApplyOnCommit(SuperDexStudio& app) const {
    if (applyOnCommit != nullptr) {
      applyOnCommit(app);
    }
  }
};

// Builds a category backed by a reflected AppSettings sub-struct.
template <auto Member>
constexpr SettingsCategory MakeSettingsCategory(
    char const* label,
    void (*applyLive)(SuperDexStudio&) = nullptr,
    void (*applyOnCommit)(SuperDexStudio&) = nullptr,
    void (*reset)(SuperDexStudio&) = [](SuperDexStudio& app) {
      app.GetAppSettings().*Member = {};
    }) {
  return {
      label,
      [](SuperDexStudio& app) {
        return ImGui::SimpleReflectionStruct(app.GetAppSettings().*Member);
      },
      reset,
      applyLive,
      applyOnCommit};
}

constexpr std::array<SettingsCategory, 7> kSettingsCategories = {{
    // Hand-drawn rather than reflection-driven: the debug draw feature list is queried from mochi
    // at runtime, so it cannot come from the settings struct. No apply hooks -- these are the
    // defaults a simulation session starts from; a running one is driven by the editor's own
    // Physics Settings window.
    SettingsCategory{
        "Physics",
        [](SuperDexStudio& app) { return app.ShowPhysicsSettings(); },
        [](SuperDexStudio& app) {
          app.GetAppSettings().physics = {};
          app.GetAppSettings().physicsDrag = {};
        },
        nullptr,
        nullptr},
    // Graphics is hand-drawn rather than reflection-driven: the MSAA and tone-mapper combos and the
    // post-processing enable/disable chain don't fall out of the field types.
    SettingsCategory{
        "Graphics",
        [](SuperDexStudio& app) { return app.ShowGraphicsSettings(); },
        [](SuperDexStudio& app) { app.GetAppSettings().graphics = {}; },
        [](SuperDexStudio& app) { app.ApplyGraphicsSettings(); },
        nullptr},
    MakeSettingsCategory<&AppSettings::viewport>(
        "Viewport",
        [](SuperDexStudio& app) { app.ApplyHighlightOverlayOpacity(); }),
    MakeSettingsCategory<&AppSettings::assetBrowser>(
        "Asset Browser",
        nullptr,
        [](SuperDexStudio& app) { app.GetAssetBrowser().Refresh(); },
        [](SuperDexStudio& app) {
          // Preferences only: rootPaths / lastPath are the user's open workspace, so a reset that
          // cleared them would close it.
          AppSettings& s = app.GetAppSettings();
          auto rootPaths = std::move(s.assetBrowser.rootPaths);
          auto lastPath = std::move(s.assetBrowser.lastPath);
          s.assetBrowser = {};
          s.assetBrowser.rootPaths = std::move(rootPaths);
          s.assetBrowser.lastPath = std::move(lastPath);
        }),
    MakeSettingsCategory<&AppSettings::logConsole>(
        "Log Console",
        [](SuperDexStudio& app) {
          LogConsole::ApplyVerboseCapture(app.GetAppSettings().logConsole);
        }),
    MakeSettingsCategory<&AppSettings::botEditor>("Bot Editor"),
    MakeSettingsCategory<&AppSettings::botVisualization>("Bot Visualization"),
}};

void SuperDexStudio::ApplyHighlightOverlayOpacity() {
  auto const opacity = static_cast<float>(_appSettings.viewport.selection.highlightOverlayOpacity);
  for (auto& editor : _assetEditors) {
    if (Viewport* viewport = editor->GetViewport()) {
      viewport->HighlightOverlayAlpha() = opacity;
    }
  }
}

void SuperDexStudio::NotifyAppSettingsChanged() {
  for (auto& editor : _assetEditors) {
    editor->OnAppSettingsChanged(_appSettings);
  }
}

void SuperDexStudio::ResetAllSettings() {
  for (auto const& category : kSettingsCategories) {
    category.reset(*this);
  }
  SaveSettings();
  for (auto const& category : kSettingsCategories) {
    category.ApplyLive(*this);
    category.ApplyOnCommit(*this);
  }
  NotifyAppSettingsChanged();
}

void SuperDexStudio::CommitPendingSettings() {
  if (_settingsPendingSaveCategory < 0 || ImGui::IsAnyItemActive()) {
    return;
  }
  auto const& category = kSettingsCategories[_settingsPendingSaveCategory];
  _settingsPendingSaveCategory = -1;
  SaveSettings();
  category.ApplyOnCommit(*this);
  NotifyAppSettingsChanged();
}

void SuperDexStudio::ShowSettingsWindow(bool* open) {
  // Same DPI rule as BuildDefaultDockLayout: macOS reports logical points, Windows/Linux pixels.
  float const fbScale = ImGui::GetIO().DisplayFramebufferScale.x;
  float const layoutScale = fbScale > 1.0f ? 1.0f : GetDpiScale();
  ImGui::SetNextWindowSize(
      ImVec2(720.0f * layoutScale, 520.0f * layoutScale), ImGuiCond_FirstUseEver);
  // Deliberately absent from BuildDefaultDockLayout so it comes up floating, but still dockable.
  ImGui::Begin(kWindowSettingsName, open);

  // Reserve the footer button row so both panes fill the remaining height.
  float const paneHeight = ImGui::GetContentRegionAvail().y - ImGui::GetFrameHeightWithSpacing();

  ImGui::BeginChild(
      "##Categories",
      ImVec2(180.0f * layoutScale, paneHeight),
      ImGuiChildFlags_Border | ImGuiChildFlags_ResizeX);
  for (int i = 0; i < static_cast<int>(kSettingsCategories.size()); ++i) {
    if (ImGui::Selectable(kSettingsCategories[i].label, i == _settingsCategoryIdx)) {
      _settingsCategoryIdx = i;
    }
  }
  ImGui::EndChild();
  ImGui::SameLine();

  // Bound after the list so a selection made this frame takes effect immediately.
  auto const& category = kSettingsCategories[_settingsCategoryIdx];

  // AlwaysUseWindowPadding keeps the reflected widgets off the panel edges.
  ImGui::BeginChild(
      "##Detail",
      ImVec2(0, paneHeight),
      ImGuiChildFlags_Border | ImGuiChildFlags_AlwaysUseWindowPadding);
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted(category.label);
  ImGui::SameLine();
  char const* const resetLabel = "Reset to Default";
  float const resetWidth =
      ImGui::CalcTextSize(resetLabel).x + ImGui::GetStyle().FramePadding.x * 2.0f;
  ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - resetWidth);
  if (ImGui::Button(resetLabel)) {
    category.reset(*this);
    SaveSettings();
    category.ApplyLive(*this);
    category.ApplyOnCommit(*this);
    NotifyAppSettingsChanged();
  }
  ImGui::Separator();
  // Nested child so the header row above stays pinned while the fields scroll.
  ImGui::BeginChild("##DetailFields");
  ImGui::PushStyleColor(ImGuiCol_Header, ImGui::GetStyleColorVec4(ImGuiCol_Button));
  ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImGui::GetStyleColorVec4(ImGuiCol_ButtonHovered));
  ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
  bool const changed = category.draw(*this);
  ImGui::PopStyleColor(3);
  ImGui::EndChild();
  ImGui::EndChild();

  if (ImGui::Button("Open Settings File Location")) {
    SaveSettings(); // The folder is always there, but the file may not be until we write it.
    auto& pio = ImGui::GetPlatformIO();
    if (pio.Platform_OpenInShellFn) {
      auto const dir = GetSettingsFilePath().GetParentPath().ToString();
      pio.Platform_OpenInShellFn(ImGui::GetCurrentContext(), dir.c_str());
    }
  }
  ImGui::SameLine();
  if (ImGui::Button("Reset All Settings...")) {
    _openResetSettingsModal = true;
  }

  // Preview edits live, but write the file (and run the expensive apply) only once the edit is
  // released, so dragging a value doesn't save or rescan every frame. The commit itself runs from
  // CommitPendingSettings, which the frame loop calls whether or not this window is open.
  if (changed) {
    _settingsPendingSaveCategory = _settingsCategoryIdx;
    category.ApplyLive(*this);
  }

  if (_openResetSettingsModal) {
    ImGui::OpenPopup("Reset All Settings");
    _openResetSettingsModal = false;
  }
  ImGui::SetNextWindowPos(
      ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
  if (ImGui::BeginPopupModal(
          "Reset All Settings",
          nullptr,
          ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove |
              ImGuiWindowFlags_NoSavedSettings)) {
    ImGui::TextUnformatted("Restore every setting to its default value?");
    ImGui::TextUnformatted("Recent files and the open workspace are kept.");
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    if (ImGui::Button("Reset All", ImVec2(140, 0))) {
      ResetAllSettings();
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel", ImVec2(140, 0))) {
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }

  ImGui::End();
}

void SuperDexStudio::ShowAssetEditorWindow() {
  ImGui::Begin("Editor");

  // No edtiors; show placeholder and skip viewport rendering and return
  if (_assetEditors.empty()) {
    ImVec2 avail = ImGui::GetContentRegionAvail();
    char const* hint = "Open a file or folder to get started (File > Open)";
    ImVec2 textSize = ImGui::CalcTextSize(hint);
    ImGui::SetCursorPos({(avail.x - textSize.x) * 0.5f, (avail.y - textSize.y) * 0.5f});
    ImGui::TextDisabled("%s", hint);
    ImGui::End(); // Begin("Editor")
    return;
  }

  // don't show all tabs in focus mode
  if (_focusMode) {
    if (auto activeEditor = GetActiveAssetEditor()) {
      activeEditor->ShowTabContents();
    }
    ImGui::End(); // Editor
    return;
  }

  // Show tab bar
  int closeEditor = -1;
  int closeOtherEditorsExcept = -1;
  bool closeAllEditors = false;
  if (ImGui::BeginTabBar(
          "EditorTabs",
          ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_FittingPolicyScroll |
              ImGuiTabBarFlags_DrawSelectedOverline)) {
    for (int iEd = 0; iEd < static_cast<int>(_assetEditors.size()); ++iEd) {
      auto* editor = _assetEditors[iEd].get();
      ImGui::PushID(iEd);
      bool tabOpen = true;
      ImGuiTabItemFlags flags = 0;
      if (editor->GetAsset()->IsDirty()) {
        flags |= ImGuiTabItemFlags_UnsavedDocument;
      }
      if (editor->_needsSelect) {
        flags |= ImGuiTabItemFlags_SetSelected;
        editor->_needsSelect = false;
      }
      auto const tabColor = editor->GetTabColor();
      ImGui::PushStyleColor(ImGuiCol_TabSelectedOverline, tabColor);
      if (ImGui::BeginTabItem(editor->GetTabDisplayName().c_str(), &tabOpen, flags)) {
        if (_activeAssetEditorIdx != iEd) {
          ActivateAssetEditor(iEd);
        }
        ImGui::EndTabItem();
      }
      if (ImGui::BeginPopupContextItem()) {
        if (ImGui::MenuItem("Close")) {
          closeEditor = iEd;
        }
        if (ImGui::MenuItem("Close Others")) {
          closeOtherEditorsExcept = iEd;
        }
        if (ImGui::MenuItem("Close All")) {
          closeAllEditors = true;
        }
        ImGui::Separator(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
        if (ImGui::MenuItem("Copy Asset Filename")) {
          ImGui::SetClipboardText(editor->GetAsset()->GetPath().GetFilename().c_str());
        }
        if (ImGui::MenuItem("Copy Asset Path")) {
          ImGui::SetClipboardText(editor->GetAsset()->GetPath().ToString().c_str());
        }
        ImGui::Separator(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
        if (ImGui::MenuItem("Select in Asset Browser")) {
          _assetBrowser->SelectAsset(editor->GetAsset());
        }
        if (ImGui::MenuItem("Reveal in File Explorer")) {
          auto dir = editor->GetAsset()->GetPath().GetParentPath().ToString();
          auto& pio = ImGui::GetPlatformIO();
          if (pio.Platform_OpenInShellFn) {
            pio.Platform_OpenInShellFn(ImGui::GetCurrentContext(), dir.c_str());
          }
        }
        ImGui::EndPopup();
      }
      ImGui::PopStyleColor();
      ImGui::PopID();
      if (!tabOpen) {
        CloseAssetEditor(iEd);
        --iEd;
      }
    }
    ImGui::EndTabBar();
  }

  // deferred close tabs
  if (closeEditor != -1) {
    CloseAssetEditor(closeEditor);
  } else if (closeOtherEditorsExcept != -1) {
    auto* keep = _assetEditors[closeOtherEditorsExcept].get();
    while (_assetEditors.size() > 1) {
      int idx = _assetEditors[0].get() == keep ? 1 : 0;
      CloseAssetEditor(idx);
    }
  } else if (closeAllEditors) {
    while (!_assetEditors.empty()) {
      CloseAssetEditor(0);
    }
  }

  // Show active editor tab contents
  if (auto* activeEditor = GetActiveAssetEditor()) {
    activeEditor->ShowTabContents();
  }

  ImGui::End(); // Begin("Editor")
}

// Docks an AssetEditor type's default windows into the given layout nodes according to each
// window's declared DockRegion. TAssetEditor must provide a static GetDefaultWindows().
template <typename TAssetEditor>
static void
DockEditorDefaultWindows(ImGuiID sidePanelTop, ImGuiID sidePanelBottom, ImGuiID bottomBar) {
  for (auto const& w : TAssetEditor::GetDefaultWindows()) {
    ImGuiID node = 0;
    switch (w.dock) {
      case AssetEditor::DockRegion::SidePanelTop:
        node = sidePanelTop;
        break;
      case AssetEditor::DockRegion::SidePanelBottom:
        node = sidePanelBottom;
        break;
      case AssetEditor::DockRegion::BottomBar:
        node = bottomBar;
        break;
      case AssetEditor::DockRegion::Floating:
        break;
    }
    if (node) {
      ImGui::DockBuilderDockWindow(w.name, node);
    }
  }
}

void SuperDexStudio::BuildDefaultDockLayout() {
  ImGuiID dockSpaceId = ImGui::GetID(k_dock_space_name);
  ImGui::DockBuilderRemoveNode(dockSpaceId);
  ImGui::DockBuilderAddNode(dockSpaceId, ImGuiDockNodeFlags_DockSpace);

  auto win_size_pix = ImGui::GetMainViewport()->Size;
  // On macOS, ImGui works in logical points (DisplayFramebufferScale > 1) — no scaling needed.
  // On Windows/Linux, ImGui works in physical pixels — scale layout to match display DPI.
  float const fbScale = ImGui::GetIO().DisplayFramebufferScale.x;
  float const layoutScale = fbScale > 1.0f ? 1.0f : GetDpiScale();
  float const k_bottom_height_pix = kDefaultWinHeight * 0.22f * layoutScale;
  float const k_right_width_pix = kDefaultWinWidth * 0.3f * layoutScale;
  float const bottom_height = k_bottom_height_pix / win_size_pix.y;
  float const right_width = k_right_width_pix / win_size_pix.x;

  ImGui::DockBuilderSetNodeSize(dockSpaceId, win_size_pix);
  // Split right panel first
  ImGuiID dock_main = dockSpaceId;
  ImGuiID dock_right =
      ImGui::DockBuilderSplitNode(dock_main, ImGuiDir_Right, right_width, nullptr, &dock_main);
  // Split the right panel into top-right ("Bot/Scene") and bottom-right ("Bot Details/Details")
  ImGuiID dock_right_bottom =
      ImGui::DockBuilderSplitNode(dock_right, ImGuiDir_Down, 0.6f, nullptr, &dock_right);
  // Split bottom panel from center (below Editor only, not full width)
  ImGuiID dock_bottom =
      ImGui::DockBuilderSplitNode(dock_main, ImGuiDir_Down, bottom_height, nullptr, &dock_main);
  // Hide the tab bar for the Editor node
  if (ImGuiDockNode* node = ImGui::DockBuilderGetNode(dock_main)) {
    node->LocalFlags |= ImGuiDockNodeFlags_NoTabBar;
  }
  // Assign app-managed windows to dock nodes.
  ImGui::DockBuilderDockWindow(kWindowEditorName, dock_main); // Center
  ImGui::DockBuilderDockWindow(kWindowAssetsName, dock_bottom); // Bottom-center
  ImGui::DockBuilderDockWindow(
      kWindowLogConsoleName, dock_bottom); // Bottom-center (tabbed with Assets)
  // Let each AssetEditor type place its own windows via its declared DockRegions. This pre-docks
  // windows for editor types that aren't open yet. Add new editor types here as they are
  // introduced.
  DockEditorDefaultWindows<ModelEditor>(dock_right, dock_right_bottom, dock_bottom);
  DockEditorDefaultWindows<BotEditor>(dock_right, dock_right_bottom, dock_bottom);
  DockEditorDefaultWindows<MochiPrefabEditor>(dock_right, dock_right_bottom, dock_bottom);
#if MOCHI_INTERNAL
  DockEditorDefaultWindows<BotSceneEditor>(dock_right, dock_right_bottom, dock_bottom);
#endif
  ImGui::DockBuilderFinish(dockSpaceId);
  // Reset all window visibility to defaults.
  _appSettings.windowVisibility.clear();
  for (auto const& w : kAppWindowDefaults) {
    _appSettings.windowVisibility[w.name] = w.showByDefault;
  }
  // Re-register the active editor's windows so they're visible after reset.
  if (auto* editor = GetActiveAssetEditor(); editor) {
    for (auto const& w : editor->GetAuxiliaryWindows()) {
      _appSettings.windowVisibility[w.name] = w.showByDefault;
    }
  }
  ImGui::MarkIniSettingsDirty();
}

} // namespace superdex::studio
