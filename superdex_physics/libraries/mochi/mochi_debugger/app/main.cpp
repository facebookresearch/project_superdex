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

#include <mochi_renderer/windows_compat.h> // Must be first — cleans Windows macros before Filament

#include "gui/gui.h"
#include "viewport/camera.h"
#include "viewport/native_gl_context.h"
#include "viewport/render_scene.h"

// gpu_selector.h must be included, even though nothing else references it. It defines variables
// that cause NVidia to use the discrete GPU (if available) not the integrated GPU (e.g. in an MSI
// laptop).
#include <imguios/gpu_selector.h>

#include <imguios/imguios.h>

#include <mochi_core/geometry/mesh_data.h>
#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/console.h>
#include <mochi_core/utils/coordinate_space.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/quaternion.h>
#include <mochi_core/utils/reflection.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/transform_rt.h>
#include <mochi_debugger/lib/command_line.h>
#include <mochi_debugger/lib/debug_client.h>
#include <mochi_debugger/lib/single_instance.h>
#include <mochi_physics/dbg/protocol.h>
#include <mochi_renderer/mochi_renderer.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>

#if MOCHI_PLATFORM_WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif // MOCHI_PLATFORM_WINDOWS

namespace mochi::dbg {

// Constants
static constexpr Int2 kDefaultMainWindowSize = {1920, 1080};
static constexpr char const* kAppName = "Mochi Debugger";
static constexpr char const* kImGuiIniName = "mochi_debugger_imgui.ini";

/*****************************************************************************
  MochiDebuggerApp
*/

namespace {
class MochiDebuggerApp final : public ImGuios::Application {
 public:
  MochiDebuggerApp(CommandLine cli, std::unique_ptr<SingleInstanceHelper> singleInstanceHelper);

  // ImGuios::Application:
  void OnInitialize() final;
  void OnUpdate() final;
  void OnShutdown() final {
    // Stop accepting launch arguments and release kSingletonPort (if bound).
    _singleInstanceHelper.reset();

    // Disconnect and destroy the client
    _client.reset();

    // Destroy the render scene before the renderer.
    _renderScene.reset();
    _mochiRenderer.reset();
  }

 private:
  void ResetUiLayout();
  void ProcessCommandLine(CommandLine const& commandLine);
  void UpdateWindowTitle();
  void UpdateConnectionState();
  void AdoptCoordinateSpace(CoordinateSpace const& space);
  void SyncSceneData();
  void SelectScene(SceneHandle handle);
  void ImportActors(SceneSyncData const& sync);

  CommandLine _commandLine;
  std::unique_ptr<SingleInstanceHelper> _singleInstanceHelper;
  std::unique_ptr<DebugClient> _client;
  std::unique_ptr<mochi_renderer::MochiRenderer> _mochiRenderer;
  RenderScenePtr _renderScene;
  UiState _uiState;

  std::string _mainWindowTitle;
  net::SocketStatus _prevStatus = net::SocketStatus::None;
};
} // namespace

MochiDebuggerApp::MochiDebuggerApp(
    CommandLine cli,
    std::unique_ptr<SingleInstanceHelper> singleInstanceHelper)
    : ImGuios::Application(
          std::make_unique<ImGuios::Window>(
              kDefaultMainWindowSize[0],
              kDefaultMainWindowSize[1],
              kAppName,
              ImGuios::WindowFlags_MSAA)),
      _commandLine(std::move(cli)),
      _singleInstanceHelper(std::move(singleInstanceHelper)) {
  MOCHI_ASSERT(_singleInstanceHelper);
}

void MochiDebuggerApp::OnInitialize() {
  // Create the Filament-based renderer, sharing the host GL context (Win/Linux) or Metal device
  // (macOS) so the offscreen render target can be displayed by ImGui.
#if MOCHI_PLATFORM_MACOS
  mochi_renderer::MochiRenderer::Config config;
  config.backend = filament::Engine::Backend::METAL;
  config.sharedContext = ImGuios::GetMetalDevice();
  config.featureLevel = filament::backend::FeatureLevel::FEATURE_LEVEL_3;
  _mochiRenderer = mochi_renderer::MochiRenderer::Create(config);
#else
  GLFWwindow* backupContext = glfwGetCurrentContext();
  void* mainOpenGlContext = GetNativeGlContext(GetMainWindow()->GetGLFW());
  glfwMakeContextCurrent(nullptr);
  mochi_renderer::MochiRenderer::Config config;
  config.backend = filament::Engine::Backend::OPENGL;
  config.sharedContext = mainOpenGlContext;
  config.featureLevel = filament::backend::FeatureLevel::FEATURE_LEVEL_3;
  _mochiRenderer = mochi_renderer::MochiRenderer::Create(config);
  glfwMakeContextCurrent(backupContext);
#endif

  _client = std::make_unique<DebugClient>();
  _renderScene = RenderScene::Create(_mochiRenderer.get());

  InitializeUi(_uiState);
  _uiState.viewport.renderScene = _renderScene.get();
  _uiState.client = _client.get();

  // Init DebugClient settings
  {
    DebugClientSettings settings;

    // Continuously sync the selected scene's actors, surface meshes, and debug-draw geometry.
    settings.sync.enabled = true;
    settings.sync.syncInterval = 1.0f / 100.0f; // Max sync rate: 100 Hz
    settings.sync.syncActors = true;
    settings.sync.syncDebugDraw = true;
    settings.sync.syncMeshes = true;
    _client->SetSettings(settings);

    // Start paused
    _client->SetSceneStepMode(StepMode::Pause);
  }

  ImGui::GetIO().IniFilename = kImGuiIniName;
  if (!std::filesystem::exists(kImGuiIniName)) {
    _uiState.rebuildLayout = true; // no saved layout yet — build the default one
  }

  // Default camera
  _uiState.viewport.camera.ResetToDefault(_renderScene->GetCoordinateSpace());

  // Select a font for the terminal
  _uiState.terminal.font = GetTerminalFont();

  ProcessCommandLine(_commandLine);

  // If no address has been specified so far (e.g. by "--connect"), then open the connect dialog.
  if (_uiState.connection.address.empty()) {
    _uiState.connectDialog.shouldOpen = true;
  }
}

void MochiDebuggerApp::ProcessCommandLine(CommandLine const& commandLine) {
  if (!commandLine.address.empty()) {
    ConnectTo(_uiState, commandLine.address, commandLine.port);
    // Close any currently open connect dialog
    _uiState.connectDialog.shouldCancel = true;
    _uiState.connectDialog.shouldOpen = false;
  }
}

// Monitor connection status: optionally auto-reconnect if disconnected, and pick up the server's
// coordinate space on each new connection.
void MochiDebuggerApp::UpdateConnectionState() {
  net::SocketStatus const status = _client->GetStatus();
  bool const justLost =
      (status == net::SocketStatus::Lost) && (_prevStatus != net::SocketStatus::Lost);
  if (_uiState.connection.autoReconnect && justLost && !_uiState.connection.address.empty()) {
    ConnectTo(_uiState, _uiState.connection.address, _uiState.connection.port);
  }
  bool const justConnected =
      (status == net::SocketStatus::Connected) && (_prevStatus != net::SocketStatus::Connected);
  if (justConnected) {
    // The space is meaningful only while connected; at any other time GetCoordinateSpace() reports
    // Filament().
    AdoptCoordinateSpace(_client->GetCoordinateSpace());
  }
  _prevStatus = status;
}

// Render in the server's coordinate space. The render scene holds the last adopted space even
// while disconnected, so reconnecting to the same server is a no-op here — reframing the camera
// would be an unwelcome surprise.
void MochiDebuggerApp::AdoptCoordinateSpace(CoordinateSpace const& space) {
  Error error;
  space.Validate(error);
  if (!error.IsOK()) {
    MOCHI_LOG_WARNING(
        "Invalid coordinate space convention received from the server: %s", error.GetDescription());
    return;
  }
  if (space == _renderScene->GetCoordinateSpace()) {
    return;
  }
  _renderScene->SetCoordinateSpace(space);
  _uiState.viewport.camera.ResetToDefault(space);
}

void MochiDebuggerApp::SelectScene(SceneHandle handle) {
  _client->SelectScene(handle);
  _uiState.scene.ResetSceneContents();
  _uiState.viewport.renderScene->ClearMeshes();
  _uiState.viewport.renderScene->ClearDebugDraw();
}

void MochiDebuggerApp::OnUpdate() {
  // Process UI commands queued by the previous frame's UI before refreshing scene state, so the
  // whole frame sees a consistent selection.
  for (auto const& cmd : _uiState.uiCommands) {
    switch (cmd.id) {
      case CommandId::Exit:
        GetMainWindow()->Close();
        break;
      case CommandId::ResetLayout:
        ResetUiLayout();
        break;
      case CommandId::SelectScene:
        SelectScene(SceneHandle{cmd.value});
        break;
    }
  }
  _uiState.uiCommands.clear();

  // Process command line arguments transmitted from other instances of the app.
  bool receivedRemoteCommandLine = false;
  while (auto cli = _singleInstanceHelper->PollCommandLine()) {
    ProcessCommandLine(*cli);
    receivedRemoteCommandLine = true;
  }
  if (receivedRemoteCommandLine) {
    // Bring the main window to the foreground
    auto* window = GetMainWindow();
    window->Show();
    window->Restore();
    window->Focus();
  }

  // Check for a new scene or new scene data.
  // This may call SelectScene().
  SyncSceneData();

  BuildUi(_uiState);

  UpdateConnectionState();
  UpdateWindowTitle();

  // Render the viewport into the offscreen target (sized during the UI build).
  if (_uiState.panels.at(kPanelNameViewport).isVisible) {
    _renderScene->SetParams(_uiState.rendering);
    _renderScene->Render(_uiState.viewport.camera);

    // Feed the mesh bounds back to the camera controls for framing.
    _uiState.viewport.focusBounds = _renderScene->GetSceneBounds();
  }

  if (_client->WasExitRequested()) {
    GetMainWindow()->Close();
  }
}

// Format a display name for an unnamed actor as: "unnamed_<type>" with the actor type lower-cased.
static std::string UnnamedActorLabel(ActorType actorType) {
  char const* const typeStr = SReflect::EnumToString(actorType);
  std::string type = (typeStr != nullptr && typeStr[0] != '\0') ? typeStr : "actor";
  for (char& c : type) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return "unnamed_" + type;
}

// Per-frame refresh of scene selection, the per-actor records, and the rendered scene. This is
// the single place that calls DebugClient::GetSceneSyncData, so the whole frame (GUI + viewport)
// sees one consistent snapshot. The synced meshes (data.actorMeshes) are non-owning views valid
// only for the callback's duration, so the render meshes and debug draw are pushed to the render
// scene inside the callback, without copying the mesh data out.
void MochiDebuggerApp::SyncSceneData() {
  ScenePanelState& sp = _uiState.scene;
  DebugClient* const client = _client.get();
  if (client == nullptr) {
    return;
  }

  // 1. Refresh the scene list. Give unnamed scenes a display name, then sort by (name, handle
  // value) and prepend the invalid "None" entry so index 0 always maps to "no selection".
  client->GetSceneList(sp.scenes);
  for (SceneInfo& info : sp.scenes) {
    if (info.name.empty()) {
      info.name = "unnamed_scene";
    }
  }
  std::ranges::sort(sp.scenes, [](auto const& a, auto const& b) {
    return std::tie(a.name, a.handle.value) < std::tie(b.name, b.handle.value);
  });

  // Insert "None" at the front of the list using push_back + rotate because DynamicArray::insert is
  // not currently implemented.
  sp.scenes.push_back(SceneInfo{{}, "None"});
  std::rotate(sp.scenes.begin(), sp.scenes.end() - 1, sp.scenes.end());

  // 2. Take a single snapshot of the synced data. The actor records and the render scene are
  // updated only when the data has changed (identified by counter value).
  client->GetSceneSyncData([&](SceneSyncData const& data) {
    if (sp.hasSyncData && data.counter == sp.lastSyncCounter) {
      return;
    }
    sp.lastSyncCounter = data.counter;
    sp.hasSyncData = true;

    ImportActors(data);

    // Copy debug draw data to the render scene (only when new data is received).
    if (_uiState.rendering.showDebugDraw) {
      protocol::DbgDrawData const& dbg = data.debugDraw;
      _renderScene->SetDebugLines(dbg.lineVertices.positions, dbg.lineVertices.colors);
      _renderScene->SetDebugSpheres(dbg.spheres.positions, dbg.spheres.radii, dbg.spheres.colors);
    }
  });

  // 3. Derive the combo index for the UI. GetSelectedScene() returns an invalid handle when nothing
  // is selected, which matches the "None" entry at index 0.
  SceneHandle const selected = client->GetSelectedScene();
  sp.selectedSceneIndex = 0;
  for (size_t i = 0; i < sp.scenes.size(); ++i) {
    if (sp.scenes[i].handle == selected) {
      sp.selectedSceneIndex = static_cast<int>(i);
      break;
    }
  }
}

// Bring the per-actor records (and the meshes they own) in line with the latest synced data, via
// mark/sweep against sp.actors. New actors get a record and a mesh (rigid -> static hard-angle
// path; deformable -> dynamic smooth-normal path), existing actors have their transforms (and
// deformable geometry) updated, and destroyed actors are removed.
// The synced meshes (sync.actorMeshes) are 1:1 with sync.actors and are non-owning views, so this
// must run inside the GetSceneSyncData callback while they are valid.
void MochiDebuggerApp::ImportActors(SceneSyncData const& sync) {
  ScenePanelState& sp = _uiState.scene;

  for (auto& [handle, info] : sp.actors) {
    info.visited = false;
  }

  // Pass 1: upsert a record per synced actor. actorMeshes is 1:1 with actors; an actor without a
  // surface mesh has an empty view.
  DynamicArray<ActorHandle> newActors;
  for (int i = 0; i < isize(sync.actors); ++i) {
    protocol::ActorSyncData const& actor = sync.actors[i];
    MeshSyncDataView const mesh =
        (i < isize(sync.actorMeshes)) ? sync.actorMeshes[i] : MeshSyncDataView{};

    auto const [it, inserted] = sp.actors.try_emplace(actor.handle);
    ActorInfo& info = it->second;
    info.visited = true;
    info.hasMesh = !mesh.coordinates.empty() && !mesh.connectivity.empty();
    if (inserted) {
      // Identity is immutable for the life of the actor, so it is captured once here.
      info.parent = actor.parent;
      info.type = actor.type;
      info.name = std::string{actor.name};
      sp.treeDirty = true;
      newActors.push_back(actor.handle);
    }
  }

  for (ActorHandle const handle : newActors) {
    ActorInfo& info = sp.actors.at(handle);
    auto const parentIt = sp.actors.find(info.parent);

    // Format the display name, removing the full parent path from child actor names.
    info.displayName = info.name;
    if (parentIt != sp.actors.end()) {
      std::string const prefix = parentIt->second.name + "/";
      if (info.displayName.starts_with(prefix)) {
        info.displayName.erase(0, prefix.size());
      }
    }
    if (info.displayName.empty()) {
      info.displayName = UnnamedActorLabel(info.type);
    }
  }

  // Hide children while their parent draws a skin, unless the user explicitly chose visibility.
  // Re-evaluate after every import because switching mesh source can change parent mesh presence.
  for (auto& [handle, info] : sp.actors) {
    if (info.visibilityOverridden) {
      continue;
    }
    auto const parentIt = sp.actors.find(info.parent);
    bool const isVisible = (parentIt == sp.actors.end()) || !parentIt->second.hasMesh;
    if (info.isVisible == isVisible) {
      continue;
    }
    info.isVisible = isVisible;
    if (info.meshId != kInvalidMeshId) {
      _renderScene->SetMeshVisible(info.meshId, isVisible);
    }
  }

  // Pass 2: reconcile each actor's render mesh.
  DynamicArray<float> normals;
  for (int i = 0; i < isize(sync.actors); ++i) {
    protocol::ActorSyncData const& actor = sync.actors[i];
    ActorInfo& info = sp.actors.at(actor.handle);
    MeshSyncDataView const mesh =
        (i < isize(sync.actorMeshes)) ? sync.actorMeshes[i] : MeshSyncDataView{};
    Span<float const> const positions = mesh.coordinates;
    Span<int const> const connectivity = mesh.connectivity;

    if (!info.hasMesh && (info.meshId != kInvalidMeshId)) {
      _renderScene->RemoveMesh(info.meshId);
      info.meshId = kInvalidMeshId;
      continue;
    }

    TransformRT const worldFromLocal{
        Quaternion(StaticCast<Real4>(actor.rotation)), StaticCast<Real3>(actor.position)};
    bool const isRigid = (actor.type == ActorType::Rigid);
    size_t const nodeCount = positions.size() / 3;

    // A dynamic mesh's in-place update requires stable topology, so recreate if the connectivity,
    // node count, or rigidity changed.
    if (info.meshId != kInvalidMeshId &&
        ((info.meshVersionCounter != mesh.versionCounter) || (info.isRigid != isRigid) ||
         (info.nodeCount != nodeCount))) {
      _renderScene->RemoveMesh(info.meshId);
      info.meshId = kInvalidMeshId;
    }

    if (info.meshId == kInvalidMeshId) {
      Color const color =
          GetRotatingColor(static_cast<int>(static_cast<uint32_t>(actor.handle.value)));
      MeshMaterialClass const materialClass =
          actor.isStatic ? MeshMaterialClass::Static : MeshMaterialClass::Dynamic;
      MeshId id = kInvalidMeshId;
      if (isRigid) {
        // Rigid: RenderScene bakes hard-angle normals from the connectivity (static path).
        id = _renderScene->AddMesh(
            positions,
            /*normals*/ {},
            connectivity,
            worldFromLocal,
            color,
            materialClass,
            /*isDynamic*/ false);
      } else {
        // Deformable: supply smooth per-node normals and update geometry each frame (dynamic path).
        ComputeSmoothNormals(positions, connectivity, normals);
        id = _renderScene->AddMesh(
            positions,
            normals,
            connectivity,
            worldFromLocal,
            color,
            materialClass,
            /*isDynamic*/ true);
      }
      _renderScene->SetMeshVisible(id, info.isVisible);
      info.meshId = id;
      info.nodeCount = nodeCount;
      info.meshVersionCounter = mesh.versionCounter;
      info.isRigid = isRigid;
    } else if (isRigid) {
      // Rigid geometry does not deform; only the transform changes.
      _renderScene->UpdateMeshTransform(info.meshId, worldFromLocal);
    } else {
      // A hidden mesh is not drawn, so skip the normal recomputation and the geometry upload; the
      // first sync after it is switched back on refreshes it.
      if (info.isVisible) {
        ComputeSmoothNormals(positions, connectivity, normals);
        _renderScene->UpdateMeshGeometry(info.meshId, positions, normals);
      }
      _renderScene->UpdateMeshTransform(info.meshId, worldFromLocal);
    }
  }

  // Sweep: drop the records (and meshes) of actors that are no longer present.
  size_t const removed = std::erase_if(sp.actors, [&](auto const& kv) {
    if (kv.second.visited) {
      return false;
    }
    _renderScene->RemoveMesh(kv.second.meshId);
    return true;
  });
  if (removed > 0) {
    sp.treeDirty = true;
  }

  if (sp.selectedActor.IsValid() && !sp.actors.contains(sp.selectedActor)) {
    sp.selectedActor = {};
  }
}

// Reflect the connection status in the window title. Only calls SetTitle when it actually changes.
void MochiDebuggerApp::UpdateWindowTitle() {
  std::string suffix;
  switch (_client->GetStatus()) {
    case net::SocketStatus::Pending:
      suffix = " - Waiting for " + _uiState.connection.FormatAddress();
      break;
    case net::SocketStatus::Connected:
      suffix = " - Connected to " + _uiState.connection.FormatAddress();
      break;
    case net::SocketStatus::None:
    case net::SocketStatus::Lost:
      suffix = " - No Connection";
      break;
  }
  std::string title = std::string(kAppName) + suffix;
  if (title != _mainWindowTitle) {
    _mainWindowTitle = title;
    GetMainWindow()->SetTitle(title.c_str());
  }
}

void MochiDebuggerApp::ResetUiLayout() {
  for (auto& [name, panel] : _uiState.panels) {
    panel.isVisible = true; // All panels are visible in the default layout
  }
  _uiState.rebuildLayout = true; // rebuilt via DockBuilder on the next BuildUi
}

static int Main(int argc, char** argv) {
  // Parse command line
  auto args = CommandLine::ArgsFromMain(argc, argv);
  auto [cli, err] = CommandLine::Parse(args);
  if (!err.empty()) {
    fprintf(stderr, "%s\n", err.c_str());
    return EXIT_FAILURE;
  }

  auto helper = std::make_unique<SingleInstanceHelper>(cli, args);
  if (helper->ShouldExit()) {
    return EXIT_SUCCESS;
  }

  mochi::dbg::MochiDebuggerApp app(std::move(cli), std::move(helper));
  app.Run();
  return 0;
}

} // namespace mochi::dbg

int main(int argc, char** argv) {
  mochi::AttachParentConsole();
  return mochi::dbg::Main(argc, argv);
}
