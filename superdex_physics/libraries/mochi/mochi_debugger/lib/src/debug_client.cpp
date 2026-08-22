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

#include <mochi_core/mochi_platform.h>
#include <mochi_core/net/message_server.h>
#include <mochi_core/net/server_list.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/span_utils.h>
#include <mochi_core/utils/string_utils.h>
#include <mochi_debugger/lib/address.h>
#include <mochi_debugger/lib/debug_client.h>
#include <mochi_physics/dbg/protocol.h>

#include <algorithm>
#include <chrono>
#include <optional>
#include <string>
#include <thread>
#include <tuple>

using namespace mochi;
using namespace mochi::dbg;

static constexpr double kCommandTimeout = 3.0; // seconds

void DebugClient::SetPrintFunction(LogFn fn) {
  MOCHI_ASSERT(
      _socket.GetStatus() == net::SocketStatus::None,
      "You must set the print function before attempting to connect");
  _printFn = std::move(fn);
}

void DebugClient::Print(
    std::string message,
    LogChannel channel,
    char const* file,
    int line,
    bool newline) {
  if (_printFn) {
    if (newline) {
      message += '\n';
    }
    _printFn(channel, message.c_str(), file, line);
  } else {
    Log(channel, std::move(message), file, line, newline);
  }
}

void DebugClient::PrintError(std::string const& message) {
  Print(message, LogChannel::Error);
}

DebugClient::DebugClient() {
  InitProtocol();
  InitCommands();

  _socket.SetStatusCallback([this](auto status) { OnStatusChange(status); });
}

DebugClient::~DebugClient() {
  Disconnect();
}

void DebugClient::Connect(std::string_view address, uint16_t port) {
  Disconnect();
  _socket.Connect(address, port);
}

void DebugClient::ConnectInProc(net::MessageServer& server) {
  Disconnect();
  _isInProc = true; // Remember that we are using an in-process connection.
  _socket.ConnectInProc(server);
}

void DebugClient::Disconnect() {
  _socket.Disconnect();
  _isInProc = false;

  // An explicit disconnect may not emit a Lost status change, so reset per-connection state here.
  ResetConnectionState();
}

uint64_t DebugClient::GetVersion() const {
  return _protocolVersion;
}

net::SocketStatus DebugClient::GetStatus() const {
  auto const status = _socket.GetStatus();
  if (status == net::SocketStatus::Connected && !_isFullyConnected.load()) {
    // The socket is up but the handshake is not complete yet.
    return net::SocketStatus::Pending;
  }
  return status;
}

void DebugClient::GetAddress(std::string& outAddress, uint16_t& outPort) const {
  _socket.GetAddress(outAddress, outPort);
}

void DebugClient::GetSceneList(DynamicArray<SceneInfo>& out) const {
  _state.Read([&](auto& state) {
    out = state.scenes; // copy
  });
}

bool DebugClient::Send(net::Message const& msg) {
  return _socket.Send(msg);
}

void DebugClient::OnStatusChange(net::SocketStatus status) {
  if (status == net::SocketStatus::Lost) {
    Print("Disconnected");
    ResetConnectionState();
  }
}

void DebugClient::ResetConnectionState() {
  _state.Mutate([&](auto& state) {
    // _isFullyConnected is modified while _state is locked to keep things in sync.
    _isFullyConnected = false;

    state.scenes.clear();
    state.coordinateSpace = CoordinateSpace::Filament(); // Matches server's default
    state.selectedScene = {};
    state.meshCache.clear();
    if (state.syncData != SceneSyncData{}) {
      state.syncData = {};
      state.syncData.counter = ++state.syncCounter; // Indicates it was changed
    }

    // Keep settings: it is a user preference that persists across connections.
  });
}

void DebugClient::ExecuteCommand(std::string_view str) {
  std::string error;
  DynamicArray<std::string> tokens = ConsoleTokenize(str, error);
  if (!error.empty()) {
    PrintError(error);
    return;
  }
  if (tokens.empty()) {
    return;
  }

  // Look up the command (first token)
  auto const it = _commands.find(tokens[0]);
  if (it == _commands.end()) {
    PrintError(Format("Unknown command: %s", tokens[0].c_str()));
    return;
  }

  // Check the number of arguments
  auto args = MakeConstSpan(tokens).subspan(1);
  auto const& cmd = it->second;
  if ((cmd.numArgs >= 0) && (cmd.numArgs != isize(args))) {
    Print("Invalid number of arguments");
    return;
  }

  // Call command handler
  cmd.fn(args);
}

bool DebugClient::IsConnected() const {
  return GetStatus() == net::SocketStatus::Connected;
}

// Format path as a string. E.g. "/a/b/c"
static std::string FormatPath(DynamicArray<std::string> const& cmdPath) {
  std::string path = "/";
  path += Join(cmdPath, "/");
  return path;
}

// Return the new path after resolving the given path argument (may be relative or absolute).
// Fails if ".." arguments would cause the path to be invalid.
DynamicArray<std::string> DebugClient::ResolvePath(std::string_view pathArg, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  DynamicArray<std::string> result;
  auto base = _state.Read(&State::cmdPath);

  bool const rooted = !pathArg.empty() && pathArg.front() == '/';
  if (!rooted) {
    result = base;
  }

  for (std::string_view token : Split(pathArg, "/")) {
    if (token == ".") {
      continue;
    } else if (token == "..") {
      if (result.empty()) {
        MOCHI_ERROR_SET(error, "Invalid path");
        return {};
      } else {
        result.pop_back();
      }
    } else {
      result.emplace_back(token);
    }
  }

  return result;
}

// Find a scene by handle. Return index or -1 if not found.
static int FindSceneIndex(Span<SceneInfo const> scenes, SceneHandle handle) {
  for (int i = 0; i < isize(scenes); ++i) {
    if (scenes[i].handle == handle) {
      return i;
    }
  }
  return -1;
}

// Find a scene by name. Return index or -1 if not found.
static int FindSceneIndex(Span<SceneInfo const> scenes, std::string_view name) {
  for (int i = 0; i < isize(scenes); ++i) {
    if (scenes[i].name == name) {
      return i;
    }
  }
  return -1;
}

// Set an error if the command path is not valid
void DebugClient::ValidatePath(Span<std::string const> path, Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (path.empty()) {
    // Empty path (root) is always valid
    return;
  }

  if (!IsConnected()) {
    MOCHI_ERROR_SET(error, "No connection");
    return;
  }

  // Empty path tokens are never valid
  for (auto const& tok : path) {
    MOCHI_ERROR_IF(tok.empty(), error, "Invalid path");
  }
  MOCHI_ERROR_RETURN(error);

  if (path.size() == 1) {
    // First token is a scene name. Check it against the cached scene list.
    std::string_view sceneName = path[0];
    int const sceneIndex =
        _state.Read([&](auto const& state) { return FindSceneIndex(state.scenes, sceneName); });
    if (sceneIndex >= 0) {
      return; // OK
    }
  }
  MOCHI_ERROR_SET(error, "Invalid path");
}

void DebugClient::InitProtocol() {
  _socket.Register<protocol::LogMessage>(
      [this](auto&& msg) { OnLogMessage(std::forward<decltype(msg)>(msg)); });
  _socket.Register<protocol::PingReply>();
  _socket.Register<protocol::PingRequest>();
  _socket.Register<protocol::SceneAddRemove>(
      [this](auto&& msg) { OnSceneAddRemove(std::forward<decltype(msg)>(msg)); });
  _socket.Register<protocol::SceneStepReply>(
      [this](auto&& msg) { OnSceneStepReply(std::forward<decltype(msg)>(msg)); });
  _socket.Register<protocol::SceneStepRequest>();
  _socket.Register<protocol::SceneSyncReply>(
      [this](auto&& msg) { OnSceneSyncReply(std::forward<decltype(msg)>(msg)); });
  _socket.Register<protocol::SceneSyncRequest>();
  _socket.Register<protocol::WelcomeMessage>(
      [this](auto&& msg) { OnWelcomeMessage(std::forward<decltype(msg)>(msg)); });

  // Set version so connection will be refused if the server does not register exactly the
  // same collection of messages.
  _protocolVersion = _socket.CalcProtocolVersionHash();
  _socket.SetVersion(_protocolVersion);
}

void DebugClient::OnWelcomeMessage(protocol::WelcomeMessage&& msg) {
  auto const& scenes = msg.scenes;
  if (scenes.handles.size() != scenes.names.size()) {
    MOCHI_ASSERT(false, "Invalid WelcomeMessage received from server. Scene array size mismatch.");
    return;
  }

  DynamicArray<protocol::SceneStepRequest> stepRequestsToSend;
  DynamicArray<protocol::SceneSyncRequest> syncRequestsToSend;
  bool const isFirstWelcome = _state.Mutate([&](auto& state) {
    // The server sends exactly one WelcomeMessage per connection. It includes information about
    // current scenes, so the client can join in progress. After this point, scenes are tracked
    // individually via SceneAddRemove.
    if (_isFullyConnected) {
      return false;
    }

    MOCHI_ASSERT_VERBOSE(
        state.scenes.empty(), "We should not track scenes until the WelcomeMessage arrives.");
    state.coordinateSpace = msg.coordinateSpace;
    state.scenes.reserve(scenes.handles.size());
    if (state.stepMode != StepMode::Pause) {
      stepRequestsToSend.reserve(scenes.handles.size());
    }

    for (int i = 0; i < isize(scenes.handles); ++i) {
      // Store information about the scene
      SceneInfo info;
      info.handle = scenes.handles[i];
      info.name = scenes.names[i];
      state.scenes.emplace_back(std::move(info));

      // Scenes start paused on the server. Update the remote StepMode if necessary.
      if (state.stepMode != StepMode::Pause) {
        protocol::SceneStepRequest req;
        req.scene = scenes.handles[i];
        req.mode = state.stepMode;
        stepRequestsToSend.push_back(req);
      }
    }

    // Maybe auto-select a scene.
    SetSelectedScene(state, FindAutoSelectableScene(state.scenes), syncRequestsToSend);

    return true;
  });

  if (!isFirstWelcome) {
    MOCHI_LOG_WARNING("Redundant WelcomeMessage received from the server. Ignoring.");
    return;
  }

  // Print a message now that this client is considered to be fully connected.
  std::string address;
  uint16_t port = 0;
  _socket.GetAddress(address, port);
  if (address.empty()) {
    Print("Connected to in-process server");
  } else {
    Print(Format("Connected to %s:%u", address.c_str(), port));
  }

  // Send queued messages
  for (auto const& req : stepRequestsToSend) {
    Send(req);
  }
  for (auto const& req : syncRequestsToSend) {
    Send(req);
  }

  _isFullyConnected = true;
}

void DebugClient::OnSceneAddRemove(protocol::SceneAddRemove&& msg) {
  std::optional<protocol::SceneStepRequest> stepRequestToSend;
  DynamicArray<protocol::SceneSyncRequest> syncRequestsToSend;
  _state.Mutate([&](auto& state) {
    // Ignore deltas until the WelcomeMessage arrives. Pre-welcome deltas are already
    // reflected in the snapshot (which the server takes, in lock order, after them).
    if (!_isFullyConnected) {
      return;
    }

    int sceneIndex = FindSceneIndex(state.scenes, msg.scene);
    if (msg.wasAdded) {
      if (sceneIndex >= 0) {
        MOCHI_ASSERT_VERBOSE(
            false, "Redundant SceneAddRemove message received from server. Scene already added.");
        return;
      }

      // Scenes start paused on the server. Update the remote StepMode if necessary.
      if (state.stepMode != StepMode::Pause) {
        stepRequestToSend.emplace();
        stepRequestToSend->scene = msg.scene;
        stepRequestToSend->mode = state.stepMode;
      }

      bool const hadSelectableScene = FindAutoSelectableScene(state.scenes).IsValid();

      SceneInfo info;
      info.handle = msg.scene;
      info.name = msg.name;
      state.scenes.emplace_back(std::move(info));

      // If there is no current selection and this is the first auto-selectable scene,
      // then select it now.
      if (!state.selectedScene.IsValid() && !hadSelectableScene &&
          IsAutoSelectableScene(state.scenes.back())) {
        SetSelectedScene(state, msg.scene, syncRequestsToSend);
      }
    } else {
      if (sceneIndex >= 0) {
        state.scenes.erase(state.scenes.begin() + sceneIndex);
      } else {
        MOCHI_ASSERT_VERBOSE(
            false, "Redundant SceneAddRemove message received from server. Scene already removed.");
      }

      // Deselect the scene and maybe select another one.
      if (msg.scene == state.selectedScene) {
        SetSelectedScene(state, FindAutoSelectableScene(state.scenes), syncRequestsToSend);
      }
    }
  });
  if (stepRequestToSend) {
    Send(*stepRequestToSend);
  }
  for (auto const& req : syncRequestsToSend) {
    Send(req);
  }
}

// Receive incoming data from SceneSyncReply to update our local mesh cache.
void DebugClient::UpdateActorMeshes(
    protocol::SceneSyncReply const& reply,
    std::unordered_map<ActorHandle, MeshInfo>& meshCache,
    DynamicArray<MeshSyncDataView>& outActorMeshes) {
  // All three data fields are required
  MOCHI_ASSERT_VERBOSE(reply.actors && reply.meshData && reply.actorMeshRanges);

  auto const& actors = *reply.actors;
  auto const& actorMeshRanges = *reply.actorMeshRanges;
  auto const& meshData = *reply.meshData;
  int const numActors = isize(actors);
  MOCHI_ASSERT_VERBOSE(isize(actorMeshRanges) == numActors, "Size mismatch");

  // Clear previous view of meshes
  outActorMeshes.clear();

  // Unmark all meshes in the cache
  for (auto&& [handle, info] : meshCache) {
    info.visited = false;
  }

  // Update the cache from this reply, tracking which actors were present.
  for (int i = 0; i < numActors; ++i) {
    auto const& actor = actors[i];
    Int2 const& coordinatesRange = actorMeshRanges[i].coordinatesRange;
    Int2 const& connectivityRange = actorMeshRanges[i].connectivityRange;
    int const coordinatesSize = coordinatesRange[1] - coordinatesRange[0];
    int const connectivitySize = connectivityRange[1] - connectivityRange[0];

    // Validation
    MOCHI_ASSERT_VERBOSE(
        (coordinatesSize >= 0) && (coordinatesSize % 3 == 0), "Expected 3 per vertex");
    MOCHI_ASSERT_VERBOSE(
        (connectivitySize >= 0) && (connectivitySize % 3 == 0), "Expected 3 per triangle");
    MOCHI_ASSERT_VERBOSE(
        (coordinatesRange[0] >= 0) && (coordinatesRange[1] <= isize(meshData.coordinates)),
        "Range out-of-bounds");
    MOCHI_ASSERT_VERBOSE(
        (connectivityRange[0] >= 0) && (connectivityRange[1] <= isize(meshData.connectivity)),
        "Range out-of-bounds");

    // Find or insert in cache
    auto& entry = meshCache[actor.handle];
    entry.visited = true;

    // Clear old mesh data if the version counter has changed.
    if (entry.versionCounter != reply.meshVersionCounter) {
      entry.mesh.coordinates.clear();
      entry.mesh.connectivity.clear();
      entry.versionCounter = reply.meshVersionCounter;
    }

    // Copy new mesh data
    if (coordinatesSize > 0) {
      entry.mesh.coordinates.assign(
          meshData.coordinates.begin() + coordinatesRange[0],
          meshData.coordinates.begin() + coordinatesRange[1]);
    }
    if (connectivitySize > 0) {
      entry.mesh.connectivity.assign(
          meshData.connectivity.begin() + connectivityRange[0],
          meshData.connectivity.begin() + connectivityRange[1]);

#if MOCHI_ASSERT_VERBOSE_ENABLED
      auto const [min, max] = MinMax(MakeConstSpan(entry.mesh.connectivity));
      int const numCoords = isize(entry.mesh.coordinates) / 3;
      MOCHI_ASSERT_VERBOSE(min >= 0 && max < numCoords, "Vertex index out-of-range");
#endif
    }

    // Add an entry to the actorMeshes array (1:1 with actors)
    MeshSyncDataView view;
    if (!entry.mesh.coordinates.empty() && !entry.mesh.connectivity.empty()) {
      view.coordinates = MakeConstSpan(entry.mesh.coordinates);
      view.connectivity = MakeConstSpan(entry.mesh.connectivity);
      view.versionCounter = entry.versionCounter;
    }
    outActorMeshes.push_back(view);
  }

  // Prune cache entries for actors that were not present in this reply (no longer in the scene).
  std::erase_if(meshCache, [](auto const& kv) { return !kv.second.visited; });
}

void DebugClient::OnSceneStepReply(protocol::SceneStepReply&& reply) {
  if (!reply.error.empty()) {
    Print(Format("Scene step request failed: %s", reply.error.c_str()), LogChannel::Warning);
  }
}

void DebugClient::OnSceneSyncReply(protocol::SceneSyncReply&& reply) {
  _state.Mutate([&](auto& state) {
    // Ignore replies for a scene that is no longer selected (stale or from another scene).
    if (reply.scene != state.selectedScene) {
      return;
    }

    // If syncing is disabled, ignore any in-flight replies that arrive after the disable request
    // was sent. This prevents resurrecting data that was just cleared in SetSceneSyncParams.
    if (!state.settings.sync.enabled) {
      return;
    }

    bool hasDataChanged = false;

    // Filter by the current params so the cached copy stays consistent with them, even if an
    // in-flight reply predates a params change.
    if (state.settings.sync.syncActors && reply.actors) {
      // Update our local mesh cache with new data from this reply
      if (state.settings.sync.syncMeshes && reply.meshData && reply.actorMeshRanges) {
        UpdateActorMeshes(reply, state.meshCache, state.syncData.actorMeshes);
      } else {
        state.syncData.actorMeshes.clear();
      }

      state.syncData.actors = std::move(*reply.actors);
      hasDataChanged = true;
    }

    if (state.settings.sync.syncDebugDraw && reply.debugDraw) {
      state.syncData.debugDraw = std::move(*reply.debugDraw);
      hasDataChanged = true;
    }
    if (hasDataChanged) {
      state.syncData.counter = ++state.syncCounter;
    }
  });
}

void DebugClient::SelectScene(SceneHandle handle) {
  DynamicArray<protocol::SceneSyncRequest> toSend;
  _state.Mutate([&](auto& state) { SetSelectedScene(state, handle, toSend); });

  // Send requests after releasing the _state lock.
  for (auto const& req : toSend) {
    Send(req);
  }
}

SceneHandle DebugClient::GetSelectedScene() const {
  return _state.Read(&State::selectedScene);
}

CoordinateSpace DebugClient::GetCoordinateSpace() const {
  return _state.Read(&State::coordinateSpace);
}

void DebugClient::SetSceneSyncParams(SceneSyncParams const& params) {
  auto newParams = params;
  newParams.syncInterval = Max(0.0f, newParams.syncInterval); // clamp

  std::optional<protocol::SceneSyncRequest> toSend;
  _state.Mutate([&](auto& state) {
    if (state.settings.sync == newParams) {
      return; // No change
    }

    // Clear cached data for any category that is being turned off, immediately.
    bool hasDataChanged = false;
    if (!newParams.enabled || !newParams.syncActors) {
      if (!state.syncData.actors.empty()) {
        state.syncData.actors.clear();
        hasDataChanged = true;
      }
    }
    if (!newParams.enabled || !newParams.syncDebugDraw) {
      if (state.syncData.debugDraw != protocol::DbgDrawData{}) {
        state.syncData.debugDraw = {};
        hasDataChanged = true;
      }
    }
    if (!newParams.enabled || !newParams.syncActors || !newParams.syncMeshes) {
      if (!state.syncData.actorMeshes.empty()) {
        state.syncData.actorMeshes.clear();
        hasDataChanged = true;
      }
    }
    if (hasDataChanged) {
      state.syncData.counter = ++state.syncCounter;
    }

    MOCHI_ASSERT_VERBOSE(
        state.syncData.actorMeshes.empty() ||
            state.syncData.actorMeshes.size() == state.syncData.actors.size(),
        "Size mismatch");

    // Store new parameters
    state.settings.sync = newParams;

    // Update syncing
    if (state.selectedScene.IsValid()) {
      toSend = MakeSyncRequest(state);
    }
  });

  // Send after releasing the _state lock
  if (toSend) {
    Send(*toSend);
  }
}

DebugClientSettings DebugClient::GetSettings() const {
  return _state.Read(&State::settings);
}

void DebugClient::SetSettings(DebugClientSettings const& settings) {
  // Apply SceneSyncParams (if changed)
  SetSceneSyncParams(settings.sync);

  // Store settings except for SceneSyncParams which have already been atomically synchronized.
  _state.Mutate([&](auto& state) {
    auto sync = state.settings.sync;
    state.settings = settings;
    state.settings.sync = sync;
  });
}

protocol::SceneSyncRequest DebugClient::MakeSyncRequest(State const& state) {
  protocol::SceneSyncRequest req;
  req.scene = state.selectedScene;
  req.enableAutoSync = state.settings.sync.enabled;
  req.syncInterval = state.settings.sync.syncInterval;
  req.syncActors = state.settings.sync.syncActors;
  req.syncDebugDraw = state.settings.sync.syncDebugDraw;
  req.syncMeshes = state.settings.sync.syncActors && state.settings.sync.syncMeshes;
  req.useVisualMesh = state.settings.sync.useVisualMesh;
  return req;
}

bool DebugClient::IsAutoSelectableScene(SceneInfo const& scene) {
  // By convention, we do not auto-select scenes with a leading underscore in the
  // name. These scenes are used for implementation details such as IK scenes, or
  // editor staging scenes.
  return scene.handle.IsValid() && !scene.name.starts_with("_");
}

SceneHandle DebugClient::FindAutoSelectableScene(Span<SceneInfo const> scenes) {
  // Pick a scene from the list to select. If there are multiple valid candidates, then pick the one
  // with the greatest handle value. This will be the scene that was created most recently since
  // handle values are assigned sequentially.
  SceneHandle best;
  for (SceneInfo const& scene : scenes) {
    if (IsAutoSelectableScene(scene) && (scene.handle.value > best.value)) {
      best = scene.handle;
    }
  }
  return best;
}

void DebugClient::SetSelectedScene(
    State& state,
    SceneHandle handle,
    DynamicArray<protocol::SceneSyncRequest>& outRequests) {
  SceneHandle newSelection;
  if (FindSceneIndex(state.scenes, handle) >= 0) {
    newSelection = handle;
  }

  if (newSelection == state.selectedScene) {
    return; // No change
  }

  SceneHandle const prevSelection = state.selectedScene;
  int const prevSceneIndex = FindSceneIndex(state.scenes, prevSelection);
  state.selectedScene = newSelection;

  if (state.settings.sync.enabled) {
    // Stop syncing the previous scene (if any)
    if (prevSceneIndex >= 0) {
      protocol::SceneSyncRequest req;
      req.scene = prevSelection;
      req.enableAutoSync = false;
      outRequests.emplace_back(std::move(req));
    }

    // Start syncing the new scene (if any)
    if (newSelection.IsValid()) {
      outRequests.emplace_back(MakeSyncRequest(state));
    }
  }

  state.meshCache.clear();
  state.syncData = {};
  state.syncData.scene = newSelection;
  state.syncData.counter = ++state.syncCounter;
}

void DebugClient::GetSceneSyncData(std::function<void(SceneSyncData const& data)> const& fn) const {
  MOCHI_ASSERT_VERBOSE(fn);
  _state.Read([&](auto const& state) { fn(state.syncData); });
}

void DebugClient::SetSceneStepMode(StepMode mode) {
  DynamicArray<protocol::SceneStepRequest> stepRequestsToSend;
  _state.Mutate([&](auto& state) {
    if (state.stepMode == mode) {
      return; // No change
    }

    // Store new step mode
    state.stepMode = mode;

    // Queue messages
    stepRequestsToSend.reserve(state.scenes.size());
    for (auto const& scene : state.scenes) {
      protocol::SceneStepRequest req;
      req.scene = scene.handle;
      req.mode = state.stepMode;
      stepRequestsToSend.push_back(req);
    }
  });
  for (auto const& msg : stepRequestsToSend) {
    Send(msg);
  }
}

StepMode DebugClient::GetSceneStepMode() const {
  return _state.Read(&State::stepMode);
}

void DebugClient::StepScene() {
  // Pause first
  SetSceneStepMode(StepMode::Pause);

  // Then send a message to the selected scene (only)
  std::optional<protocol::SceneStepRequest> msg;
  _state.Mutate([&](auto& state) {
    int const index = FindSceneIndex(state.scenes, state.selectedScene);
    if (index >= 0) {
      msg.emplace();
      msg->scene = state.selectedScene;
      msg->stepForward = true;
    }
  });
  if (msg) {
    Send(*msg);
  }
}

void DebugClient::RestoreSceneState() {
  std::optional<protocol::SceneStepRequest> msg;
  _state.Read([&](auto const& state) {
    int const index = FindSceneIndex(state.scenes, state.selectedScene);
    if (index >= 0) {
      msg.emplace();
      msg->scene = state.selectedScene;
      msg->restoreInitialState = true;
    }
  });
  if (msg) {
    Send(*msg);
  }
}

void DebugClient::OnLogMessage(protocol::LogMessage&& msg) {
  // If there is no user-installed print function and we are connected to an in-proc server,
  // then do not print anything. It has already been printed by the server, in the same process.
  if (_isInProc && !_printFn) {
    return;
  }

  // The message already ends with a newline, so don't add a redundant one.
  Print(
      std::string{msg.message},
      msg.channel,
      msg.file.c_str(),
      msg.line,
      /*newline*/ false);
}

void DebugClient::InitCommands() {
  auto addCommand = [this](std::string const& name, int numArgs, CommandFn fn) {
    auto& cmd = _commands[name];
    cmd.fn = std::move(fn);
    cmd.numArgs = numArgs;
  };
  addCommand("connect", -1, [this](auto args) { CmdConnect(args); });
  addCommand("disconnect", 0, [this](auto /*args*/) { Disconnect(); });
  addCommand("echo", -1, [this](auto args) { Print(Join(args, " ")); });
  addCommand("help", -1, [this](auto /*args*/) { CmdHelp(); });
  addCommand("cd", 1, [this](auto args) { CmdCd(args); });
  addCommand("pwd", 0, [this](auto /*args*/) { CmdPwd(); });
  addCommand("ls", -1, [this](auto args) { CmdLs(args); });
  addCommand("ping", 0, [this](auto /*args*/) { CmdPing(); });
  addCommand("servers", 0, [this](auto /*args*/) { CmdServers(); });
  addCommand("quit", -1, [this](auto /*args*/) { CmdQuit(); });
}

bool DebugClient::WasExitRequested() const {
  return _state.Read(&State::exitRequested);
}

std::string DebugClient::GetCommandPath() const {
  return _state.Read([](State const& state) { return FormatPath(state.cmdPath); });
}

std::string DebugClient::GetCommandPrompt() const {
  return GetCommandPath();
}

void DebugClient::CmdCd(Span<std::string const> args) {
  MOCHI_ASSERT_VERBOSE(args.size() == 1); // Because numArgs == 1
  auto const& pathArg = args[0];
  Error error;
  auto resolved = ResolvePath(pathArg, error);
  ValidatePath(resolved, error);
  if (error.IsOK()) {
    _state.Mutate([&](auto& state) { state.cmdPath = std::move(resolved); });
  } else {
    PrintError(error.GetDescription());
  }
}

void DebugClient::CmdPwd() {
  Print(GetCommandPath());
}

void DebugClient::PrintSceneList() {
  DynamicArray<std::string> names;
  _state.Read([&](auto const& state) {
    for (auto const& s : state.scenes) {
      names.emplace_back(s.name);
    }
  });
  if (!names.empty()) {
    std::ranges::sort(names);
    Print(Join(names, "\n"));
  }
}

void DebugClient::PrintActorList(std::string_view sceneName, Error& error) {
  MOCHI_ERROR_RETURN(error);
  // TODO: Deal with non-unique scene names
  SceneHandle sceneHandle = _state.Read([&](auto const& state) {
    int idx = FindSceneIndex(state.scenes, sceneName);
    return idx >= 0 ? state.scenes[idx].handle : SceneHandle{};
  });
  MOCHI_ERROR_IF(!sceneHandle.IsValid(), error, "Invalid scene");
  MOCHI_ERROR_RETURN(error);

  // Then get the list of actors
  protocol::SceneSyncRequest req;
  req.scene = sceneHandle;
  req.syncActors = true;
  auto reply = SendAndAwaitReply(req, kCommandTimeout, error);
  MOCHI_ERROR_RETURN(error);
  if (!reply.error.empty()) {
    // The reply came back, but with an error. Log it.
    PrintError(std::string{reply.error});
    return;
  }
  if (!reply.actors) {
    return;
  }

  // Sort by handle (the first column).
  std::ranges::sort(
      *reply.actors, [](auto const& a, auto const& b) { return a.handle.value < b.handle.value; });

  // Build an aligned "handle  type  name" table.
  auto const& actors = *reply.actors;
  DynamicArray<std::string> types;
  DynamicArray<std::string> names;
  for (auto const& a : actors) {
    char const* const typeStr = SReflect::EnumToString(a.type);
    types.emplace_back((typeStr != nullptr && typeStr[0] != '\0') ? typeStr : "?");
    names.emplace_back(a.name.empty() ? "unnamed" : a.name.c_str());
  }

  auto maxWidth = [](DynamicArray<std::string> const& col) {
    size_t width = 0;
    for (auto const& s : col) {
      width = Max(width, s.size());
    }
    return width;
  };
  int const w1 = static_cast<int>(maxWidth(types)) + 2;

  std::string text;
  for (int i = 0; i < isize(actors); ++i) {
    if (!text.empty()) {
      text += '\n';
    }
    text += Format(
        "%016" PRIx64 "  %-*s%s", actors[i].handle.value, w1, types[i].c_str(), names[i].c_str());
  }
  if (!text.empty()) {
    Print(text);
  }
}

void DebugClient::CmdLs(Span<std::string const> args) {
  if (args.size() > 1) {
    PrintError("Invalid number of arguments");
    return;
  }

  if (!IsConnected()) {
    PrintError("No connection");
    return;
  }

  Error error;
  auto target = ResolvePath(args.empty() ? "" : args[0], error);
  ValidatePath(target, error);
  if (!error.IsOK()) {
    PrintError(error.GetDescription());
    return;
  }

  // We can currently only list contents at the root scope (the list of scenes)
  if (target.empty()) {
    PrintSceneList();
  } else if (target.size() == 1) {
    PrintActorList(target[0], error);
  }

  if (!error.IsOK()) {
    PrintError(error.GetDescription());
  }
}

void DebugClient::CmdHelp() {
  Print(R"(Commands:
  help               Show this help.
  servers            List available servers.
  connect            Connect to localhost on the default port.
  connect <ip>       Connect to <ip> on the default port.
  connect <port>     Connect to localhost on <port>.
  connect <ip:port>  Connect to <ip> on <port>.
  cd <path>          Change current working directory.
  pwd                Print current working directory.
  ls                 List current directory contents.
  ls <path>          List contents of specified directory.
  ping               Measure round-trip time to the server.
  echo <message>     Print message to the console.
  disconnect         Disconnect from the current server.
  quit               Exit.)");
}

void DebugClient::CmdConnect(Span<std::string const> args) {
  if (args.empty()) {
    Connect("127.0.0.1", dbg::kDefaultDebugServerPort);
  } else if (args.size() == 1) {
    auto const& arg = args[0];
    Error error;
    auto const [address, port] = ParseAddressAndPort(arg, error);
    if (!error.IsOK()) {
      PrintError(error.GetDescription());
      return;
    }
    Connect(address, port);
  } else {
    PrintError("Invalid number of arguments");
  }
}

void DebugClient::CmdPing() {
  if (!IsConnected()) {
    PrintError("No connection");
    return;
  }

  constexpr int kNumPings = 6;
  Error error;
  for (int i = 0; i < kNumPings; ++i) {
    auto reply = SendAndAwaitReply(protocol::PingRequest{}, kCommandTimeout, error);
    if (!error.IsOK()) {
      PrintError(Format("ping: %s", error.GetDescription()));
      return;
    }
    constexpr uint64_t kNanosPerMilli = 1000000;
    uint64_t const elapsedMs = (reply.recvTimeNs - reply.sendTimeNs) / kNanosPerMilli;
    Print(Format("ping: %" PRIu64 " ms", elapsedMs));
  }
}

void DebugClient::CmdServers() {
  // Discover servers
  net::ServerList serverList(kDiscoveryPort);
  serverList.Refresh();
  // UDP discovery is asynchronous with no completion signal; wait briefly for responses to arrive.
  // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  DynamicArray<net::ServerInfo> servers;
  serverList.GetServers(servers);

  // Sort by address, then by port
  std::ranges::sort(servers, [](auto const& a, auto const& b) {
    return std::tie(a.address, a.port) < std::tie(b.address, b.port);
  });

  // Print table
  if (servers.empty()) {
    Print("No servers found.");
  } else {
    std::string text = "Servers:\n";
    for (auto const& s : servers) {
      std::string addrPort = Format("%s:%u", s.address.c_str(), static_cast<unsigned>(s.port));
      std::string suffix;
      if (s.version != _socket.CalcProtocolVersionHash()) {
        suffix = " (incompatible)";
      } else if (s.numClients == s.maxClients) {
        suffix = " (busy)";
      }
      text += Format("  %.21s  %s%s\n", addrPort.c_str(), s.label.c_str(), suffix.c_str());
    }
    Print(text);
  }
}

void DebugClient::CmdQuit() {
  _state.Mutate([](auto& state) { state.exitRequested = true; });
}
