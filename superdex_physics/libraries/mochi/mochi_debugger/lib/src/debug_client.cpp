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
#include <mochi_debugger/lib/console_utils.h>
#include <mochi_debugger/lib/debug_client.h>
#include <mochi_physics/dbg/console_path.h>
#include <mochi_physics/dbg/protocol.h>
#include <simple_reflection/simple_reflection_console.h>

#include <algorithm>
#include <chrono>
#include <exception>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

using namespace mochi;
using namespace mochi::dbg;

uint64_t StepProfileHistory::GetBeginSequence() const {
  return empty() ? 0 : front().sequence;
}

uint64_t StepProfileHistory::GetEndSequence() const {
  return empty() ? 0 : back().sequence + 1;
}

void StepProfileHistory::Append(DynamicArray<protocol::StepProfileData> const& profiles) {
  for (auto const& profile : profiles) {
    _profiles.push_back(profile);
  }
}

void StepProfileHistory::Trim(int maxSize) {
  int const excess = isize(_profiles) - maxSize;
  if (excess > 0) {
    _profiles.erase(_profiles.begin(), _profiles.begin() + excess);
  }
}

void StepProfileHistory::Clear() {
  _profiles.clear();
}

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
  _worker = std::thread([this] { ProcessQueuedCommands(); });
}

DebugClient::~DebugClient() {
  _commandQueue.Mutate([](auto& state) {
    state.isShuttingDown = true;
    state.commands.clear();
  });
  _commandsChanged.notify_all();
  Disconnect(); // Cancels the request that the running command may be waiting for
  _worker.join();
  Disconnect(); // The last command may have connected again
}

void DebugClient::Connect(std::string_view address, uint16_t port) {
  std::scoped_lock lock(_connectionMutex);
  DisconnectLocked();
  _socket.Connect(address, port);
}

void DebugClient::ConnectInProc(net::MessageServer& server) {
  std::scoped_lock lock(_connectionMutex);
  DisconnectLocked();
  _isInProc = true; // Remember that we are using an in-process connection.
  _socket.ConnectInProc(server);
}

void DebugClient::Disconnect() {
  std::scoped_lock lock(_connectionMutex);
  DisconnectLocked();
}

void DebugClient::DisconnectLocked() {
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
    state.sleepParams.reset();
    state.profileHistory.Clear();

    // Keep settings: it is a user preference that persists across connections.
  });
}

void DebugClient::QueueCommand(std::string_view str) {
  QueueCommand([this, command = std::string{str}] { ExecuteCommand(command); });
}

void DebugClient::QueueCommand(std::function<void()> fn) {
  MOCHI_ASSERT_VERBOSE(fn, "Null task");
  _commandQueue.Mutate([&](auto& state) {
    if (state.isShuttingDown) {
      return;
    }
    state.commands.push_back(std::move(fn));
  });
  _commandsChanged.notify_all();
}

void DebugClient::WaitForQueuedCommands() {
  MOCHI_ASSERT(
      std::this_thread::get_id() != _worker.get_id(), "A queued task cannot wait for itself");
  auto completed = std::make_shared<std::promise<void>>();
  auto ready = completed->get_future();
  QueueCommand([completed = std::move(completed)] { completed->set_value(); });
  ready.wait(); // A discarded completion command makes the future ready with a broken promise
}

void DebugClient::ProcessQueuedCommands() {
  for (;;) {
    auto command = _commandQueue.MutateWithLock(
        [&](auto& lock, auto& state) -> std::optional<std::function<void()>> {
          _commandsChanged.wait(
              lock, [&] { return state.isShuttingDown || !state.commands.empty(); });
          if (state.isShuttingDown) {
            return std::nullopt;
          }
          auto next = std::move(state.commands.front());
          state.commands.pop_front();
          return next;
        });
    if (!command) {
      return;
    }
    try {
      (*command)();
    } catch (std::exception const& e) {
      PrintError(Format("Exception: %s", e.what()));
    } catch (...) {
      PrintError("Unknown exception");
    }
  }
}

void DebugClient::ExecuteCommand(std::string_view str) {
  std::string error;
  size_t errorOffset = 0;
  auto const tokens = SReflect::detail::ConsoleTokenize(str, error, &errorOffset);
  if (!error.empty()) {
    PrintError(SReflect::detail::FormatConsoleError(error, str, errorOffset));
    return;
  }
  if (tokens.empty()) {
    return;
  }

  // The server interprets commands that have no client-side handler.
  auto const it = _commands.find(tokens[0]);
  if (it == _commands.end()) {
    CmdRemote(str);
    return;
  }

  // Check the number of arguments
  auto const& cmd = it->second;
  if ((cmd.numArgsAllowed >= 0) && (cmd.numArgsAllowed != isize(tokens) - 1)) {
    PrintError("Invalid number of arguments");
    return;
  }

  // Call command handler
  cmd.fn(tokens);
}

bool DebugClient::IsConnected() const {
  return GetStatus() == net::SocketStatus::Connected;
}

std::optional<protocol::CommandReply> DebugClient::ExecuteRemoteCommandAndWait(
    std::string_view command) {
  MOCHI_ASSERT(
      std::this_thread::get_id() == _worker.get_id(),
      "Remote commands must run on the command worker");
  if (_socket.GetStatus() != net::SocketStatus::Connected) {
    PrintError("No connection");
    return std::nullopt;
  }
  Error error;
  protocol::CommandRequest request;
  request.command = command;
  request.workingDirectory = GetCommandPath();

  double const timeout =
      _state.Read([](auto const& state) { return state.settings.commandTimeout; });
  auto reply = SendAndAwaitReply(std::move(request), timeout, error);
  if (!error.IsOK()) {
    PrintError(Format("No reply from the server: %s", error.GetDescription()));
    return std::nullopt;
  }
  if (!reply.error.empty()) {
    PrintError(std::string{reply.error});
    return std::nullopt;
  }
  return reply;
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

static protocol::DbgDrawFeatures ReconcileDebugDrawFeatures(
    protocol::DbgDrawFeatures newState,
    protocol::DbgDrawFeatures const& oldState) {
  newState.masterEnabled = oldState.masterEnabled;
  for (auto& newFeature : newState.features) {
    auto const* oldFeature =
        std::ranges::find(oldState.features, newFeature.name, &protocol::DbgDrawFeature::name);
    newFeature.enabled = oldFeature != oldState.features.end() ? oldFeature->enabled : false;
  }
  return newState;
}

void DebugClient::InitProtocol() {
  _socket.Register<protocol::CommandReply>();
  _socket.Register<protocol::CommandRequest>();
  _socket.Register<protocol::DebugDrawReply>(
      [this](auto&& msg) { OnDebugDrawReply(std::forward<decltype(msg)>(msg)); });
  _socket.Register<protocol::DebugDrawRequest>();
  _socket.Register<protocol::LogMessage>(
      [this](auto&& msg) { OnLogMessage(std::forward<decltype(msg)>(msg)); });
  _socket.Register<protocol::SceneAddRemove>(
      [this](auto&& msg) { OnSceneAddRemove(std::forward<decltype(msg)>(msg)); });
  _socket.Register<protocol::SceneSettingsReply>(
      [this](auto&& msg) { OnSceneSettingsReply(std::forward<decltype(msg)>(msg)); });
  _socket.Register<protocol::SceneSettingsRequest>();
  _socket.Register<protocol::SceneStepReply>(
      [this](auto&& msg) { OnSceneStepReply(std::forward<decltype(msg)>(msg)); });
  _socket.Register<protocol::SceneStepRequest>();
  _socket.Register<protocol::SceneSyncReply>(
      [this](auto&& msg) { OnSceneSyncReply(std::forward<decltype(msg)>(msg)); });
  _socket.Register<protocol::SceneSyncRequest>();
  _socket.Register<protocol::SleepParamsReply>(
      [this](auto&& msg) { OnSleepParamsReply(std::forward<decltype(msg)>(msg)); });
  _socket.Register<protocol::SleepParamsRequest>();
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
  DynamicArray<protocol::DebugDrawRequest> debugDrawRequestsToSend;
  DynamicArray<protocol::SleepParamsRequest> sleepRequestsToSend;
  DynamicArray<protocol::SceneSettingsRequest> settingsRequestsToSend;
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
    state.debugDraw = ReconcileDebugDrawFeatures(std::move(msg.debugDraw), state.debugDraw);
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
    SetSelectedScene(
        state,
        state.settings.autoSelectScene ? FindAutoSelectableScene(state.scenes) : SceneHandle{},
        syncRequestsToSend,
        debugDrawRequestsToSend,
        sleepRequestsToSend,
        settingsRequestsToSend);

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
  for (auto const& req : debugDrawRequestsToSend) {
    Send(req);
  }
  for (auto const& req : settingsRequestsToSend) {
    Send(req);
  }
  for (auto const& req : syncRequestsToSend) {
    Send(req);
  }
  for (auto const& req : sleepRequestsToSend) {
    Send(req);
  }

  _isFullyConnected = true;
}

void DebugClient::OnSceneAddRemove(protocol::SceneAddRemove&& msg) {
  std::optional<protocol::SceneStepRequest> stepRequestToSend;
  DynamicArray<protocol::SceneSyncRequest> syncRequestsToSend;
  DynamicArray<protocol::DebugDrawRequest> debugDrawRequestsToSend;
  DynamicArray<protocol::SleepParamsRequest> sleepRequestsToSend;
  DynamicArray<protocol::SceneSettingsRequest> settingsRequestsToSend;
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
      if (state.settings.autoSelectScene && !state.selectedScene.IsValid() && !hadSelectableScene &&
          IsAutoSelectableScene(state.scenes.back())) {
        SetSelectedScene(
            state,
            msg.scene,
            syncRequestsToSend,
            debugDrawRequestsToSend,
            sleepRequestsToSend,
            settingsRequestsToSend);
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
        SetSelectedScene(
            state,
            state.settings.autoSelectScene ? FindAutoSelectableScene(state.scenes) : SceneHandle{},
            syncRequestsToSend,
            debugDrawRequestsToSend,
            sleepRequestsToSend,
            settingsRequestsToSend);
      }
    }
  });
  if (stepRequestToSend) {
    Send(*stepRequestToSend);
  }
  for (auto const& req : debugDrawRequestsToSend) {
    Send(req);
  }
  for (auto const& req : settingsRequestsToSend) {
    Send(req);
  }
  for (auto const& req : syncRequestsToSend) {
    Send(req);
  }
  for (auto const& req : sleepRequestsToSend) {
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

    if (state.settings.sync.syncPerformance && reply.performance) {
      state.profileHistory.Append(*reply.performance);
      state.profileHistory.Trim(state.settings.maxProfileHistory);
    }
    if (hasDataChanged) {
      state.syncData.counter = ++state.syncCounter;
    }
  });
}

void DebugClient::SelectScene(SceneHandle handle) {
  DynamicArray<protocol::SceneSyncRequest> syncToSend;
  DynamicArray<protocol::DebugDrawRequest> debugDrawToSend;
  DynamicArray<protocol::SleepParamsRequest> sleepToSend;
  DynamicArray<protocol::SceneSettingsRequest> settingsToSend;
  _state.Mutate([&](auto& state) {
    SetSelectedScene(state, handle, syncToSend, debugDrawToSend, sleepToSend, settingsToSend);
  });

  // Send requests after releasing the _state lock.
  for (auto const& req : debugDrawToSend) {
    Send(req);
  }
  for (auto const& req : settingsToSend) {
    Send(req);
  }
  for (auto const& req : syncToSend) {
    Send(req);
  }
  for (auto const& req : sleepToSend) {
    Send(req);
  }
}

SceneHandle DebugClient::GetSelectedScene() const {
  return _state.Read(&State::selectedScene);
}

CoordinateSpace DebugClient::GetCoordinateSpace() const {
  return _state.Read(&State::coordinateSpace);
}

void DebugClient::ClearProfileHistory() {
  _state.Mutate([](auto& state) { state.profileHistory.Clear(); });
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
  std::optional<protocol::SceneSettingsRequest> settingsToSend;
  _state.Mutate([&](auto& state) {
    // Send only the changed scene settings to the selected scene.
    SceneSettings const& oldScene = state.settings.scene;
    SceneSettings const& newScene = settings.scene;
    if (state.selectedScene.IsValid() && oldScene != newScene) {
      auto& req = settingsToSend.emplace();
      req.scene = state.selectedScene;
      if (oldScene.forceSingleThreaded != newScene.forceSingleThreaded) {
        req.forceSingleThreaded = newScene.forceSingleThreaded;
      }
      if (oldScene.forceSingleIsland != newScene.forceSingleIsland) {
        req.forceSingleIsland = newScene.forceSingleIsland;
      }
    }

    auto sync = state.settings.sync;
    state.settings = settings;
    state.settings.sync = sync;
    state.settings.maxProfileHistory = Max(0, settings.maxProfileHistory);
    state.profileHistory.Trim(state.settings.maxProfileHistory);
  });

  // Send after releasing the _state lock
  if (settingsToSend) {
    Send(*settingsToSend);
  }
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
  req.syncPerformance = state.settings.sync.syncPerformance;
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
    DynamicArray<protocol::SceneSyncRequest>& outSyncRequests,
    DynamicArray<protocol::DebugDrawRequest>& outDebugDrawRequests,
    DynamicArray<protocol::SleepParamsRequest>& outSleepRequests,
    DynamicArray<protocol::SceneSettingsRequest>& outSettingsRequests) {
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
      outSyncRequests.emplace_back(std::move(req));
    }

    // Start syncing the new scene (if any)
    if (newSelection.IsValid()) {
      outSyncRequests.emplace_back(MakeSyncRequest(state));
    }
  }

  // Debug draw follows the selection: stop the previous scene and apply the client's complete
  // state to the new one, including an all-disabled state.
  if (prevSceneIndex >= 0) {
    protocol::DebugDrawRequest req;
    req.scene = prevSelection;
    req.featureEnable.resize(state.debugDraw.features.size(), false);
    outDebugDrawRequests.emplace_back(std::move(req));
  }
  if (newSelection.IsValid()) {
    outDebugDrawRequests.emplace_back(MakeDebugDrawRequest(state));
  }

  // Scene settings follow the selection too: release the previous scene's overrides and apply the
  // client's settings to the new one.
  if (prevSceneIndex >= 0) {
    outSettingsRequests.emplace_back(MakeSceneSettingsRequest(prevSelection, SceneSettings{}));
  }
  if (newSelection.IsValid()) {
    outSettingsRequests.emplace_back(MakeSceneSettingsRequest(newSelection, state.settings.scene));
  }

  state.meshCache.clear();
  state.profileHistory.Clear();
  state.syncData = {};
  state.syncData.scene = newSelection;
  state.syncData.counter = ++state.syncCounter;

  // Read the sleep params of the new scene. They may differ between scenes.
  state.sleepParams.reset();
  if (newSelection.IsValid()) {
    protocol::SleepParamsRequest req;
    req.scene = newSelection;
    outSleepRequests.emplace_back(std::move(req));
  }
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

DynamicArray<protocol::DbgDrawFeature> DebugClient::GetDebugDrawFeatures() const {
  return _state.Read([](auto const& state) { return state.debugDraw.features; });
}

bool DebugClient::IsDebugDrawEnabled() const {
  return _state.Read([](auto const& state) { return state.debugDraw.masterEnabled; });
}

protocol::DebugDrawRequest DebugClient::MakeDebugDrawRequest(State const& state) {
  protocol::DebugDrawRequest req;
  req.scene = state.selectedScene;
  req.masterEnable = state.debugDraw.masterEnabled;
  req.featureEnable.reserve(state.debugDraw.features.size());
  for (auto const& feature : state.debugDraw.features) {
    req.featureEnable.push_back(feature.enabled);
  }
  return req;
}

protocol::SceneSettingsRequest DebugClient::MakeSceneSettingsRequest(
    SceneHandle scene,
    SceneSettings const& settings) {
  protocol::SceneSettingsRequest req;
  req.scene = scene;
  req.forceSingleThreaded = settings.forceSingleThreaded;
  req.forceSingleIsland = settings.forceSingleIsland;
  return req;
}

void DebugClient::OnSceneSettingsReply(protocol::SceneSettingsReply&& reply) {
  if (!reply.error.empty()) {
    Print(Format("Scene settings request failed: %s", reply.error.c_str()), LogChannel::Warning);
  }
}

void DebugClient::EnableDebugDraw(bool enable) {
  std::optional<protocol::DebugDrawRequest> toSend;
  _state.Mutate([&](auto& state) {
    if (state.debugDraw.masterEnabled == enable) {
      return; // No change
    }
    state.debugDraw.masterEnabled = enable;
    if (state.selectedScene.IsValid()) {
      toSend = MakeDebugDrawRequest(state);
    }
  });

  // Send after releasing the _state lock
  if (toSend) {
    Send(*toSend);
  }
}

void DebugClient::EnableDebugDrawFeature(std::string_view featureName, bool enable) {
  std::optional<protocol::DebugDrawRequest> toSend;
  _state.Mutate([&](auto& state) {
    for (auto& feature : state.debugDraw.features) {
      if (feature.name == featureName) {
        bool const enableMaster = enable && !state.debugDraw.masterEnabled;
        if ((feature.enabled == enable) && !enableMaster) {
          return; // No change
        }
        feature.enabled = enable;
        if (enableMaster) {
          state.debugDraw.masterEnabled = true;
        }
        if (state.selectedScene.IsValid()) {
          toSend = MakeDebugDrawRequest(state);
        }
        break;
      }
    }
  });

  // Send after releasing the _state lock
  if (toSend) {
    Send(*toSend);
  }
}

void DebugClient::OnDebugDrawReply(protocol::DebugDrawReply&& reply) {
  if (!reply.error.empty()) {
    Print(Format("Debug draw request failed: %s", reply.error.c_str()), LogChannel::Warning);
  }
}

std::optional<experimental::SleepParams> DebugClient::GetSleepParams() const {
  return _state.Read(&State::sleepParams);
}

void DebugClient::SetSleepParams(experimental::SleepParams const& params) {
  std::optional<protocol::SleepParamsRequest> toSend;
  _state.Mutate([&](auto& state) {
    if (!state.sleepParams) {
      return; // Not received from the server yet
    }
    if (*state.sleepParams == params) {
      return; // No change
    }
    state.sleepParams = params;
    toSend.emplace();
    toSend->scene = state.selectedScene;
    toSend->params = params;
  });

  // Send after releasing the _state lock
  if (toSend) {
    Send(*toSend);
  }
}

void DebugClient::OnSleepParamsReply(protocol::SleepParamsReply&& reply) {
  if (!reply.error.empty()) {
    Print(Format("Sleep params request failed: %s", reply.error.c_str()), LogChannel::Warning);
  }
  _state.Mutate([&](auto& state) {
    // Only the first reply after selecting a scene is used. After that, the client's copy is
    // authoritative, so later replies cannot make the UI jump back to an older value.
    if (reply.scene == state.selectedScene && !state.sleepParams) {
      state.sleepParams = reply.params;
    }
  });
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
  // Register each command that will be handled on this client. Others will be forwarded to the
  // server directly. numArgsAllowed == -1 means "any number".
  auto addCommand = [this](std::string const& name, int numArgsAllowed, CommandFn fn) {
    auto& cmd = _commands[name];
    cmd.fn = std::move(fn);
    cmd.numArgsAllowed = numArgsAllowed;
  };
  addCommand("connect", -1, [this](auto tokens) { CmdConnect(tokens.subspan(1)); });
  addCommand("disconnect", 0, [this](auto /*tokens*/) { Disconnect(); });
  addCommand("echo", -1, [this](auto tokens) { Print(Join(tokens.subspan(1), " ")); });
  addCommand("help", -1, [this](auto /*tokens*/) { CmdHelp(); });
  addCommand("cd", 1, [this](auto tokens) { CmdCd(tokens.subspan(1)); });
  addCommand("pwd", 0, [this](auto /*tokens*/) { CmdPwd(); });
  addCommand("ls", -1, [this](auto tokens) { CmdLs(tokens.subspan(1)); });
  addCommand("ping", 0, [this](auto /*tokens*/) { CmdPing(); });
  addCommand("servers", 0, [this](auto /*tokens*/) { CmdServers(); });
  addCommand("quit", -1, [this](auto /*tokens*/) { CmdQuit(); });
}

bool DebugClient::WasExitRequested() const {
  return _state.Read(&State::exitRequested);
}

std::string DebugClient::GetCommandPath() const {
  return _state.Read(&State::cmdPath);
}

std::string DebugClient::GetCommandPrompt() const {
  return GetCommandPath();
}

void DebugClient::CmdCd(Span<std::string const> args) {
  std::string const cwd = GetCommandPath();
  std::string path;
  std::string error;
  size_t errorOffset = 0;
  if (!ResolveConsolePath(cwd, args[0], path, error, errorOffset)) {
    PrintError(
        SReflect::detail::FormatConsoleError(
            error, "cd", {args.data(), args.size()}, 1, errorOffset));
    return;
  }
  // Validate the path by attempting an "ls" command.
  // Exception: You are always allowed to shortten the path.
  bool const isAncestor = path == "/" || cwd.starts_with(path + "/");
  if (!isAncestor && !ExecuteRemoteCommandAndWait("ls " + SReflect::ToJsonString(path, false))) {
    return;
  }
  _state.Mutate([&](auto& state) { state.cmdPath = std::move(path); });
}

void DebugClient::CmdPwd() {
  Print(GetCommandPath());
}

void DebugClient::CmdLs(Span<std::string const> args) {
  bool detailed = false;
  bool options = true;
  std::string const* path = nullptr;
  for (auto const& arg : args) {
    if (options && arg == "--") {
      options = false;
      continue;
    }
    if (options && arg.size() > 1 && arg[0] == '-') {
      for (char flag : std::string_view{arg}.substr(1)) {
        switch (flag) {
          case 'l':
            detailed = true;
            break;
          default:
            PrintError("Usage: ls [<path>] [-l]");
            return;
        }
      }
    } else if (path == nullptr) {
      path = &arg;
    } else {
      PrintError("Usage: ls [<path>] [-l]");
      return;
    }
  }
  std::string command = "ls";
  if (path != nullptr) {
    command += " " + SReflect::ToJsonString(*path, false);
  }
  CmdRemote(command, detailed);
}

void DebugClient::CmdRemote(std::string_view command, bool detailed) {
  auto const reply = ExecuteRemoteCommandAndWait(command);
  if (reply) {
    if (reply->listing && !reply->listing->empty()) {
      ConsoleFormatParams params;
      params._showTypes = detailed;
      Print(FormatConsoleList(*reply->listing, params) + "\n");
    } else if (!reply->output.empty()) {
      Print(std::string{reply->output});
    }
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
  ls                 List current directory methods and properties.
  ls <path>          List methods and properties of specified directory.
  ls [<path>] -l     List with property types.
  <path>             Print the value at <path>.
  <path> <value>     Set the value at <path>, for example: gravity [0, -9.81, 0]
  <path> [args...]   Call the method at <path>.
  ping               Measure round-trip time to the server.
  echo <message>     Print message to the console.
  disconnect         Disconnect from the current server.
  quit               Exit.
)");
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
  for (int i = 0; i < kNumPings; ++i) {
    using clock = std::chrono::steady_clock;
    auto const start = clock::now();
    if (!ExecuteRemoteCommandAndWait("")) {
      return;
    }
    auto const elapsed = clock::now() - start;
    Print(Format(
        "ping: %lld ms",
        static_cast<long long>(
            std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count())));
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
