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

#include "mochi_context.h"
#include "mochi_ecs.h"
#include "mochi_scene.h"
#include "mochi_scene_debugger.h"

#include <mochi_core/net/message_server.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/dynamic_string.h>
#include <mochi_core/utils/guarded.h>
#include <mochi_core/utils/log.h>
#include <mochi_physics/dbg/debug_server_internal.h>
#include <mochi_physics/dbg/protocol.h>

#include <memory>

using namespace mochi;
using namespace mochi::dbg;

namespace {

// Implements the DebugServer and DebugServerInternal APIs
class DebugServerImpl : public DebugServerInternal {
  MOCHI_DECLARE_NO_COPY_NO_MOVE(DebugServerImpl);

 public:
  DebugServerImpl(ContextImpl* context);
  ~DebugServerImpl() override;

  // DebugServer API
  void Start(uint16_t preferredPort) override;
  void Stop() override;
  bool HasStarted() const override;
  bool HasConnection() const override;
  uint16_t GetPort() const override;
  void SetCoordinateSpace(CoordinateSpace const& space) override;

  // DebugServerInternal API
  void StartInProc() override;
  void OnAddScene(Scene* scene) override;
  void OnRemoveScene(Scene* scene) override;
  net::MessageServer& GetMessageServer_ForTestingOnly() override {
    return _server;
  }

  struct SceneInfo {
    // Raw pointer to the Scene. Safe because OnRemoveFromScene will remove it
    // before the Scene is destroyed.
    SceneImpl* scene = nullptr;

    // The SceneDebugger is used to queue scene-specific messages that must be
    // processed on the scene's owning thread.
    std::shared_ptr<SceneDebugger> debugger;
  };

  struct State {
    DynamicArray<SceneInfo> scenes;
    net::ClientId client = {};
    // For now, the default matches Mochi's historical default (X-right, Y-up, Z-backward)
    CoordinateSpace coordinateSpace = CoordinateSpace::Filament();
    bool hasRegistered = false;
    bool hasStarted = false;
    bool isInProcSocket = false;
  };

 private:
  void StartImpl(uint16_t preferredPort, bool inProc);
  void OnClientEvent(net::ClientId client, net::ClientEvent event);
  void OnClientConnect(net::ClientId client);
  void OnClientDisconnect(net::ClientId client);
  void RegisterProtocol();
  static void SetSceneStepMode(SceneInfo const& info, StepMode mode);

  template <class RequestT>
  void RegisterSceneRequest();

  static int FindScene(State const& state, SceneHandle handle);
  static void
  AttachSceneDebuggerForClient(SceneInfo& info, net::ClientId client, net::MessageServer& server);
  static void ReleaseSceneDebugger(SceneInfo& info);

  // Message Handlers:
  void OnPingRequest(net::ClientId client, protocol::PingRequest const& request);
  void OnSceneRequest(
      net::ClientId client,
      std::unique_ptr<protocol::SceneRequest>&& request,
      protocol::SceneReply&& reply);

  net::MessageServer _server;
  RecursiveGuarded<State> _state;
};

} // namespace

// Format a human readable label for UDP discovery.
static DynamicString FormatLabel(DebugServerImpl::State const& state) {
  // Pick one scene name to show. It is best if code matches DebugClient's auto-selection logic
  // so that the scene we show in the label is the one the client would auto-select after
  // joining. DebugClient::IsAutoSelectableScene is the canonical definition of that rule.
  char const* sceneName = nullptr;
  SceneHandle sceneHandle;
  for (auto&& info : state.scenes) {
    char const* name = info.scene->GetName();
    char const* displayName = (name && *name) ? name : "unnamed";
    auto const handle = info.scene->GetHandle();
    if ((*displayName != '_') && (handle.value > sceneHandle.value)) {
      sceneName = displayName;
      sceneHandle = handle;
    }
  }
  if (!sceneName) {
    return DynamicString{
        Format("%d scene%s", isize(state.scenes), (state.scenes.size() == 1) ? "" : "s")};
  } else if (state.scenes.size() == 1) {
    return DynamicString{sceneName};
  } else {
    return DynamicString{Format("%s (and %d others)", sceneName, isize(state.scenes) - 1)};
  }
}

DebugServerImpl::DebugServerImpl([[maybe_unused]] ContextImpl* context) {
  // In the future, we will need to store the context pointer. For now, it is unused.
  MOCHI_ASSERT_VERBOSE(context);

  // NOTE: Protocol registration is deferred until StartImpl to minimize the cost
  // of an unused DebugServer.
}

DebugServerImpl::~DebugServerImpl() {
  Stop();
}

void DebugServerImpl::StartImpl(uint16_t preferredPort, bool inProc) {
  _state.Mutate([&](auto& state) {
    // Register message types before we start hosting for the first time.
    if (!state.hasRegistered) {
      RegisterProtocol();
      state.hasRegistered = true;
    }

    if (state.hasStarted) {
      // Switching transports requires an explicit Stop() first, so that the
      // receive-thread join happens outside this lock (see Stop()).
      MOCHI_ASSERT(
          state.isInProcSocket == inProc,
          "DebugServer already started with a different transport; call Stop() before switching transports.");
      return;
    }

    // Track client connections
    _server.SetClientCallback(
        [this](net::ClientId client, net::ClientEvent event) { OnClientEvent(client, event); });

    // Format a human readable label for UDP discovery
    auto label = FormatLabel(state);

    // Start accepting connections
    int constexpr kMaxClients = 1;
    if (inProc) {
      _server.StartInProc(kMaxClients, label);
    } else {
      _server.SetDiscoveryPort(kDiscoveryPort);
      _server.Start(preferredPort, kMaxClients, label);
    }

    // Set a log callback so that we can broadcast LogMessage to the clients.
    // Logging will also pass through to the default logging function.
    //
    // TODO: This is currently not very robust because:
    //   * We could override someone else's custom logging function.
    //   * Someone else could later override our logging function, preventing the broadcast.
    //   * A better long term solution is needed (e.g. install more than one logging callback
    //     at the same time).
    SetLogCallback([this](LogChannel channel, char const* message, char const* file, int line) {
      log_impl::DefaultLogFn(channel, message, file, line); // Log it locally
      _server.Broadcast(protocol::LogMessage{channel, message, file, line}); // Send to client(s)
    });

    state.hasStarted = true;
    state.isInProcSocket = inProc;
  });
}

void DebugServerImpl::Start(uint16_t preferredPort) {
  StartImpl(preferredPort, /*inProc*/ false);
}

void DebugServerImpl::StartInProc() {
  StartImpl(/*preferredPort*/ 0, /*inProc*/ true);
}

void DebugServerImpl::Stop() {
  // Lock the state and clean up the state.
  bool const needsStop = _state.Mutate([&](auto& state) {
    if (!state.hasStarted) {
      return false;
    }

    // Restore the default logging function.
    SetLogCallback({});

    // Detach the client-event callback so a disconnect flushed during the pending
    // _server.Stop() cannot re-enter _state on the receive thread.
    _server.SetClientCallback(nullptr);

    // Clear client state, including SceneDebugger instances.
    if (state.client != 0) {
      OnClientDisconnect(state.client);
      MOCHI_ASSERT_VERBOSE(state.client == 0);
    }

    state.hasStarted = false;
    state.isInProcSocket = false;
    return true;
  });

  // Then, stop the underlying server outside the lock.
  if (needsStop) {
    _server.Stop();
  }
}

bool DebugServerImpl::HasStarted() const {
  return _state.Read([](auto const& state) { return state.hasStarted; });
}

bool DebugServerImpl::HasConnection() const {
  return _state.Read([](auto const& state) { return state.client != 0; });
}

uint16_t DebugServerImpl::GetPort() const {
  return _server.GetPort(); // Thread-safe
}

void DebugServerImpl::SetCoordinateSpace(CoordinateSpace const& space) {
  space.Validate(ErrorAssert{});
  _state.Mutate([&](auto& state) {
    MOCHI_ASSERT(
        !state.hasStarted, "DebugServer::SetCoordinateSpace must be called before Start().");
    state.coordinateSpace = space;
  });
}

void DebugServerImpl::OnClientEvent(net::ClientId client, net::ClientEvent event) {
  MOCHI_ASSERT(client != 0, "Invalid ClientId");
  switch (event) {
    case net::ClientEvent::Connected:
      OnClientConnect(client);
      break;
    case net::ClientEvent::Disconnected:
      OnClientDisconnect(client);
      break;
  }
}

static void FillSceneList(DebugServerImpl::State const& state, protocol::SceneList& outList) {
  auto const numScenes = isize(state.scenes);
  outList.handles.resize(numScenes);
  outList.names.resize(numScenes);
  for (int i = 0; i < numScenes; ++i) {
    outList.handles[i] = state.scenes[i].scene->GetHandle();
    outList.names[i] = state.scenes[i].scene->GetName();
  }
}

void DebugServerImpl::OnClientConnect(net::ClientId client) {
  _state.Mutate([&](auto& state) {
    MOCHI_ASSERT_VERBOSE(state.client == 0, "Currently only supports one client at a time");
    state.client = client;

    for (auto& sceneInfo : state.scenes) {
      AttachSceneDebuggerForClient(sceneInfo, client, _server);
    }

    // Push the authoritative snapshot to the newly connected client, under the lock, so it is
    // totally ordered with respect to the SceneAddRemove deltas broadcast from
    // OnAddScene/OnRemoveScene (which also hold the lock). The single client connection delivers
    // messages in order, keeping the client's list consistent.
    protocol::WelcomeMessage welcome;
    FillSceneList(state, welcome.scenes);
    welcome.coordinateSpace = state.coordinateSpace;
    _server.SendTo(client, welcome);
  });
}

void DebugServerImpl::AttachSceneDebuggerForClient(
    SceneInfo& info,
    net::ClientId client,
    net::MessageServer& server) {
  MOCHI_ASSERT_VERBOSE(client != 0, "Invalid ClientId{}");
  MOCHI_ASSERT_VERBOSE(info.scene != nullptr);
  MOCHI_ASSERT_VERBOSE(!info.debugger, "Scene already has a debugger");

  info.debugger = dbg::CreateSceneDebugger(client, info.scene, server);
  MOCHI_ASSERT_VERBOSE(info.debugger);

  // Install it on the scene (thread-safe)
  info.scene->SetDebugger(info.debugger);
}

void DebugServerImpl::ReleaseSceneDebugger(SceneInfo& info) {
  if (info.debugger) {
    // Call PreShutdown (thread-safe). This prevents any future messages from being sent
    info.debugger->PreShutdownAsync(info.scene);

    // Release our shared_ptr
    info.debugger.reset();

    // Notify the scene. It will clear its active debugger pointer, and queue final cleanup
    // on its own thread.
    info.scene->SetDebugger(nullptr);
  }
}

void DebugServerImpl::OnClientDisconnect([[maybe_unused]] net::ClientId client) {
  _state.Mutate([&](auto& state) {
    MOCHI_ASSERT_VERBOSE(state.client == client, "Disconnected client is not the active client");
    state.client = 0;

    // Clear any scene debuggers
    for (auto& s : state.scenes) {
      ReleaseSceneDebugger(s);
    }
  });
}

template <class RequestT>
void DebugServerImpl::RegisterSceneRequest() {
  using ReplyT = typename RequestT::Reply;
  static_assert(std::is_base_of_v<protocol::SceneRequest, RequestT>);
  static_assert(std::is_base_of_v<protocol::SceneReply, ReplyT>);
  _server.Register<RequestT>([this](net::ClientId client, auto&& request) {
    // Move the message into a std::unique_ptr that can be used polymorphically.
    // Also construct an empty reply message, in case the request fails immediately.
    OnSceneRequest(
        client, std::make_unique<RequestT>(std::forward<decltype(request)>(request)), ReplyT{});
  });
}

void DebugServerImpl::RegisterProtocol() {
  _server.Register<protocol::LogMessage>();
  _server.Register<protocol::PingReply>();
  _server.Register<protocol::PingRequest>(
      [this](net::ClientId client, auto&& request) { OnPingRequest(client, request); });
  _server.Register<protocol::SceneAddRemove>();
  _server.Register<protocol::SceneStepReply>();
  RegisterSceneRequest<protocol::SceneStepRequest>();
  _server.Register<protocol::SceneSyncReply>();
  RegisterSceneRequest<protocol::SceneSyncRequest>();
  _server.Register<protocol::WelcomeMessage>();

  // Set version so connection will be refused if the client does not register exactly the
  // same collection of messages.
  _server.SetVersion(_server.CalcProtocolVersionHash());
}

void DebugServerImpl::OnPingRequest(net::ClientId client, protocol::PingRequest const& request) {
  // Reply immediately
  _server.SendTo(client, protocol::PingReply{request});
}

void DebugServerImpl::OnSceneRequest(
    net::ClientId client,
    std::unique_ptr<protocol::SceneRequest>&& request,
    protocol::SceneReply&& reply) {
  MOCHI_ASSERT_VERBOSE(request);
  bool sceneExists = false;
  _state.Mutate([&](auto& state) {
    int idx = FindScene(state, request->scene);
    if (idx >= 0) {
      sceneExists = true;
      auto& sceneInfo = state.scenes[idx];
      MOCHI_ASSERT_VERBOSE(
          sceneInfo.debugger, "Every scene must have a SceneDebugger while a client is connected");
      sceneInfo.debugger->OnReceiveAsync(std::move(request));
    }
  });
  if (!sceneExists) {
    // Send failure reply now
    reply.requestId = request->requestId;
    reply.scene = request->scene;
    reply.error = "No such scene";
    _server.SendTo(client, reply);
  }
}

int DebugServerImpl::FindScene(State const& state, SceneHandle handle) {
  for (int i = 0; i < isize(state.scenes); ++i) {
    if (state.scenes[i].scene->GetHandle() == handle) {
      return i;
    }
  }
  return -1;
}

void DebugServerImpl::OnAddScene(Scene* scene) {
  MOCHI_ASSERT(scene != nullptr);
  auto handle = scene->GetHandle();
  _state.Mutate([&](auto& state) {
    MOCHI_ASSERT(FindScene(state, handle) == -1, "Already added");
    auto& info = state.scenes.push_back();
    info.scene = assert_cast<SceneImpl*>(scene);
    if (state.hasStarted) {
      _server.SetLabel(FormatLabel(state));

      // Broadcast the delta under the lock so it is ordered with WelcomeMessage::scenes.
      if (state.client != 0) {
        AttachSceneDebuggerForClient(info, state.client, _server);

        // All new scenes start out paused.
        SetSceneStepMode(info, StepMode::Pause);

        protocol::SceneAddRemove delta;
        delta.scene = handle;
        delta.name = scene->GetName();
        delta.wasAdded = true;
        _server.Broadcast(delta);
      }
    }
  });
}

void DebugServerImpl::OnRemoveScene(Scene* scene) {
  MOCHI_ASSERT(scene != nullptr);
  auto handle = scene->GetHandle();
  _state.Mutate([&](auto& state) {
    int idx = FindScene(state, handle);
    MOCHI_ASSERT(idx >= 0, "Unknown scene");

    // Broadcast the delta (before erasing) under the lock so it is ordered with
    // WelcomeMessage::scenes.
    if (state.hasStarted && state.client != 0) {
      protocol::SceneAddRemove delta;
      delta.scene = handle;
      delta.name = scene->GetName();
      delta.wasAdded = false;
      _server.Broadcast(delta);
    }

    auto& sceneInfo = state.scenes[idx];
    ReleaseSceneDebugger(sceneInfo);

    state.scenes.erase(state.scenes.begin() + idx);
    if (state.hasStarted) {
      _server.SetLabel(FormatLabel(state));
    }
  });
}

void DebugServerImpl::SetSceneStepMode(SceneInfo const& sceneInfo, StepMode mode) {
  // Send a SceneStepRequest message that the scene handles on its own thread.
  MOCHI_ASSERT_VERBOSE(sceneInfo.scene && sceneInfo.debugger, "Null pointer");
  auto request = std::make_unique<protocol::SceneStepRequest>();
  request->scene = sceneInfo.scene->GetHandle();
  request->mode = mode;
  request->sendReply = false; // Do not send reply message to the client
  sceneInfo.debugger->OnReceiveAsync(std::move(request));
}

std::unique_ptr<DebugServerInternal> dbg::CreateDebugServer(Context* context) {
  return std::make_unique<DebugServerImpl>(assert_cast<ContextImpl*>(context));
}
