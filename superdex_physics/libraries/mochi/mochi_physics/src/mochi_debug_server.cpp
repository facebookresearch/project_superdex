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
#include "mochi_debug_draw.h"
#include "mochi_ecs.h"
#include "mochi_scene.h"
#include "mochi_scene_console.h"
#include "mochi_scene_debugger.h"

#include <mochi_core/net/message_server.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/dynamic_string.h>
#include <mochi_core/utils/guarded.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/string_utils.h>
#include <mochi_physics/dbg/console_path.h>
#include <mochi_physics/dbg/debug_server_internal.h>
#include <mochi_physics/dbg/protocol.h>
#include <picojson/picojson.h>
#include <simple_reflection/simple_reflection_console.h>

#include <algorithm>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

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
  void Start(uint16_t preferredPort, bool allowRemoteConnection) override;
  void Stop() override;
  bool HasStarted() const override;
  bool HasConnection() const override;
  uint16_t GetPort() const override;
  void SetCoordinateSpace(CoordinateSpace const& space) override;

  // DebugServerInternal API
  void StartInProc() override;
  void OnAddScene(Scene* scene) override;
  void OnRemoveScene(Scene* scene) override;
  void SetConsoleRoot_ForTestingOnly(void* object, SReflect::TypeInfo const& typeInfo) override;
  net::MessageServer& GetMessageServer_ForTestingOnly() override {
    return _server;
  }

  struct SceneInfo {
    // Raw pointer to the Scene. Safe because OnRemoveFromScene will remove it
    // before the Scene is destroyed.
    SceneImpl* scene = nullptr;

    // Unique among the scenes. It needs no quotes as a path step, so `cd scenes/<name>` works in
    // the terminal.
    std::string debugName;

    // The SceneDebugger is used to queue scene-specific messages that must be
    // processed on the scene's owning thread.
    std::shared_ptr<SceneDebugger> debugger;
  };

  struct State {
    DynamicArray<SceneInfo> scenes;
    // The next suffix to try for each debug name base that has been taken (see MakeDebugName)
    std::unordered_map<std::string, int> nextDebugNameSuffix;
    void* consoleRoot = nullptr;
    SReflect::TypeInfo const* consoleRootType = nullptr;
    net::ClientId client = {};
    // For now, the default matches Mochi's historical default (X-right, Y-up, Z-backward)
    CoordinateSpace coordinateSpace = CoordinateSpace::Filament();
    bool hasRegistered = false;
    bool hasStarted = false;
    bool isInProcSocket = false;
  };

 private:
  void StartImpl(uint16_t preferredPort, bool inProc, bool allowRemoteConnection);
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

  // Forward a request to the scene with the given debug name and return true, leaving request
  // moved from. Return false and leave request untouched if there is no such scene.
  template <class RequestT>
  bool ForwardToScene(std::string_view debugName, RequestT& request);

  // Call fn(object, typeInfo) with the object that console paths outside "scenes" start from. The
  // context is bound to this thread, and shapes made by console members stay registered, until fn
  // returns.
  template <class Fn>
  void WithConsoleRoot(Fn const& fn);

  // Message Handlers:
  void OnSceneRequest(
      net::ClientId client,
      std::unique_ptr<protocol::SceneRequest>&& request,
      protocol::SceneReply&& reply);
  void OnConsoleRequest(net::ClientId client, protocol::CommandRequest request);
  void OnAutoCompleteRequest(net::ClientId client, protocol::AutoCompleteRequest request);

  ContextImpl* const _context;
  net::MessageServer _server;
  RecursiveGuarded<State> _state;

 public:
  MOCHI_STRUCT_BEGIN(DebugServerImpl)
  MOCHI_BASE_CLASS(mochi::DebugServer)
  MOCHI_STRUCT_END()
};

} // namespace

// Make a debug name from a scene's name that no scene in state uses: the sanitized name, or
// "scene" if that is empty, followed by the first free number from 2 up if it is taken.
static std::string MakeDebugName(DebugServerImpl::State& state, std::string_view sceneName) {
  std::string base = SanitizeConsoleName(sceneName);
  if (base.empty()) {
    base = "scene";
  }
  return MakeUniqueDebugName(base, state.nextDebugNameSuffix, [&](std::string const& name) {
    return std::ranges::find(state.scenes, name, &DebugServerImpl::SceneInfo::debugName) !=
        state.scenes.end();
  });
}

// Format a human readable label for UDP discovery.
static DynamicString FormatLabel(DebugServerImpl::State const& state) {
  // Pick one scene name to show. It is best if code matches DebugClient's auto-selection logic
  // so that the scene we show in the label is the one the client would auto-select after
  // joining. DebugClient::IsAutoSelectableScene is the canonical definition of that rule.
  char const* sceneName = nullptr;
  SceneHandle sceneHandle;
  for (auto&& info : state.scenes) {
    auto const handle = info.scene->GetHandle();
    if (!info.debugName.starts_with('_') && (handle.value > sceneHandle.value)) {
      sceneName = info.debugName.c_str();
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

DebugServerImpl::DebugServerImpl(ContextImpl* context) : _context(context) {
  MOCHI_ASSERT_VERBOSE(context);

  // NOTE: Protocol registration is deferred until StartImpl to minimize the cost
  // of an unused DebugServer.
}

DebugServerImpl::~DebugServerImpl() {
  Stop();
}

void DebugServerImpl::StartImpl(uint16_t preferredPort, bool inProc, bool allowRemoteConnection) {
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
      _server.Start(preferredPort, kMaxClients, label, allowRemoteConnection);
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

void DebugServerImpl::Start(uint16_t preferredPort, bool allowRemoteConnection) {
  StartImpl(preferredPort, /*inProc*/ false, allowRemoteConnection);
}

void DebugServerImpl::StartInProc() {
  StartImpl(/*preferredPort*/ 0, /*inProc*/ true, /*allowRemoteConnection*/ false);
}

void DebugServerImpl::SetConsoleRoot_ForTestingOnly(
    void* object,
    SReflect::TypeInfo const& typeInfo) {
  _state.Mutate([&](auto& state) {
    MOCHI_ASSERT(
        !state.hasStarted, "The console root must be installed while the server is stopped");
    state.consoleRoot = object;
    state.consoleRootType = object ? &typeInfo : nullptr;
  });
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
    outList.names[i] = state.scenes[i].debugName;
  }
}

static SReflect::ConsoleListEntry MakeDirectoryEntry(std::string_view name) {
  return {
      ._kind = SReflect::ConsoleEntryKind::Directory,
      ._name = std::string{name},
      ._coreType = SReflect::CoreType::CT_struct};
}

// List the scenes by name, sorted like map keys in SReflect::ConsoleList.
static void ListScenes(
    DebugServerImpl::State const& state,
    DynamicArray<SReflect::ConsoleListEntry>& outEntries) {
  for (auto const& info : state.scenes) {
    auto entry = MakeDirectoryEntry(info.debugName);
    entry._value = Format("(handle=%llu)", info.scene->GetHandle().value);
    outEntries.push_back(std::move(entry));
  }
  std::ranges::sort(outEntries, {}, &SReflect::ConsoleListEntry::_name);
}

// Copy the debug draw feature catalog into the message. Scenes start with debug draw off, and the
// client is authoritative from then on, so all the flags are false.
static void FillDbgDrawFeatures(protocol::DbgDrawFeatures& outFeatures) {
  auto const catalog = GetDebugDrawFeatureCatalog();
  outFeatures.features.resize(catalog.size());
  for (int i = 0; i < isize(catalog); ++i) {
    outFeatures.features[i].name = catalog[i].name;
    outFeatures.features[i].description = catalog[i].description;
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
    FillDbgDrawFeatures(welcome.debugDraw);
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
    info.debugger->PreShutdownAsync();

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
  _server.Register<protocol::AutoCompleteReply>();
  _server.Register<protocol::AutoCompleteRequest>([this](net::ClientId client, auto&& request) {
    OnAutoCompleteRequest(client, std::forward<decltype(request)>(request));
  });
  _server.Register<protocol::CommandReply>();
  _server.Register<protocol::CommandRequest>([this](net::ClientId client, auto&& request) {
    OnConsoleRequest(client, std::forward<decltype(request)>(request));
  });
  _server.Register<protocol::DebugDrawReply>();
  RegisterSceneRequest<protocol::DebugDrawRequest>();
  _server.Register<protocol::LogMessage>();
  _server.Register<protocol::SceneAddRemove>();
  _server.Register<protocol::SceneSettingsReply>();
  RegisterSceneRequest<protocol::SceneSettingsRequest>();
  _server.Register<protocol::SceneStepReply>();
  RegisterSceneRequest<protocol::SceneStepRequest>();
  _server.Register<protocol::SceneSyncReply>();
  RegisterSceneRequest<protocol::SceneSyncRequest>();
  _server.Register<protocol::SleepParamsReply>();
  RegisterSceneRequest<protocol::SleepParamsRequest>();
  _server.Register<protocol::WelcomeMessage>();

  // Set version so connection will be refused if the client does not register exactly the
  // same collection of messages.
  _server.SetVersion(_server.CalcProtocolVersionHash());
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

// Split a resolved absolute path into steps, with offsets into the whole path.
static std::vector<SReflect::detail::ConsolePathStep> SplitResolvedPath(std::string_view resolved) {
  std::vector<SReflect::detail::ConsolePathStep> steps;
  [[maybe_unused]] bool const isValid =
      SReflect::detail::SplitConsolePath(resolved.substr(1), steps);
  MOCHI_ASSERT(isValid, "Resolved path must be valid");
  for (auto& step : steps) {
    ++step.offset;
  }
  return steps;
}

// Where the part of a resolved path inside scenes/<name> starts
static size_t ScenePathStart(
    Span<SReflect::detail::ConsolePathStep const> steps,
    std::string_view resolved) {
  return steps.size() > 2 ? steps[2].offset : resolved.size();
}

template <class Fn>
void DebugServerImpl::WithConsoleRoot(Fn const& fn) {
  ScopedSchedulerBinding schedulerBinding(_context->GetTaskScheduler());
  ContextImpl::ScopedShapeHandleRetention retainShapes(_context);
  auto const [object, typeInfo] = _state.Read([&](auto const& state) {
    using Root = std::pair<void*, SReflect::TypeInfo const*>;
    if (state.consoleRoot) {
      return Root{state.consoleRoot, state.consoleRootType};
    }
    return Root{_context, &SReflect::GetTypeInfo<ContextImpl>()};
  });
  fn(object, *typeInfo);
}

void DebugServerImpl::OnConsoleRequest(net::ClientId client, protocol::CommandRequest request) {
  protocol::CommandReply reply(request);
  auto const fail = [&](std::string_view error) {
    reply.error = error;
    _server.SendTo(client, reply);
  };
  std::string error;
  size_t errorOffset = 0;
  std::vector<std::string> tokens;
  for (auto& token : SReflect::detail::ConsoleTokenize(request.command, error, &errorOffset)) {
    tokens.push_back(std::move(token.text));
  }
  if (!error.empty()) {
    fail(SReflect::detail::FormatConsoleError(error, request.command, errorOffset));
    return;
  }
  if (tokens.empty()) {
    _server.SendTo(client, reply); // An empty command is a connection check
    return;
  }

  static constexpr std::string_view kCommands[] = {"ls"};
  request.isList = std::ranges::find(kCommands, tokens.front()) != std::ranges::end(kCommands);
  size_t const pathIndex = request.isList ? 1 : 0;
  if (request.isList && tokens.size() > 2) {
    fail("Invalid number of arguments for ls");
    return;
  }
  std::string const path = tokens.size() > pathIndex ? tokens[pathIndex] : "";
  std::string resolved;
  if (!ResolveConsolePath(request.workingDirectory, path, resolved, error, errorOffset)) {
    SReflect::Span<std::string const> const args{tokens.data() + 1, tokens.size() - 1};
    fail(SReflect::detail::FormatConsoleError(error, tokens.front(), args, pathIndex, errorOffset));
    return;
  }
  request.path = resolved;
  reply.path = resolved;
  if (!request.isList) {
    request.args.assign(tokens.begin() + 1, tokens.end());
  }

  auto const steps = SplitResolvedPath(resolved);
  auto const args = DynamicArray<std::string>(request.args.begin(), request.args.end());
  if (!steps.empty() && steps[0].name == "scenes") {
    if (steps.size() >= 2) {
      request.pathStart = ScenePathStart(steps, resolved);
      if (ForwardToScene(steps[1].name, request)) {
        return;
      }
      error = SReflect::ToJsonString(steps[1].name, false) + " not found";
      errorOffset = steps[1].offset;
    } else if (request.isList) {
      reply.listing.emplace();
      _state.Read([&](auto const& state) { ListScenes(state, *reply.listing); });
    } else if (!args.empty()) {
      error = "Cannot set value because \"scenes\" is read-only";
      errorOffset = steps[0].offset;
    } else {
      // Each scene maps to an empty object, because scenes are not serialized recursively.
      picojson::object scenes;
      _state.Read([&](auto const& state) {
        for (auto const& info : state.scenes) {
          scenes.emplace(info.debugName, picojson::object{});
        }
      });
      std::string output = picojson::value{std::move(scenes)}.serialize(/*prettify*/ true);
      output.pop_back(); // Pretty JSON ends with a newline
      reply.output = output;
    }
    if (!error.empty()) {
      reply.error = SReflect::detail::FormatConsoleError(error, resolved, args, 0, errorOffset);
    }
  } else {
    WithConsoleRoot([&](void* object, SReflect::TypeInfo const& typeInfo) {
      if (request.isList) {
        auto listing = SReflect::ConsoleList(object, typeInfo, resolved, /*params*/ {}, &error, 1);
        if (error.empty()) {
          if (steps.empty()) {
            auto scenes = MakeDirectoryEntry("scenes");
            scenes._coreType = SReflect::CoreType::CT_map;
            scenes._containerSize =
                _state.Read([](auto const& state) { return state.scenes.size(); });
            listing.insert(listing.begin(), std::move(scenes));
          }
          reply.listing = std::move(listing);
        }
      } else {
        std::string output;
        SReflect::detail::ConsoleExecute(object, typeInfo, resolved, args, &output, &error, 1);
        reply.output = output;
      }
    });
    reply.error = error;
  }
  _server.SendTo(client, reply);
}

void DebugServerImpl::OnAutoCompleteRequest(
    net::ClientId client,
    protocol::AutoCompleteRequest request) {
  using SReflect::detail::ConsoleCompletionTarget;
  protocol::AutoCompleteReply reply(request);
  auto completion =
      SReflect::detail::FindConsoleCompletion(request.command, request.pathTokenIndex);
  std::string resolved;
  std::string error;
  size_t errorOffset = 0;
  // Only a path in the first token takes arguments. A command such as `cd <path>` takes none.
  if (!completion ||
      (completion->target == ConsoleCompletionTarget::Argument && request.pathTokenIndex != 0) ||
      !ResolveConsolePath(
          request.workingDirectory, completion->path, resolved, error, errorOffset)) {
    _server.SendTo(client, reply);
    return;
  }
  completion->path = resolved;
  bool const isStep = completion->target == ConsoleCompletionTarget::PathStep;
  std::string_view const typed = std::string_view{request.command}.substr(completion->offset);
  SReflect::ConsoleAutoCompleteParams params;
  params._directoriesOnly = request.directoriesOnly;

  auto const steps = SplitResolvedPath(resolved);
  if (!steps.empty() && steps[0].name == "scenes") {
    if (steps.size() >= 2) {
      request.pathStart = ScenePathStart(steps, resolved);
      request.completion = std::move(*completion);
      if (ForwardToScene(steps[1].name, request)) {
        return;
      }
    } else if (isStep) {
      DynamicArray<SReflect::ConsoleListEntry> scenes;
      _state.Read([&](auto const& state) { ListScenes(state, scenes); });
      AddDirectoryCandidates(reply.result, scenes, typed, params);
      // Scene names starting with '_' denote implementation details, such as IK scenes.
      auto const& candidates = reply.result._candidates;
      auto const it = std::ranges::find_if(
          candidates, [](std::string const& name) { return !name.starts_with('_'); });
      reply.result._defaultIndex =
          it != candidates.end() ? static_cast<size_t>(it - candidates.begin()) : 0;
    }
  } else {
    WithConsoleRoot([&](void* object, SReflect::TypeInfo const& typeInfo) {
      bool const ok = SReflect::detail::ConsoleAutoComplete(
          object, typeInfo, *completion, typed, params, reply.result, &error, 1);
      if (ok && isStep && steps.empty()) {
        SReflect::detail::AddConsoleStepCandidates(reply.result, {"scenes"}, typed, params);
      }
    });
  }
  _server.SendTo(client, reply);
}

template <class RequestT>
bool DebugServerImpl::ForwardToScene(std::string_view debugName, RequestT& request) {
  std::optional<uint64_t> number;
  try {
    number = ParseConsoleHandle(debugName);
  } catch (std::invalid_argument const&) {
    return false;
  }
  if (!number && debugName.starts_with('@')) {
    debugName.remove_prefix(1);
  }
  return _state.Mutate([&](auto& state) {
    for (auto& sceneInfo : state.scenes) {
      if (number ? sceneInfo.scene->GetHandle().value == *number
                 : sceneInfo.debugName == debugName) {
        if (!sceneInfo.debugger) { // In case it was released just before _state was locked.
          return false;
        }
        sceneInfo.debugger->OnReceiveAsync(std::make_unique<RequestT>(std::move(request)));
        return true;
      }
    }
    return false;
  });
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
    std::string debugName = MakeDebugName(state, scene->GetName());
    auto& info = state.scenes.push_back();
    info.scene = assert_cast<SceneImpl*>(scene);
    info.debugName = std::move(debugName);
    if (state.hasStarted) {
      _server.SetLabel(FormatLabel(state));

      // Broadcast the delta under the lock so it is ordered with WelcomeMessage::scenes.
      if (state.client != 0) {
        AttachSceneDebuggerForClient(info, state.client, _server);

        // All new scenes start out paused.
        SetSceneStepMode(info, StepMode::Pause);

        protocol::SceneAddRemove delta;
        delta.scene = handle;
        delta.name = info.debugName;
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
      delta.name = state.scenes[idx].debugName;
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
