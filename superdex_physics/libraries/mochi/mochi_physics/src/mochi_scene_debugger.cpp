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

#include "mochi_scene_debugger.h"
#include "mochi_active_collisions.h"
#include "mochi_common_components.h"
#include "mochi_constraint.h"
#include "mochi_island.h"
#include "mochi_scene.h"
#include "mochi_scene_profiler.h"
#include "mochi_sleep.h"

#include <mochi_core/geometry/mesh_data.h>
#include <mochi_core/net/message_dispatcher.h>
#include <mochi_core/net/message_server.h>
#include <mochi_core/utils/container_utils.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/guarded.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/span_utils.h>
#include <mochi_physics/cpp_api/mochi_actor.h>
#include <mochi_physics/dbg/protocol.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>

using namespace mochi;
using namespace mochi::dbg;

static SceneDebugger::ClockFn MakeDefaultClock() {
  auto const startTime = std::chrono::steady_clock::now();
  return [startTime] {
    auto const now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - startTime).count();
  };
}

static bool IsPaused(StepMode mode) {
  return mode == StepMode::Pause;
}

static bool TracksRealTime(StepMode mode) {
  return mode == StepMode::Play;
}

static bool IsActorAwake(entt::registry const& reg, entt::entity e) {
  auto const* memberInfo = reg.try_get<CIslandMemberInfo const>(e);
  return memberInfo && sleep::IsIslandAwake(reg, memberInfo->island);
}

// A constraint belongs to the island of its actors. Constraints nested in compounds are not listed
// by CIslandMembers.
static int32_t CountActiveConstraints(entt::registry const& reg) {
  int32_t numConstraints = 0;
  for (auto&& [e, info] : reg.view<CConstraintInfo const>().each()) {
    bool const isActive = std::any_of(info.actors.begin(), info.actors.end(), [&](entt::entity a) {
      return IsActorAwake(reg, a);
    });
    numConstraints += isActive ? 1 : 0;
  }
  return numConstraints;
}

template <bool kIsSync>
static int32_t CountActiveContacts(entt::registry const& reg) {
  int32_t numContacts = 0;
  for (auto&& [island, descendants] :
       reg.view<CIslandDescendants const, TagIslandIsAwake const>().each()) {
    for (entt::entity e : descendants.actors) {
      auto const* collisions = reg.try_get<CActiveCollisions<kIsSync, TimeStep::Current> const>(e);
      if (collisions) {
        for (ActiveCollision const& collision : *collisions) {
          numContacts += isize(collision.collisionResult.sampleIndices);
        }
      }
    }
  }
  return numContacts;
}

static protocol::StepProfileData RecordStepProfileData(
    SceneImpl const& scene,
    StepInfo const& step,
    StepProfileTimers const& stepProfile) {
  entt::registry const& reg = scene.GetRegistry();
  protocol::StepProfileData profile;
  profile.simTimeStepSec = StaticCast<float>(step.timeStepSec);
  profile.sceneStepSec = StaticCast<float>(stepProfile.stepSec);
  CIslandStats totals;
  bool const hasBreakdown = step.timeStepSec > 0.0 && stepProfile.numWorkerThreads == 0;
  for (auto&& [island, members, dofs] :
       reg.view<CIslandMembers const, CIslandDofInfo const, TagIslandIsAwake const>().each()) {
    ++profile.activeIslands;
    profile.activeActors += isize(members.actors);
    profile.activeDofs += dofs.dofsSize;
    if (hasBreakdown) {
      if (auto const* stats = reg.try_get<CIslandStats const>(island)) {
        totals.contactSec += stats->contactSec;
        totals.assemblySec += stats->assemblySec;
        totals.solveSec += stats->solveSec;
        totals.queriesSec += stats->queriesSec;
      }
    }
  }
  profile.activeConstraints = CountActiveConstraints(reg);
  profile.activeContacts = CountActiveContacts<false>(reg) + CountActiveContacts<true>(reg);
  profile.convergenceStatus = scene.GetSolverStats().convergenceStatus;
  profile.numWorkerThreads = stepProfile.numWorkerThreads;
  profile.sceneContactSec = StaticCast<float>(totals.contactSec);
  profile.sceneAssemblySec = StaticCast<float>(totals.assemblySec);
  profile.sceneSolveSec = StaticCast<float>(totals.solveSec);
  profile.sceneQueriesSec = StaticCast<float>(totals.queriesSec);
  return profile;
}

namespace {

class SceneDebuggerImpl final : public SceneDebugger {
  MOCHI_DECLARE_NO_COPY_NO_MOVE(SceneDebuggerImpl);

 public:
  explicit SceneDebuggerImpl(net::ClientId client, SceneImpl* scene, net::MessageServer& server)
      : _client(client), _sceneHandle(scene->GetHandle()), _server(&server) {
    // WARNING: Only call thread-safe functions on the scene pointer. This is NOT its owning thread!
    // WARNING: Do not store the scene pointer for later use.
    Init(scene);
  }

  ~SceneDebuggerImpl() override = default;

  // SceneDebugger API
  void UpdateOnSceneThread(SceneImpl* scene) override;
  void OnPreStep(SceneImpl* scene, StepInfo const& step) override;
  bool IsForceSingleThreaded() const override {
    return !IsShutdown() && _forceSingleThreaded;
  }
  void CaptureStepProfile(
      SceneImpl* scene,
      StepInfo const& step,
      StepProfileTimers const& stepProfile) override;
  void OnPostStep(SceneImpl* scene, StepInfo const& step) override;
  StepMode GetStepMode() const override {
    return _stepMode.load();
  }
  void SetClock(ClockFn clock) override;
  void OnReceiveAsync(std::unique_ptr<net::Message> msg) override;
  void PreShutdownAsync() override;
  void ShutdownOnSceneThread(SceneImpl* scene) override;

 private:
  void Init(SceneImpl* scene);
  void UpdateOnSceneThreadImpl(SceneImpl* scene);
  bool IsShutdown() const;
  bool IsClientConnected() const;
  void WaitForRealTime(SceneImpl* scene);
  void SendToClient(net::Message const& msg) const;

  // Build and send a SceneSyncReply for the requested categories. Runs on the scene's thread.
  void SendSyncReply(
      SceneImpl* scene,
      bool syncActors,
      bool syncDebugDraw,
      bool syncMeshes,
      bool useVisualMesh,
      bool syncPerformance,
      uint64_t requestId);

  // Reconcile per-actor mesh-tracking state against the live actor set. When meshesEnabled is
  // false, cancels all queries and clears the tracking map so a later re-enable re-sends meshes.
  // Runs on the scene's thread.
  void UpdateMeshQueries(SceneImpl* scene, bool meshesEnabled, bool useVisualMesh);

  // Actor mesh synced to the client.
  enum class MeshSource { Surface, Visual, ContactSkin };

  // Per-actor mesh-tracking state (scene-thread only).
  struct ActorInfo {
    QueryHandle query; // Valid only for non-rigid actors (live node-positions query).
    MeshSource meshSource = MeshSource::Surface;
    bool meshSent = false; // Whether this actor's connectivity was already sent.
    bool marked = false; // Transient; used by UpdateMeshQueries mark-and-sweep only.
  };

  // Append an actor's surface mesh to meshData and record its ranges in outRanges. Connectivity is
  // sent once per actor (it never changes); rigid actors send nothing after that, non-rigid actors
  // keep sending updated coordinates. Uses/updates info.
  static void AppendActorMesh(
      Actor const& actor,
      ActorInfo& info,
      protocol::MeshSyncData& meshData,
      protocol::MeshSyncRanges& outRanges);

  // Send a periodic SceneSyncReply if one is due. Runs on the scene's thread.
  void SendSyncIfRequested(SceneImpl* scene);

  // Message Handlers (called on the scene's thread)
  void OnDebugDrawRequest(SceneImpl* scene, protocol::DebugDrawRequest&& request);
  void OnSceneSettingsRequest(SceneImpl* scene, protocol::SceneSettingsRequest&& request);
  void OnSceneStepRequest(SceneImpl* scene, protocol::SceneStepRequest&& request);
  void OnSceneSyncRequest(SceneImpl* scene, protocol::SceneSyncRequest&& request);
  void OnSleepParamsRequest(SceneImpl* scene, protocol::SleepParamsRequest&& request);

  net::ClientId const _client;
  SceneHandle const _sceneHandle;
  Guarded<net::MessageServer*> _server;
  std::mutex _sceneThreadMutex;
  net::MessageDispatcher<SceneImpl*> _dispatcher;
  Guarded<DynamicArray<std::unique_ptr<net::Message>>> _inbox;
  DynamicArray<std::unique_ptr<net::Message>> _newMessages;

  // Counts simulation steps. Incremented in OnPostStep.
  uint64_t _stepCounter = 1;

  uint64_t _profileSequenceCounter = 0;

  // Sync state. Only touched on the scene's owning (step) thread.
  struct SyncState {
    float interval = 0;
    bool syncActors = false;
    bool syncDebugDraw = false;
    bool syncMeshes = false;
    bool useVisualMesh = true;
    bool syncPerformance = false;
    double lastSyncTime = 0.0;
    uint64_t lastSyncStep = 0;
  };
  std::optional<SyncState> _sync;

  // Step profiles not yet sent to the client, oldest first (scene-thread only). Filled while
  // performance syncing is enabled. Capped so it cannot grow without bound between syncs.
  static constexpr int kMaxPendingProfiles = 4096;
  std::deque<protocol::StepProfileData> _pendingProfileData;

  // Mesh syncing state (scene-thread only). One entry per actor while mesh syncing is enabled.
  std::unordered_map<ActorHandle, ActorInfo> _actorInfo;

  // Identifies the version of the mesh data being sent. Incremented whenever previously sent
  // meshes become obsolete. Starts at 1 because the protocol reserves 0 to mean "no mesh".
  uint64_t _meshVersionCounter = 1;

  ClockFn _clock = MakeDefaultClock();

  std::atomic<StepMode> _stepMode = StepMode::Pause;
  bool _singleStepRequested = false;

  bool _forceSingleThreaded = false; // Scene-thread only

  // Whether this instance enabled the scene profiler (scene-thread only).
  bool _profilerEnabled = false;

  // Real-time tracking state
  struct RealTimeTracking {
    std::optional<double> realTimeStart; // Real start time for real-time tracking
    std::optional<double> lastPreStepTime; // Real time when the previous step was about to run.
    std::optional<double> lastTimeStepSec; // Sim time elapsed by the previous step.
    double simTimeTotal = 0.0; // Total simulation time elapsed during real-time tracking

    void ResetTime() {
      realTimeStart.reset();
      lastPreStepTime.reset();
      lastTimeStepSec.reset();
      simTimeTotal = 0.0;
    }
  };

  RealTimeTracking _realTimeTracking;

  struct InitialState {
    bool shouldCapture = true;
    DynamicArray<uint8_t> captureData;
    DynamicString error;
  };

  InitialState _initialState;
};

} // namespace

void SceneDebuggerImpl::Init(SceneImpl* scene) {
  // WARNING: Only call thread-safe functions on the scene pointer. This is NOT its owning thread!
  // WARNING: Do not store the scene pointer for later use.

  MOCHI_ASSERT(scene != nullptr);
  MOCHI_ASSERT(_client != 0, "Invalid ClientId{}");
  MOCHI_ASSERT(_sceneHandle.IsValid(), "Invalid SceneHandle");

  // Register message handlers
  _dispatcher.Register<protocol::DebugDrawRequest>([this](auto* scene, auto&& msg) {
    OnDebugDrawRequest(scene, std::forward<decltype(msg)>(msg));
  });
  _dispatcher.Register<protocol::SceneSettingsRequest>([this](auto* scene, auto&& msg) {
    OnSceneSettingsRequest(scene, std::forward<decltype(msg)>(msg));
  });
  _dispatcher.Register<protocol::SceneStepRequest>([this](auto* scene, auto&& msg) {
    OnSceneStepRequest(scene, std::forward<decltype(msg)>(msg));
  });
  _dispatcher.Register<protocol::SceneSyncRequest>([this](auto* scene, auto&& msg) {
    OnSceneSyncRequest(scene, std::forward<decltype(msg)>(msg));
  });
  _dispatcher.Register<protocol::SleepParamsRequest>([this](auto* scene, auto&& msg) {
    OnSleepParamsRequest(scene, std::forward<decltype(msg)>(msg));
  });
}

void SceneDebuggerImpl::SendToClient(net::Message const& msg) const {
  // Send a message to the client. Ignore if the server pointer has already been cleared by
  // PreShutdown.
  _server.Read([&](auto* server) {
    if (server) {
      server->SendTo(_client, msg);
    }
  });
}

void SceneDebuggerImpl::UpdateOnSceneThread(SceneImpl* scene) {
  std::lock_guard lock{_sceneThreadMutex};
  if (IsShutdown()) {
    return;
  }
  UpdateOnSceneThreadImpl(scene);
}

// Set a custom clock function for unit tests.
void SceneDebuggerImpl::SetClock(ClockFn clock) {
  std::lock_guard lock{_sceneThreadMutex};
  _clock = clock ? std::move(clock) : MakeDefaultClock();
  _realTimeTracking.ResetTime();
  if (_sync) {
    _sync->lastSyncTime = _clock();
  }
}

void SceneDebuggerImpl::UpdateOnSceneThreadImpl(SceneImpl* scene) {
  MOCHI_ASSERT_VERBOSE(scene);

  // Enabled here rather than in the constructor, which does not run on the scene's thread.
  if (!_profilerEnabled) {
    profiler::Enable(scene->GetRegistry());
    _profilerEnabled = true;
  }

  // Acquire new messages from the inbox
  _newMessages.clear();
  _inbox.Mutate([&](auto& inbox) { std::swap(_newMessages, inbox); });

  // Process them
  for (auto& msg : _newMessages) {
    MOCHI_ASSERT_VERBOSE(msg);
    // Use the dispatcher to call the appropriate handler using the most derived type.
    auto const& msgType = msg->GetFinalTypeInfo();
    bool wasDispatched = _dispatcher.DispatchVoid(scene, msgType._typeId, msg.get());
    if (!wasDispatched) {
      MOCHI_LOG_WARNING(
          "[SceneDebugger] Received message of unknown type `%s`. Ignoring it.",
          msgType._nameWithNamespace);
    }
  }

  // Transmit periodic sync data if it is due for this step.
  SendSyncIfRequested(scene);
}

bool SceneDebuggerImpl::IsShutdown() const {
  return _server.Read([](auto* server) { return server == nullptr; });
}

bool SceneDebuggerImpl::IsClientConnected() const {
  return _server.Read(
      [this](auto* server) { return server && Contains(server->GetClients(), _client); });
}

void SceneDebuggerImpl::WaitForRealTime(SceneImpl* scene) {
  if (!TracksRealTime(GetStepMode())) {
    return;
  }

  if (!_realTimeTracking.realTimeStart.has_value()) {
    _realTimeTracking.realTimeStart = _clock();
  }

  double constexpr kTimeToleranceSec = 1e-5; // 0.01 ms
  double constexpr kSleepThresholdSec = 3e-3; // 3 ms
  double constexpr kSleepTimeSec = 1e-3; // 1 ms
  double constexpr kBusyWaitThresholdSec = 5e-4; // 0.5 ms

  // Loop until we have slowed down the simulation enough to not exceed real time.
  for (;;) {
    // Cancel if client disconnected
    if (!IsClientConnected()) {
      break;
    }

    // Cancel if the mode changed or a handled message reset real-time tracking.
    if (!TracksRealTime(GetStepMode()) || !_realTimeTracking.realTimeStart.has_value()) {
      return;
    }

    double const now = _clock();
    double const realTimeElapsed = now - *_realTimeTracking.realTimeStart;
    double const realTimeRemaining = _realTimeTracking.simTimeTotal - realTimeElapsed;
    if (realTimeRemaining <= kTimeToleranceSec) {
      return;
    }

    // If we have enough time to kill, then sleep for a bit. This is not exact because the operating
    // system may not wake us exactly when we ask.
    if (realTimeRemaining > kSleepThresholdSec) {
      // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
      std::this_thread::sleep_for(std::chrono::duration<double>(kSleepTimeSec));
    }

    // Handle network messages while we wait, unless we are already very close to the target time.
    if (realTimeRemaining > kBusyWaitThresholdSec) {
      UpdateOnSceneThreadImpl(scene);
    }
  }
}

void SceneDebuggerImpl::OnPreStep(SceneImpl* scene, StepInfo const& step) {
  std::lock_guard lock{_sceneThreadMutex};
  if (IsShutdown()) {
    return;
  }
  double const preStepTime = _clock();

  if (TracksRealTime(GetStepMode()) && _realTimeTracking.lastPreStepTime.has_value() &&
      _realTimeTracking.lastTimeStepSec.has_value() &&
      preStepTime - *_realTimeTracking.lastPreStepTime > *_realTimeTracking.lastTimeStepSec) {
    _realTimeTracking.ResetTime();
  }

  // Poll for messages before checking pause/step state.
  UpdateOnSceneThreadImpl(scene);

  // Optionally slow the simulation to match real time.
  // NOTE: IK scene step with an infinite time step. No real time tracking in that case.
  if (TracksRealTime(GetStepMode()) && !_singleStepRequested && IsFinite(step.timeStepSec)) {
    WaitForRealTime(scene);
    if (TracksRealTime(GetStepMode())) {
      // Sample AFTER the throttle wait so the next step's stall check measures the true external
      // inter-step gap, not our own throttling (which pads each step to ~timeStepSec).
      _realTimeTracking.lastPreStepTime = _clock();
      _realTimeTracking.lastTimeStepSec = step.timeStepSec;
    }
  } else {
    _realTimeTracking.lastPreStepTime.reset();
    _realTimeTracking.lastTimeStepSec.reset();
  }

  // Hold here if paused. A single-step request passes through this pause for one step.
  bool wasPaused = false;
  while (::IsPaused(GetStepMode()) && !_singleStepRequested && IsClientConnected()) {
    // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    UpdateOnSceneThreadImpl(scene);
    wasPaused = true;
  }

  if (wasPaused || _singleStepRequested) {
    _realTimeTracking.ResetTime();
  }

  _singleStepRequested = false;

  // Capture state just before the first simulation step. If it fails,
  // then remember the reasons so we can report it to the client later.
  if (_initialState.shouldCapture) {
    Error error;
    scene->CaptureStateToBytes(_initialState.captureData, error);
    if (!error.IsOK()) {
      _initialState.captureData.clear(); // Just in case
    }
    _initialState.error = error.IsOK() ? "" : error.GetDescription();
    _initialState.shouldCapture = false;
  }
}

void SceneDebuggerImpl::CaptureStepProfile(
    SceneImpl* scene,
    StepInfo const& step,
    StepProfileTimers const& stepProfile) {
  std::lock_guard lock{_sceneThreadMutex};
  if (IsShutdown() || step.timeStepSec == 0.0) {
    return;
  }
  uint64_t const sequence = ++_profileSequenceCounter;
  if (!_sync || !_sync->syncPerformance) {
    return;
  }

  if (isize(_pendingProfileData) >= kMaxPendingProfiles) {
    _pendingProfileData.pop_front();
  }
  auto& profile =
      _pendingProfileData.emplace_back(RecordStepProfileData(*scene, step, stepProfile));
  profile.sequence = sequence;
}

void SceneDebuggerImpl::OnPostStep(SceneImpl* scene, StepInfo const& step) {
  // It is safe to process messages and access the scene at this time.
  std::lock_guard lock{_sceneThreadMutex};

  if (IsShutdown()) {
    return;
  }

  ++_stepCounter;
  if (!IsFinite(step.timeStepSec)) {
    _realTimeTracking.ResetTime();
  } else if (TracksRealTime(GetStepMode())) {
    _realTimeTracking.simTimeTotal += step.timeStepSec;
  }

  // A sync sent below includes the profile captured for this step.
  UpdateOnSceneThreadImpl(scene);
}

void SceneDebuggerImpl::OnReceiveAsync(std::unique_ptr<net::Message> msg) {
  // TODO: This inbox could fill without bound if messages are never processed on the scene thread.
  // Possible solutions:
  // - Some later messages may supersede earlier ones. If so, prune the inbox before emplacing.
  // - Use a watchdog thread to check for message expiration. Send failure replies asynchronously.
  // - Put a limit on the number of messages, just in case.

  MOCHI_ASSERT_VERBOSE(msg, "Null message");
  _inbox.Mutate([&](auto& inbox) { inbox.emplace_back(std::move(msg)); });
}

void SceneDebuggerImpl::OnDebugDrawRequest(SceneImpl* scene, protocol::DebugDrawRequest&& request) {
  MOCHI_ASSERT_VERBOSE(request.scene == _sceneHandle, "Received by the wrong scene");
  protocol::DebugDrawReply reply(request);

  auto& debugDraw = scene->GetDebugDraw();
  int const numFeatures = debugDraw.GetNumFeatures();
  if (isize(request.featureEnable) != numFeatures) {
    reply.error = Format(
        "Expected %d debug draw feature flags but received %d",
        numFeatures,
        isize(request.featureEnable));
    MOCHI_LOG_WARNING("[SceneDebugger] Ignoring DebugDrawRequest. %s.", reply.error.c_str());
  } else {
    bool didChange = debugDraw.IsEnabled() != request.masterEnable;
    debugDraw.Enable(request.masterEnable);
    for (int i = 0; i < numFeatures; ++i) {
      didChange |= debugDraw.IsFeatureEnabled(i) != request.featureEnable[i];
      debugDraw.EnableFeature(i, request.featureEnable[i]);
    }

    // The visible state of the scene changed, so the client should receive fresh data on the next
    // pump, even while paused.
    if (didChange) {
      ++_stepCounter;
    }
  }

  if (request.sendReply) {
    SendToClient(reply);
  }
}

void SceneDebuggerImpl::OnSceneSettingsRequest(
    SceneImpl* scene,
    protocol::SceneSettingsRequest&& request) {
  MOCHI_ASSERT_VERBOSE(request.scene == _sceneHandle, "Received by the wrong scene");
  entt::registry& reg = scene->GetRegistry();

  if (request.forceSingleThreaded.has_value()) {
    _forceSingleThreaded = *request.forceSingleThreaded; // Takes effect on the next step
  }

  bool forceSingleIsland = island::GetForceSingleIslandByDebugger(reg);
  if (request.forceSingleIsland.has_value() && *request.forceSingleIsland != forceSingleIsland) {
    forceSingleIsland = *request.forceSingleIsland;
    island::SetForceSingleIslandByDebugger(reg, forceSingleIsland);
    ++_stepCounter; // Send a fresh sync even while paused
  }

  if (request.sendReply) {
    protocol::SceneSettingsReply reply(request);
    reply.forceSingleThreaded = _forceSingleThreaded;
    reply.forceSingleIsland = forceSingleIsland;
    SendToClient(reply);
  }
}

void SceneDebuggerImpl::OnSceneStepRequest(SceneImpl* scene, protocol::SceneStepRequest&& request) {
  protocol::SceneStepReply reply(request);

  if (request.mode.has_value()) {
    if (GetStepMode() != *request.mode) {
      _realTimeTracking.ResetTime();
    }
    _stepMode.store(*request.mode);
    _singleStepRequested = false;
  }

  if (request.stepForward) {
    _realTimeTracking.ResetTime();
    _stepMode.store(StepMode::Pause);
    _singleStepRequested = true;
  }

  if (request.restoreInitialState) {
    if (_initialState.captureData.empty()) {
      reply.error = _initialState.error.empty()
          ? "Initial scene state is not available"
          : Format(
                "Failed to restore state because the initial capture failed. Reason: %s",
                _initialState.error.c_str());
    } else {
      MOCHI_ASSERT_VERBOSE(
          _initialState.error.empty(), "If there was an error, then captureData should be empty.");
      Error error;
      scene->RestoreStateFromBytes(_initialState.captureData, error);
      if (error.IsOK()) {
        // This is a discontinuity in the timeline
        _realTimeTracking.ResetTime();

        // Increment the step counter because the state of the scene has changed
        // and should thus be synced to client.
        ++_stepCounter;

        // Re-process queries after RestoreStateFromBytes.
        Error queryError;
        scene->UpdateQueries(queryError);
        if (!queryError.IsOK()) {
          MOCHI_LOG_WARNING_ONCE(
              "[SceneDebugger] UpdateQueries: %s", queryError.ToString().c_str());
        }
      } else {
        reply.error =
            Format("Failed to restore initial scene state. Reason: %s", error.GetDescription());
      }
    }
  }

  if (request.sendReply) {
    SendToClient(reply);
  }
}

void SceneDebuggerImpl::OnSceneSyncRequest(SceneImpl* scene, protocol::SceneSyncRequest&& request) {
  MOCHI_ASSERT_VERBOSE(request.scene == _sceneHandle, "Received by the wrong scene");
  bool const wasSyncingMeshes = _sync && _sync->syncActors && _sync->syncMeshes;
  bool const wasSyncingPerformance = _sync && _sync->syncPerformance;

  // If enableAutoSync.has_value() then enable/disable auto-syncing. Otherwise this is a one-time
  // sync request that must not affect the active syncing state.
  if (request.enableAutoSync.has_value()) {
    if (*request.enableAutoSync) {
      bool const meshSourceChanged = _sync && _sync->useVisualMesh != request.useVisualMesh;
      bool const meshSyncStarting = request.syncMeshes && (!_sync || !_sync->syncMeshes);
      if (meshSourceChanged || meshSyncStarting) {
        // Existing client meshes remain visible until the immediate reply starts this version.
        ++_meshVersionCounter;
        UpdateMeshQueries(scene, /*meshesEnabled=*/false, /*useVisualMesh=*/false);
      }
      _sync = SyncState{
          .interval = request.syncInterval,
          .syncActors = request.syncActors,
          .syncDebugDraw = request.syncDebugDraw,
          .syncMeshes = request.syncMeshes,
          .useVisualMesh = request.useVisualMesh,
          .syncPerformance = request.syncPerformance,
          .lastSyncTime = _clock(),
          .lastSyncStep = _stepCounter};

    } else {
      _sync = std::nullopt;
    }
  } else {
    // Mesh and performance data are only supported for periodic syncing.
    request.syncMeshes = false;
    request.syncPerformance = false;
  }

  // Samples gathered under a previous setting are stale.
  bool const isSyncingPerformance = _sync && _sync->syncPerformance;
  if (wasSyncingPerformance != isSyncingPerformance) {
    _pendingProfileData.clear();
  }
  request.syncPerformance = request.syncPerformance && isSyncingPerformance;

  bool const isSyncingMeshes = _sync && _sync->syncActors && _sync->syncMeshes;
  bool const preserveMeshQueries = wasSyncingMeshes && isSyncingMeshes;
  if (!preserveMeshQueries) {
    UpdateMeshQueries(scene, /*meshesEnabled=*/false, /*useVisualMesh=*/false);
  }
  request.syncMeshes = request.syncMeshes && isSyncingMeshes;

  // Send a reply immediately
  if (request.sendReply) {
    SendSyncReply(
        scene,
        request.syncActors,
        request.syncDebugDraw,
        request.syncMeshes,
        request.useVisualMesh,
        request.syncPerformance,
        request.requestId);
  }
}

void SceneDebuggerImpl::OnSleepParamsRequest(
    SceneImpl* scene,
    protocol::SleepParamsRequest&& request) {
  MOCHI_ASSERT_VERBOSE(request.scene == _sceneHandle, "Received by the wrong scene");
  protocol::SleepParamsReply reply(request);

  if (request.params.has_value()) {
    Error error;
    scene->SetSleepParams(*request.params, error);
    if (!error.IsOK()) {
      reply.error = Format("Failed to set sleep params. Reason: %s", error.GetDescription());
    }
  }
  reply.params = scene->GetSleepParams();

  if (request.sendReply) {
    SendToClient(reply);
  }
}

void SceneDebuggerImpl::SendSyncReply(
    SceneImpl* scene,
    bool syncActors,
    bool syncDebugDraw,
    bool syncMeshes,
    bool useVisualMesh,
    bool syncPerformance,
    uint64_t requestId) {
  protocol::SceneSyncReply reply;
  reply.requestId = requestId;
  reply.scene = _sceneHandle;

  // Meshes ride on actor syncing. Ensure the required per-actor queries exist before gathering.
  // Only the periodic auto-sync path gathers meshes; one-off requests force syncMeshes off.
  bool const gatherMeshes = syncActors && syncMeshes;
  if (gatherMeshes) {
    UpdateMeshQueries(scene, true, useVisualMesh);
  }

  if (syncActors) {
    reply.actors.emplace();
    if (gatherMeshes) {
      reply.meshData.emplace();
      reply.actorMeshRanges.emplace();
      reply.meshVersionCounter = _meshVersionCounter;
    }
    scene->ForEachActor([&](Actor const* actor) {
      protocol::ActorSyncData a;
      a.type = actor->GetType();
      a.handle = actor->GetHandle();
      a.name = actor->GetName();
      auto const& rt = actor->GetRootTransform();
      a.position = StaticCast<Float3>(rt.GetTranslation());
      a.rotation = StaticCast<Float4>(rt.GetRotation().ToReal4());
      a.isStatic = actor->IsStatic();
      // Nested link/soft actors report their parent articulated actor; standalone actors have none.
      if (actor->IsNestedLinkActor() || actor->IsNestedSoftActor()) {
        a.parent = actor->GetArticulatedActor(ErrorAssert{});
      }
      if (gatherMeshes) {
        // UpdateMeshQueries(scene, true) ran above, so every live actor is tracked.
        auto const it = _actorInfo.find(a.handle);
        MOCHI_ASSERT_VERBOSE(it != _actorInfo.end(), "Actor missing from mesh tracking map");
        protocol::MeshSyncRanges ranges;
        AppendActorMesh(*actor, it->second, *reply.meshData, ranges);
        reply.actorMeshRanges->emplace_back(ranges);
      }
      reply.actors->emplace_back(std::move(a));
    });
  }

  if (syncDebugDraw) {
    reply.debugDraw.emplace();
    auto const& drawData = scene->GetDebugDraw().GatherData();
    auto const& inLines = drawData.lineVertices;
    auto const& inSpheres = drawData.spheres;
    auto& outLines = reply.debugDraw->lineVertices;
    auto& outSpheres = reply.debugDraw->spheres;

    // Copy data and cast to float (in case of double precision)
    outLines.positions.resize_noinit(Flatten(inLines.positions).size());
    StaticCast<float>(Flatten(inLines.positions), MakeSpan(outLines.positions));
    outLines.colors = Flatten(inLines.colors);

    outSpheres.positions.resize_noinit(Flatten(inSpheres.positions).size());
    outSpheres.radii.resize_noinit(inSpheres.radii.size());
    StaticCast<float>(Flatten(inSpheres.positions), MakeSpan(outSpheres.positions));
    StaticCast<float>(MakeConstSpan(inSpheres.radii), MakeSpan(outSpheres.radii));
    outSpheres.colors = Flatten(inSpheres.colors);
  }

  if (syncPerformance) {
    reply.performance.emplace(_pendingProfileData.begin(), _pendingProfileData.end());
    _pendingProfileData.clear();
  }

  // Send the message unless Shutdown() was called.
  SendToClient(reply);
}

void SceneDebuggerImpl::SendSyncIfRequested(SceneImpl* scene) {
  if (!_sync) {
    return; // Periodic syncing not enabled
  }

  // Transmit at most once per simulation step.
  if (_stepCounter == _sync->lastSyncStep) {
    return;
  }

  // Limit transmission rate using _sync->interval (seconds)
  double const now = _clock();
  bool const intervalElapsed =
      (_sync->interval == 0.0f) || (now - _sync->lastSyncTime >= _sync->interval);
  if (!intervalElapsed) {
    return;
  }

  SendSyncReply(
      scene,
      _sync->syncActors,
      _sync->syncDebugDraw,
      _sync->syncMeshes,
      _sync->useVisualMesh,
      _sync->syncPerformance,
      /*requestId*/ 0);
  _sync->lastSyncTime = now;
  _sync->lastSyncStep = _stepCounter;
}

void SceneDebuggerImpl::UpdateMeshQueries(
    SceneImpl* scene,
    bool meshesEnabled,
    bool useVisualMesh) {
  if (!meshesEnabled) {
    // Cancel all queries and forget which meshes were sent, so a later re-enable re-sends them.
    for (auto const& [handle, info] : _actorInfo) {
      if (info.query.IsValid()) {
        if (Actor* actor = scene->GetActor(handle)) {
          actor->CancelQuery(info.query);
        }
      }
    }
    _actorInfo.clear();
    return;
  }

  // Mark phase: clear the transient mark on every tracked actor.
  for (auto& [handle, info] : _actorInfo) {
    info.marked = false;
  }

  // Reconcile phase: mark live actors, inserting tracking state (and registering the positions
  // query for non-rigid actors) for any that are newly seen.
  scene->ForEachActor([&](Actor* actor) {
    ActorHandle const handle = actor->GetHandle();
    auto const it = _actorInfo.find(handle);
    if (it != _actorInfo.end()) {
      it->second.marked = true;
      return;
    }

    ActorInfo info;
    info.marked = true;
    // Non-rigid actors need a live node-positions query; rigid meshes are static.
    if (actor->GetType() == ActorType::Rigid) {
      MeshDataView const visualMesh = actor->GetVisualMesh();
      if (useVisualMesh && !visualMesh.coordinates.empty() && !visualMesh.connectivity.empty()) {
        info.meshSource = MeshSource::Visual;
      }
    } else {
      Error error;
      QueryHandle query;
      // Attempt to use the visual mesh, if requested.
      if (useVisualMesh) {
        query = actor->RegisterQueryAndCompute(QueryType::VisualNodePositions, error);
        if (error.IsOK()) {
          info.meshSource = MeshSource::Visual;
        }
      }
      // Then the contact skin
      if (info.meshSource == MeshSource::Surface) {
        error = {}; // Clear previous error
        query = actor->RegisterQueryAndCompute(QueryType::ContactSkinNodePositions, error);
        if (error.IsOK()) {
          info.meshSource = MeshSource::ContactSkin;
        }
      }
      // Fall back on the normal surface mesh
      if (info.meshSource == MeshSource::Surface) {
        error = {}; // Clear previous error
        query = actor->RegisterQueryAndCompute(QueryType::SurfaceNodePositions, error);
      }
      if (error.IsOK()) {
        info.query = query;
      }
    }
    _actorInfo.emplace(handle, info);
  });

  // Sweep phase: drop tracking state for actors that no longer exist, cancelling their queries.
  for (auto it = _actorInfo.begin(); it != _actorInfo.end();) {
    if (!it->second.marked) {
      if (it->second.query.IsValid()) {
        if (Actor* actor = scene->GetActor(it->first)) {
          actor->CancelQuery(it->second.query);
        }
      }
      it = _actorInfo.erase(it);
    } else {
      ++it;
    }
  }
}

void SceneDebuggerImpl::AppendActorMesh(
    Actor const& actor,
    ActorInfo& info,
    protocol::MeshSyncData& meshData,
    protocol::MeshSyncRanges& outRanges) {
  outRanges = {};
  bool const isRigid = (actor.GetType() == ActorType::Rigid);

  // Rigid meshes are static: send once, then rely on the client-side cache.
  if (isRigid && info.meshSent) {
    return;
  }

  MeshDataView const mesh = info.meshSource == MeshSource::Visual ? actor.GetVisualMesh()
      : info.meshSource == MeshSource::ContactSkin                ? actor.GetContactSkinMesh()
                                                                  : actor.GetSurfaceMesh();
  Span<int const> const connectivity = mesh.connectivity;
  if (connectivity.empty()) {
    return; // This actor has no surface mesh.
  }

  // Rigid actors use the baked surface mesh; non-rigid actors use the live positions query.
  Span<real const> coordinates;
  if (isRigid) {
    coordinates = mesh.coordinates;
  } else {
    Error error;
    coordinates = info.meshSource == MeshSource::Visual
        ? actor.GetVisualMeshNodePositionsLocal(error)
        : info.meshSource == MeshSource::ContactSkin
        ? actor.GetContactSkinMeshNodePositionsLocal(error)
        : actor.GetSurfaceMeshNodePositionsLocal(error);
    if (!error.IsOK() || coordinates.empty()) {
      return; // Positions query unavailable (e.g. a one-off request without registration).
    }
  }

  // Append coordinates (cast real->float), recording the range for this actor.
  int const coordBegin = isize(meshData.coordinates);
  meshData.coordinates.resize_noinit(static_cast<size_t>(coordBegin) + coordinates.size());
  StaticCast<float>(coordinates, MakeSpan(meshData.coordinates).subspan(coordBegin));
  outRanges.coordinatesRange = Int2{coordBegin, isize(meshData.coordinates)};

  // Connectivity never changes, so send it only on the first sync for this actor. Values stay
  // 0-based relative to this actor's own coordinate block.
  if (!info.meshSent) {
    int const connBegin = isize(meshData.connectivity);
    meshData.connectivity.append(connectivity.begin(), connectivity.end());
    outRanges.connectivityRange = Int2{connBegin, isize(meshData.connectivity)};
  }

  info.meshSent = true;
}

void SceneDebuggerImpl::PreShutdownAsync() {
  // WARNING: Called on the DebugServer's thread. All operations must be thread-safe.

  // Clear the server pointer first. This releases a paused or throttled OnPreStep even when
  // DebugServer::Stop is running before MessageServer has removed the client.
  _server.Mutate([](auto& server) { server = nullptr; });

  // Wait for any in-progress scene-thread work to finish.
  {
    std::lock_guard lock{_sceneThreadMutex};
  }

  // Cancel any pending messages
  _inbox.Mutate([](auto& inbox) { inbox.clear(); });
}

void SceneDebuggerImpl::ShutdownOnSceneThread(SceneImpl* scene) {
  std::lock_guard lock{_sceneThreadMutex};

  // No more syncing
  _sync = std::nullopt;
  _pendingProfileData.clear();

  // The debugger owns debug draw while connected. Return the scene to a fully disabled state.
  auto& debugDraw = scene->GetDebugDraw();
  debugDraw.Enable(false);
  for (int i = 0; i < debugDraw.GetNumFeatures(); ++i) {
    debugDraw.EnableFeature(i, false);
  }

  // Release the debugger's scene settings. Single-threaded mode is step-scoped, so it needs no
  // cleanup.
  island::SetForceSingleIslandByDebugger(scene->GetRegistry(), false);

  // Cancel all queries
  UpdateMeshQueries(scene, /*meshesEnabled*/ false, /*useVisualMesh*/ false);

  if (_profilerEnabled) {
    profiler::Disable(scene->GetRegistry());
    _profilerEnabled = false;
  }
}

std::shared_ptr<SceneDebugger>
dbg::CreateSceneDebugger(net::ClientId client, SceneImpl* scene, net::MessageServer& server) {
  return std::make_shared<SceneDebuggerImpl>(client, scene, server);
}
