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

#pragma once

#include <mochi_core/net/message.h>
#include <mochi_core/utils/coordinate_space.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/dynamic_string.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/reflection.h>
#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/mochi_physics_experimental.h>

#include <cstdint>
#include <optional>

namespace mochi::dbg {
static constexpr uint16_t kDiscoveryPort = 7330;
static constexpr uint16_t kSingletonPort = 7331;
static constexpr uint16_t kDefaultDebugServerPort = 7333;

/// @brief Playback mode for scene stepping, requested via @ref SceneStepRequest.
enum class StepMode {
  Pause, ///< Halt stepping until another mode is requested.
  Play, ///< Run simulation, but throttle playback to not exceed real time.
  FastForward, ///< Run unthrottled, as fast as possible.
};

} // namespace mochi::dbg

// NOTE: This could move somewhere else if it is needed in multiple places.
MOCHI_ENUM_BEGIN(mochi::LogChannel)
MOCHI_ENUM_ITEM(Verbose)
MOCHI_ENUM_ITEM(Info)
MOCHI_ENUM_ITEM(Warning)
MOCHI_ENUM_ITEM(Error)
MOCHI_ENUM_COUNT(Count)
MOCHI_ENUM_END()

MOCHI_ENUM_BEGIN(mochi::dbg::StepMode)
MOCHI_ENUM_ITEM(Pause)
MOCHI_ENUM_ITEM(Play)
MOCHI_ENUM_ITEM(FastForward)
MOCHI_ENUM_END()

namespace mochi::dbg::protocol {

// Forwards
struct DebugDrawReply;
struct PingReply;
struct SceneSettingsReply;
struct SceneStepReply;
struct SceneSyncReply;
struct SleepParamsReply;

//------------------------------------------------------------------------------
// Base Classes
//------------------------------------------------------------------------------

/// @brief [S<--C] Base class for a @ref net::RequestMessage that pertains to a single scene.
struct SceneRequest : net::RequestMessage {
  SceneHandle scene; // Scene that the request pertains to.
  bool sendReply = true; // Non-serialized field. If false, no reply should be sent.

  MOCHI_STRUCT_BEGIN(mochi::dbg::protocol::SceneRequest)
  MOCHI_BASE_CLASS(net::RequestMessage)
  MOCHI_FIELD(scene)
  MOCHI_STRUCT_END()
};

/// @brief [S-->C] Base class for a @ref net::ReplyMessage that pertains to a single scene.
struct SceneReply : net::ReplyMessage {
  SceneReply() = default;
  explicit SceneReply(SceneRequest const& request)
      : net::ReplyMessage(request), scene(request.scene) {}

  SceneHandle scene; // Scene that the reply pertains to.
  DynamicString error; // Empty means success

  MOCHI_STRUCT_BEGIN(mochi::dbg::protocol::SceneReply)
  MOCHI_BASE_CLASS(net::ReplyMessage)
  MOCHI_FIELD(scene)
  MOCHI_FIELD(error)
  MOCHI_STRUCT_END()
};

//------------------------------------------------------------------------------
// Payload data used within messages
//------------------------------------------------------------------------------

// Like mochi::DebugDrawLineVertices but with owning DynamicArrays
struct DbgLineVertices {
  DynamicArray<float> positions; // 3 per vert
  DynamicArray<uint8_t> colors; // 4 per vert (RGBA)

  bool operator==(DbgLineVertices const& rhs) const = default;

  MOCHI_STRUCT_BEGIN(mochi::dbg::protocol::DbgLineVertices)
  MOCHI_FIELD(positions)
  MOCHI_FIELD(colors)
  MOCHI_STRUCT_END()
};

// Like mochi::DebugDrawSpheres but with owning DynamicArrays
struct DbgSpheres {
  DynamicArray<float> positions; // 3 per sphere
  DynamicArray<float> radii; // 1 per sphere
  DynamicArray<uint8_t> colors; // 4 per sphere (RGBA)

  bool operator==(DbgSpheres const& rhs) const = default;

  MOCHI_STRUCT_BEGIN(mochi::dbg::protocol::DbgSpheres)
  MOCHI_FIELD(positions)
  MOCHI_FIELD(radii)
  MOCHI_FIELD(colors)
  MOCHI_STRUCT_END()
};

struct DbgDrawData {
  DbgLineVertices lineVertices;
  DbgSpheres spheres;

  bool operator==(DbgDrawData const& rhs) const = default;

  MOCHI_STRUCT_BEGIN(mochi::dbg::protocol::DbgDrawData)
  MOCHI_FIELD(lineVertices)
  MOCHI_FIELD(spheres)
  MOCHI_STRUCT_END()
};

/// @brief One entry in the debug draw feature catalog.
struct DbgDrawFeature {
  DynamicString name;
  DynamicString description;
  bool enabled = false;

  MOCHI_STRUCT_BEGIN(mochi::dbg::protocol::DbgDrawFeature)
  MOCHI_FIELD(name)
  MOCHI_FIELD(description)
  MOCHI_FIELD(enabled)
  MOCHI_STRUCT_END()
};

struct DbgDrawFeatures {
  bool masterEnabled = false;
  DynamicArray<DbgDrawFeature> features;

  MOCHI_STRUCT_BEGIN(mochi::dbg::protocol::DbgDrawFeatures)
  MOCHI_FIELD(masterEnabled)
  MOCHI_FIELD(features)
  MOCHI_STRUCT_END()
};

struct MeshSyncRanges {
  // Offsets into MeshSyncData.
  Int2 coordinatesRange = {}; // [begin, end)
  Int2 connectivityRange = {}; // [begin, end)

  bool operator==(MeshSyncRanges const& rhs) const = default;

  MOCHI_STRUCT_BEGIN(mochi::dbg::protocol::MeshSyncRanges)
  MOCHI_FIELD(coordinatesRange)
  MOCHI_FIELD(connectivityRange)
  MOCHI_STRUCT_END()
};

struct MeshSyncData {
  // NOTE: Multiple meshes may be concatenated within these arrays
  DynamicArray<float> coordinates; // 3 per vertex
  DynamicArray<int> connectivity; // 3 per triangle

  bool operator==(MeshSyncData const& rhs) const = default;

  MOCHI_STRUCT_BEGIN(mochi::dbg::protocol::MeshSyncData)
  MOCHI_FIELD(coordinates)
  MOCHI_FIELD(connectivity)
  MOCHI_STRUCT_END()
};

struct ActorSyncData {
  ActorType type = {};
  ActorHandle handle = {};
  ActorHandle parent = {}; // Parent articulated actor for nested link/soft actors; invalid if none.
  DynamicString name = {};
  Float3 position; // XYZ
  Float4 rotation; // Quaternion XYZW
  bool isStatic = false;

  bool operator==(ActorSyncData const& rhs) const = default;

  MOCHI_STRUCT_BEGIN(mochi::dbg::protocol::ActorSyncData)
  MOCHI_FIELD(type)
  MOCHI_FIELD(handle)
  MOCHI_FIELD(parent)
  MOCHI_FIELD(name)
  MOCHI_FIELD(position)
  MOCHI_FIELD(rotation)
  MOCHI_FIELD(isStatic)
  MOCHI_STRUCT_END()
};

struct SceneList {
  DynamicArray<SceneHandle> handles;
  DynamicArray<DynamicString> names;

  MOCHI_STRUCT_BEGIN(mochi::dbg::protocol::SceneList)
  MOCHI_FIELD(handles)
  MOCHI_FIELD(names)
  MOCHI_STRUCT_END()
};

/// @brief Profiling data for one simulation step. "Active" counts only include awake islands.
struct StepProfileData {
  uint64_t sequence = 0; ///< Monotonically increasing sequence within the scene debugger
  float simTimeStepSec = 0.0f; ///< Simulated time step [s]
  float sceneStepSec = 0.0f; ///< Total simulation duration, excluding callbacks [s]
  // Exclusive category durations [s]. Available only when numWorkerThreads is zero.
  float sceneContactSec = 0.0f;
  float sceneAssemblySec = 0.0f;
  float sceneSolveSec = 0.0f;
  float sceneQueriesSec = 0.0f;
  int32_t activeActors = 0; ///< Top-level actors (direct island members)
  int32_t activeConstraints = 0;
  int32_t activeContacts = 0; ///< Contact points, both sync and async
  int32_t activeDofs = 0;
  int32_t activeIslands = 0;
  ConvergenceStatus convergenceStatus = ConvergenceStatus::None; ///< Worst over active islands
  /// Effective task scheduler worker threads during the step. Zero if single-threaded.
  int32_t numWorkerThreads = 0;

  bool operator==(StepProfileData const& rhs) const = default;

  MOCHI_STRUCT_BEGIN(mochi::dbg::protocol::StepProfileData)
  MOCHI_FIELD(sequence)
  MOCHI_FIELD(simTimeStepSec)
  MOCHI_FIELD(sceneStepSec)
  MOCHI_FIELD(sceneContactSec)
  MOCHI_FIELD(sceneAssemblySec)
  MOCHI_FIELD(sceneSolveSec)
  MOCHI_FIELD(sceneQueriesSec)
  MOCHI_FIELD(activeActors)
  MOCHI_FIELD(activeConstraints)
  MOCHI_FIELD(activeContacts)
  MOCHI_FIELD(activeDofs)
  MOCHI_FIELD(activeIslands)
  MOCHI_FIELD(convergenceStatus)
  MOCHI_FIELD(numWorkerThreads)
  MOCHI_STRUCT_END()
};

//------------------------------------------------------------------------------
// Message Types
//------------------------------------------------------------------------------

/// @brief [S-->C] A single message from Mochi logging.
struct LogMessage : net::Message {
  LogMessage() = default;
  LogMessage(LogChannel channel_, DynamicString message_, DynamicString file_, int32_t line_)
      : channel(channel_), message(std::move(message_)), file(std::move(file_)), line(line_) {}

  LogChannel channel = LogChannel::Info;
  DynamicString message;
  DynamicString file;
  int32_t line = 0;

  MOCHI_STRUCT_BEGIN(mochi::dbg::protocol::LogMessage)
  MOCHI_BASE_CLASS(net::Message)
  MOCHI_FIELD(channel)
  MOCHI_FIELD(message)
  MOCHI_FIELD(file)
  MOCHI_FIELD(line)
  MOCHI_STRUCT_END()
};

/// @brief [S<--C] Health-check message. Recipient responds with @ref PingReply.
struct PingRequest : net::RequestMessage {
  using Reply = PingReply;
  PingRequest();
  uint64_t sendTimeNs = 0; // Send time [ns]

  MOCHI_STRUCT_BEGIN(mochi::dbg::protocol::PingRequest)
  MOCHI_BASE_CLASS(net::RequestMessage)
  MOCHI_FIELD(sendTimeNs)
  MOCHI_STRUCT_END()
};

/// @brief [S-->C] Reply to @ref PingRequest.
struct PingReply : net::ReplyMessage {
  PingReply();
  explicit PingReply(PingRequest const& request);

  uint64_t sendTimeNs = 0; // Returns the value from @ref PingRequest [ns]
  uint64_t recvTimeNs = 0; // Set by constructor. Not serialized.

  MOCHI_STRUCT_BEGIN(mochi::dbg::protocol::PingReply)
  MOCHI_BASE_CLASS(net::ReplyMessage)
  MOCHI_FIELD(sendTimeNs)
  MOCHI_STRUCT_END()
};

/// @brief [S-->C] Broadcast when a scene is added to or removed from the context.
///
/// @warning Receiving this notification does not grant ownership of the referenced scene or extend
/// its lifetime.
/// @warning For an in-process connection, the receive callback may run on the thread calling
/// @ref Context::CreateScene or @ref Context::DestroyScene before that call returns. For an add
/// notification, do not destroy the announced scene or pass it to an ownership-taking API until
/// the originating @ref Context::CreateScene call returns, and only then if the caller controls the
/// scene's lifetime. After a removal notification, do not resolve or access the announced scene.
struct SceneAddRemove : net::Message {
  SceneHandle scene;
  DynamicString name;
  bool wasAdded = false; ///< False if the scene was removed

  MOCHI_STRUCT_BEGIN(mochi::dbg::protocol::SceneAddRemove)
  MOCHI_BASE_CLASS(net::Message)
  MOCHI_FIELD(scene)
  MOCHI_FIELD(name)
  MOCHI_FIELD(wasAdded)
  MOCHI_STRUCT_END()
};

/// @brief [S<--C] Request the scene to apply the client's complete debug draw state.
struct DebugDrawRequest : SceneRequest {
  using Reply = DebugDrawReply;

  bool masterEnable = false;
  DynamicArray<bool> featureEnable;

  MOCHI_STRUCT_BEGIN(mochi::dbg::protocol::DebugDrawRequest)
  MOCHI_BASE_CLASS(SceneRequest)
  MOCHI_FIELD(masterEnable)
  MOCHI_FIELD(featureEnable)
  MOCHI_STRUCT_END()
};

/// @brief [S-->C] Reply to @ref DebugDrawRequest.
struct DebugDrawReply : SceneReply {
  using SceneReply::SceneReply;
  MOCHI_STRUCT_WITH_BASE(mochi::dbg::protocol::DebugDrawReply, SceneReply)
};

/// @brief [S<--C] Request the scene to change its play/pause/step behavior.
struct SceneStepRequest : SceneRequest {
  using Reply = SceneStepReply;

  /// Optionally modify the playback mode.
  std::optional<StepMode> mode;

  /// Optionally advance the simulation forward by one step and pause the scene (overrides the
  /// specified @ref mode to @ref StepMode::Pause).
  bool stepForward = false;

  /// Optionally restore the scene to the debugger's initial state snapshot.
  /// If combined with stepForward, then one step will be taken after restore.
  bool restoreInitialState = false;

  MOCHI_STRUCT_BEGIN(mochi::dbg::protocol::SceneStepRequest)
  MOCHI_BASE_CLASS(SceneRequest)
  MOCHI_FIELD(mode)
  MOCHI_FIELD(stepForward)
  MOCHI_FIELD(restoreInitialState)
  MOCHI_STRUCT_END()
};

/// @brief [S-->C] Reply to @ref SceneStepRequest.
struct SceneStepReply : SceneReply {
  using SceneReply::SceneReply;
  MOCHI_STRUCT_WITH_BASE(mochi::dbg::protocol::SceneStepReply, SceneReply)
};

/// @brief [S<--C] Request the scene to transmit a @ref SceneSyncReply once, or periodically.
struct SceneSyncRequest : SceneRequest {
  using Reply = SceneSyncReply;

  // If a value is specified, then this message will enable or disable automatic syncing of scene
  // state via repeated SceneSyncReply messages. Else, this message will be treated as a one-off
  // request, resulting in a single reply message and no change of automatic syncing behavior.
  std::optional<bool> enableAutoSync;

  // Minimum time between automatic replies [seconds]. Used to limit the data rate.
  // Only affects auto syncing behavior if enableAutoSync.has_value().
  float syncInterval = 0.0f;

  // Should actors be included in the reply message(s)? Always determines the content of the
  // immediate one-off reply, as well as periodic replies when auto-syncing is enabled.
  bool syncActors = false;

  // Should debug draw data be included in the reply message(s)? Always determines the content of
  // the immediate one-off reply, as well as periodic replies when auto-syncing is enabled.
  bool syncDebugDraw = false;

  // Should actor meshes be included in the reply message(s)? Requires syncActors. Only
  // supported when auto-syncing is enabled (not for one-off requests). See useVisualMesh for
  // which mesh is sent.
  bool syncMeshes = false;

  // Prefer actor visual meshes when available. Otherwise, synchronize simulation surface meshes.
  // Requires syncMeshes. Changing this setting starts a new mesh version.
  bool useVisualMesh = true;

  // Should per-step profiling data be included in periodic replies? Samples accumulate on the
  // server between replies. Only affects auto syncing behavior if enableAutoSync.has_value().
  bool syncPerformance = false;

  MOCHI_STRUCT_BEGIN(mochi::dbg::protocol::SceneSyncRequest)
  MOCHI_BASE_CLASS(SceneRequest)
  MOCHI_FIELD(enableAutoSync)
  MOCHI_FIELD(syncInterval)
  MOCHI_FIELD(syncActors)
  MOCHI_FIELD(syncDebugDraw)
  MOCHI_FIELD(syncMeshes)
  MOCHI_FIELD(useVisualMesh)
  MOCHI_FIELD(syncPerformance)
  MOCHI_STRUCT_END()
};

/// @brief [S-->C] Replies to @ref SceneSyncRequest with information about the state of the scene
struct SceneSyncReply : SceneReply {
  using SceneReply::SceneReply;

  std::optional<DynamicArray<ActorSyncData>> actors; // Sent if syncActors requested
  std::optional<MeshSyncData> meshData; // Sent if syncMeshes requested. Concatenated mesh buffers.
  // Per-actor ranges into meshData. When present, 1:1 with actors: actorMeshRanges[i] locates the
  // selected mesh for actors[i]. Sent only when syncMeshes is requested.
  std::optional<DynamicArray<MeshSyncRanges>> actorMeshRanges;
  // Identifies the version of the mesh data in this reply. Incremented when mesh syncing starts,
  // and when meshes are re-sent after a settings change, so a client observing a new value must
  // discard everything it cached for the previous version. Live values start at 1; 0 is reserved
  // to mean "no mesh".
  uint64_t meshVersionCounter = 0;
  std::optional<DbgDrawData> debugDraw; // Sent if syncDebugDraw requested
  // Steps completed since the previous reply, oldest first. Sent if syncPerformance is enabled.
  std::optional<DynamicArray<StepProfileData>> performance;

  MOCHI_STRUCT_BEGIN(mochi::dbg::protocol::SceneSyncReply)
  MOCHI_BASE_CLASS(SceneReply)
  MOCHI_FIELD(actors)
  MOCHI_FIELD(meshData)
  MOCHI_FIELD(actorMeshRanges)
  MOCHI_FIELD(meshVersionCounter)
  MOCHI_FIELD(debugDraw)
  MOCHI_FIELD(performance)
  MOCHI_STRUCT_END()
};

/// @brief [S<--C] Query, and optionally modify, the scene's @ref experimental::SleepParams.
struct SleepParamsRequest : SceneRequest {
  using Reply = SleepParamsReply;

  std::optional<experimental::SleepParams> params; // If nullopt, only query the current value.

  MOCHI_STRUCT_BEGIN(mochi::dbg::protocol::SleepParamsRequest)
  MOCHI_BASE_CLASS(SceneRequest)
  MOCHI_FIELD(params)
  MOCHI_STRUCT_END()
};

/// @brief [S-->C] Reply to @ref SleepParamsRequest.
struct SleepParamsReply : SceneReply {
  using SceneReply::SceneReply;

  experimental::SleepParams params; // The scene's value after applying the request.

  MOCHI_STRUCT_BEGIN(mochi::dbg::protocol::SleepParamsReply)
  MOCHI_BASE_CLASS(SceneReply)
  MOCHI_FIELD(params)
  MOCHI_STRUCT_END()
};

/// @brief [S<--C] Query, and optionally modify, the debugger's overrides of scene settings.
///
/// @details Each field is applied only if it has a value, so an empty request is a query. The
/// overrides are owned by the debugger and are released when it detaches from the scene.
struct SceneSettingsRequest : SceneRequest {
  using Reply = SceneSettingsReply;

  /// Step the scene on a single thread, regardless of the application's threading mode.
  std::optional<bool> forceSingleThreaded;

  /// Merge all islands into one, in addition to the application's @ref Scene::SetForceSingleIsland.
  std::optional<bool> forceSingleIsland;

  MOCHI_STRUCT_BEGIN(mochi::dbg::protocol::SceneSettingsRequest)
  MOCHI_BASE_CLASS(SceneRequest)
  MOCHI_FIELD(forceSingleThreaded)
  MOCHI_FIELD(forceSingleIsland)
  MOCHI_STRUCT_END()
};

/// @brief [S-->C] Reply to @ref SceneSettingsRequest with the debugger's values after the request
/// is applied.
struct SceneSettingsReply : SceneReply {
  using SceneReply::SceneReply;

  bool forceSingleThreaded = false;
  bool forceSingleIsland = false;

  MOCHI_STRUCT_BEGIN(mochi::dbg::protocol::SceneSettingsReply)
  MOCHI_BASE_CLASS(SceneReply)
  MOCHI_FIELD(forceSingleThreaded)
  MOCHI_FIELD(forceSingleIsland)
  MOCHI_STRUCT_END()
};

/// @brief [S-->C] First message sent from the server to the client. The client does not consider
/// themselves to be fully connected until this arrives.
struct WelcomeMessage : net::Message {
  DbgDrawFeatures debugDraw; ///< Debug draw feature catalog; enable fields are initially false.
  SceneList scenes; ///< Information about all scenes that already exist.
  CoordinateSpace coordinateSpace = {}; ///< The server's coordinate space convention.

  MOCHI_STRUCT_BEGIN(mochi::dbg::protocol::WelcomeMessage)
  MOCHI_BASE_CLASS(net::Message)
  MOCHI_FIELD(debugDraw)
  MOCHI_FIELD(scenes)
  MOCHI_FIELD(coordinateSpace)
  MOCHI_STRUCT_END()
};

} // namespace mochi::dbg::protocol
