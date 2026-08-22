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

#include <mochi_core/geometry/mesh_data.h>
#include <mochi_core/net/message.h>
#include <mochi_core/net/message_client.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/guarded.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/span.h>
#include <mochi_physics/cpp_api/mochi_handle.h>
#include <mochi_physics/dbg/protocol.h>

#include <cinttypes>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

// Forwards:
namespace mochi::net {
class MessageServer;
} // namespace mochi::net

namespace mochi::dbg {

/// @brief An entry in the client's cached scene list.
struct SceneInfo {
  SceneHandle handle;
  std::string name;
};

/// @brief Parameters to control automatic synchronization of scene state
struct SceneSyncParams {
  bool enabled = false; ///< Enable/disable automatic synchronization of scene state
  float syncInterval =
      0.0f; ///< Minimum time between transmissions [seconds]. Used to throttle data rate.
  bool syncActors = false; ///< Should the actors be included? Ignored if not @ref enabled.
  bool syncDebugDraw = false; ///< Should debug draw data be included? Ignored if not @ref enabled.
  bool syncMeshes = false; ///< Should actor meshes be included? Requires @ref syncActors.
                           ///< Ignored if not @ref enabled.
  bool useVisualMesh = true; ///< Prefer actor visual meshes when available. Otherwise, synchronize
                             ///< simulation surface meshes. Requires @ref syncMeshes.

  bool operator==(SceneSyncParams const& rhs) const = default;
};

/// @brief Customizable settings for DebugClient
struct DebugClientSettings {
  /// @brief Parameters to control automatic synchronization of scene state
  SceneSyncParams sync;

  bool operator==(DebugClientSettings const& rhs) const = default;
};

using MeshSyncData = protocol::MeshSyncData;

/// @brief A non-owning view into a mesh's data, in actor-local coordinates.
struct MeshSyncDataView {
  Span<float const> coordinates; // 3 per vertex
  Span<int const> connectivity; // 3 per triangle
  // Server-assigned version of this mesh, shared by every mesh in a sync. It changes when the
  // server starts syncing meshes or switches mesh source, meaning previously cached data for
  // this actor is obsolete. Only comparable within a scene selection; compare for inequality
  // rather than ordering. Zero means no mesh.
  uint64_t versionCounter = 0;

  bool operator==(MeshSyncDataView const& rhs) const = default;
};

/// @brief Scene data synced from the server.
struct SceneSyncData {
  /// Which scene this data pertains to
  SceneHandle scene;

  /// Incremented each time the cached data changes.
  uint64_t counter = 0;

  /// Information about each actor or empty. Requires @ref SceneSyncParams::syncActors.
  DynamicArray<protocol::ActorSyncData> actors;

  /// Meshes to render for each actor (1:1) or empty. Requires @ref SceneSyncParams::syncMeshes.
  ///
  /// @warning These are non-owning views. Data can be read inside @ref
  /// DebugClient::GetSceneSyncData, but it is not safe to copy the view for later use.
  DynamicArray<MeshSyncDataView> actorMeshes;

  /// Debug draw data or empty. Requires @ref SceneSyncParams::syncDebugDraw. Else empty.
  protocol::DbgDrawData debugDraw;

  bool operator==(SceneSyncData const& rhs) const = default;
};

/**
 * @brief Mochi debugger client. Connects to @ref DebugServer over a network socket.
 */
class DebugClient {
  // Cannot move because of callbacks that capture `this`.
  MOCHI_DECLARE_NO_COPY_NO_MOVE(DebugClient);

 public:
  DebugClient();
  ~DebugClient();

  /**
   * @brief Connect to a server over TCP.
   *
   * @param address Server IP address (e.g. "127.0.0.1")
   * @param port Server port number (e.g. 7333)
   *
   * @note Non-blocking. Use @ref GetStatus to discover status.
   */
  void Connect(std::string_view address, uint16_t port);

  /**
   * @brief Connect to a server for in-process testing (no TCP sockets)
   *
   * @param server A @ref net::MessageServer on which @ref net::MessageServer::StartInProc has been
   * called.
   */
  void ConnectInProc(net::MessageServer& server);

  /**
   * Disconnect from the server
   */
  void Disconnect();

  /**
   * @brief Return the client's protocol version (must match the server's version).
   */
  uint64_t GetVersion() const;

  /**
   * @brief Return this client connection status.
   *
   * @note This client is not considered to be fully connected until the server's @ref
   * protocol::WelcomeMessage has been received.
   */
  net::SocketStatus GetStatus() const;

  /**
   * @brief Get the address and port that this client is connecting to (if any).
   *
   * @param[out] outAddress Returns the server IP address, or empty string.
   * @param[out] outPort Returns the server port, or zero.
   */
  void GetAddress(std::string& outAddress, uint16_t& outPort) const;

  /**
   * @brief Enqueue a message to the server. Thread-safe.
   *
   * @param[in] msg A message to send. Must derive from @ref net::Message and support reflection
   * serialization.
   * @return True if the client is connected and the message was enqueued. False otherwise.
   */
  bool Send(net::Message const& msg);

  /**
   * @brief Send a request and block until the matching reply arrives.
   *
   * @tparam RequestT A message type deriving from @ref RequestMessage with a @c Reply type alias
   * naming the expected reply type (which must derive from @ref ReplyMessage).
   * @param[in] request Request payload. Its @ref RequestMessage::id is assigned before sending.
   * @param[in] timeoutSeconds Timeout [s].
   * @param[in,out] error Check @ref Error::IsOK for status. Fails on no connection or timeout.
   * @return Matching reply message.
   *
   * @note The reply type must be registered via @ref Register with no callback function, in order
   * for it to be received this way.
   *
   * @warning Illegal to call from within a message receive callback (see @ref Register).
   */
  template <class RequestT>
  auto SendAndAwaitReply(RequestT request, double timeoutSeconds, Error& error) {
    return _socket.SendAndAwaitReply<RequestT>(std::move(request), timeoutSeconds, error);
  }

  /**
   * @brief Register an external message handler
   *
   * @tparam MessageT the message class to register
   * @param fn The callback function.
   *
   * @note The callback may be fired on another thread.
   * @note Illegal to register messages after the first dispatch.
   */
  template <class MessageT, class Fn>
  void Register(Fn&& fn) {
    _socket.Register<MessageT>(std::forward<Fn>(fn));
  }

  /**
   * @brief Copy the list of scenes.
   *
   * @details The list is kept up to date from the server. It is empty when the client is not
   * connected.
   */
  void GetSceneList(DynamicArray<SceneInfo>& out) const;

  /**
   * @brief Set the selected scene.
   *
   * @param handle Handle of a scene to select, or an invalid handle to clear the selection.
   *
   * @details If scene synchronization is enabled when the scene selection changes, then
   * any previously synchronized data will be cleared, synchronization of the old scene (if any)
   * will stop and synchronization of the new scene (if any) will start.
   */
  void SelectScene(SceneHandle handle);

  /**
   * @brief Return the currently selected scene, or an invalid handle if no selection.
   *
   * @details The client may automatically select a scene on connect, when a scene is added and
   * nothing selectable existed before, and when the selected scene is removed. By convention a
   * scene name starting with "_" will not be selected automatically, but it could still be selected
   * explicitly.
   */
  [[nodiscard]] SceneHandle GetSelectedScene() const;

  /**
   * @brief Return the coordinate space the server's scene data is expressed in.
   *
   * @details Reported by the server in its @ref protocol::WelcomeMessage, so this is only
   * meaningful once this client is fully connected. Reports @ref CoordinateSpace::Filament when
   * not connected, matching the Debug Server default.
   */
  [[nodiscard]] CoordinateSpace GetCoordinateSpace() const;

  /**
   * @brief Get the current settings.
   */
  DebugClientSettings GetSettings() const;

  /**
   * @brief Apply new settings (if changed)
   */
  void SetSettings(DebugClientSettings const& settings);

  /**
   * @brief Access the latest synchronized data for the selected scene via callback.
   *
   * @param fn Callback function to be called immediately with access to the latest data.
   *
   * @details When polling for changes, the caller may wish to copy the data only when the
   * sync counter has changed. This can be done within the callback.
   */
  void GetSceneSyncData(std::function<void(SceneSyncData const& data)> const& fn) const;

  /**
   * @brief Set the playback mode for all scenes.
   *
   * @details The mode is remembered while disconnected, and is applied to each scene as the
   * client learns about it, including scenes created after connecting.
   */
  void SetSceneStepMode(StepMode mode);

  /** @brief Get the playback mode shared by all scenes. */
  StepMode GetSceneStepMode() const;

  /** @brief Pause all scenes, then take a single step forward in the selected scene. */
  void StepScene();

  /** @brief Restore the selected scene to its initial debugger snapshot. */
  void RestoreSceneState();

  /**
   * @brief Execute a console command
   *
   * @param str A command string. First token is the command. Additional tokens are arguments.
   */
  void ExecuteCommand(std::string_view str);

  /**
   * @brief Get the current console command prompt (for terminal UI).
   */
  std::string GetCommandPrompt() const;

  /**
   * @brief Return true if exit was requested via console command
   */
  bool WasExitRequested() const;

  /**
   * @brief Set a custom function to receive console output.
   *
   * @param fn Callback invoked for each printed message with its channel, message text, source
   * file, and line. Pass an empty @ref LogFn to restore default output via Mochi's log system.
   *
   * @note Must be called before @ref Connect or @ref ConnectInProc to prevent race conditions. The
   * user-provided function may be called concurrently from multiple threads.
   */
  void SetPrintFunction(LogFn fn);

 private:
  using CommandFn = std::function<void(Span<std::string const> tokens)>;

  struct CommandInfo {
    CommandFn fn;
    int numArgs = 0; // -1 means any number
  };

  struct MeshInfo {
    MeshSyncData mesh;
    // Mesh version this entry holds. See MeshSyncDataView::versionCounter.
    uint64_t versionCounter = 0;
    bool visited = false;
  };

  struct State {
    bool exitRequested = false;
    StepMode stepMode = StepMode::Pause; // Current step mode (applies to all scenes)
    DynamicArray<std::string> cmdPath; // Current console command path
    DynamicArray<SceneInfo> scenes; // Cached list of scenes, kept in sync with the server
    CoordinateSpace coordinateSpace = CoordinateSpace::Filament(); // Synced from the server
    SceneHandle selectedScene; // Currently selected scene (if any)
    SceneSyncData syncData; // Latest sync data received from the server.
    uint64_t syncCounter = 0; // Incremented when sync data changes.
    std::unordered_map<ActorHandle, MeshInfo> meshCache; // Per-actor mesh cache.
    DebugClientSettings settings;
  };

  void InitProtocol();
  void InitCommands();
  bool IsConnected() const;
  void OnStatusChange(net::SocketStatus status);
  void SetSceneSyncParams(SceneSyncParams const& params);
  void ResetConnectionState(); // Clear per-connection client state (e.g. cached scene list)
  std::string GetCommandPath() const;
  DynamicArray<std::string> ResolvePath(std::string_view pathArg, Error& error);
  void ValidatePath(Span<std::string const> path, Error& error);
  static protocol::SceneSyncRequest MakeSyncRequest(State const& state);
  static bool IsAutoSelectableScene(SceneInfo const& scene);
  static SceneHandle FindAutoSelectableScene(Span<SceneInfo const> scenes);
  static void SetSelectedScene(
      State& state,
      SceneHandle handle,
      DynamicArray<protocol::SceneSyncRequest>& outRequests);
  static void UpdateActorMeshes(
      protocol::SceneSyncReply const& reply,
      std::unordered_map<ActorHandle, MeshInfo>& meshCache,
      DynamicArray<MeshSyncDataView>& outActorMeshes);

  // Output helpers. Route through the custom print function if set (see @ref SetPrintFunction),
  // otherwise fall back to Mochi's logging system.
  void Print(
      std::string message,
      LogChannel channel = LogChannel::Info,
      char const* file = "",
      int line = 0,
      bool newline = true);
  void PrintError(std::string const& message);
  void PrintSceneList();
  void PrintActorList(std::string_view sceneName, Error& error);

  // Message handlers
  void OnLogMessage(protocol::LogMessage&& msg);
  void OnWelcomeMessage(protocol::WelcomeMessage&& msg);
  void OnSceneAddRemove(protocol::SceneAddRemove&& msg);
  void OnSceneStepReply(protocol::SceneStepReply&& reply);
  void OnSceneSyncReply(protocol::SceneSyncReply&& reply);

  // Console command handlers
  void CmdConnect(Span<std::string const> args);
  void CmdCd(Span<std::string const> args);
  void CmdPwd();
  void CmdLs(Span<std::string const> args);
  void CmdHelp();
  void CmdPing();
  void CmdServers();
  void CmdQuit();

  uint64_t _protocolVersion = 0;
  net::MessageClient _socket;
  std::unordered_map<std::string, CommandInfo> _commands;
  RecursiveGuarded<State> _state;
  std::atomic<bool> _isFullyConnected{false}; // True after WelcomeMessage is received.
  LogFn _printFn;
  std::atomic<bool> _isInProc = false;
};

} // namespace mochi::dbg
