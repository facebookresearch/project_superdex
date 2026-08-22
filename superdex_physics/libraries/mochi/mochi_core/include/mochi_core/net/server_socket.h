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

#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/dynamic_string.h>
#include <mochi_core/utils/reflection.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string_view>

namespace mochi::net {

// -------------------------------------------------------------------------------------------------
// Types
// -------------------------------------------------------------------------------------------------

using ClientId = uint32_t;

enum class ClientEvent {
  Connected,
  Disconnected,
};

// -------------------------------------------------------------------------------------------------
// ServerSocket
// -------------------------------------------------------------------------------------------------

/**
 * @brief TCP server with support for real sockets and in-process loopback.
 *
 * Transport mode is determined by which method is used to start hosting.
 * - @ref Start binds a real TCP socket (spawns a network thread)
 * - @ref StartInProc hosts within the local process, without a true network socket
 */
class ServerSocket final {
  MOCHI_DECLARE_MOVE_ONLY(ServerSocket);

 public:
  ServerSocket();
  ~ServerSocket();

  /**
   * @brief Start the server on a TCP port.
   *
   * @details Spawns a network thread. Auto-increments the port if the requested one is already in
   * use.
   *
   * @param[in] preferredPort Preferred TCP port. Actual port may differ if already in use.
   * @param[in] maxClients Maximum number of simultaneous client connections.
   * @param[in] label Human-readable label identifying the server.
   *
   * @see GetPort
   */
  void Start(uint16_t preferredPort, int maxClients, std::string_view label);

  /**
   * @brief Start the server in the local process, without a true network socket.
   *
   * @param[in] maxClients Maximum number of simultaneous client connections.
   * @param[in] label Human-readable label for identification.
   */
  void StartInProc(int maxClients, std::string_view label);

  /**
   * @brief Set an opaque user-defined version that connecting clients must match.
   *
   * @details Optional. Not calling it is equivalent to @c SetVersion(0). On connect, this version
   * is exchanged in addition to the internal socket protocol version; a client is accepted only if
   * both the protocol version and this user-defined version match exactly. It is also advertised
   * over UDP discovery (see @ref ServerInfo::version).
   *
   * @param[in] version Opaque application-defined version.
   *
   * @note Must be called before @ref Start / @ref StartInProc; calling it after the server has
   * started hosting is illegal.
   */
  void SetVersion(uint64_t version);

  /**
   * @brief Set the UDP discovery port used on the next @ref Start.
   *
   * @details Must be called before @ref Start / @ref StartInProc or after @ref Stop.
   *
   * @param[in] discoveryPort UDP discovery port. 0 disables discovery.
   */
  void SetDiscoveryPort(uint16_t discoveryPort);

  /**
   * @brief Stop the server and disconnect all clients.
   *
   * @warning Do not call this from a receive or client-event callback; doing so triggers a
   * @ref MOCHI_ASSERT (for the in-process transport) or terminates (for TCP).
   */
  void Stop();

  /**
   * @brief Get the actual TCP port the server is listening on.
   *
   * May differ from the requested port if it was already in use. Returns 0 if the server is not
   * listening on a TCP socket, including in-process servers started via @ref StartInProc.
   */
  [[nodiscard]] uint16_t GetPort() const;

  /** @brief Update the server's human-readable label. */
  void SetLabel(std::string_view label);

  /** @brief Get the list of currently connected client IDs. */
  [[nodiscard]] DynamicArray<ClientId> GetClients() const;

  /**
   * @brief Send a message to a specific client. Thread-safe.
   *
   * @param[in] client Target client ID.
   * @param[in] data Pointer to the message payload.
   * @param[in] size Size of the payload [bytes].
   * @return True if the client is currently connected and the message was enqueued, false
   * otherwise.
   */
  bool SendTo(ClientId client, void const* data, size_t size);

  /**
   * @brief Send a message to all connected clients. Thread-safe, fire-and-forget.
   *
   * @param[in] data Pointer to the message payload.
   * @param[in] size Size of the payload [bytes].
   */
  void Broadcast(void const* data, size_t size);

  /**
   * @brief Set a callback invoked for each message received from any client.
   *
   * @details The callback may happen on another thread. The data pointer is only valid for the
   * duration of the call — copy it if you need to retain it. It is safe to call @ref SendTo or @ref
   * Broadcast from the callback function.
   *
   * @param[in] callback Invoked with the sender's @ref ClientId, a pointer to the payload, and its
   * size [bytes].
   */
  void SetReceiveCallback(std::function<void(ClientId, void const* data, size_t size)> callback);

  /**
   * @brief Set a callback invoked on client connect/disconnect
   *
   * @details The callback may happen on another thread (same thread as the receive callback). It is
   * safe to call @ref SendTo or @ref Broadcast from the callback function.
   */
  void SetClientCallback(std::function<void(ClientId, ClientEvent)> callback);

 private:
  friend class ClientSocket;
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

} // namespace mochi::net
