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

// Forwards:
class ServerSocket;

enum class SocketStatus {
  None,
  Pending,
  Connected,
  Lost,
};

// -------------------------------------------------------------------------------------------------
// ClientSocket
// -------------------------------------------------------------------------------------------------

/**
 * @brief TCP client with support for real sockets and in-process loopback.
 *
 * @details Transport mode is determined by which connection method is called:
 * - @ref Connect uses real TCP sockets (spawns a network thread)
 * - @ref ConnectInProc connects within the local process, without a true network socket
 */
class ClientSocket {
  MOCHI_DECLARE_MOVE_ONLY(ClientSocket);

 public:
  ClientSocket();
  ~ClientSocket();

  /**
   * @brief Connect to a server via TCP.
   *
   * @details Non-blocking. Spawns a network thread that retries until the initial connection
   * succeeds or @ref Disconnect is called. After an established connection is lost, the client
   * transitions to Lost and does not reconnect automatically. SocketStatus transitions: None ->
   * Pending -> Connected -> Lost.
   *
   * @param[in] address Server address (e.g. "127.0.0.1").
   * @param[in] port Server TCP port.
   */
  void Connect(std::string_view address, uint16_t port);

  /**
   * @brief Connect to a server in the local process, without a true network socket.
   *
   * @param[in] server Server instance to connect to.
   *
   * @note It is safe for @ref ClientSocket or @ref ServerSocket to be destroyed in either order.
   */
  void ConnectInProc(ServerSocket& server);

  /**
   * @brief Set an opaque user-defined version that must match the server's for a connection to be
   * accepted.
   *
   * @details Optional. Not calling it is equivalent to @c SetVersion(0). On connect, this version
   * is exchanged in addition to the internal socket protocol version; the server accepts the
   * connection only if both the protocol version and this user-defined version match exactly.
   *
   * @param[in] version Opaque application-defined version.
   *
   * @note Must be called before @ref Connect / @ref ConnectInProc; calling it after a connection
   * has been initiated is illegal.
   */
  void SetVersion(uint64_t version);

  /**
   * @brief Disconnect and stop the network thread.
   *
   * @warning Do not call this from a receive or status callback; doing so triggers a
   * @ref MOCHI_ASSERT (for the in-process transport) or terminates (for TCP).
   */
  void Disconnect();

  /** @brief Get the current connection status. */
  [[nodiscard]] SocketStatus GetStatus() const;

  /**
   * @brief Get the address and port that this client is connecting to (if any)
   *
   * @param[out] outAddress Returns the server IP address, or empty string.
   * @param[out] outPort Returns the server port, or zero.
   */
  void GetAddress(std::string& outAddress, uint16_t& outPort) const;

  /**
   * @brief Enqueue a message to the server. Thread-safe.
   *
   * @param[in] data Pointer to the message payload.
   * @param[in] size Size of the payload [bytes].
   * @return True if the message was queued or delivered. False if connection has not been
   * initiated, or has been lost.
   */
  bool Send(void const* data, size_t size);

  /**
   * @brief Set a callback invoked for each message received from the server.
   *
   * @details The callback may happen on another thread. The data pointer is only valid for the
   * duration of the call — copy it if you need to retain it. It is safe to call @ref Send from the
   * callback function.
   *
   * @param[in] callback Invoked with a pointer to the payload and its size [bytes].
   */
  void SetReceiveCallback(std::function<void(void const* data, size_t size)> callback);

  /**
   * @brief Set a callback invoked on status changes.
   *
   * @details The callback may happen on another thread (same thread as the receive callback). It is
   * safe to call @ref Send from the callback function.
   */
  void SetStatusCallback(std::function<void(SocketStatus)> callback);

 private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

} // namespace mochi::net
