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
#include <mochi_core/net/message_dispatcher.h>
#include <mochi_core/net/server_socket.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/reflection.h>

#include <cstdint>
#include <functional>
#include <string_view>
#include <type_traits>

#if !MOCHI_USE_REFLECTION
#error "This file requires MOCHI_USE_REFLECTION=1"
#endif

namespace mochi::net {

/**
 * @brief Wraps a @ref ServerSocket. Instead of sending and receiving raw byte buffers, this class
 * uses message structs and reflection serialization.
 *
 * @details Each received message is routed to a per-type handler that also receives the sender's
 * @ref ClientId, so the server can distinguish which client a message came from.
 */
class MessageServer {
  MOCHI_DECLARE_NO_COPY_NO_MOVE(MessageServer);

 public:
  MessageServer();
  ~MessageServer();

  /**
   * @brief Optionally set a version number before calling @ref Start or @ref StartInProc.
   *
   * @details Clients must match this version number or connection will be refused.
   *
   * @param[in] version Version number (default 0)
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

  /** @brief Stop the server and disconnect all clients. */
  void Stop();

  /**
   * @brief Get the actual TCP port the server is listening on.
   *
   * May differ from the requested port if it was already in use. Returns 0 if the server is not
   * listening on a TCP socket, including in-process servers started via @ref StartInProc.
   */
  [[nodiscard]] uint16_t GetPort() const;

  /** @brief Get the list of currently connected client IDs. */
  [[nodiscard]] DynamicArray<ClientId> GetClients() const;

  /** @brief Update the server's human-readable label. */
  void SetLabel(std::string_view label);

  /**
   * @brief Set a callback invoked on client connect/disconnect.
   *
   * @details The callback may happen on another thread (same thread as the receive callback).
   */
  void SetClientCallback(std::function<void(ClientId, ClientEvent)> callback);

  /**
   * @brief Enqueue a message to a specific client. Thread-safe.
   *
   * @param[in] client Target client ID.
   * @param[in] msg A message to send. Must derive from @ref Message and support reflection
   * serialization.
   * @return True if the client is connected and the message was enqueued. False otherwise.
   */
  bool SendTo(ClientId client, Message const& msg);

  /**
   * @brief Enqueue a message to all connected clients. Thread-safe, fire-and-forget.
   *
   * @param[in] msg A message to send. Must derive from @ref Message and support reflection
   * serialization.
   *
   * @note Message will be dropped if no clients are connected.
   */
  void Broadcast(Message const& msg);

  /**
   * @brief Register a message class with optional callback.
   *
   * @tparam MessageT A message class derived from @ref Message, with reflection support.
   * @param onReceive Function to call when the message is received. May be called on another
   * thread. Can be omitted for message types that are only sent to the client.
   *
   * @note It is illegal to register additional message handlers after @ref Start / @ref
   * StartInProc.
   */
  template <class MessageT>
  void Register(std::function<void(ClientId, MessageT&&)> onReceive = {}) {
    using MessageType = std::decay_t<MessageT>;
    static_assert(
        std::is_default_constructible_v<MessageType>,
        "All message types must be default constructible");
    static_assert(
        std::is_base_of_v<Message, MessageType>,
        "Message classes must derive from mochi::net::Message");
    if (onReceive) {
      _dispatcher.Register<MessageType>(std::move(onReceive));
    } else {
      _dispatcher.Register<MessageType>();
    }
    ValidateMessageType(
        MessageType::GetTypeInfo(),
        std::is_base_of_v<RequestMessage, MessageType>,
        std::is_base_of_v<ReplyMessage, MessageType>);
  }

  // For internal use and unit tests.
  ServerSocket& GetSocket_InternalUseOnly() {
    return _socket;
  }

  /**
   * @brief Calculate a hash of all of the registered message types and all their reflection
   * metadata.
   *
   * @details If both client and server pass this value to @ref SetVersion, then they will only
   * be allowed to connect if they fully agree on all details of all registered message types.
   */
  uint64_t CalcProtocolVersionHash() const;

 private:
  static void
  ValidateMessageType(SReflect::StructTypeInfo const& typeInfo, bool isRequest, bool isReply);
  void OnReceive(ClientId client, void const* data, size_t size);
  ServerSocket _socket;
  MessageDispatcher<ClientId> _dispatcher;
};

} // namespace mochi::net
