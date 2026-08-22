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

#include <mochi_core/net/client_socket.h>
#include <mochi_core/net/message.h>
#include <mochi_core/net/message_dispatcher.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/guarded.h>
#include <mochi_core/utils/reflection.h>

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <type_traits>

#if !MOCHI_USE_REFLECTION
#error "This file requires MOCHI_USE_REFLECTION=1"
#endif

namespace mochi::net {

// Forwards:
class MessageServer;

/**
 * @brief Wraps a @ref ClientSocket. Instead of sending and receiving raw byte buffers, this class
 * uses message structs and reflection serialization.
 */
class MessageClient {
  MOCHI_DECLARE_NO_COPY_NO_MOVE(MessageClient);

 public:
  MessageClient();
  ~MessageClient();

  /**
   * @brief Optionally set a version number before calling @ref Connect or @ref ConnectInProc.
   *
   * @details Must match the server's version or connection will be refused.
   *
   * @param[in] version Version number (default 0)
   */
  void SetVersion(uint64_t version);

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
   * @note It is safe for @ref MessageClient or @ref MessageServer to be destroyed in either order.
   */
  void ConnectInProc(MessageServer& server);

  /**
   * @brief Connect to a generic @ref ServerSocket for unit tests.
   *
   * @param[in] server Server socket to connect to.
   *
   * @note It is safe for @ref MessageClient or @ref ServerSocket to be destroyed in either order.
   */
  void ConnectInProc(ServerSocket& server);

  /** @brief Disconnect and stop the network thread. */
  void Disconnect();

  /** @brief Get the current connection status. */
  [[nodiscard]] SocketStatus GetStatus() const;

  /**
   * @brief Get the address and port that this client is connecting to (if any).
   *
   * @param[out] outAddress Returns the server IP address, or empty string.
   * @param[out] outPort Returns the server port, or zero.
   */
  void GetAddress(std::string& outAddress, uint16_t& outPort) const;

  /**
   * @brief Set a callback invoked on status changes.
   *
   * @details The callback may happen on another thread (same thread as the receive callback).
   */
  void SetStatusCallback(std::function<void(SocketStatus)> callback);

  /**
   * @brief Enqueue a message to the server. Thread-safe.
   *
   * @param[in] msg A message to send. Must derive from @ref Message and support reflection
   * serialization.
   * @return True if the client is connected and the message was enqueued. False otherwise.
   */
  bool Send(Message const& msg);

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
   * @warning Illegal to call from within a message receive callback (see @ref Register) or a status
   * callback (see @ref SetStatusCallback).
   */
  template <class RequestT>
  auto SendAndAwaitReply(RequestT request, double timeoutSeconds, Error& error) {
    using ReplyT = typename RequestT::Reply;
    static_assert(std::is_base_of_v<RequestMessage, RequestT>);
    static_assert(std::is_base_of_v<ReplyMessage, ReplyT>);
    ReplyT reply;
    SendAndAwaitReplyImpl(request, reply, timeoutSeconds, error);
    return reply;
  }

  /**
   * @brief Register a message class with optional callback.
   *
   * @tparam MessageT A message class derived from @ref Message, with reflection support.
   * @param onReceive Function to call when the message is received. May be called on another
   * thread. Can be omitted for message types that are only sent to the server, or only received
   * via @ref SendAndAwaitReply.
   *
   * @note It is illegal to register additional message handlers after @ref Connect.
   */
  template <class MessageT>
  void Register(std::function<void(MessageT&&)> onReceive = {}) {
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
  void OnReceive(void const* data, size_t size);
  void OnStatus(SocketStatus status);
  void SendAndAwaitReplyImpl(
      RequestMessage& request,
      ReplyMessage& outReply,
      double timeoutSeconds,
      Error& error);

  ClientSocket _socket;
  MessageDispatcher<> _dispatcher;
  RecursiveGuarded<std::function<void(SocketStatus)>> _statusCallback;
  struct ReqRepHelper;
  std::unique_ptr<ReqRepHelper> _reqrep;
};

} // namespace mochi::net
