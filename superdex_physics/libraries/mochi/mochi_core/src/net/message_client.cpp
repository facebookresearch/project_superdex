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

#include "message_validation.h"
#include "reqrep_helper.h"

#include <mochi_core/net/message_client.h>
#include <mochi_core/net/message_serialization.h>
#include <mochi_core/net/message_server.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/dynamic_array.h>

using namespace mochi;
using namespace net;

#if MOCHI_ASSERT_VERBOSE_ENABLED
// True while the calling thread is inside OnReceive/OnStatus.
// Used to catch an illegal re-entry.
thread_local static bool g_thisThreadIsInSocketCallback = false;
#endif

//--------------------------------------------------------------------------------------
// MessageClient
//--------------------------------------------------------------------------------------

MessageClient::MessageClient() {
  _reqrep = std::make_unique<ReqRepHelper>(*this);
  _socket.SetReceiveCallback([this](void const* data, size_t size) { OnReceive(data, size); });
  _socket.SetStatusCallback([this](auto status) { OnStatus(status); });
}

MessageClient::~MessageClient() {
  _socket.SetStatusCallback({});
  _socket.SetReceiveCallback({});
  Disconnect();
}

void MessageClient::SetVersion(uint64_t version) {
  _socket.SetVersion(version);
}

void MessageClient::Connect(std::string_view address, uint16_t port) {
  _socket.Connect(address, port);
}

void MessageClient::ConnectInProc(MessageServer& server) {
  _socket.ConnectInProc(server.GetSocket_InternalUseOnly());
}

void MessageClient::ConnectInProc(ServerSocket& server) {
  _socket.ConnectInProc(server);
}

void MessageClient::Disconnect() {
  _socket.Disconnect();
  _reqrep->Cancel();
}

SocketStatus MessageClient::GetStatus() const {
  return _socket.GetStatus();
}

void MessageClient::GetAddress(std::string& outAddress, uint16_t& outPort) const {
  _socket.GetAddress(outAddress, outPort);
}

void MessageClient::SetStatusCallback(std::function<void(SocketStatus)> callback) {
  _statusCallback.Store(std::move(callback));
}

bool MessageClient::Send(Message const& msg) {
  // Start with a modest capacity while still allowing large messages to grow safely.
  int constexpr kReserveSize = 4 * 1024;
  DynamicArray<uint8_t> bytes;
  bytes.reserve(kReserveSize);

  // Serialize header + payload
  SerializeMessage(msg, bytes);

  // Enqueue send (if connected)
  return _socket.Send(bytes.data(), bytes.size());
}

void MessageClient::SendAndAwaitReplyImpl(
    RequestMessage& request,
    ReplyMessage& outReply,
    double timeoutSeconds,
    Error& error) {
  MOCHI_ASSERT_VERBOSE(
      !g_thisThreadIsInSocketCallback,
      "SendAndAwaitReply must not be called from within OnReceive/OnStatus on the same "
      "thread. This will result in a deadlock.");
  _reqrep->SendAndAwaitReply(request, outReply, timeoutSeconds, error);
}

void MessageClient::ValidateMessageType(
    [[maybe_unused]] SReflect::StructTypeInfo const& typeInfo,
    bool isRequest,
    bool isReply) {
  net::ValidateMessageType(typeInfo, isRequest, isReply);
}

void MessageClient::OnReceive(void const* data, size_t size) {
#if MOCHI_ASSERT_VERBOSE_ENABLED
  MOCHI_ASSERT_VERBOSE(
      !g_thisThreadIsInSocketCallback,
      "We do not expect nested socket callbacks on the same thread.")
  g_thisThreadIsInSocketCallback = true;
  MOCHI_DEFER(g_thisThreadIsInSocketCallback = false);
#endif

  Error error;
  auto tryGetTypeInfo = [this](auto id) { return _dispatcher.TryGetTypeInfo(id); };
  Message* msg = DeserializeMessage(data, size, tryGetTypeInfo, error);
  if (!error.IsOK()) {
    MOCHI_LOG_WARNING(
        "[Client] Received an invalid network message. Ignoring. Error: %s",
        error.GetDescription());
    return;
  }
  MOCHI_DEFER(DeleteMessage(msg));

  // If this is a ReplyMessage to a valid ID, then route it to the ReqRepHelper.
  // It may or may not be received, depending on whether their request was canceled or timed out.
  if (auto* reply = dynamic_cast<ReplyMessage*>(msg)) {
    if (reply->requestId != 0) {
      _reqrep->DispatchReply(std::move(*reply));
      return;
    }
  }

  // Use MessageDispatcher to route all other message types.
  auto const& typeInfo = msg->GetFinalTypeInfo();
  bool wasReceived = _dispatcher.DispatchVoid(typeInfo._typeId, msg);

  if (!wasReceived) [[unlikely]] {
    MOCHI_LOG_VERBOSE(
        "[Client] Received a message of type %s, but no callback was registered for it. ",
        typeInfo._nameWithNamespace);
  }
}

void MessageClient::OnStatus(SocketStatus status) {
#if MOCHI_ASSERT_VERBOSE_ENABLED
  MOCHI_ASSERT_VERBOSE(
      !g_thisThreadIsInSocketCallback,
      "We do not expect nested socket callbacks on the same thread.")
  g_thisThreadIsInSocketCallback = true;
  MOCHI_DEFER(g_thisThreadIsInSocketCallback = false);
#endif

  // Cancel all request/reply pairs if connection was lost.
  if (status == SocketStatus::Lost) {
    _reqrep->Cancel();
  }

  // Optionally forward this notification to the user
  _statusCallback.Read([status](auto const& fn) {
    if (fn) {
      fn(status);
    }
  });
}

uint64_t MessageClient::CalcProtocolVersionHash() const {
  return _dispatcher.CalcProtocolVersionHash();
}
