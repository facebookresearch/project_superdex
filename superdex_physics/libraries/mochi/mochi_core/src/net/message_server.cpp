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

#include <mochi_core/net/message_serialization.h>
#include <mochi_core/net/message_server.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/dynamic_array.h>

using namespace mochi;
using namespace net;

MessageServer::MessageServer() {
  _socket.SetReceiveCallback(
      [this](ClientId id, void const* data, size_t size) { OnReceive(id, data, size); });
}

MessageServer::~MessageServer() {
  Stop();
}

void MessageServer::SetVersion(uint64_t version) {
  _socket.SetVersion(version);
}

void MessageServer::SetDiscoveryPort(uint16_t discoveryPort) {
  _socket.SetDiscoveryPort(discoveryPort);
}

void MessageServer::Start(uint16_t preferredPort, int maxClients, std::string_view label) {
  _socket.Start(preferredPort, maxClients, label);
}

void MessageServer::StartInProc(int maxClients, std::string_view label) {
  _socket.StartInProc(maxClients, label);
}

void MessageServer::Stop() {
  _socket.Stop();
}

uint16_t MessageServer::GetPort() const {
  return _socket.GetPort();
}

DynamicArray<ClientId> MessageServer::GetClients() const {
  return _socket.GetClients();
}

void MessageServer::SetLabel(std::string_view label) {
  _socket.SetLabel(label);
}

void MessageServer::SetClientCallback(std::function<void(ClientId, ClientEvent)> callback) {
  _socket.SetClientCallback(std::move(callback));
}

bool MessageServer::SendTo(ClientId client, Message const& msg) {
  // Start with a modest capacity while still allowing large messages to grow safely.
  int constexpr kReserveSize = 4 * 1024;
  DynamicArray<uint8_t> bytes;
  bytes.reserve(kReserveSize);

  // Serialize header + payload
  SerializeMessage(msg, bytes);

  // Enqueue send. Ignored if client is not connected.
  return _socket.SendTo(client, bytes.data(), bytes.size());
}

void MessageServer::Broadcast(Message const& msg) {
  // Start with a modest capacity while still allowing large messages to grow safely.
  int constexpr kReserveSize = 4 * 1024;
  DynamicArray<uint8_t> bytes;
  bytes.reserve(kReserveSize);

  // Serialize header + payload
  SerializeMessage(msg, bytes);

  // Enqueue send. Ignored if no clients are connected.
  _socket.Broadcast(bytes.data(), bytes.size());
}

void MessageServer::ValidateMessageType(
    [[maybe_unused]] SReflect::StructTypeInfo const& typeInfo,
    bool isRequest,
    bool isReply) {
  net::ValidateMessageType(typeInfo, isRequest, isReply);
}

void MessageServer::OnReceive(ClientId client, void const* data, size_t size) {
  Error error;
  auto tryGetTypeInfo = [this](auto id) { return _dispatcher.TryGetTypeInfo(id); };
  Message* msg = DeserializeMessage(data, size, tryGetTypeInfo, error);
  if (!error.IsOK()) {
    MOCHI_LOG_WARNING(
        "[Server] Received an invalid network message. Ignoring. Error: %s",
        error.GetDescription());
    return;
  }
  MOCHI_DEFER(DeleteMessage(msg));
  [[maybe_unused]] bool const wasDispatched =
      _dispatcher.DispatchVoid(client, msg->GetFinalTypeId(), msg);
  MOCHI_ASSERT_VERBOSE(
      wasDispatched,
      "Message should have been dispatched. If it were not registered then TryGetTypeInfo should have failed.");
}

uint64_t MessageServer::CalcProtocolVersionHash() const {
  return _dispatcher.CalcProtocolVersionHash();
}
