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

#include <mochi_core/net/message.h>
#include <mochi_core/net/message_serialization.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/stream.h>

#include <cstring>

using namespace mochi;
using namespace mochi::net;

void net::SerializeMessage(Message const& msg, DynamicArray<uint8_t>& outBytes) {
  // Get the most-derived TypeInfo for the concrete message type
  auto const& typeInfo = msg.GetFinalTypeInfo();

  // Serialize header
  uint64_t const typeId = typeInfo._typeId.value;
  outBytes.resize_noinit(sizeof(typeId));
  std::memcpy(outBytes.data(), &typeId, sizeof(typeId));

  // Serialize message payload
  DynamicArrayStreamWriter writer(outBytes);
  [[maybe_unused]] bool const ok = typeInfo.SerializeToBytes(&msg, writer);
  MOCHI_ASSERT_VERBOSE(ok, "Binary in-memory serialization should always succeed");
}

Message* net::DeserializeMessage(
    void const* buffer,
    size_t size,
    std::function<SReflect::TypeInfo const*(SReflect::TypeId)> const& tryGetTypeInfo,
    Error& error) {
  MOCHI_ERROR_IF(buffer == nullptr, error, "Null message buffer");
  MOCHI_ERROR_IF(size < sizeof(uint64_t), error, "Insufficient message size");
  MOCHI_ERROR_RETURN(error, nullptr);

  // Deserialize header
  size_t constexpr kHeaderSize = sizeof(uint64_t);
  uint64_t typeId = 0;
  std::memcpy(&typeId, buffer, kHeaderSize);

  // Lookup the TypeInfo
  auto const* typeInfo = tryGetTypeInfo(SReflect::TypeId{typeId});
  MOCHI_ERROR_IF(typeInfo == nullptr, error, "Unknown message type");
  MOCHI_ERROR_RETURN(error, nullptr);
  MOCHI_ASSERT_VERBOSE(
      assert_cast<SReflect::StructTypeInfo const*>(typeInfo)->IsSameOrDerivedFrom(
          SReflect::GetTypeId<Message>()),
      "Invalid message type. Must be a struct that derives from mochi::net::Message."
      "MessageClient and MessageServer enforce this at registration time.");

  // Create a message of the correct type
  void* msg = typeInfo->New();
  MOCHI_DEFER(if (!error.IsOK()) { typeInfo->Delete(msg); });

  // Deserialize message payload
  uint8_t const* payloadData = static_cast<uint8_t const*>(buffer) + kHeaderSize;
  size_t const payloadSize = size - kHeaderSize;
  SpanStreamReader reader{Span<uint8_t const>(payloadData, payloadSize)};
  bool const ok = typeInfo->DeserializeFromBytes(reader, msg);
  MOCHI_ERROR_IF(!ok, error, "Malformed message payload");
  MOCHI_ERROR_IF(reader.GetNumBytesRemaining() != 0, error, "Trailing bytes in message payload");
  MOCHI_ERROR_RETURN(error, nullptr);

  // The cast to Message* is safe because we know that the object derives from Message (see above),
  // and we know that no pointer offset is required because Simple Reflection does not allow
  // multiple inheritance of non-empty base classes.
  return static_cast<Message*>(msg);
}

void net::DeleteMessage(Message* msg) {
  if (msg != nullptr) {
    msg->GetFinalTypeInfo().Delete(msg);
  }
}
