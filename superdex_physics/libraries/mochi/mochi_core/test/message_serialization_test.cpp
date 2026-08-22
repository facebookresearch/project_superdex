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
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/error.h>

#include <cstdint>

using namespace mochi;
using namespace mochi::net;

namespace {

struct TestMessage : Message {
  TestMessage() = default;
  explicit TestMessage(int v) : value(v) {}

  int value = 0;

  MOCHI_STRUCT_BEGIN(TestMessage)
  MOCHI_BASE_CLASS(Message)
  MOCHI_FIELD(value)
  MOCHI_STRUCT_END()
};

} // namespace

TEST(NetMessageSerialization, SerializeMessage) {
  DynamicArray<uint8_t> bytes;
  SerializeMessage(TestMessage{42}, bytes);
  EXPECT_GT(bytes.size(), sizeof(uint64_t)); // Expected 8 byte header + payload
}

TEST(NetMessageSerialization, RoundTrip) {
  DynamicArray<uint8_t> bytes;
  SerializeMessage(TestMessage{42}, bytes);

  auto const& typeInfo = SReflect::GetTypeInfo<TestMessage>();
  auto const tryGetTypeInfo = [&](SReflect::TypeId id) -> SReflect::TypeInfo const* {
    return (id == typeInfo._typeId) ? &typeInfo : nullptr;
  };

  test::ExpectOK error;
  Message* const msg = DeserializeMessage(bytes.data(), bytes.size(), tryGetTypeInfo, error);
  ASSERT_NE(msg, nullptr);
  auto* const typedMsg = assert_cast<TestMessage*>(msg);
  EXPECT_EQ(typedMsg->value, 42);
  DeleteMessage(msg);
  DeleteMessage(nullptr); // No-op
}

TEST(NetMessageSerialization, DeserializeMessage_FailsOnIncorrectBufferSize) {
  DynamicArray<uint8_t> bytes;
  SerializeMessage(TestMessage{42}, bytes);

  auto const& typeInfo = SReflect::GetTypeInfo<TestMessage>();
  auto const tryGetTypeInfo = [&](SReflect::TypeId id) -> SReflect::TypeInfo const* {
    return (id == typeInfo._typeId) ? &typeInfo : nullptr;
  };

  // Fail if the buffer is too small
  for (size_t sz = 0; sz < bytes.size(); ++sz) {
    Message* const msg = DeserializeMessage(bytes.data(), sz, tryGetTypeInfo, test::ExpectNotOK{});
    EXPECT_EQ(nullptr, msg);
  }

  // Fail if the buffer has unused bytes at the end
  {
    bytes.push_back(0);
    Message* const msg =
        DeserializeMessage(bytes.data(), bytes.size(), tryGetTypeInfo, test::ExpectNotOK{});
    EXPECT_EQ(nullptr, msg);
  }

  // Succed if the buffer is exactly the right size
  bytes.pop_back();
  {
    Message* const msg =
        DeserializeMessage(bytes.data(), bytes.size(), tryGetTypeInfo, test::ExpectOK{});
    EXPECT_NE(nullptr, assert_cast<TestMessage*>(msg));
    DeleteMessage(msg);
  }
}

TEST(NetMessageSerialization, DeserializeMessage_FailsIfTypeInfoNotFound) {
  DynamicArray<uint8_t> bytes;
  SerializeMessage(TestMessage{42}, bytes);

  auto const tryGetTypeInfo = [](SReflect::TypeId) -> SReflect::TypeInfo const* { return nullptr; };

  Message* const msg =
      DeserializeMessage(bytes.data(), bytes.size(), tryGetTypeInfo, test::ExpectNotOK{});
  EXPECT_EQ(msg, nullptr);
}
