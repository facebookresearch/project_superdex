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

#include <mochi_core/net/message_dispatcher.h>
#include <mochi_core/test/mochi_test_helpers.h>

using namespace mochi;
using namespace mochi::net;

namespace {
struct TestMessageA {
  int value = 0;
  MOCHI_STRUCT_BEGIN(TestMessageA)
  MOCHI_FIELD(value)
  MOCHI_STRUCT_END()
};

struct TestMessageB {
  int value = 0;
  MOCHI_STRUCT_BEGIN(TestMessageB)
  MOCHI_FIELD(value)
  MOCHI_STRUCT_END()
};

struct TestMessageC {
  std::string str;
  MOCHI_STRUCT_BEGIN(TestMessageC)
  MOCHI_FIELD(str)
  MOCHI_STRUCT_END()
};

TEST(NetMessageDispatcher, Register) {
  MessageDispatcher<> dispatcher;
  auto const& typeA = SReflect::GetTypeInfo<TestMessageA>();
  auto const& typeB = SReflect::GetTypeInfo<TestMessageB>();

  // No registered types
  EXPECT_TRUE(dispatcher.IsEmpty());
  EXPECT_FALSE(dispatcher.HasReceiver(typeA._typeId));
  EXPECT_FALSE(dispatcher.HasReceiver(typeB._typeId));
  EXPECT_EQ(nullptr, dispatcher.TryGetTypeInfo(typeA._typeId));
  EXPECT_EQ(nullptr, dispatcher.TryGetTypeInfo(typeB._typeId));

  // One registered type
  dispatcher.Register<TestMessageA>([](TestMessageA&&) {});
  EXPECT_FALSE(dispatcher.IsEmpty());
  EXPECT_TRUE(dispatcher.HasReceiver(typeA._typeId));
  EXPECT_FALSE(dispatcher.HasReceiver(typeB._typeId));
  EXPECT_EQ(&typeA, dispatcher.TryGetTypeInfo(typeA._typeId));
  EXPECT_EQ(nullptr, dispatcher.TryGetTypeInfo(typeB._typeId));

  // Two registered types
  dispatcher.Register<TestMessageB>([](TestMessageB&&) {});
  EXPECT_FALSE(dispatcher.IsEmpty());
  EXPECT_TRUE(dispatcher.HasReceiver(typeA._typeId));
  EXPECT_TRUE(dispatcher.HasReceiver(typeB._typeId));
  EXPECT_EQ(&typeA, dispatcher.TryGetTypeInfo(typeA._typeId));
  EXPECT_EQ(&typeB, dispatcher.TryGetTypeInfo(typeB._typeId));
}

TEST(NetMessageDispatcher, Dispatch) {
  MessageDispatcher<> dispatcher;
  std::string log;

  // Message handlers append to the 'log'
  dispatcher.Register<TestMessageA>(
      [&](auto&& msg) { log += Format("TestMessageA(%d)", msg.value); });
  dispatcher.Register<TestMessageC>(
      [&](auto&& msg) { log += Format("TestMessageC(%s)", msg.str.c_str()); });

  EXPECT_TRUE(dispatcher.Dispatch(TestMessageA{123}));
  EXPECT_STREQ("TestMessageA(123)", log.c_str());
  log.clear();

  // TestMessageB was not registered
  EXPECT_FALSE(dispatcher.Dispatch(TestMessageB{}));
  EXPECT_STREQ("", log.c_str());

  EXPECT_TRUE(dispatcher.Dispatch(TestMessageC{"cool"}));
  EXPECT_STREQ("TestMessageC(cool)", log.c_str());
  log.clear();

  // Via DispatchVoid
  TestMessageA msgA{911};
  EXPECT_TRUE(dispatcher.DispatchVoid(SReflect::GetTypeId<TestMessageA>(), &msgA));
  EXPECT_STREQ("TestMessageA(911)", log.c_str());
  log.clear();
  EXPECT_FALSE(
      dispatcher.DispatchVoid(SReflect::GetTypeId<TestMessageB>(), &msgA)); // Not registered
}

TEST(NetMessageDispatcher, RegisterWithoutCallback) {
  MessageDispatcher<> dispatcher;
  auto const& typeA = SReflect::GetTypeInfo<TestMessageA>();

  // Registering without a callback makes the type known for deserialization...
  dispatcher.Register<TestMessageA>();
  EXPECT_FALSE(dispatcher.IsEmpty());
  EXPECT_EQ(&typeA, dispatcher.TryGetTypeInfo(typeA._typeId));

  // ...but dispatch fires nothing and reports that no callback ran.
  TestMessageA msg{123};
  EXPECT_FALSE(dispatcher.Dispatch(TestMessageA{123}));
  EXPECT_FALSE(dispatcher.DispatchVoid(SReflect::GetTypeId<TestMessageA>(), &msg));
}

TEST(NetMessageDispatcher, CalcProtocolVersionHash) {
  MessageDispatcher<> dispatcher;
  uint64_t const emptyHash = dispatcher.CalcProtocolVersionHash();

  dispatcher.Register<TestMessageA>([](TestMessageA&&) {});
  uint64_t const hashWithA = dispatcher.CalcProtocolVersionHash();
  EXPECT_NE(emptyHash, hashWithA);

  dispatcher.Register<TestMessageB>();
  uint64_t const hashWithAB = dispatcher.CalcProtocolVersionHash();
  EXPECT_NE(hashWithA, hashWithAB);

  MessageDispatcher<> sameOrder;
  sameOrder.Register<TestMessageA>([](TestMessageA&&) {});
  sameOrder.Register<TestMessageB>();
  EXPECT_EQ(hashWithAB, sameOrder.CalcProtocolVersionHash());

  MessageDispatcher<> reverseOrder;
  reverseOrder.Register<TestMessageB>();
  reverseOrder.Register<TestMessageA>([](TestMessageA&&) {});
  EXPECT_EQ(hashWithAB, reverseOrder.CalcProtocolVersionHash());
}

TEST(NetMessageDispatcher, MoveSemantics) {
  MessageDispatcher<> dispatcher;
  std::string text;

  dispatcher.Register<TestMessageC>([&](auto&& msg) { text = std::move(msg.str); });

  // The std::string within TestMessageC should be moved, not copied. Use a string long enough to
  // exceed the small-string-optimization buffer so it heap-allocates.
  std::string const longStr = "this string is long enough to avoid small string optimization";
  TestMessageC msg{longStr};
  char const* ptr = msg.str.c_str();
  EXPECT_TRUE(dispatcher.Dispatch(std::move(msg)));
  EXPECT_EQ(longStr, text);
  EXPECT_EQ(ptr, text.c_str()); // Address that was moved

  // Via DispatchVoid
  msg.str = longStr;
  ptr = msg.str.c_str();
  EXPECT_TRUE(dispatcher.DispatchVoid(SReflect::GetTypeId<TestMessageC>(), &msg));
  EXPECT_EQ(longStr, text);
  EXPECT_EQ(ptr, text.c_str()); // Address that was moved
}

TEST(NetMessageDispatcher, ExtraArgs) {
  MessageDispatcher<int, float> dispatcher;
  std::string log;

  // Message handlers append to the 'log'
  dispatcher.Register<TestMessageA>([&](int i, float f, auto&& msg) {
    log += Format("TestMessageA(%d, %g, %d)", i, f, msg.value);
  });
  dispatcher.Register<TestMessageB>([&](int i, float f, auto&& msg) {
    log += Format("TestMessageB(%d, %g, %d)", i, f, msg.value);
  });

  // Dispatch with extra arguments
  EXPECT_TRUE(dispatcher.Dispatch(-1, 1.0f, TestMessageA{123}));
  EXPECT_STREQ("TestMessageA(-1, 1, 123)", log.c_str());
  log.clear();
  EXPECT_TRUE(dispatcher.Dispatch(3, 1.5f, TestMessageB{456}));
  EXPECT_STREQ("TestMessageB(3, 1.5, 456)", log.c_str());
  log.clear();
}

} // namespace
