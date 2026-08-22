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

#include <mochi_core/test/allocator_test_helpers.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/dynamic_string.h>
#include <mochi_core/utils/reflection.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/stream.h>

#include <unordered_map>
#include <unordered_set>

using namespace mochi;
using namespace mochi::test;

// Some build configurations enable STL iterator debugging in debug builds.
// Some implementations of iterator debugging perform additional heap allocations.
// Therefore, we only check the TestAllocator::s_allocate (counter) in non-debug builds.
// There are other specific macros we could check, but they vary by compiler.
#if MOCHI_DEBUG
#define EXPECT_TEST_ALLOCATION_COUNT(count)
#else
#define EXPECT_TEST_ALLOCATION_COUNT(count) EXPECT_EQ(count, TestAllocator::s_allocate)
#endif

static constexpr char const* kLongStringLiteral =
    "This is a string that should be long enough to exceed the small memory "
    "optimization within any implementation of std::string.";

TEST(DynamicString, Empty) {
  TestAllocator::ResetCounters();
  TestAllocator alloc;

  // Default construct
  {
    DynamicString s;
    EXPECT_STREQ("", s.c_str());
    EXPECT_EQ(GetDefaultAllocator(), s.get_allocator());
  }

  // Construct with allocator
  {
    DynamicString s(&alloc);
    EXPECT_TEST_ALLOCATION_COUNT(0);
    EXPECT_STREQ("", s.c_str());
    EXPECT_EQ(&alloc,
              s.get_allocator().get_allocator()); // Get Allocator* from StlAllocator*
  }
}

TEST(DynamicString, Copy) {
  TestAllocator::ResetCounters();
  TestAllocator alloc, alloc2;

  // Copy from string literal
  {
    DynamicString s("hello");
    EXPECT_STREQ("hello", s.c_str());
    EXPECT_EQ(GetDefaultAllocator(), s.get_allocator().get_allocator());
  }

  // Copy from string literal + allocator
  {
    DynamicString s(kLongStringLiteral, &alloc);
    EXPECT_TEST_ALLOCATION_COUNT(1);
    EXPECT_STREQ(kLongStringLiteral, s.c_str());
    EXPECT_EQ(&alloc, s.get_allocator().get_allocator());
  }
  EXPECT_EQ(0, TestAllocator::s_bytes);
  TestAllocator::ResetCounters();

  // Copy from std::string
  {
    DynamicString s(std::string{kLongStringLiteral});
    EXPECT_STREQ(kLongStringLiteral, s.c_str());
    EXPECT_EQ(GetDefaultAllocator(), s.get_allocator().get_allocator());
  }

  // Copy from std::string + allocator
  {
    DynamicString s(std::string{kLongStringLiteral}, &alloc);
    EXPECT_TEST_ALLOCATION_COUNT(1);
    EXPECT_STREQ(kLongStringLiteral, s.c_str());
    EXPECT_EQ(&alloc, s.get_allocator().get_allocator());
  }
  EXPECT_EQ(0, TestAllocator::s_bytes);
  TestAllocator::ResetCounters();

  // Copy from another DynamicString
  {
    DynamicString s(kLongStringLiteral, &alloc);
    EXPECT_TEST_ALLOCATION_COUNT(1);
    DynamicString s2(s); // Copy using the same allocator
    EXPECT_TEST_ALLOCATION_COUNT(2);
    EXPECT_STREQ(kLongStringLiteral, s2.c_str());
    EXPECT_EQ(&alloc, s2.get_allocator().get_allocator());
  }
  EXPECT_EQ(0, TestAllocator::s_bytes);
  TestAllocator::ResetCounters();

  // Copy from another DynamicString + allocator
  {
    DynamicString s(kLongStringLiteral, &alloc);
    EXPECT_TEST_ALLOCATION_COUNT(1);
    DynamicString s2(s, &alloc2);
    EXPECT_TEST_ALLOCATION_COUNT(2);
    EXPECT_STREQ(kLongStringLiteral, s2.c_str());
    EXPECT_EQ(&alloc2, s2.get_allocator().get_allocator());
  }
  EXPECT_EQ(0, TestAllocator::s_bytes);
  TestAllocator::ResetCounters();
}

TEST(DynamicString, Move) {
  TestAllocator::ResetCounters();
  TestAllocator alloc, alloc2;
  TestAllocator::s_compatibleWithOtherInstances = true;

  // Move construct. Use RHS allocator.
  {
    DynamicString s(kLongStringLiteral, &alloc);
    EXPECT_TEST_ALLOCATION_COUNT(1);
    DynamicString s2(std::move(s));
    EXPECT_TEST_ALLOCATION_COUNT(1); // No change
    EXPECT_STREQ(kLongStringLiteral, s2.c_str());
    EXPECT_EQ(&alloc, s2.get_allocator().get_allocator());
  }
  EXPECT_EQ(0, TestAllocator::s_bytes);
  TestAllocator::ResetCounters();

  // Move construct with compatible allocator.
  {
    DynamicString s(kLongStringLiteral, &alloc);
    EXPECT_TEST_ALLOCATION_COUNT(1);
    DynamicString s2(std::move(s), &alloc2);
    EXPECT_TEST_ALLOCATION_COUNT(1); // No change
    EXPECT_STREQ(kLongStringLiteral, s2.c_str());
    EXPECT_EQ(&alloc2, s2.get_allocator().get_allocator());
  }
  EXPECT_EQ(0, TestAllocator::s_bytes);
  TestAllocator::ResetCounters();

  // Move construct with incompatible allocator.
  {
    TestAllocator::s_compatibleWithOtherInstances = false;
    DynamicString s(kLongStringLiteral, &alloc);
    EXPECT_TEST_ALLOCATION_COUNT(1);
    DynamicString s2(std::move(s), &alloc2);
    EXPECT_TEST_ALLOCATION_COUNT(2); // String was copied to new memory
    EXPECT_STREQ(kLongStringLiteral, s2.c_str());
    EXPECT_EQ(&alloc2, s2.get_allocator().get_allocator());
    TestAllocator::s_compatibleWithOtherInstances = true; // restore
  }
  EXPECT_EQ(0, TestAllocator::s_bytes);
  TestAllocator::ResetCounters();
}

TEST(DynamicString, CopyAssign) {
  TestAllocator::ResetCounters();
  TestAllocator alloc, alloc2;

  {
    DynamicString s(kLongStringLiteral, &alloc);
    DynamicString s2(&alloc2);
    EXPECT_TEST_ALLOCATION_COUNT(1);
    s2 = s;
    EXPECT_TEST_ALLOCATION_COUNT(2); // String was copied
    EXPECT_STREQ(kLongStringLiteral, s.c_str());
    EXPECT_EQ(&alloc, s.get_allocator().get_allocator()); // No change
  }
  EXPECT_EQ(0, TestAllocator::s_bytes);
  TestAllocator::ResetCounters();
}

TEST(DynamicString, MoveAssign) {
  TestAllocator::ResetCounters();
  TestAllocator alloc, alloc2;

  // Move assign with compatible allocators
  TestAllocator::s_compatibleWithOtherInstances = true;
  {
    DynamicString s(kLongStringLiteral, &alloc);
    EXPECT_TEST_ALLOCATION_COUNT(1);
    DynamicString s2(&alloc2);
    s2 = std::move(s);
    EXPECT_TEST_ALLOCATION_COUNT(1); // No change
    EXPECT_STREQ(kLongStringLiteral, s2.c_str());
    EXPECT_EQ(&alloc2, s2.get_allocator().get_allocator());
  }
  EXPECT_EQ(0, TestAllocator::s_bytes);
  TestAllocator::ResetCounters();

  // Move assign with incompatible allocators
  TestAllocator::s_compatibleWithOtherInstances = false;
  {
    DynamicString s(kLongStringLiteral, &alloc);
    EXPECT_TEST_ALLOCATION_COUNT(1);
    DynamicString s2(&alloc2);
    s2 = std::move(s);
    EXPECT_TEST_ALLOCATION_COUNT(2); // DynamicString was copied using alloc2
    EXPECT_STREQ(kLongStringLiteral, s2.c_str());
    EXPECT_EQ(&alloc2, s2.get_allocator().get_allocator());
  }
  EXPECT_EQ(0, TestAllocator::s_bytes);
  TestAllocator::ResetCounters();

  // Restore default
  TestAllocator::s_compatibleWithOtherInstances = true;
}

TEST(DynamicString, UnorderedContainers) {
  // Verify std::hash<DynamicString> enables use in unordered containers
  {
    std::unordered_set<DynamicString> set;
    set.insert(DynamicString("alpha"));
    set.insert(DynamicString("beta"));
    set.insert(DynamicString("alpha")); // duplicate
    EXPECT_EQ(2u, set.size());
    EXPECT_EQ(1u, set.count(DynamicString("alpha")));
    EXPECT_EQ(1u, set.count(DynamicString("beta")));
    EXPECT_EQ(0u, set.count(DynamicString("gamma")));
    EXPECT_NE(set.find(DynamicString("alpha")), set.end());
    EXPECT_EQ(set.find(DynamicString("gamma")), set.end());
  }

  {
    std::unordered_map<DynamicString, int> map;
    map[DynamicString("one")] = 1;
    map[DynamicString("two")] = 2;
    map[DynamicString("three")] = 3;
    EXPECT_EQ(3u, map.size());
    EXPECT_EQ(1, map[DynamicString("one")]);
    EXPECT_EQ(2, map[DynamicString("two")]);
    EXPECT_EQ(3, map[DynamicString("three")]);
    EXPECT_EQ(1u, map.count(DynamicString("one")));
    EXPECT_EQ(0u, map.count(DynamicString("four")));
  }
}

TEST(DynamicString, Reflection) {
  // Type info
  auto const& ti = SReflect::GetTypeInfo<DynamicString>();
  EXPECT_EQ(SReflect::CoreType::CT_string, ti._coreType);
  EXPECT_STREQ("DynamicString", ti._name);
  EXPECT_STREQ("mochi::DynamicString", ti._nameWithNamespace);
  EXPECT_FALSE(ti._isReadOnly);
  EXPECT_TRUE(ti._isNullTerminated);
  EXPECT_EQ(alignof(DynamicString), ti._alignment);
  EXPECT_EQ(sizeof(DynamicString), ti._sizeInBytes);
  EXPECT_EQ(SReflect::ComputeTypeId(ti._nameWithNamespace), ti._typeId);

  // Non-template factory methods
  DynamicString src = "hello";
  void* pStr = ti.New();
  ti.Set(&src, pStr); // Copy "hello"
  void* pStr2 = ti.Clone(pStr);
  DynamicString dst;
  ti.Set(pStr2, &dst);
  EXPECT_STREQ("hello", dst.c_str());
  ti.Delete(pStr);
  ti.Delete(pStr2);

  // JSON serialization (just like std::string)
  EXPECT_STREQ("\"\"", SReflect::ToJsonString(DynamicString(), /*pretty*/ false).c_str());
  EXPECT_STREQ(
      "\"hello\"", SReflect::ToJsonString(DynamicString("hello"), /*pretty*/ false).c_str());
  std::string longJson = SReflect::ToJsonString(std::string{kLongStringLiteral});
  EXPECT_STREQ(longJson.c_str(), SReflect::ToJsonString(DynamicString{kLongStringLiteral}).c_str());
  EXPECT_STREQ("", SReflect::FromJsonString<DynamicString>("\"\"").c_str());
  EXPECT_STREQ("woot", SReflect::FromJsonString<DynamicString>("\"woot\"").c_str());
  EXPECT_STREQ(
      SReflect::FromJsonString<std::string>(longJson).c_str(),
      SReflect::FromJsonString<DynamicString>(longJson).c_str());

  // Binary serialization (just like std::string)
  {
    DynamicArray<uint8_t> expectedData, actualData;
    DynamicArrayStreamWriter expectedWriter(expectedData), actualWriter(actualData);
    SReflect::ToBytes(std::string{}, expectedWriter);
    SReflect::ToBytes(DynamicString{}, actualWriter);
    EXPECT_EQ(
        expectedData, actualData); // Expected same bytes from std::string and mochi::DynamicString
    SpanStreamReader reader(MakeConstSpan(actualData));
    dst = "nope";
    SReflect::FromBytes(reader, dst);
    EXPECT_STREQ("", dst.c_str());
  }
  {
    DynamicArray<uint8_t> expectedData, actualData;
    DynamicArrayStreamWriter expectedWriter(expectedData), actualWriter(actualData);
    SReflect::ToBytes(std::string{"before"}, expectedWriter);
    SReflect::ToBytes(std::string{kLongStringLiteral}, expectedWriter);
    SReflect::ToBytes(std::string{"after"}, expectedWriter);
    SReflect::ToBytes(DynamicString{"before"}, actualWriter);
    SReflect::ToBytes(DynamicString{kLongStringLiteral}, actualWriter);
    SReflect::ToBytes(DynamicString{"after"}, actualWriter);
    EXPECT_EQ(
        expectedData, actualData); // Expected same bytes from std::string and mochi::DynamicString
    SpanStreamReader reader(MakeConstSpan(actualData));
    dst = "nope";
    SReflect::FromBytes(reader, dst);
    EXPECT_STREQ("before", dst.c_str());
    SReflect::FromBytes(reader, dst);
    EXPECT_STREQ(kLongStringLiteral, dst.c_str());
    SReflect::FromBytes(reader, dst);
    EXPECT_STREQ("after", dst.c_str());
  }
}
