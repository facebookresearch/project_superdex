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

#include <mochi_core/memory/filo_allocator.h>
#include <mochi_core/test/allocator_test_helpers.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/dynamic_array.h>

#include <picojson/picojson.h>

#include <list>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

using namespace mochi;
using namespace mochi::test;

namespace {
// A class with static counters to find out if user-defined types are
// being constructed and copied correctly
class Snoop {
 public:
  inline static int s_defaultConstruct = 0;
  inline static int s_copyConstruct = 0;
  inline static int s_moveConstruct = 0;
  inline static int s_userConstruct = 0;
  inline static int s_copyAssign = 0;
  inline static int s_moveAssign = 0;
  inline static int s_destroy = 0;
  alignas(64) int value = 123;

  Snoop() {
    ++s_defaultConstruct;
  }
  Snoop(Snoop const& rhs) {
    ++s_copyConstruct;
    value = rhs.value;
  }
  Snoop(Snoop&& rhs) noexcept {
    ++s_moveConstruct;
    value = rhs.value;
  }
  Snoop(int val) : value(val) { // arbitrary non-default constructor
    ++s_userConstruct;
  }
  ~Snoop() {
    ++s_destroy;
  }
  Snoop& operator=(Snoop const& rhs) {
    ++s_copyAssign;
    value = rhs.value;
    return *this;
  }
  Snoop& operator=(Snoop&& rhs) noexcept {
    ++s_moveAssign;
    value = rhs.value;
    return *this;
  }
  bool operator==(Snoop const& rhs) const {
    return value == rhs.value;
  }
  bool operator!=(Snoop const& rhs) const {
    return value != rhs.value;
  }
  static void ResetCounters() {
    s_defaultConstruct = s_copyConstruct = s_moveConstruct = s_userConstruct = s_copyAssign =
        s_moveAssign = s_destroy = 0;
  }
  static void ExpectCounters(
      int defaultConstruct,
      int copyConstruct,
      int moveConstruct,
      int userConstruct,
      int copyAssign,
      int moveAssign,
      int destroy) {
    EXPECT_EQ(s_defaultConstruct, defaultConstruct);
    EXPECT_EQ(s_copyConstruct, copyConstruct);
    EXPECT_EQ(s_moveConstruct, moveConstruct);
    EXPECT_EQ(s_userConstruct, userConstruct);
    EXPECT_EQ(s_copyAssign, copyAssign);
    EXPECT_EQ(s_moveAssign, moveAssign);
    EXPECT_EQ(s_destroy, destroy);
  }
};

static_assert(alignof(Snoop) == 64);
static_assert(sizeof(Snoop) == 64);

} // namespace

TEST(DynamicArray, Default) {
  DynamicArray<int> a;
  EXPECT_EQ(0, a.size());
  EXPECT_EQ(0, a.capacity());
  EXPECT_TRUE(a.empty());
  EXPECT_EQ((int*)nullptr, a.data());
  EXPECT_EQ(GetDefaultAllocator(), a.get_allocator());
  a.resize(1, 123);
  EXPECT_EQ(1, a.size());
  EXPECT_EQ(1, a.capacity());
  EXPECT_EQ(123, a[0]);
  EXPECT_FALSE(a.empty());
  EXPECT_NE((int*)nullptr, a.data());
}

TEST(DynamicArray, TestAllocator) {
  TestAllocator::ResetCounters();
  TestAllocator alloc;
  {
    DynamicArray<Snoop> a(&alloc);
    EXPECT_EQ(0, a.size());
    EXPECT_EQ(0, a.capacity());
    EXPECT_TRUE(a.empty());
    EXPECT_EQ((Snoop*)nullptr, a.data());
    EXPECT_EQ(&alloc, a.get_allocator());
    EXPECT_EQ(0, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(0, TestAllocator::s_bytes);
    a.resize(1);
    EXPECT_EQ(1, a.size());
    EXPECT_EQ(1, a.capacity());
    EXPECT_FALSE(a.empty());
    EXPECT_NE((Snoop*)nullptr, a.data());
    EXPECT_EQ(1, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(a.capacity() * sizeof(Snoop), TestAllocator::s_bytes);
    EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastAllocAlignment);
  }
  EXPECT_EQ(1, TestAllocator::s_allocate);
  EXPECT_EQ(1, TestAllocator::s_deallocate);
  EXPECT_EQ(0, TestAllocator::s_bytes);
  EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastDeallocAlignment);
}

TEST(DynamicArray, FiloAllocator) {
  // This test proves that FiloAllocator can be used with DynamicArray.
  // Requires array elements to be moved correctly and destroyed in reverse.

  struct TestObj {
    Allocator* alloc;
    int* ptr = nullptr;
    TestObj(Allocator* alloc_)
        : alloc(alloc_), ptr(static_cast<int*>(alloc->allocate(sizeof(int), alignof(int)))) {
      *ptr = 888;
    }
    TestObj(TestObj const& rhs) : alloc(rhs.alloc) {
      ptr = static_cast<int*>(alloc->allocate(sizeof(int), alignof(int)));
      *ptr = *rhs.ptr;
    }
    TestObj(TestObj&& rhs) noexcept : alloc(rhs.alloc), ptr(rhs.ptr) {
      rhs.ptr = nullptr;
    }
    ~TestObj() {
      if (ptr) {
        alloc->deallocate(ptr, sizeof(int), alignof(int));
        ptr = nullptr;
      }
    }
  };

  constexpr size_t kStackMemSize = 3 * sizeof(TestObj) + 3 * sizeof(TestObj);
  MOCHI_FILO_STACK_ALLOCATOR(alloc, kStackMemSize);

  TestObj obj(&alloc);
  *obj.ptr = 123;

  {
    DynamicArray<TestObj> a(&alloc);

    // Resize and duplicate obj
    a.resize(3, obj);
    EXPECT_EQ(3, a.size());
    EXPECT_EQ(3, a.capacity());
    EXPECT_NE((TestObj*)nullptr, a.data());

    // Each object should have copied the value of (*obj.ptr) into a new allocation
    EXPECT_EQ(123, *a[0].ptr);
    EXPECT_EQ(123, *a[1].ptr);
    EXPECT_EQ(123, *a[2].ptr);
    *a[0].ptr = 234;
    *a[1].ptr = 345;
    *a[2].ptr = 456;
    EXPECT_EQ(234, *a[0].ptr);
    EXPECT_EQ(345, *a[1].ptr);
    EXPECT_EQ(456, *a[2].ptr);

    // The array memory should have come from the bytes stored within the buffer declared by
    // MOCHI_FILO_STACK_ALLOCATOR (naming convention is "stack_memory_for_##name").
    auto allocBegin = reinterpret_cast<size_t>(stack_memory_for_alloc);
    auto allocEnd = allocBegin + sizeof(stack_memory_for_alloc);
    EXPECT_LE(allocBegin, reinterpret_cast<size_t>(a.data()));
    EXPECT_GE(allocEnd, reinterpret_cast<size_t>(a.data()) + 3 * sizeof(TestObj));

    // Expect array elements to be cleaned up in reverse order
  }

  // This time, use emplace_back to add objects.
  {
    DynamicArray<TestObj> a(&alloc);
    for (int i = 0; i < 3; ++i) {
      a.emplace_back(&alloc);
      EXPECT_EQ(888, *a.back().ptr); // default value
      EXPECT_TRUE(i == 0 || (a[i].ptr != a[i - 1].ptr)); // unique pointers
    }
    // Expect everything to clean up nicely
  }
}

TEST(DynamicArray, InitialSize) {
  Snoop::ResetCounters();

  // Default value, default allocator
  {
    DynamicArray<Snoop> a(2);
    EXPECT_EQ(2, a.size());
    EXPECT_EQ(2, a.capacity());
    EXPECT_NE((Snoop*)nullptr, a.data());
    EXPECT_EQ(GetDefaultAllocator(), a.get_allocator());
    Snoop::ExpectCounters(2, 0, 0, 0, 0, 0, 0); // Default construct 2
    EXPECT_EQ(Snoop{}, a[0]);
    EXPECT_EQ(Snoop{}, a[1]);
    a[0].value = 11;
    a[1].value = 22;
    EXPECT_EQ(11, a[0].value);
    EXPECT_EQ(22, a[1].value);
    Snoop::ResetCounters();
  }

  Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 2); // Destroy 2
  Snoop::ResetCounters();
  TestAllocator::ResetCounters();

  // Custom value, default allocator
  {
    DynamicArray<Snoop> a(2, Snoop{555});
    EXPECT_EQ(2, a.size());
    EXPECT_EQ(2, a.capacity());
    EXPECT_NE((Snoop*)nullptr, a.data());
    EXPECT_EQ(GetDefaultAllocator(), a.get_allocator());
    Snoop::ExpectCounters(0, 2, 0, 1, 0, 0, 1); // User construct 1, copy 2, destroy 1
    EXPECT_EQ(Snoop{555}, a[0]);
    EXPECT_EQ(Snoop{555}, a[1]);
    a[0].value = 11;
    a[1].value = 22;
    EXPECT_EQ(11, a[0].value);
    EXPECT_EQ(22, a[1].value);
    Snoop::ResetCounters();
  }

  Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 2); // Destroy 2
  Snoop::ResetCounters();
  TestAllocator::ResetCounters();

  // Default value, custom allocator
  {
    TestAllocator alloc;
    DynamicArray<Snoop> a(2, &alloc);
    EXPECT_EQ(2, a.size());
    EXPECT_EQ(2, a.capacity());
    EXPECT_NE((Snoop*)nullptr, a.data());
    EXPECT_EQ(&alloc, a.get_allocator());
    EXPECT_EQ(1, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(2 * sizeof(Snoop), TestAllocator::s_bytes);
    EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastAllocAlignment);
    Snoop::ExpectCounters(2, 0, 0, 0, 0, 0, 0); // Default construct 2
    EXPECT_EQ(Snoop{}, a[0]);
    EXPECT_EQ(Snoop{}, a[1]);
    a[0].value = 11;
    a[1].value = 22;
    EXPECT_EQ(11, a[0].value);
    EXPECT_EQ(22, a[1].value);
    Snoop::ResetCounters();
  }

  Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 2); // Destroy 2
  Snoop::ResetCounters();
  TestAllocator::ResetCounters();

  // Custom value, custom allocator
  {
    TestAllocator alloc;
    DynamicArray<Snoop> a(2, Snoop{888}, &alloc);
    EXPECT_EQ(2, a.size());
    EXPECT_EQ(2, a.capacity());
    EXPECT_NE((Snoop*)nullptr, a.data());
    EXPECT_EQ(&alloc, a.get_allocator());
    EXPECT_EQ(1, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(2 * sizeof(Snoop), TestAllocator::s_bytes);
    EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastAllocAlignment);
    Snoop::ExpectCounters(0, 2, 0, 1, 0, 0, 1); // User construct 1, copy 2, destroy 1
    EXPECT_EQ(Snoop{888}, a[0]);
    EXPECT_EQ(Snoop{888}, a[1]);
    a[0].value = 11;
    a[1].value = 22;
    EXPECT_EQ(11, a[0].value);
    EXPECT_EQ(22, a[1].value);
    Snoop::ResetCounters();
  }

  Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 2); // Destroy 2
  EXPECT_EQ(1, TestAllocator::s_allocate);
  EXPECT_EQ(1, TestAllocator::s_deallocate);
  EXPECT_EQ(0, TestAllocator::s_bytes);
  EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastAllocAlignment);
  EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastDeallocAlignment);
}

TEST(DynamicArray, ConstructFromRange) {
  // Use iterators from another container, just to show we can.
  std::list<Snoop> inList{Snoop{11}, Snoop{22}, Snoop{33}};
  DynamicArray<Snoop> inArr{Snoop{44}, Snoop{55}, Snoop{66}};
  auto const& cinList = inList; // const access
  auto const& cinArr = inArr; // const access

  TestAllocator alloc;
  TestAllocator::ResetCounters();
  Snoop::ResetCounters();

  // Empty range
  {
    DynamicArray<Snoop> a((Snoop*)nullptr, (Snoop*)nullptr);
    EXPECT_EQ(0, a.size());
    EXPECT_EQ(0, a.capacity());
    DynamicArray<Snoop> b(inList.begin(), inList.begin());
    EXPECT_EQ(0, a.size());
    EXPECT_EQ(0, a.capacity());
    EXPECT_EQ(0, TestAllocator::s_allocate);
    Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 0); // nothing happened
  }

  // Non-empty range of pointers
  {
    DynamicArray<Snoop> a(cinArr.data(), cinArr.data() + cinArr.size(), &alloc);
    EXPECT_EQ(3, a.size());
    EXPECT_EQ(3, a.capacity());
    EXPECT_EQ(44, a[0].value);
    EXPECT_EQ(55, a[1].value);
    EXPECT_EQ(66, a[2].value);
    EXPECT_EQ(1, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(3 * sizeof(Snoop), TestAllocator::s_bytes);
    EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastAllocAlignment);
    Snoop::ExpectCounters(0, 3, 0, 0, 0, 0, 0); // copy construct 3
    Snoop::ResetCounters();
  }

  Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 3); // destroy 3
  Snoop::ResetCounters();
  EXPECT_EQ(1, TestAllocator::s_allocate);
  EXPECT_EQ(1, TestAllocator::s_deallocate);
  EXPECT_EQ(0, TestAllocator::s_bytes);
  EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastDeallocAlignment);
  TestAllocator::ResetCounters();

  // Non-empty range of iterators
  {
    DynamicArray<Snoop> a(cinList.begin(), cinList.end(), &alloc);
    EXPECT_EQ(3, a.size());
    EXPECT_EQ(3, a.capacity());
    EXPECT_EQ(11, a[0].value);
    EXPECT_EQ(22, a[1].value);
    EXPECT_EQ(33, a[2].value);
    EXPECT_EQ(1, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(3 * sizeof(Snoop), TestAllocator::s_bytes);
    EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastAllocAlignment);
    Snoop::ExpectCounters(0, 3, 0, 0, 0, 0, 0); // copy construct 3
    Snoop::ResetCounters();
  }

  Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 3); // destroy 3
  Snoop::ResetCounters();
  EXPECT_EQ(1, TestAllocator::s_allocate);
  EXPECT_EQ(1, TestAllocator::s_deallocate);
  EXPECT_EQ(0, TestAllocator::s_bytes);
  EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastDeallocAlignment);
  TestAllocator::ResetCounters();
}

TEST(DynamicArray, InitializerList) {
  TestAllocator alloc;
  TestAllocator::ResetCounters();
  Snoop::ResetCounters();

  // Default allocator
  {
    DynamicArray<Snoop> a{Snoop{11}, Snoop{22}, Snoop{33}};
    Snoop::ExpectCounters(0, 3, 0, 3, 0, 0, 3); // User construct 3, copy 3, destroy 3
    Snoop::ResetCounters();
    EXPECT_EQ(3, a.size());
    EXPECT_EQ(3, a.capacity());
    EXPECT_EQ(11, a[0].value);
    EXPECT_EQ(22, a[1].value);
    EXPECT_EQ(33, a[2].value);
  }

  Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 3); // Destroy 3

  // Custom allocator
  {
    Snoop::ResetCounters();
    DynamicArray<Snoop> a{{Snoop{11}, Snoop{22}, Snoop{33}}, &alloc};
    Snoop::ExpectCounters(0, 3, 0, 3, 0, 0, 3); // User construct 3, copy 3, destroy 3
    Snoop::ResetCounters();
    EXPECT_EQ(3, a.size());
    EXPECT_EQ(3, a.capacity());
    EXPECT_EQ(11, a[0].value);
    EXPECT_EQ(22, a[1].value);
    EXPECT_EQ(33, a[2].value);
    EXPECT_EQ(1, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(3 * sizeof(Snoop), TestAllocator::s_bytes);
    EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastAllocAlignment);
  }

  Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 3); // Destroy 3
  Snoop::ResetCounters();
  EXPECT_EQ(1, TestAllocator::s_allocate);
  EXPECT_EQ(1, TestAllocator::s_deallocate);
  EXPECT_EQ(0, TestAllocator::s_bytes);
  EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastDeallocAlignment);
  TestAllocator::ResetCounters();
}

TEST(DynamicArray, BeginEnd) {
  DynamicArray<int> a;
  auto const& ca = a;

  EXPECT_EQ(0, ca.size());
  EXPECT_EQ((int*)nullptr, a.data());
  EXPECT_EQ((int*)nullptr, a.begin());
  EXPECT_EQ((int*)nullptr, a.end());
  EXPECT_EQ((int*)nullptr, a.cbegin());
  EXPECT_EQ((int*)nullptr, a.cend());
  EXPECT_EQ((int*)nullptr, ca.data());
  EXPECT_EQ((int*)nullptr, ca.begin());
  EXPECT_EQ((int*)nullptr, ca.end());
  EXPECT_EQ((int*)nullptr, ca.cbegin());
  EXPECT_EQ((int*)nullptr, ca.cend());
  a.resize(3);
  a[0] = 11;
  a[1] = 22;
  a[2] = 33;
  EXPECT_NE((int*)nullptr, a.data());
  EXPECT_NE((int*)nullptr, a.begin());
  EXPECT_NE((int*)nullptr, a.end());
  EXPECT_NE((int*)nullptr, a.cbegin());
  EXPECT_NE((int*)nullptr, a.cend());
  EXPECT_EQ(a.data(), ca.data());
  EXPECT_EQ(a.begin(), ca.begin());
  EXPECT_EQ(a.begin(), ca.cbegin());
  EXPECT_EQ(a.end(), ca.end());
  EXPECT_EQ(a.end(), ca.cend());
  EXPECT_EQ(3, ca.end() - ca.begin());
  EXPECT_EQ(11, ca.begin()[0]);
  EXPECT_EQ(22, ca.begin()[1]);
  EXPECT_EQ(33, ca.begin()[2]);

  // Const DynamicArray gives const iterators (pointers)
  static_assert(!std::is_const_v<std::remove_reference_t<decltype(*a.data())>>);
  static_assert(!std::is_const_v<std::remove_reference_t<decltype(*a.begin())>>);
  static_assert(!std::is_const_v<std::remove_reference_t<decltype(*a.end())>>);
  static_assert(std::is_const_v<std::remove_reference_t<decltype(*ca.data())>>);
  static_assert(std::is_const_v<std::remove_reference_t<decltype(*ca.begin())>>);
  static_assert(std::is_const_v<std::remove_reference_t<decltype(*ca.end())>>);

  // Range-for
  for (int& ref : a) {
    ++ref;
  }
  int i = 1;
  for (int const& cref : ca) {
    EXPECT_EQ(i * 11 + 1, cref);
    ++i;
  }
  i = 1;
  for (int x : ca) {
    EXPECT_EQ(i * 11 + 1, x);
    ++i;
  }
}

TEST(DynamicArray, FrontBack) {
  DynamicArray<int> a;
  auto const& ca = a;

  a.resize(1);
  a[0] = 11;
  EXPECT_EQ(ca.data(), &a.front());
  EXPECT_EQ(ca.data(), &ca.front());
  EXPECT_EQ(ca.data(), &a.back());
  EXPECT_EQ(ca.data(), &ca.back());
  EXPECT_EQ(11, a.front());
  EXPECT_EQ(11, ca.front());
  EXPECT_EQ(11, a.back());
  EXPECT_EQ(11, ca.back());

  a.push_back(22);
  EXPECT_EQ(ca.data(), &a.front());
  EXPECT_EQ(ca.data(), &ca.front());
  EXPECT_EQ(ca.data() + 1, &a.back());
  EXPECT_EQ(ca.data() + 1, &ca.back());
  EXPECT_EQ(11, a.front());
  EXPECT_EQ(11, ca.front());
  EXPECT_EQ(22, a.back());
  EXPECT_EQ(22, ca.back());

  // Const DynamicArray gives const references
  static_assert(!std::is_const_v<std::remove_reference_t<decltype(a.front())>>);
  static_assert(!std::is_const_v<std::remove_reference_t<decltype(a.back())>>);
  static_assert(std::is_const_v<std::remove_reference_t<decltype(ca.front())>>);
  static_assert(std::is_const_v<std::remove_reference_t<decltype(ca.back())>>);
}

TEST(DynamicArray, Reserve) {
  Snoop::ResetCounters();
  TestAllocator::ResetCounters();
  TestAllocator alloc;

  // Reserve while empty
  {
    DynamicArray<Snoop> a(&alloc);
    EXPECT_EQ(0, a.size());
    EXPECT_EQ(0, a.capacity());
    EXPECT_EQ(0, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(0, TestAllocator::s_bytes);
    Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 0);
    a.reserve(3);
    EXPECT_EQ(0, a.size());
    EXPECT_EQ(3, a.capacity());
    EXPECT_EQ(1, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(a.capacity() * sizeof(Snoop), TestAllocator::s_bytes);
    Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 0);
    a.reserve(2); // smaller than capacity
    EXPECT_EQ(0, a.size());
    EXPECT_EQ(3, a.capacity());
    EXPECT_EQ(1, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(a.capacity() * sizeof(Snoop), TestAllocator::s_bytes);
    Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 0);
    auto requestedCapacity = a.capacity() + 1;
    a.reserve(requestedCapacity); // Requires reallocation
    EXPECT_EQ(0, a.size());
    EXPECT_EQ(requestedCapacity, a.capacity());
    EXPECT_EQ(2, TestAllocator::s_allocate);
    EXPECT_EQ(1, TestAllocator::s_deallocate);
    EXPECT_EQ(a.capacity() * sizeof(Snoop), TestAllocator::s_bytes);
    Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 0);
  }
  EXPECT_EQ(2, TestAllocator::s_allocate);
  EXPECT_EQ(2, TestAllocator::s_deallocate);
  EXPECT_EQ(0, TestAllocator::s_bytes);
  TestAllocator::ResetCounters();
  Snoop::ResetCounters();

  // Reserve while not empty
  {
    DynamicArray<Snoop> a(&alloc);
    EXPECT_EQ(0, a.size());
    EXPECT_EQ(0, a.capacity());
    EXPECT_EQ(0, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(0, TestAllocator::s_bytes);
    Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 0);
    a.resize(2); // default construct 2
    EXPECT_EQ(2, a.size());
    EXPECT_LE(2, a.capacity());
    EXPECT_EQ(1, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(a.capacity() * sizeof(Snoop), TestAllocator::s_bytes);
    Snoop::ExpectCounters(2, 0, 0, 0, 0, 0, 0);
    auto* prevPtr = a.data();
    auto prevCapacity = a.capacity();
    a.reserve(1); // no change
    EXPECT_EQ(2, a.size());
    EXPECT_EQ(prevCapacity, a.capacity());
    EXPECT_EQ(1, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(prevCapacity * sizeof(Snoop), TestAllocator::s_bytes);
    Snoop::ExpectCounters(2, 0, 0, 0, 0, 0, 0);
    Snoop::ResetCounters();
    a.reserve(prevCapacity + 1); // New allocation
    EXPECT_EQ(2, a.size());
    EXPECT_EQ(prevCapacity + 1, a.capacity());
    EXPECT_EQ(2, TestAllocator::s_allocate);
    EXPECT_EQ(1, TestAllocator::s_deallocate);
    EXPECT_EQ(a.capacity() * sizeof(Snoop), TestAllocator::s_bytes);
    EXPECT_NE(prevPtr, a.data());
    Snoop::ExpectCounters(0, 0, 2, 0, 0, 0, 2); // move construct, then destroy 2
    Snoop::ResetCounters();
  }
  Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 2); // destroy 2
  EXPECT_EQ(2, TestAllocator::s_allocate);
  EXPECT_EQ(2, TestAllocator::s_deallocate);
  EXPECT_EQ(0, TestAllocator::s_bytes);
}

TEST(DynamicArray, Resize) {
  Snoop::ResetCounters();
  TestAllocator::ResetCounters();
  TestAllocator alloc;
  {
    DynamicArray<Snoop> a(&alloc);
    EXPECT_EQ(0, a.size());
    EXPECT_EQ(0, a.capacity());
    EXPECT_EQ(0, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(0, TestAllocator::s_bytes);
    Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 0);

    // First resize allocates and default constructs 3
    a.resize(3);
    EXPECT_EQ(3, a.size());
    EXPECT_EQ(3, a.capacity());
    EXPECT_EQ(1, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(a.capacity() * sizeof(Snoop), TestAllocator::s_bytes);
    EXPECT_EQ(123, a[0].value); // default value
    EXPECT_EQ(123, a[1].value); // default value
    EXPECT_EQ(123, a[2].value); // default value
    Snoop::ExpectCounters(3, 0, 0, 0, 0, 0, 0);
    Snoop::ResetCounters();

    // Resize down to 2
    a.resize(2);
    EXPECT_EQ(2, a.size());
    EXPECT_EQ(3, a.capacity());
    EXPECT_EQ(1, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(a.capacity() * sizeof(Snoop), TestAllocator::s_bytes);
    EXPECT_EQ(123, a[0].value); // default value
    EXPECT_EQ(123, a[1].value); // default value
    Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 1); // destroy 1
    Snoop::ResetCounters();

    // Resize down to 1 (passing an unnecessary argument)
    Snoop tmp(456);
    Snoop::ExpectCounters(0, 0, 0, 1, 0, 0, 0); // user construct 1
    Snoop::ResetCounters();
    a.resize(1, tmp); // Extra argument is ignored
    EXPECT_EQ(1, a.size());
    EXPECT_EQ(3, a.capacity());
    EXPECT_EQ(1, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(a.capacity() * sizeof(Snoop), TestAllocator::s_bytes);
    EXPECT_EQ(123, a[0].value); // default value
    Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 1); // destroy 1
    Snoop::ResetCounters();

    // Resize back to 3. This time, copy a non-default value.
    a.resize(3, tmp);
    EXPECT_EQ(3, a.size());
    EXPECT_EQ(3, a.capacity());
    EXPECT_EQ(1, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(a.capacity() * sizeof(Snoop), TestAllocator::s_bytes);
    EXPECT_EQ(123, a[0].value); // no change
    EXPECT_EQ(456, a[1].value); // Copied from tmp
    EXPECT_EQ(456, a[2].value); // Copied from tmp
    Snoop::ExpectCounters(0, 2, 0, 0, 0, 0, 0); // copy 2
    Snoop::ResetCounters();

    // Resize to (capacity + 1). This forces existing values to be moved
    // to a new allocation.
    tmp.value = 789;
    auto newSize = a.capacity() + 1;
    a.resize(newSize, tmp);
    EXPECT_EQ(newSize, a.size());
    EXPECT_LE(newSize, a.capacity());
    EXPECT_EQ(2, TestAllocator::s_allocate);
    EXPECT_EQ(1, TestAllocator::s_deallocate);
    EXPECT_EQ(a.capacity() * sizeof(Snoop), TestAllocator::s_bytes);
    EXPECT_EQ(123, a[0].value); // no change
    EXPECT_EQ(456, a[1].value); // no change
    EXPECT_EQ(456, a[2].value); // no change
    for (int i = 3; i < newSize; ++i) {
      EXPECT_EQ(789, a[i].value); // Copied from tmp
    }
    Snoop::ExpectCounters(0, newSize - 3, 3, 0, 0, 0, 3); // move 3, destroy 3, copy new values
    Snoop::ResetCounters();
  }
  EXPECT_EQ(2, TestAllocator::s_allocate);
  EXPECT_EQ(2, TestAllocator::s_deallocate);
  EXPECT_EQ(0, TestAllocator::s_bytes);
}

TEST(DynamicArray, ResizeNoInit) {
  // Chose an int size that happens to be the same size as real so we can more
  // easily test for NaNs.
  using IntT = std::conditional_t<sizeof(real) == sizeof(float), int32_t, int64_t>;
  DynamicArray<IntT> a;
  a.reserve(3);
  a.resize_noinit(1);
  EXPECT_EQ(1, a.size());
  EXPECT_EQ(3, a.capacity());
  a[0] = 11;

  // Writing to a[1] and a[2] is not normally allowed, but it should be safe enough for
  // this test because we reserved the memory up front.
  *(a.data() + 1) = 22;
  *(a.data() + 2) = 33;

  // Now call resize_noinit
  a.resize_noinit(3);
  EXPECT_EQ(3, a.size());
  EXPECT_EQ(3, a.capacity());
  EXPECT_EQ(11, a[0]);
#if MOCHI_DARRAY_DEBUG
  // Expect the memory of the new elements to be filled with NaNs
  real asReal[3];
  memcpy(asReal, a.data() + 1, sizeof(real) * 2);
  EXPECT_FALSE(IsFinite(asReal[0]));
  EXPECT_FALSE(IsFinite(asReal[1]));
  a[1] = 22;
  a[2] = 33;
#else
  // Expect the memory of the new elements to be untouched
  EXPECT_EQ(22, a[1]);
  EXPECT_EQ(33, a[2]);
#endif

  // Shrink
  a.resize_noinit(1);
  EXPECT_EQ(1, a.size());
  EXPECT_EQ(3, a.capacity());
  EXPECT_EQ(11, a[0]);
#if MOCHI_DARRAY_DEBUG
  // Expect the memory of the destroyed elements to be filled with NaNs
  memcpy(asReal, a.data() + 1, sizeof(real) * 2);
  EXPECT_FALSE(IsFinite(asReal[0]));
  EXPECT_FALSE(IsFinite(asReal[1]));
#else
  // Expect the memory of the destroyed elements to be untouched.
  EXPECT_EQ(22, *(a.data() + 1));
  EXPECT_EQ(33, *(a.data() + 2));
#endif

  // In contrast, resize(3) should zero-initialize
  a.resize(3);
  EXPECT_EQ(3, a.size());
  EXPECT_EQ(3, a.capacity());
  EXPECT_EQ(11, a[0]);
  EXPECT_EQ(0, a[1]);
  EXPECT_EQ(0, a[2]);
  a[1] = 44;
  a[2] = 55;

  // Clear
  a.clear();
  EXPECT_EQ(0, a.size());
  EXPECT_EQ(3, a.capacity());
#if MOCHI_DARRAY_DEBUG
  // Expect the memory of the destroyed elements to be filled with NaNs
  memcpy(asReal, a.data(), sizeof(real) * 3);
  EXPECT_FALSE(IsFinite(asReal[0]));
  EXPECT_FALSE(IsFinite(asReal[1]));
  EXPECT_FALSE(IsFinite(asReal[2]));
#else
  // Expect the memory of the destroyed elements to be untouched.
  EXPECT_EQ(11, *(a.data() + 0));
  EXPECT_EQ(44, *(a.data() + 1));
  EXPECT_EQ(55, *(a.data() + 2));
#endif
}

TEST(DynamicArray, Clear) {
  Snoop::ResetCounters();

  DynamicArray<Snoop> a;
  a.resize(3);
  EXPECT_EQ(3, a.size());
  EXPECT_EQ(3, a.capacity());
  Snoop::ExpectCounters(3, 0, 0, 0, 0, 0, 0); // default 3
  Snoop::ResetCounters();
  a.clear();
  EXPECT_EQ(0, a.size());
  EXPECT_EQ(3, a.capacity());
  Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 3); // destroy 3
}

TEST(DynamicArray, PushPopEmplace) {
  Snoop::ResetCounters();
  TestAllocator::ResetCounters();

  TestAllocator alloc;
  DynamicArray<Snoop> a(&alloc);
  a.reserve(6);
  EXPECT_EQ(0, a.size());
  EXPECT_EQ(6, a.capacity());
  EXPECT_EQ(1, TestAllocator::s_allocate);
  EXPECT_EQ(0, TestAllocator::s_deallocate);
  EXPECT_EQ(6 * sizeof(Snoop), TestAllocator::s_bytes);
  EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastAllocAlignment);
  Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 0); // none yet

  Snoop temp(999);
  Snoop::ExpectCounters(0, 0, 0, 1, 0, 0, 0); // user construct 1
  Snoop::ResetCounters();

  // push_back (copy)
  a.push_back(temp);
  EXPECT_EQ(1, a.size());
  EXPECT_EQ(6, a.capacity());
  EXPECT_EQ(1, TestAllocator::s_allocate);
  EXPECT_EQ(0, TestAllocator::s_deallocate);
  EXPECT_EQ(999, a[0].value); // copied from temp
  Snoop::ExpectCounters(0, 1, 0, 0, 0, 0, 0); // copy construct 1
  Snoop::ResetCounters();

  // push_back (move)
  temp.value = 888;
  a.push_back(std::move(temp));
  EXPECT_EQ(2, a.size());
  EXPECT_EQ(6, a.capacity());
  EXPECT_EQ(1, TestAllocator::s_allocate);
  EXPECT_EQ(0, TestAllocator::s_deallocate);
  EXPECT_EQ(999, a[0].value); // no change
  EXPECT_EQ(888, a[1].value); // moved from temp
  Snoop::ExpectCounters(0, 0, 1, 0, 0, 0, 0); // move construct 1
  Snoop::ResetCounters();

  // emplace_back (copy constructor)
  temp.value = 777; // NOLINT(bugprone-use-after-move)
  a.emplace_back(temp);
  EXPECT_EQ(3, a.size());
  EXPECT_EQ(6, a.capacity());
  EXPECT_EQ(1, TestAllocator::s_allocate);
  EXPECT_EQ(0, TestAllocator::s_deallocate);
  EXPECT_EQ(999, a[0].value); // no change
  EXPECT_EQ(888, a[1].value); // no change
  EXPECT_EQ(777, a[2].value); // copied from temp
  Snoop::ExpectCounters(0, 1, 0, 0, 0, 0, 0); // copy construct 1
  Snoop::ResetCounters();

  // emplace_back (move constructor)
  temp.value = 666;
  a.emplace_back(std::move(temp));
  EXPECT_EQ(4, a.size());
  EXPECT_EQ(6, a.capacity());
  EXPECT_EQ(1, TestAllocator::s_allocate);
  EXPECT_EQ(0, TestAllocator::s_deallocate);
  EXPECT_EQ(999, a[0].value); // no change
  EXPECT_EQ(888, a[1].value); // no change
  EXPECT_EQ(777, a[2].value); // no change
  EXPECT_EQ(666, a[3].value); // copied from temp
  Snoop::ExpectCounters(0, 0, 1, 0, 0, 0, 0); // move construct 1
  Snoop::ResetCounters();

  // emplace_back (user constructor)
  a.emplace_back(555);
  EXPECT_EQ(5, a.size());
  EXPECT_EQ(6, a.capacity());
  EXPECT_EQ(1, TestAllocator::s_allocate);
  EXPECT_EQ(0, TestAllocator::s_deallocate);
  EXPECT_EQ(999, a[0].value); // no change
  EXPECT_EQ(888, a[1].value); // no change
  EXPECT_EQ(777, a[2].value); // no change
  EXPECT_EQ(666, a[3].value); // no change
  EXPECT_EQ(555, a[4].value); // copied from temp
  Snoop::ExpectCounters(0, 0, 0, 1, 0, 0, 0); // user construct 1
  Snoop::ResetCounters();

  // push_back (empty)
  auto& newRef = a.push_back();
  EXPECT_EQ(6, a.size());
  EXPECT_EQ(6, a.capacity());
  EXPECT_EQ(1, TestAllocator::s_allocate);
  EXPECT_EQ(0, TestAllocator::s_deallocate);
  EXPECT_EQ(999, a[0].value); // no change
  EXPECT_EQ(888, a[1].value); // no change
  EXPECT_EQ(777, a[2].value); // no change
  EXPECT_EQ(666, a[3].value); // no change
  EXPECT_EQ(555, a[4].value); // copied from temp
  EXPECT_EQ(123, a[5].value); // default value
  EXPECT_EQ(&newRef, &a.back()); // push_back() returned a reference to the new element
  Snoop::ExpectCounters(1, 0, 0, 0, 0, 0, 0); // default construct 1
  Snoop::ResetCounters();

  // pop_back
  a.pop_back();
  EXPECT_EQ(5, a.size());
  EXPECT_EQ(6, a.capacity());
  EXPECT_EQ(1, TestAllocator::s_allocate);
  EXPECT_EQ(0, TestAllocator::s_deallocate);
  EXPECT_EQ(999, a[0].value); // no change
  EXPECT_EQ(888, a[1].value); // no change
  EXPECT_EQ(777, a[2].value); // no change
  EXPECT_EQ(666, a[3].value); // no change
  EXPECT_EQ(555, a[4].value); // no change
  Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 1); // destroy 1
  Snoop::ResetCounters();
}

TEST(DynamicArray, CapacityGrowthByEmplaceBack) {
  // Call emplace_back() repeatedly and record the changes in capacity.
  DynamicArray<int> capacities;
  DynamicArray<int> a;
  auto prevCapacity = (size_t)-1;
  for (int i = 0; i < 100; ++i) {
    a.emplace_back(0);
    if (a.capacity() != prevCapacity) {
      capacities.push_back(static_cast<int>(a.capacity()));
      prevCapacity = a.capacity();
    }
  }

  // For reference, std::vector on Windows grows by: 1, 2, 3, 4, 6, 9, 13, 19, 28, 42, 63, 94, 141
  // DynamicArray favors bigger jumps with fewer reallocations.
  EXPECT_EQ((DynamicArray<int>{8, 12, 18, 27, 40, 60, 90, 135}), capacities);
}

TEST(DynamicArray, CapacityGrowthByAppend) {
  // Call append() to add 1 element at a time. Record changes in capacity.
  DynamicArray<int> capacities;
  DynamicArray<int> a;

  auto prevCapacity = (size_t)-1;
  for (int i = 0; i < 100; ++i) {
    int value = 0;
    a.append(&value, &value + 1);
    if (a.capacity() != prevCapacity) {
      capacities.push_back(static_cast<int>(a.capacity()));
      prevCapacity = a.capacity();
    }
  }

  // Same results as emplace_back (see above)
  EXPECT_EQ((DynamicArray<int>{8, 12, 18, 27, 40, 60, 90, 135}), capacities);
}

TEST(DynamicArray, CapacityGrowthByResize) {
  // Call append() to add 1 element at a time. Record changes in capacity.
  DynamicArray<int> capacities;
  DynamicArray<int> a;
  auto prevCapacity = (size_t)-1;
  for (int i = 0; i < 100; ++i) {
    a.resize(i + 1);
    if (a.capacity() != prevCapacity) {
      capacities.push_back(static_cast<int>(a.capacity()));
      prevCapacity = a.capacity();
    }
  }

  // Same results as emplace_back (see above), except that the first resize is exact.
  EXPECT_EQ((DynamicArray<int>{1, 8, 12, 18, 27, 40, 60, 90, 135}), capacities);
}

TEST(DynamicArray, ArrayOperator) {
  DynamicArray<int> a;
  auto const& ca = a;
  a.push_back(123);
  a.push_back(123);

  EXPECT_EQ(123, a[0]);
  EXPECT_EQ(123, ca[0]);
  EXPECT_EQ(&a[0], &ca[0]);
  a[1] = 234;
  EXPECT_EQ(234, a[1]);
  EXPECT_EQ(234, ca[1]);
  EXPECT_EQ(&a[1], &ca[1]);

  // Const DynamicArray gives const references
  static_assert(!std::is_const_v<std::remove_reference_t<decltype(a[0])>>);
  static_assert(!std::is_const_v<std::remove_reference_t<decltype(a[0])>>);
  static_assert(std::is_const_v<std::remove_reference_t<decltype(ca[0])>>);
  static_assert(std::is_const_v<std::remove_reference_t<decltype(ca[0])>>);
}

TEST(DynamicArray, CopyConstruct) {
  TestAllocator::ResetCounters();
  Snoop::ResetCounters();

  TestAllocator alloc;
  DynamicArray<Snoop> a(&alloc);
  a.resize(1, Snoop(321));
  EXPECT_EQ(1, a.size());
  EXPECT_EQ(1, a.capacity());
  EXPECT_EQ(321, a[0].value);
  EXPECT_EQ(1, TestAllocator::s_allocate);
  EXPECT_EQ(0, TestAllocator::s_deallocate);
  EXPECT_EQ(sizeof(Snoop), TestAllocator::s_bytes);
  EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastAllocAlignment);
  Snoop::ExpectCounters(
      0, 1, 0, 1, 0, 0, 1); // user construct temp, then copy temp, then destroy temp
  Snoop::ResetCounters();

  DynamicArray<Snoop> b(a); // copy construct
  EXPECT_EQ(321, a[0].value); // no change
  EXPECT_NE(a.data(), b.data()); // different addresses
  EXPECT_EQ(1, b.size());
  EXPECT_EQ(1, b.capacity());
  EXPECT_EQ(321, b[0].value); // copied value
  EXPECT_EQ(2, TestAllocator::s_allocate);
  EXPECT_EQ(0, TestAllocator::s_deallocate);
  EXPECT_EQ(2 * sizeof(Snoop), TestAllocator::s_bytes);
  EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastAllocAlignment);
  EXPECT_EQ(a.get_allocator(), b.get_allocator()); // same allocator
  Snoop::ExpectCounters(0, 1, 0, 0, 0, 0, 0); // copy 1
  Snoop::ResetCounters();

  // Copy with a different allocator
  TestAllocator alloc2;
  DynamicArray<Snoop> c(a, &alloc2);
  EXPECT_EQ(321, a[0].value); // no change
  EXPECT_NE(a.data(), c.data()); // different addresses
  EXPECT_EQ(1, c.size());
  EXPECT_EQ(1, c.capacity());
  EXPECT_EQ(321, c[0].value); // copied value
  EXPECT_EQ(3, TestAllocator::s_allocate);
  EXPECT_EQ(0, TestAllocator::s_deallocate);
  EXPECT_EQ(3 * sizeof(Snoop), TestAllocator::s_bytes);
  EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastAllocAlignment);
  EXPECT_EQ(&alloc2, c.get_allocator()); // same allocator
  Snoop::ExpectCounters(0, 1, 0, 0, 0, 0, 0); // copy 1
  Snoop::ResetCounters();

  TestAllocator::ResetCounters();
}

TEST(DynamicArray, CopyConstructFromOtherArray) {
  Snoop::ResetCounters();
  TestAllocator::ResetCounters();
  TestAllocator alloc;

  // Construct from Span<Snoop const> + default allocator
  {
    Snoop a(321);
    EXPECT_EQ(0, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    Snoop::ResetCounters();
    DynamicArray<Snoop> b(Span<Snoop const>(&a, 1));
    EXPECT_EQ(1, b.size());
    EXPECT_EQ(1, b.capacity());
    EXPECT_EQ(321, b[0].value);
    EXPECT_EQ(0, TestAllocator::s_allocate); // no change
    EXPECT_EQ(0, TestAllocator::s_deallocate); // no change
    Snoop::ExpectCounters(0, 1, 0, 0, 0, 0, 0); // Copy construct 1
    Snoop::ResetCounters();
  }
  EXPECT_EQ(0, TestAllocator::s_allocate);
  EXPECT_EQ(0, TestAllocator::s_deallocate);
  TestAllocator::ResetCounters();
  Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 2); // Destroy 2
  Snoop::ResetCounters();

  // Construct from Span<Snoop> + default allocator
  {
    Snoop a(321);
    EXPECT_EQ(0, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    Snoop::ResetCounters();
    DynamicArray<Snoop> b(Span<Snoop>(&a, 1));
    EXPECT_EQ(1, b.size());
    EXPECT_EQ(1, b.capacity());
    EXPECT_EQ(321, b[0].value);
    EXPECT_EQ(0, TestAllocator::s_allocate); // no change
    EXPECT_EQ(0, TestAllocator::s_deallocate); // no change
    Snoop::ExpectCounters(0, 1, 0, 0, 0, 0, 0); // Copy construct 1
    Snoop::ResetCounters();
  }
  EXPECT_EQ(0, TestAllocator::s_allocate);
  EXPECT_EQ(0, TestAllocator::s_deallocate);
  TestAllocator::ResetCounters();
  Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 2); // Destroy 2
  Snoop::ResetCounters();

  // Construct from DynamicArray<Snoop> + custom allocator
  {
    std::vector<Snoop> a;
    a.emplace_back(321);
    EXPECT_EQ(0, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    Snoop::ResetCounters();
    DynamicArray<Snoop> b(a, &alloc);
    EXPECT_EQ(1, b.size());
    EXPECT_EQ(1, b.capacity());
    EXPECT_EQ(321, b[0].value);
    EXPECT_EQ(1, TestAllocator::s_allocate); // 1 allocation for b
    EXPECT_EQ(0, TestAllocator::s_deallocate); // no change
    Snoop::ExpectCounters(0, 1, 0, 0, 0, 0, 0); // Copy construct 1
    Snoop::ResetCounters();
  }
  EXPECT_EQ(1, TestAllocator::s_allocate);
  EXPECT_EQ(1, TestAllocator::s_deallocate);
  Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 2); // Destroy 2
}

TEST(DynamicArray, MoveConstruct) {
  TestAllocator::ResetCounters();
  Snoop::ResetCounters();

  TestAllocator alloc;
  DynamicArray<Snoop> a(&alloc);
  a.resize(1, Snoop(321));
  EXPECT_EQ(1, a.size());
  EXPECT_EQ(1, a.capacity());
  EXPECT_EQ(321, a[0].value);
  EXPECT_EQ(1, TestAllocator::s_allocate);
  EXPECT_EQ(0, TestAllocator::s_deallocate);
  EXPECT_EQ(sizeof(Snoop), TestAllocator::s_bytes);
  EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastAllocAlignment);
  Snoop::ExpectCounters(
      0, 1, 0, 1, 0, 0, 1); // user construct temp, then copy temp, then destroy temp
  Snoop::ResetCounters();
  auto* data = a.data();

  DynamicArray<Snoop> b(std::move(a)); // move construct
  EXPECT_EQ(0, a.size()); // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(0, a.capacity()); // NOLINT(bugprone-use-after-move)
  EXPECT_EQ((Snoop*)nullptr, a.data()); // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(data, b.data()); // previously a.data()
  EXPECT_EQ(1, b.size());
  EXPECT_EQ(1, b.capacity());
  EXPECT_EQ(321, b[0].value);
  EXPECT_EQ(1, TestAllocator::s_allocate);
  EXPECT_EQ(0, TestAllocator::s_deallocate);
  EXPECT_EQ(sizeof(Snoop), TestAllocator::s_bytes);
  EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastAllocAlignment);
  EXPECT_EQ(a.get_allocator(), b.get_allocator()); // same allocator
  Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 0); // Elements unaffected
  Snoop::ResetCounters();

  // Move construct with different-but-compatible allocators
  TestAllocator alloc2;
  TestAllocator::s_compatibleWithOtherInstances = true;
  DynamicArray<Snoop> c(std::move(b), &alloc2); // move construct
  EXPECT_EQ(0, b.size()); // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(0, b.capacity()); // NOLINT(bugprone-use-after-move)
  EXPECT_EQ((Snoop*)nullptr, b.data()); // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(data, c.data()); // previously a.data()
  EXPECT_EQ(1, c.size());
  EXPECT_EQ(1, c.capacity());
  EXPECT_EQ(321, c[0].value);
  EXPECT_EQ(1, TestAllocator::s_allocate);
  EXPECT_EQ(0, TestAllocator::s_deallocate);
  EXPECT_EQ(sizeof(Snoop), TestAllocator::s_bytes);
  EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastAllocAlignment);
  EXPECT_EQ(&alloc2, c.get_allocator()); // same allocator
  Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 0); // Elements unaffected
  Snoop::ResetCounters();

  // Fall back on per-element move for incompatible allocators
  TestAllocator alloc3;
  TestAllocator::s_compatibleWithOtherInstances = false;
  auto const* cdata = c.data();
  DynamicArray<Snoop> d(std::move(c), &alloc3); // move construct
  EXPECT_EQ(0, c.size()); // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(1, c.capacity()); // NOLINT(bugprone-use-after-move)
  EXPECT_NE(cdata, d.data()); // Different pointers
  EXPECT_EQ(1, d.size());
  EXPECT_EQ(1, d.capacity());
  EXPECT_EQ(321, d[0].value);
  EXPECT_EQ(2, TestAllocator::s_allocate);
  EXPECT_EQ(0, TestAllocator::s_deallocate);
  EXPECT_EQ(2 * sizeof(Snoop), TestAllocator::s_bytes);
  EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastAllocAlignment);
  EXPECT_EQ(&alloc3, d.get_allocator()); // same allocator
  Snoop::ExpectCounters(0, 0, 1, 0, 0, 0, 1); // Move construct 1, destroy 1
  Snoop::ResetCounters();

  // Cleanup
  TestAllocator::s_compatibleWithOtherInstances = true;
  TestAllocator::ResetCounters();
}

TEST(DynamicArray, CopyAssign) {
  Snoop::ResetCounters();
  TestAllocator::ResetCounters();
  TestAllocator alloc;
  {
    DynamicArray<Snoop> a(&alloc);
    auto const& ca = a;
    a.reserve(3);
    a.resize(1, Snoop(321));
    Snoop::ResetCounters();

    // Copy assign while b is empty
    DynamicArray<Snoop> b(&alloc);
    b = ca;
    EXPECT_EQ(1, a.size()); // no change
    EXPECT_EQ(321, a[0].value); // no change
    EXPECT_NE(a.data(), b.data()); // different addresses
    EXPECT_EQ(1, b.size());
    EXPECT_EQ(1, b.capacity());
    EXPECT_EQ(321, b[0].value); // copied value
    EXPECT_EQ(2, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(4 * sizeof(Snoop), TestAllocator::s_bytes);
    EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastAllocAlignment);
    Snoop::ExpectCounters(0, 1, 0, 0, 0, 0, 0); // copy 1
    Snoop::ResetCounters();

    // Copy assign while b is non-empty but has sufficient capacity
    a.resize(2);
    a[0] = 432;
    a[1] = 543;
    b.reserve(2);
    EXPECT_EQ(3, TestAllocator::s_allocate);
    EXPECT_EQ(1, TestAllocator::s_deallocate);
    EXPECT_EQ(5 * sizeof(Snoop), TestAllocator::s_bytes);
    EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastAllocAlignment);
    EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastDeallocAlignment);
    Snoop::ResetCounters();
    b = ca; // copy assign
    EXPECT_EQ(2, a.size()); // no change
    EXPECT_EQ(432, a[0].value); // no change
    EXPECT_NE(a.data(), b.data()); // different addresses
    EXPECT_EQ(2, b.size());
    EXPECT_EQ(2, b.capacity());
    EXPECT_EQ(432, b[0].value); // copied value
    EXPECT_EQ(543, b[1].value); // copied value
    EXPECT_EQ(3, TestAllocator::s_allocate); // no change
    EXPECT_EQ(1, TestAllocator::s_deallocate); // no change
    Snoop::ExpectCounters(0, 1, 0, 0, 1, 0, 0); // copy assign 1, copy construct 1

    // Copy assign while b is non-empty and does NOT have sufficient capacity
    a.resize(3);
    a[0] = 987;
    a[1] = 876;
    a[2] = 765;
    Snoop::ResetCounters();
    b = ca; // copy assign
    EXPECT_EQ(3, a.size()); // no change
    EXPECT_EQ(987, a[0].value); // no change
    EXPECT_NE(a.data(), b.data()); // different addresses
    EXPECT_EQ(3, b.size());
    EXPECT_EQ(3, b.capacity());
    EXPECT_EQ(987, b[0].value); // copied value
    EXPECT_EQ(876, b[1].value); // copied value
    EXPECT_EQ(765, b[2].value); // copied value
    EXPECT_EQ(4, TestAllocator::s_allocate); // no change
    EXPECT_EQ(2, TestAllocator::s_deallocate); // no change
    Snoop::ExpectCounters(
        0, 1, 2, 0, 2, 0, 2); // move construct 2, destroy 2, copy assign 2, copy construct 1

    // Copy assign to reduce size by 1
    a.resize(2);
    a[0] = 111;
    a[1] = 222;
    Snoop::ResetCounters();
    b = ca; // copy assign
    EXPECT_EQ(2, a.size()); // no change
    EXPECT_EQ(111, a[0].value); // no change
    EXPECT_NE(a.data(), b.data()); // different addresses
    EXPECT_EQ(2, b.size());
    EXPECT_EQ(3, b.capacity()); // no change
    EXPECT_EQ(111, b[0].value); // copied value
    EXPECT_EQ(222, b[1].value); // copied value
    EXPECT_EQ(4, TestAllocator::s_allocate); // no change
    EXPECT_EQ(2, TestAllocator::s_deallocate); // no change
    Snoop::ExpectCounters(0, 0, 0, 0, 2, 0, 1); // copy assign 2, destroy 1

    // Copy assign to empty
    a.clear();
    Snoop::ResetCounters();
    b = ca; // copy assign
    EXPECT_EQ(0, a.size()); // no change
    EXPECT_EQ(3, a.capacity()); // no change
    EXPECT_NE(a.data(), b.data()); // different addresses
    EXPECT_EQ(0, b.size());
    EXPECT_EQ(3, b.capacity()); // no change
    EXPECT_EQ(4, TestAllocator::s_allocate); // no change
    EXPECT_EQ(2, TestAllocator::s_deallocate); // no change
    Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 2); // destroy 2
    Snoop::ResetCounters();
  }

  // Expect proper cleanup
  Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 0); // no change
  EXPECT_EQ(4, TestAllocator::s_allocate);
  EXPECT_EQ(4, TestAllocator::s_deallocate);
  EXPECT_EQ(0, TestAllocator::s_bytes);
  EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastDeallocAlignment);
  TestAllocator::ResetCounters();

  // Repeat with DynamicArray<int> to test memcpy path (mostly copy/paste)
  {
    DynamicArray<int> a(&alloc);
    auto const& ca = a;
    a.reserve(3);
    a.resize(1, 321);

    // Copy assign while b is empty
    DynamicArray<int> b(&alloc);
    b = ca;
    EXPECT_EQ(1, a.size()); // no change
    EXPECT_EQ(321, a[0]); // no change
    EXPECT_NE(a.data(), b.data()); // different addresses
    EXPECT_EQ(1, b.size());
    EXPECT_EQ(1, b.capacity());
    EXPECT_EQ(321, b[0]); // copied value
    EXPECT_EQ(2, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(4 * sizeof(int), TestAllocator::s_bytes);
    EXPECT_EQ(alignof(int), TestAllocator::s_lastAllocAlignment);

    // Copy assign while b is non-empty but has sufficient capacity
    a.resize(2);
    a[0] = 432;
    a[1] = 543;
    b.reserve(2);
    EXPECT_EQ(3, TestAllocator::s_allocate);
    EXPECT_EQ(1, TestAllocator::s_deallocate);
    EXPECT_EQ(5 * sizeof(int), TestAllocator::s_bytes);
    EXPECT_EQ(alignof(int), TestAllocator::s_lastAllocAlignment);
    EXPECT_EQ(alignof(int), TestAllocator::s_lastDeallocAlignment);
    b = ca; // copy assign
    EXPECT_EQ(2, a.size()); // no change
    EXPECT_EQ(432, a[0]); // no change
    EXPECT_NE(a.data(), b.data()); // different addresses
    EXPECT_EQ(2, b.size());
    EXPECT_EQ(2, b.capacity());
    EXPECT_EQ(432, b[0]); // copied value
    EXPECT_EQ(543, b[1]); // copied value
    EXPECT_EQ(3, TestAllocator::s_allocate); // no change
    EXPECT_EQ(1, TestAllocator::s_deallocate); // no change

    // Copy assign while b is non-empty and does NOT have sufficient capacity
    a.resize(3);
    a[0] = 987;
    a[1] = 876;
    a[2] = 765;
    b = ca; // copy assign
    EXPECT_EQ(3, a.size()); // no change
    EXPECT_EQ(987, a[0]); // no change
    EXPECT_NE(a.data(), b.data()); // different addresses
    EXPECT_EQ(3, b.size());
    EXPECT_EQ(3, b.capacity());
    EXPECT_EQ(987, b[0]); // copied value
    EXPECT_EQ(876, b[1]); // copied value
    EXPECT_EQ(765, b[2]); // copied value
    EXPECT_EQ(4, TestAllocator::s_allocate); // no change
    EXPECT_EQ(2, TestAllocator::s_deallocate); // no change

    // Copy assign to reduce size by 1
    a.resize(2);
    a[0] = 111;
    a[1] = 222;
    b = ca; // copy assign
    EXPECT_EQ(2, a.size()); // no change
    EXPECT_EQ(111, a[0]); // no change
    EXPECT_NE(a.data(), b.data()); // different addresses
    EXPECT_EQ(2, b.size());
    EXPECT_EQ(3, b.capacity()); // no change
    EXPECT_EQ(111, b[0]); // copied value
    EXPECT_EQ(222, b[1]); // copied value
    EXPECT_EQ(4, TestAllocator::s_allocate); // no change
    EXPECT_EQ(2, TestAllocator::s_deallocate); // no change

    // Copy assign to empty
    a.clear();
    b = ca; // copy assign
    EXPECT_EQ(0, a.size()); // no change
    EXPECT_EQ(3, a.capacity()); // no change
    EXPECT_NE(a.data(), b.data()); // different addresses
    EXPECT_EQ(0, b.size());
    EXPECT_EQ(3, b.capacity()); // no change
    EXPECT_EQ(4, TestAllocator::s_allocate); // no change
    EXPECT_EQ(2, TestAllocator::s_deallocate); // no change
  }

  // Expect proper cleanup
  EXPECT_EQ(4, TestAllocator::s_allocate);
  EXPECT_EQ(4, TestAllocator::s_deallocate);
  EXPECT_EQ(0, TestAllocator::s_bytes);
  EXPECT_EQ(alignof(int), TestAllocator::s_lastDeallocAlignment);
}

TEST(DynamicArray, CopyAssignFromOtherArray) {
  Snoop::ResetCounters();
  TestAllocator::ResetCounters();
  TestAllocator alloc;

  {
    DynamicArray<Snoop> a(&alloc);
    DynamicArray<Snoop> b(&alloc);
    a.resize(1, Snoop(321));
    EXPECT_EQ(1, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    Snoop::ResetCounters();

    // Assign from Span<Snoop const>
    b = MakeConstSpan(a);
    EXPECT_EQ(1, b.size());
    EXPECT_EQ(1, b.capacity());
    EXPECT_EQ(2, TestAllocator::s_allocate); // +1 allocation for b
    EXPECT_EQ(0, TestAllocator::s_deallocate); // no change
    Snoop::ExpectCounters(0, 1, 0, 0, 0, 0, 0); // Copy construct 1
    Snoop::ResetCounters();

    // Assign from Span<Snoop>
    b.clear();
    Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 1); // Destroy 1
    Snoop::ResetCounters();
    b = MakeSpan(a);
    EXPECT_EQ(1, b.size());
    EXPECT_EQ(1, b.capacity());
    EXPECT_EQ(2, TestAllocator::s_allocate); // no change
    EXPECT_EQ(0, TestAllocator::s_deallocate); // no change
    Snoop::ExpectCounters(0, 1, 0, 0, 0, 0, 0); // Copy construct 1
    Snoop::ResetCounters();

    // Assign from std::vector<Snoop>
    std::vector<Snoop> c;
    c.emplace_back(123);
    b.clear();
    Snoop::ResetCounters();
    b = c;
    EXPECT_EQ(1, b.size());
    EXPECT_EQ(1, b.capacity());
    EXPECT_EQ(2, TestAllocator::s_allocate); // +1 allocation for b
    EXPECT_EQ(0, TestAllocator::s_deallocate); // no change
    Snoop::ExpectCounters(0, 1, 0, 0, 0, 0, 0); // Copy construct 1
    Snoop::ResetCounters();

    // Assign from std::initializer_list<Snoop>
    auto list = std::initializer_list<Snoop>{Snoop(1), Snoop(2), Snoop(3)};
    b.clear();
    Snoop::ResetCounters();
    b = list;
    EXPECT_EQ(3, b.size());
    EXPECT_EQ(3, b.capacity());
    EXPECT_EQ(3, TestAllocator::s_allocate); // +1 allocation for b
    EXPECT_EQ(1, TestAllocator::s_deallocate); // no change
    Snoop::ExpectCounters(0, 3, 0, 0, 0, 0, 0); // Copy construct 3
    Snoop::ResetCounters();
  }

  EXPECT_EQ(3, TestAllocator::s_allocate); // no change
  EXPECT_EQ(3, TestAllocator::s_deallocate); // a and b deallocated
  Snoop::ExpectCounters(
      0, 0, 0, 0, 0, 0, 8); // Destroy 2 (1 from a, 3 from b, 1 from c, 3 from list)
}

TEST(DynamicArray, CopyAssignRange) {
  Snoop::ResetCounters();
  TestAllocator::ResetCounters();
  TestAllocator alloc;
  {
    DynamicArray<Snoop> a(3, &alloc);
    auto const& ca = a;
    a[0].value = 123;
    a[1].value = 234;
    a[2].value = 345;
    Snoop::ResetCounters();

    // Assign range while b is empty with zero capacity
    DynamicArray<Snoop> b(&alloc);
    b.assign(ca.begin(), ca.begin()); // empty range
    EXPECT_EQ(0, b.size());
    b.assign((Snoop*)nullptr, (Snoop*)nullptr); // empty range
    EXPECT_EQ(0, b.size());
    b.assign(ca.begin(), ca.begin() + 1);
    EXPECT_NE(a.data(), b.data()); // different addresses
    EXPECT_EQ(1, b.size());
    EXPECT_EQ(1, b.capacity());
    EXPECT_EQ(123, b[0].value); // copied value
    EXPECT_EQ(2, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(4 * sizeof(Snoop), TestAllocator::s_bytes);
    EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastAllocAlignment);
    Snoop::ExpectCounters(0, 1, 0, 0, 0, 0, 0); // copy 1
    Snoop::ResetCounters();

    // Assign range while b has sufficient capacity
    b.reserve(2);
    Snoop::ExpectCounters(0, 0, 1, 0, 0, 0, 1); // move construct 1, destroy 1
    Snoop::ResetCounters();
    b.assign(ca.begin(), ca.begin()); // assign empty range
    Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 1); // destroy 1
    Snoop::ResetCounters();
    EXPECT_EQ(0, b.size());
    EXPECT_EQ(2, b.capacity());
    b.assign(ca.begin(), ca.begin() + 1); // assign range of 1
    Snoop::ExpectCounters(0, 1, 0, 0, 0, 0, 0); // copy 1
    Snoop::ResetCounters();
    EXPECT_EQ(1, b.size());
    EXPECT_EQ(2, b.capacity());
    EXPECT_EQ(123, b[0].value); // copied value
    b.assign(ca.begin(), ca.begin() + 2); // assign range of 2
    Snoop::ExpectCounters(0, 1, 0, 0, 1, 0, 0); // assign 1, copy 1
    Snoop::ResetCounters();
    EXPECT_EQ(2, b.size());
    EXPECT_EQ(2, b.capacity());
    EXPECT_EQ(123, b[0].value); // copied value
    EXPECT_EQ(234, b[1].value); // copied value
    EXPECT_EQ(3, TestAllocator::s_allocate);
    EXPECT_EQ(1, TestAllocator::s_deallocate);
    EXPECT_EQ(5 * sizeof(Snoop), TestAllocator::s_bytes);
    EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastAllocAlignment);
    EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastDeallocAlignment);

    // Copy assign while b is non-empty and does NOT have sufficient capacity
    a[0].value = 111;
    a[1].value = 222;
    a[2].value = 333;
    b.assign(ca.begin(), ca.end());
    Snoop::ExpectCounters(
        0, 1, 2, 0, 2, 0, 2); // move construct 2, destroy 2, assign 2, copy construct 1
    Snoop::ResetCounters();
    EXPECT_EQ(3, b.size());
    EXPECT_EQ(3, b.capacity());
    EXPECT_EQ(111, b[0].value); // copied value
    EXPECT_EQ(222, b[1].value); // copied value
    EXPECT_EQ(333, b[2].value); // copied value
    EXPECT_EQ(4, TestAllocator::s_allocate);
    EXPECT_EQ(2, TestAllocator::s_deallocate);

    // Copy assign to reduce size by 1
    a[0].value = 444;
    a[1].value = 555;
    a[2].value = 666;
    b.assign(ca.begin() + 1, ca.end());
    Snoop::ExpectCounters(0, 0, 0, 0, 2, 0, 1); // copy assign 2, destroy 1
    Snoop::ResetCounters();
    EXPECT_EQ(2, b.size());
    EXPECT_EQ(3, b.capacity()); // no change
    EXPECT_EQ(555, b[0].value); // copied value
    EXPECT_EQ(666, b[1].value); // copied value
    EXPECT_EQ(4, TestAllocator::s_allocate); // no change
    EXPECT_EQ(2, TestAllocator::s_deallocate); // no change

    // Copy assign to empty
    b.assign(ca.end(), ca.end());
    Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 2); // destroy 2
    Snoop::ResetCounters();
    EXPECT_EQ(0, b.size());
    EXPECT_EQ(3, b.capacity()); // no change
    EXPECT_EQ(4, TestAllocator::s_allocate); // no change
    EXPECT_EQ(2, TestAllocator::s_deallocate); // no change
  }

  // Expect proper cleanup
  Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 3); // destroy 3 from a
  EXPECT_EQ(4, TestAllocator::s_allocate);
  EXPECT_EQ(4, TestAllocator::s_deallocate);
  EXPECT_EQ(0, TestAllocator::s_bytes);
  EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastDeallocAlignment);
  TestAllocator::ResetCounters();
}

TEST(DynamicArray, MoveAssign) {
  Snoop::ResetCounters();
  TestAllocator::ResetCounters();
  TestAllocator alloc1;
  TestAllocator alloc2;

  // alloc1 and alloc2 should be interchangeable (for now)
  TestAllocator::s_compatibleWithOtherInstances = true;
  EXPECT_TRUE(alloc1.is_equal(alloc1));
  EXPECT_TRUE(alloc1.is_equal(alloc2));
  EXPECT_TRUE(alloc2.is_equal(alloc1));

  // Move assign with compatible allocators while empty
  {
    DynamicArray<Snoop> a(&alloc1);
    DynamicArray<Snoop> b(&alloc2);
    a.reserve(2);
    a.resize(1, Snoop(11));
    auto* data = a.data();
    Snoop::ResetCounters();
    EXPECT_EQ(1, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(2 * sizeof(Snoop), TestAllocator::s_bytes);
    EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastAllocAlignment);
    b = std::move(a);
    EXPECT_EQ(0, a.size()); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(0, a.capacity()); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ((Snoop*)nullptr, a.data()); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(1, b.size()); // moved from a
    EXPECT_EQ(2, b.capacity()); // moved from a
    EXPECT_EQ(data, b.data()); // moved from a
    EXPECT_EQ(11, b[0].value);
    EXPECT_EQ(1, TestAllocator::s_allocate); // no change
    EXPECT_EQ(0, TestAllocator::s_deallocate); // no change
    EXPECT_EQ(&alloc1, a.get_allocator());
    EXPECT_EQ(&alloc2, b.get_allocator());
    Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 0); // no change
  }
  Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 1); // destroy 1
  Snoop::ResetCounters();
  EXPECT_EQ(1, TestAllocator::s_allocate);
  EXPECT_EQ(1, TestAllocator::s_deallocate);
  EXPECT_EQ(0, TestAllocator::s_bytes);
  EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastDeallocAlignment);
  TestAllocator::ResetCounters();

  // Move assign with compatible allocators while not empty
  {
    DynamicArray<Snoop> a(&alloc1);
    DynamicArray<Snoop> b(&alloc2);
    a.reserve(2);
    a.resize(1, Snoop(11));
    b.resize(3, Snoop(22));
    auto* data = a.data();
    Snoop::ResetCounters();
    EXPECT_EQ(2, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(5 * sizeof(Snoop), TestAllocator::s_bytes);
    EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastAllocAlignment);
    b = std::move(a);
    EXPECT_EQ(0, a.size()); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(0, a.capacity()); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ((Snoop*)nullptr, a.data()); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(1, b.size()); // moved from a
    EXPECT_EQ(2, b.capacity()); // moved from a
    EXPECT_EQ(data, b.data()); // moved from a
    EXPECT_EQ(11, b[0].value);
    EXPECT_EQ(2, TestAllocator::s_allocate); // no change
    EXPECT_EQ(1, TestAllocator::s_deallocate); // up by 1
    EXPECT_EQ(2 * sizeof(Snoop), TestAllocator::s_bytes); // down by 3 * sizeof(Snoop)
    EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastAllocAlignment);
    EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastDeallocAlignment);
    EXPECT_EQ(&alloc1, a.get_allocator());
    EXPECT_EQ(&alloc2, b.get_allocator());
    Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 3); // destroy 3
    Snoop::ResetCounters();
  }
  Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 1); // destroy 1
  Snoop::ResetCounters();
  EXPECT_EQ(2, TestAllocator::s_allocate);
  EXPECT_EQ(2, TestAllocator::s_deallocate);
  EXPECT_EQ(0, TestAllocator::s_bytes);
  EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastDeallocAlignment);
  TestAllocator::ResetCounters();

  // Now we specify that allocations from alloc1 cannot be freed by alloc2 and visa versa
  TestAllocator::s_compatibleWithOtherInstances = false;
  EXPECT_TRUE(alloc1.is_equal(alloc1));
  EXPECT_FALSE(alloc1.is_equal(alloc2));
  EXPECT_FALSE(alloc2.is_equal(alloc1));

  // Move assign non-empty rhs to empty lhs with incompatible allocators.
  // This allocates with the lhs allocator, then moves each element, then clears the rhs array.
  {
    DynamicArray<Snoop> a(&alloc1);
    DynamicArray<Snoop> b(&alloc2);
    a.reserve(2);
    a.resize(1, Snoop(11));
    auto const* adata = a.data();
    Snoop::ResetCounters();
    EXPECT_EQ(1, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(2 * sizeof(Snoop), TestAllocator::s_bytes);
    EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastAllocAlignment);
    b = std::move(a);
    EXPECT_EQ(0, a.size()); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(2, a.capacity()); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(1, b.size()); // Based on a.size() before assignment
    EXPECT_EQ(1, b.capacity()); // just enough for the values
    EXPECT_NE(adata, b.data()); // different pointers
    EXPECT_EQ(11, b[0].value);
    EXPECT_EQ(2, TestAllocator::s_allocate); // up by one
    EXPECT_EQ(0, TestAllocator::s_deallocate); // no change
    EXPECT_EQ(3 * sizeof(Snoop), TestAllocator::s_bytes);
    EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastAllocAlignment);
    EXPECT_EQ(&alloc1, a.get_allocator());
    EXPECT_EQ(&alloc2, b.get_allocator());
    Snoop::ExpectCounters(0, 0, 1, 0, 0, 0, 1); // Move construct 1 (in a), destroy 1 (in a)
    Snoop::ResetCounters();
  }
  Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 1); // destroy 1 (in b)
  Snoop::ResetCounters();
  EXPECT_EQ(2, TestAllocator::s_allocate);
  EXPECT_EQ(2, TestAllocator::s_deallocate);
  EXPECT_EQ(0, TestAllocator::s_bytes);
  EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastDeallocAlignment);
  TestAllocator::ResetCounters();

  // Move asign with same incompatible allocator while not empty. This allocates the new size
  // using the lhs allocator, moves each element, and clear the rhs array.
  {
    DynamicArray<Snoop> a(&alloc1);
    DynamicArray<Snoop> b(&alloc2);
    a.reserve(2);
    a.resize(1, Snoop(11));
    b.resize(3, Snoop(22));
    auto* adata = a.data();
    auto* bdata = b.data();
    EXPECT_NE(adata, bdata);
    Snoop::ResetCounters();
    EXPECT_EQ(2, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(5 * sizeof(Snoop), TestAllocator::s_bytes);
    EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastAllocAlignment);
    b = std::move(a);
    EXPECT_EQ(0, a.size()); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(2, a.capacity()); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(1, b.size()); // copied from a
    EXPECT_EQ(3, b.capacity()); // no change
    EXPECT_EQ(bdata, b.data()); // no change
    EXPECT_EQ(11, b[0].value);
    EXPECT_EQ(2, TestAllocator::s_allocate); // no change
    EXPECT_EQ(0, TestAllocator::s_deallocate); // no change
    EXPECT_EQ(&alloc1, a.get_allocator());
    EXPECT_EQ(&alloc2, b.get_allocator());
    Snoop::ExpectCounters(
        0, 0, 1, 0, 0, 0, 4); // Move construct 1 (in b), destroy 4 (1 in a, 3 in b)
    Snoop::ResetCounters();
  }
  Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 1); // destroy 1 (in b)
  Snoop::ResetCounters();
  EXPECT_EQ(2, TestAllocator::s_allocate);
  EXPECT_EQ(2, TestAllocator::s_deallocate);
  EXPECT_EQ(0, TestAllocator::s_bytes);
  EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastDeallocAlignment);
  TestAllocator::ResetCounters();
}

TEST(DynamicArray, AppendRange) {
  TestAllocator alloc;
  DynamicArray<Snoop> a(3, &alloc);
  auto const& ca = a;
  a[0].value = 111;
  a[1].value = 222;
  a[2].value = 333;

  TestAllocator::ResetCounters();
  Snoop::ResetCounters();

  // Append while b is empty
  {
    DynamicArray<Snoop> b(&alloc);
    a.append((Snoop*)nullptr, (Snoop*)nullptr); // empty range
    EXPECT_EQ(0, b.size());
    EXPECT_EQ(0, b.capacity());
    b.append(ca.begin(), ca.begin()); // also empty
    EXPECT_EQ(0, b.size());
    EXPECT_EQ(0, b.capacity());
    b.append(ca.begin(), ca.begin() + 2); // Append 2
    Snoop::ExpectCounters(0, 2, 0, 0, 0, 0, 0); // copy construct 2
    Snoop::ResetCounters();
    EXPECT_EQ(2, b.size());
    EXPECT_LE(2, b.capacity());
    EXPECT_EQ(111, b[0].value);
    EXPECT_EQ(222, b[1].value);
    EXPECT_EQ(1, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(b.capacity() * sizeof(Snoop), TestAllocator::s_bytes);
    EXPECT_EQ(alignof(Snoop), TestAllocator::s_lastAllocAlignment);
    auto bCapacity = b.capacity();
    b.append(ca.end(), ca.end()); // empty range
    EXPECT_EQ(2, b.size());
    EXPECT_EQ(bCapacity, b.capacity()); // no change
    Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 0);
  }

  Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 2); // destroy 2
  Snoop::ResetCounters();
  EXPECT_EQ(1, TestAllocator::s_allocate);
  EXPECT_EQ(1, TestAllocator::s_deallocate);
  EXPECT_EQ(0, TestAllocator::s_bytes);
  TestAllocator::ResetCounters();

  // Append while b has sufficient capacity
  {
    DynamicArray<Snoop> b(&alloc);
    b.reserve(3);
    b.emplace_back(999);
    Snoop::ExpectCounters(0, 0, 0, 1, 0, 0, 0); // user construct 1
    Snoop::ResetCounters();
    b.append(ca.begin() + 1, ca.end());
    Snoop::ExpectCounters(0, 2, 0, 0, 0, 0, 0); // copy consturct 2
    Snoop::ResetCounters();
    EXPECT_EQ(3, b.size());
    EXPECT_EQ(3, b.capacity());
    EXPECT_EQ(999, b[0].value);
    EXPECT_EQ(222, b[1].value);
    EXPECT_EQ(333, b[2].value);
    EXPECT_EQ(1, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(3 * sizeof(Snoop), TestAllocator::s_bytes);
  }

  Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 3); // destroy 3
  Snoop::ResetCounters();
  EXPECT_EQ(1, TestAllocator::s_allocate);
  EXPECT_EQ(1, TestAllocator::s_deallocate);
  EXPECT_EQ(0, TestAllocator::s_bytes);
  TestAllocator::ResetCounters();

  // Append while b does not have sufficient capacity
  {
    DynamicArray<Snoop> b(&alloc);
    b.reserve(3);
    b.emplace_back(999);
    Snoop::ExpectCounters(0, 0, 0, 1, 0, 0, 0); // user construct 1
    Snoop::ResetCounters();
    b.append(ca.begin(), ca.end());
    Snoop::ExpectCounters(0, 3, 1, 0, 0, 0, 1); // move construct 1, destroy 1, copy construct 3
    Snoop::ResetCounters();
    EXPECT_EQ(4, b.size());
    EXPECT_LE(4, b.capacity());
    EXPECT_EQ(999, b[0].value);
    EXPECT_EQ(111, b[1].value);
    EXPECT_EQ(222, b[2].value);
    EXPECT_EQ(333, b[3].value);
    EXPECT_EQ(2, TestAllocator::s_allocate);
    EXPECT_EQ(1, TestAllocator::s_deallocate);
    EXPECT_EQ(b.capacity() * sizeof(Snoop), TestAllocator::s_bytes);
  }

  Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 4); // destroy 4
  Snoop::ResetCounters();
  EXPECT_EQ(2, TestAllocator::s_allocate);
  EXPECT_EQ(2, TestAllocator::s_deallocate);
  EXPECT_EQ(0, TestAllocator::s_bytes);
  TestAllocator::ResetCounters();

  // Append utility function (from container)
  {
    DynamicArray<Snoop> b(&alloc);
    b.reserve(6);
    Snoop::ResetCounters();
    Append(b, a);
    Snoop::ExpectCounters(0, 3, 0, 0, 0, 0, 0); // copy construct 3
    Snoop::ResetCounters();
    Append(b, a);
    Snoop::ExpectCounters(0, 3, 0, 0, 0, 0, 0); // copy construct 3
    Snoop::ResetCounters();
    EXPECT_EQ(6, b.size());
    EXPECT_EQ(111, b[0].value);
    EXPECT_EQ(222, b[1].value);
    EXPECT_EQ(333, b[2].value);
    EXPECT_EQ(111, b[3].value);
    EXPECT_EQ(222, b[4].value);
    EXPECT_EQ(333, b[5].value);
  }

  // Append utility function (from iterator range)
  {
    DynamicArray<Snoop> b(&alloc);
    b.reserve(4);
    Snoop::ResetCounters();
    b.append(a.begin(), a.end());
    Snoop::ExpectCounters(0, 3, 0, 0, 0, 0, 0); // copy construct 3
    Snoop::ResetCounters();
    b.append(a.begin() + 1, a.begin() + 2);
    Snoop::ExpectCounters(0, 1, 0, 0, 0, 0, 0); // copy construct 1
    Snoop::ResetCounters();
    EXPECT_EQ(4, b.size());
    EXPECT_LE(4, b.capacity());
    EXPECT_EQ(111, b[0].value);
    EXPECT_EQ(222, b[1].value);
    EXPECT_EQ(333, b[2].value);
    EXPECT_EQ(222, b[3].value);
  }
}

TEST(DynamicArray, AppendSum) {
  DynamicArray<int> a({1, 2, 3, 4});
  DynamicArray<int> b;
  AppendSum(b, a, 10);
  EXPECT_EQ(4, b.size());
  EXPECT_EQ(11, b[0]);
  EXPECT_EQ(12, b[1]);
  EXPECT_EQ(13, b[2]);
  EXPECT_EQ(14, b[3]);
  b.resize(1);
  AppendSum(b, a, 100);
  EXPECT_EQ(5, b.size());
  EXPECT_EQ(11, b[0]);
  EXPECT_EQ(101, b[1]);
  EXPECT_EQ(102, b[2]);
  EXPECT_EQ(103, b[3]);
  EXPECT_EQ(104, b[4]);
}

TEST(DynamicArray, Comparison) {
  // Arrays of int
  {
    DynamicArray<int> a;
    DynamicArray<int> b;
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
    a.push_back(11);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    b.push_back(0);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    b.back() = 11;
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
    a.push_back(22);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    b.push_back(0);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    b.back() = 22;
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
    a.clear();
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    b.clear();
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
  }

  // Arrays of objects
  {
    DynamicArray<Snoop> a;
    DynamicArray<Snoop> b;
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
    a.emplace_back(11);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    b.emplace_back(0);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    b.back().value = 11;
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
    a.emplace_back(22);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    b.emplace_back(0);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    b.back().value = 22;
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
    a.clear();
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    b.clear();
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
  }
}

TEST(DynamicArray, NonCopyableType) {
  // Default construct
  DynamicArray<std::unique_ptr<int>> a;
  a.push_back(std::make_unique<int>(123));
  EXPECT_EQ(1, a.size());
  EXPECT_EQ(123, *a[0]);

  // Move construct
  DynamicArray<std::unique_ptr<int>> b(std::move(a));
  EXPECT_EQ(0, a.size()); // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(1, b.size());
  EXPECT_EQ(123, *b[0]);

  // Move assign
  DynamicArray<std::unique_ptr<int>> c;
  c = std::move(b);
  EXPECT_EQ(0, b.size()); // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(1, c.size());
  EXPECT_EQ(123, *c[0]);
}

TEST(DynamicArray, Reset) {
  TestAllocator alloc;
  DynamicArray<Snoop> a(&alloc);
  DynamicArray<Snoop> b(&alloc);

  // Reset from non-empty to empty
  a = {Snoop{1}, Snoop{2}};
  Snoop::ResetCounters();
  TestAllocator::ResetCounters();
  a.reset(&alloc);
  EXPECT_EQ(0, a.size());
  EXPECT_EQ(0, a.capacity());
  Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 2); // Destroy 2
  EXPECT_EQ(0, TestAllocator::s_allocate);
  EXPECT_EQ(1, TestAllocator::s_deallocate);

  // Reset from non-empty to copy
  a = {Snoop{1}, Snoop{2}};
  b = {Snoop(3)};
  Snoop::ResetCounters();
  TestAllocator::ResetCounters();
  a.reset(b);
  EXPECT_EQ(1, a.size());
  EXPECT_EQ(1, a.capacity());
  EXPECT_EQ(3, a[0].value);
  Snoop::ExpectCounters(0, 1, 0, 0, 0, 0, 2); // Copy 1, Destroy 2
  EXPECT_EQ(1, TestAllocator::s_allocate);
  EXPECT_EQ(1, TestAllocator::s_deallocate);

  // Reset from non-empty to move
  a = {Snoop{1}, Snoop{2}};
  b = {Snoop(3)};
  Snoop::ResetCounters();
  TestAllocator::ResetCounters();
  a.reset(std::move(b));
  EXPECT_EQ(1, a.size());
  EXPECT_EQ(1, a.capacity());
  EXPECT_EQ(3, a[0].value);
  Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 2); // Destroy 2
  EXPECT_EQ(0, TestAllocator::s_allocate);
  EXPECT_EQ(1, TestAllocator::s_deallocate);

  // Reset from empty to initializer list
  a.reset(&alloc); // clear and free
  auto list = {Snoop(11)};
  Snoop::ResetCounters();
  TestAllocator::ResetCounters();
  a.reset(list, &alloc);
  EXPECT_EQ(1, a.size());
  EXPECT_EQ(1, a.capacity());
  EXPECT_EQ(11, a[0].value);
  Snoop::ExpectCounters(0, 1, 0, 0, 0, 0, 0); // Copy 1
  EXPECT_EQ(1, TestAllocator::s_allocate);
  EXPECT_EQ(0, TestAllocator::s_deallocate);
}

TEST(DynamicArray, Erase) {
  // POD type
  {
    DynamicArray<int> a = {1, 2, 3, 4, 5};
    auto const* ptr = a.data();
    EXPECT_EQ(5, a.capacity());

    a.erase(a.begin());
    EXPECT_EQ((DynamicArray<int>{2, 3, 4, 5}), a);
    EXPECT_EQ(5, a.capacity()); // No change

    a.erase(a.begin() + 1); // middle
    EXPECT_EQ((DynamicArray<int>{2, 4, 5}), a);
    EXPECT_EQ(5, a.capacity()); // No change

    a.erase(a.begin() + 2); // end
    EXPECT_EQ((DynamicArray<int>{2, 4}), a);
    EXPECT_EQ(5, a.capacity()); // No change

    a.erase(a.begin());
    a.erase(a.begin());
    EXPECT_EQ(0, a.size());
    EXPECT_EQ(5, a.capacity()); // No change
    EXPECT_EQ(ptr, a.data()); // No change
  }

  // Snoop
  {
    DynamicArray<Snoop> a{Snoop{1}, Snoop{2}, Snoop{3}, Snoop{4}, Snoop{5}};
    Snoop::ResetCounters();
    auto const* ptr = a.data();
    EXPECT_EQ(5, a.capacity());

    a.erase(a.begin());
    Snoop::ExpectCounters(0, 0, 0, 0, 0, 4, 1); // Move assign 4, destroy 1
    EXPECT_EQ((DynamicArray<Snoop>{Snoop{2}, Snoop{3}, Snoop{4}, Snoop{5}}), a);
    EXPECT_EQ(5, a.capacity()); // No change
    Snoop::ResetCounters();

    a.erase(a.begin() + 1); // middle
    Snoop::ExpectCounters(0, 0, 0, 0, 0, 2, 1); // Move assign 2, destroy 1
    EXPECT_EQ((DynamicArray<Snoop>{Snoop{2}, Snoop{4}, Snoop{5}}), a);
    EXPECT_EQ(5, a.capacity()); // No change
    Snoop::ResetCounters();

    a.erase(a.begin() + 2); // end
    Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 1); // Destroy 1
    EXPECT_EQ((DynamicArray<Snoop>{Snoop{2}, Snoop{4}}), a);
    EXPECT_EQ(5, a.capacity()); // No change
    Snoop::ResetCounters();

    a.erase(a.begin());
    Snoop::ExpectCounters(0, 0, 0, 0, 0, 1, 1); // Move assign 1, destroy 1
    Snoop::ResetCounters();
    a.erase(a.begin());
    Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 1); // Destroy 1
    Snoop::ResetCounters();
    EXPECT_EQ(0, a.size());
    EXPECT_EQ(5, a.capacity()); // No change
    EXPECT_EQ(ptr, a.data()); // No change
  }
}

TEST(DynamicArray, EraseRange) {
  // POD type
  {
    DynamicArray<int> a = {1, 2, 3, 4, 5, 6, 7, 8};
    auto const* ptr = a.data();
    EXPECT_EQ(8, a.capacity());

    a.erase(a.begin(), a.begin());
    EXPECT_EQ((DynamicArray<int>{1, 2, 3, 4, 5, 6, 7, 8}), a); // No change
    EXPECT_EQ(8, a.capacity()); // No change

    a.erase(a.begin(), a.begin() + 2);
    EXPECT_EQ((DynamicArray<int>{3, 4, 5, 6, 7, 8}), a);
    EXPECT_EQ(8, a.capacity()); // No change

    a.erase(a.begin() + 1, a.begin() + 3); // middle
    EXPECT_EQ((DynamicArray<int>{3, 6, 7, 8}), a);
    EXPECT_EQ(8, a.capacity()); // No change

    a.erase(a.begin() + 2, a.end());
    EXPECT_EQ((DynamicArray<int>{3, 6}), a);
    EXPECT_EQ(8, a.capacity()); // No change

    a.erase(a.begin(), a.end());
    EXPECT_EQ(0, a.size());
    EXPECT_EQ(8, a.capacity()); // No change

    a.erase(a.begin(), a.end());
    EXPECT_EQ(0, a.size()); // No change
    EXPECT_EQ(8, a.capacity()); // No change
    EXPECT_EQ(ptr, a.data()); // No change
  }

  // Snoop
  {
    DynamicArray<Snoop> a{
        Snoop{1}, Snoop{2}, Snoop{3}, Snoop{4}, Snoop{5}, Snoop{6}, Snoop{7}, Snoop{8}};
    Snoop::ResetCounters();
    auto const* ptr = a.data();
    EXPECT_EQ(8, a.capacity());

    a.erase(a.begin(), a.begin());
    Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 0); // No operations
    EXPECT_EQ(
        (DynamicArray<Snoop>{
            Snoop{1}, Snoop{2}, Snoop{3}, Snoop{4}, Snoop{5}, Snoop{6}, Snoop{7}, Snoop{8}}),
        a); // No change
    EXPECT_EQ(8, a.capacity()); // No change
    Snoop::ResetCounters();

    a.erase(a.begin(), a.begin() + 2);
    Snoop::ExpectCounters(0, 0, 0, 0, 0, 6, 2); // Move assign 6, destroy 2
    EXPECT_EQ((DynamicArray<Snoop>{Snoop{3}, Snoop{4}, Snoop{5}, Snoop{6}, Snoop{7}, Snoop{8}}), a);
    EXPECT_EQ(8, a.capacity()); // No change
    Snoop::ResetCounters();

    a.erase(a.begin() + 1, a.begin() + 3); // middle
    Snoop::ExpectCounters(0, 0, 0, 0, 0, 3, 2); // Move assign 3, destroy 2
    EXPECT_EQ((DynamicArray<Snoop>{Snoop{3}, Snoop{6}, Snoop{7}, Snoop{8}}), a);
    EXPECT_EQ(8, a.capacity()); // No change
    Snoop::ResetCounters();

    a.erase(a.begin() + 2, a.end());
    Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 2); // Destroy 2
    EXPECT_EQ((DynamicArray<Snoop>{Snoop{3}, Snoop{6}}), a);
    EXPECT_EQ(8, a.capacity()); // No change
    Snoop::ResetCounters();

    a.erase(a.begin(), a.end());
    Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 2); // Destroy 2
    Snoop::ResetCounters();

    a.erase(a.begin(), a.end());
    Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 0); // Nothing
    EXPECT_EQ(0, a.size());
    EXPECT_EQ(8, a.capacity()); // No change
    EXPECT_EQ(ptr, a.data()); // No change
  }
}

TEST(DynamicArray, EraseUnordered) {
  // POD type
  {
    DynamicArray<int> a = {1, 2, 3, 4, 5};
    auto const* ptr = a.data();
    EXPECT_EQ(5, a.capacity());

    a.erase_unordered(a.begin());
    EXPECT_EQ((DynamicArray<int>{5, 2, 3, 4}), a);
    EXPECT_EQ(5, a.capacity()); // No change

    a.erase_unordered(a.begin() + 1); // middle
    EXPECT_EQ((DynamicArray<int>{5, 4, 3}), a);
    EXPECT_EQ(5, a.capacity()); // No change

    a.erase_unordered(a.begin() + 2); // end
    EXPECT_EQ((DynamicArray<int>{5, 4}), a);
    EXPECT_EQ(5, a.capacity()); // No change

    a.erase_unordered(a.begin());
    a.erase_unordered(a.begin());
    EXPECT_EQ(0, a.size());
    EXPECT_EQ(5, a.capacity()); // No change
    EXPECT_EQ(ptr, a.data()); // No change
  }

  // Snoop
  {
    DynamicArray<Snoop> a{Snoop{1}, Snoop{2}, Snoop{3}, Snoop{4}, Snoop{5}};
    Snoop::ResetCounters();
    auto const* ptr = a.data();
    EXPECT_EQ(5, a.capacity());

    a.erase_unordered(a.begin());
    Snoop::ExpectCounters(0, 0, 0, 0, 0, 1, 1); // Move assign 1, destroy 1
    EXPECT_EQ((DynamicArray<Snoop>{Snoop{5}, Snoop{2}, Snoop{3}, Snoop{4}}), a);
    EXPECT_EQ(5, a.capacity()); // No change
    Snoop::ResetCounters();

    a.erase_unordered(a.begin() + 1); // middle
    Snoop::ExpectCounters(0, 0, 0, 0, 0, 1, 1); // Move assign 1, destroy 1
    EXPECT_EQ((DynamicArray<Snoop>{Snoop{5}, Snoop{4}, Snoop{3}}), a);
    EXPECT_EQ(5, a.capacity()); // No change
    Snoop::ResetCounters();

    a.erase_unordered(a.begin() + 2); // end
    Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 1); // Destroy 1
    EXPECT_EQ((DynamicArray<Snoop>{Snoop{5}, Snoop{4}}), a);
    EXPECT_EQ(5, a.capacity()); // No change
    Snoop::ResetCounters();

    a.erase_unordered(a.begin());
    Snoop::ExpectCounters(0, 0, 0, 0, 0, 1, 1); // Move assign 1, destroy 1
    Snoop::ResetCounters();
    a.erase_unordered(a.begin());
    Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 1); // Destroy 1
    Snoop::ResetCounters();
    EXPECT_EQ(0, a.size());
    EXPECT_EQ(5, a.capacity()); // No change
    EXPECT_EQ(ptr, a.data()); // No change
  }
}

TEST(DynamicArray, ShrinkToFit) {
  TestAllocator::ResetCounters();
  TestAllocator alloc;

  // POD type
  {
    DynamicArray<int> a{{1, 2, 3, 4, 5}, &alloc};
    auto const* ptr = a.data();
    EXPECT_EQ(5, a.capacity());
    EXPECT_EQ(1, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(5 * sizeof(int), TestAllocator::s_bytes);

    a.shrink_to_fit();
    EXPECT_EQ(5, a.capacity()); // no change
    EXPECT_EQ(1, TestAllocator::s_allocate); // no change
    EXPECT_EQ(0, TestAllocator::s_deallocate); // no change
    EXPECT_EQ(5 * sizeof(int), TestAllocator::s_bytes); // no change
    EXPECT_EQ((DynamicArray<int>{1, 2, 3, 4, 5}), a); // no change
    EXPECT_EQ(ptr, a.data()); // no change

    a.pop_back();
    a.shrink_to_fit();
    EXPECT_EQ(4, a.capacity());
    EXPECT_EQ(2, TestAllocator::s_allocate);
    EXPECT_EQ(1, TestAllocator::s_deallocate);
    EXPECT_EQ(4 * sizeof(int), TestAllocator::s_bytes);
    EXPECT_EQ((DynamicArray<int>{1, 2, 3, 4}), a);
    EXPECT_NE(ptr, a.data()); // Different address
  }

  EXPECT_EQ(2, TestAllocator::s_allocate);
  EXPECT_EQ(2, TestAllocator::s_deallocate);
  EXPECT_EQ(0, TestAllocator::s_bytes); // No leaks
  TestAllocator::ResetCounters();

  // Snoop
  {
    DynamicArray<Snoop> a{{Snoop{1}, Snoop{2}, Snoop{3}, Snoop{4}, Snoop{5}}, &alloc};
    Snoop::ResetCounters();
    auto const* ptr = a.data();
    EXPECT_EQ(5, a.capacity());
    EXPECT_EQ(1, TestAllocator::s_allocate);
    EXPECT_EQ(0, TestAllocator::s_deallocate);
    EXPECT_EQ(5 * sizeof(Snoop), TestAllocator::s_bytes);

    a.shrink_to_fit();
    EXPECT_EQ(5, a.capacity()); // no change
    EXPECT_EQ(1, TestAllocator::s_allocate); // no change
    EXPECT_EQ(0, TestAllocator::s_deallocate); // no change
    EXPECT_EQ(5 * sizeof(Snoop), TestAllocator::s_bytes); // no change
    Snoop::ExpectCounters(0, 0, 0, 0, 0, 0, 0); // nothing happened
    EXPECT_EQ(
        (DynamicArray<Snoop>{Snoop{1}, Snoop{2}, Snoop{3}, Snoop{4}, Snoop{5}}), a); // no change
    EXPECT_EQ(ptr, a.data()); // no change

    a.pop_back();
    Snoop::ResetCounters();
    a.shrink_to_fit();
    EXPECT_EQ(4, a.capacity());
    EXPECT_EQ(2, TestAllocator::s_allocate);
    EXPECT_EQ(1, TestAllocator::s_deallocate);
    EXPECT_EQ(4 * sizeof(Snoop), TestAllocator::s_bytes);
    Snoop::ExpectCounters(0, 0, 4, 0, 0, 0, 4); // Move construct 4, destroy 4
    EXPECT_EQ((DynamicArray<Snoop>{Snoop{1}, Snoop{2}, Snoop{3}, Snoop{4}}), a);
    EXPECT_NE(ptr, a.data()); // Different address
  }

  EXPECT_EQ(2, TestAllocator::s_allocate);
  EXPECT_EQ(2, TestAllocator::s_deallocate);
  EXPECT_EQ(0, TestAllocator::s_bytes); // No leaks
}

TEST(DynamicArray, IsDynamicArray) {
  static_assert(kIsDynamicArray<DynamicArray<int>>);
  static_assert(kIsDynamicArray<DynamicArray<Snoop>>);
  static_assert(!kIsDynamicArray<std::vector<int>>);
  static_assert(!kIsDynamicArray<std::vector<Snoop>>);
  static_assert(!kIsDynamicArray<int>);
  static_assert(!kIsDynamicArray<Snoop>);
}

TEST(DynamicArray, Reflection) {
  using ArrayType = DynamicArray<int32_t>;
  ArrayType arr;
  arr.push_back(123);
  arr.push_back(456);

  auto const& typeInfo = SReflect::GetTypeInfo<ArrayType>();
  auto const& innerType = SReflect::GetTypeInfo<int32_t>();

  // Serialization
  EXPECT_STREQ("[123,456]", SReflect::ToJsonString(arr, false).c_str());
  EXPECT_EQ(arr, SReflect::FromJsonString<ArrayType>("[123,456]"));

  // Type Introspection
  EXPECT_STREQ("DynamicArray<int32>", typeInfo._name);
  EXPECT_STREQ("mochi::DynamicArray<int32>", typeInfo._nameWithNamespace);
  EXPECT_EQ(SReflect::CoreType::CT_array, typeInfo._coreType);
  EXPECT_EQ(sizeof(ArrayType), typeInfo._sizeInBytes);
  EXPECT_EQ(alignof(ArrayType), typeInfo._alignment);
  EXPECT_EQ(&innerType, typeInfo._innerTypeInfo);
  EXPECT_EQ(2, typeInfo.GetNumElements(&arr));
  EXPECT_EQ(123, innerType.ToUInt64(typeInfo.GetElement(&arr, 0)));
  EXPECT_EQ(456, innerType.ToUInt64(typeInfo.GetElement(&arr, 1)));
  EXPECT_TRUE(typeInfo.CanResize());

  // Factor Creation (does not require compile-time knowledge of ArrayType)
  void* newObj = typeInfo.New();
  MOCHI_DEFER(typeInfo.Delete(newObj));
  picojson::value json = picojson::object();
  typeInfo.Serialize(newObj, json);
  EXPECT_STREQ("[]", json.serialize(false).c_str());

  // We can also modify the array through reflection without compile-time knowledge of ArrayType.
  EXPECT_TRUE(typeInfo.SetNumElements(newObj, 3));
  EXPECT_EQ(3, typeInfo.GetNumElements(newObj));
  typeInfo.Serialize(newObj, json);
  EXPECT_STREQ("[0,0,0]", json.serialize(false).c_str());
  int32_t newVal = 42;
  void* elem = typeInfo.GetElement(newObj, 1);
  innerType.Set(&newVal, elem);
  typeInfo.Serialize(newObj, json);
  EXPECT_STREQ("[0,42,0]", json.serialize(false).c_str());
}
