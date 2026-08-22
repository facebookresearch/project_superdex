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

#include <mochi_core/memory/monotonic_allocator.h>
#include <mochi_core/test/mochi_test_helpers.h>

using namespace mochi;

#define MOCHI_EXPECT_BAD_ALLOC(expression)       \
  {                                              \
    bool badAlloc = false;                       \
    try {                                        \
      [[maybe_unused]] auto result = expression; \
    } catch (std::bad_alloc) {                   \
      badAlloc = true;                           \
    }                                            \
    EXPECT_TRUE(badAlloc);                       \
  }

TEST(MonotonicAllocator, BasicUsage) {
  // Aligned buffer
  constexpr int kMaxAllocatedBytes = 34;
  alignas(16) char buffer[kMaxAllocatedBytes + 1];
  EXPECT_EQ(0, reinterpret_cast<std::intptr_t>(buffer) % 16);

  // Non-aligned pointer
  char* base = buffer + 1;

  MonotonicAllocator mem(base, kMaxAllocatedBytes);

  EXPECT_EQ(buffer + 1, mem.allocate(1, 1));
  EXPECT_EQ(buffer + 2, mem.allocate(1, 1));
  mem.deallocate(buffer + 1, 1, 1); // nop
  EXPECT_EQ(buffer + 3, mem.allocate(4, 1));
  EXPECT_EQ(buffer + 8, mem.allocate(4, 4)); // rounded up for alignment
  EXPECT_EQ(buffer + 12, mem.allocate(4, 4));
  mem.deallocate(buffer + 3, 4, 1); // nop
  EXPECT_EQ(buffer + 16, mem.allocate(1, 1));
  EXPECT_EQ(buffer + 24, mem.allocate(8, 8)); // rounded up for alignment
  EXPECT_EQ(buffer + 32, mem.allocate(1, 1)); // Allocate last byte

  // If you allocate zero bytes, it will return a unique non-null pointer, as per the standard for
  // new/delete. MonotonicAllocator implements this by returning a 1 byte allocation.
  EXPECT_EQ(buffer + 33, mem.allocate(0, 1));
  EXPECT_EQ(buffer + 34, mem.allocate(0, 1));
  mem.deallocate(buffer + 33, 0, 1);
  mem.deallocate(buffer + 34, 0, 1);

  // Deallocate remainder in any order
  mem.deallocate(buffer + 32, 1, 1);
  mem.deallocate(buffer + 24, 8, 8);
  mem.deallocate(buffer + 2, 1, 1);
  mem.deallocate(buffer + 12, 4, 4);
  mem.deallocate(buffer + 16, 1, 1);
  mem.deallocate(buffer + 8, 4, 4);

  // Reset and allocate all bytes again
  mem.Reset();
  EXPECT_EQ(buffer + 1, mem.allocate(32, 1));

  // Let MonotonicAllocator go out-of-scope without deallocating
}

TEST(MonotonicAllocator, OutOfMemory) {
  // Aligned buffer
  alignas(16) char buffer[32 + 1];
  EXPECT_EQ(0, reinterpret_cast<std::intptr_t>(buffer) % 16);

  // Non-aligned pointer
  char* base = buffer + 1;

  MonotonicAllocator mem(base, 32);

  // Try to allocate too much
  MOCHI_EXPECT_BAD_ALLOC(mem.allocate(33, 1));

  // Succeed with one byte less
  EXPECT_EQ(base, mem.allocate(32, 1));

  // Fail when already full
  MOCHI_EXPECT_BAD_ALLOC(mem.allocate(1, 1));

  // Fail due to alignment
  mem.Reset();
  EXPECT_EQ(base, mem.allocate(28, 1)); // succeed
  MOCHI_EXPECT_BAD_ALLOC(mem.allocate(4, 4)); // fail

  // Succeed for same size, but (alignment == 1)
  EXPECT_EQ(base + 28, mem.allocate(4, 1));
}

TEST(MonotonicAllocator, Equality) {
  mochi::MonotonicAllocator mem1(nullptr, 0);
  mochi::MonotonicAllocator mem2(nullptr, 0);
  EXPECT_TRUE(mem1.is_equal(mem1));
  EXPECT_TRUE(mem2.is_equal(mem2));
  EXPECT_FALSE(mem1.is_equal(mem2));
  EXPECT_FALSE(mem2.is_equal(mem1));
  EXPECT_FALSE(mem1.is_equal(*mochi::GetDefaultAllocator()));
}
