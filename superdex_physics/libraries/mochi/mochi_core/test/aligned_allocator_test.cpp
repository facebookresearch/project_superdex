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

#include <mochi_core/memory/aligned_allocator.h>
#include <mochi_core/memory/filo_allocator.h>
#include <mochi_core/test/mochi_test_helpers.h>

#include <cstdint>
#include <cstring>

using namespace mochi;

template <size_t kAlignment>
static void TestAlignedAllocator() {
  AlignedAllocator<kAlignment> alloc;
  size_t const sizes[] = {0, 1, 2, 7, 13, 16, 31, 32, 33, 64, 100, 128, 255, 256, 1000};
  size_t const alignments[] = {1, 2, 4, 8, 16, 32, 64, 128};
  for (auto size : sizes) {
    for (auto alignment : alignments) {
      size_t const allocatedSize = (size + alignment - 1) & ~(alignment - 1);
      void* ptr = alloc.allocate(allocatedSize, alignment);
      EXPECT_NE(nullptr, ptr);
      size_t const effectiveAlignment = Max(alignment, kAlignment);
      EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(ptr) % effectiveAlignment)
          << "kAlignment=" << kAlignment << " alignment=" << alignment << " size=" << size;
      size_t const memsetSize = (allocatedSize + kAlignment - 1) & ~(kAlignment - 1);
      memset(ptr, 0xAB, memsetSize);
      alloc.deallocate(ptr, allocatedSize, alignment);
    }
  }
}

TEST(AlignedAllocator, DefaultConstructor) {
  AlignedAllocator<64> alloc;
  void* ptr = alloc.allocate(128);
  EXPECT_NE(nullptr, ptr);
  EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(ptr) % 64);
  memset(ptr, 0, 128);
  alloc.deallocate(ptr, 128);
}

TEST(AlignedAllocator, ExplicitConstructor) {
  AlignedAllocator<64> alloc(GetDefaultAllocator());
  void* ptr = alloc.allocate(64);
  EXPECT_NE(nullptr, ptr);
  EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(ptr) % 64);
  memset(ptr, 0, 64);
  alloc.deallocate(ptr, 64);
}

TEST(AlignedAllocator, AlignmentGuarantee) {
  TestAlignedAllocator<1>();
  TestAlignedAllocator<2>();
  TestAlignedAllocator<4>();
  TestAlignedAllocator<8>();
  TestAlignedAllocator<16>();
  TestAlignedAllocator<32>();
  TestAlignedAllocator<64>();
  TestAlignedAllocator<128>();
}

TEST(AlignedAllocator, Equality) {
  // Same alignment, same inner allocator
  AlignedAllocator<64> a;
  AlignedAllocator<64> b;
  EXPECT_TRUE(a.is_equal(b));
  EXPECT_TRUE(b.is_equal(a));

  // Different alignment
  AlignedAllocator<32> c;
  EXPECT_FALSE(a.is_equal(c));
  EXPECT_FALSE(c.is_equal(a));

  // Same alignment, different inner allocator
  FiloAllocator filo;
  AlignedAllocator<64> d(&filo);
  EXPECT_FALSE(a.is_equal(d));
  EXPECT_FALSE(d.is_equal(a));

  // AlignedAllocator vs plain DefaultAllocator
  EXPECT_FALSE(a.is_equal(*GetDefaultAllocator()));
  EXPECT_FALSE(GetDefaultAllocator()->is_equal(a));
}

TEST(AlignedAllocator, MultipleAllocations) {
  AlignedAllocator<64> alloc;
  size_t constexpr kNumAllocs = 16;
  void* ptrs[kNumAllocs];
  size_t const sizes[kNumAllocs] = {1, 2, 4, 8, 16, 32, 64, 128, 3, 7, 13, 33, 65, 100, 255, 512};

  for (size_t i = 0; i < kNumAllocs; ++i) {
    ptrs[i] = alloc.allocate(sizes[i]);
    EXPECT_NE(nullptr, ptrs[i]);
    EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(ptrs[i]) % 64);
    memset(ptrs[i], static_cast<int>(i & 0xFF), sizes[i]);
  }

  for (size_t i = 0; i < kNumAllocs; ++i) {
    alloc.deallocate(ptrs[i], sizes[i]);
  }
}
