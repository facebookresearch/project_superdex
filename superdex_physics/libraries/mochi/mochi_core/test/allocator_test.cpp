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

#include <mochi_core/memory/allocator.h>
#include <mochi_core/test/mochi_test_helpers.h>

#include <string>
#include <vector>

using namespace mochi;

TEST(Allocator, DefaultAllocator) {
  size_t constexpr kNumSizes = 64;
  size_t constexpr kNumAlignments = 13; // [2^0, 2^12] byte alignment

  auto* allocator = GetDefaultAllocator();
  EXPECT_NE((Allocator*)nullptr, allocator);

  // All instances of DefaultAllocator are "equal" (interoperable)
  EXPECT_TRUE(allocator->is_equal(*allocator));
  EXPECT_TRUE(allocator->is_equal(*GetDefaultAllocator()));

  // Allocate several aligned blocks of memory
  std::vector<void*> ptrs;
  std::vector<size_t> sizes;
  std::vector<size_t> alignments;
  ptrs.reserve(kNumSizes * kNumAlignments);
  for (size_t mult = 0; mult < kNumSizes; ++mult) {
    for (size_t power = 0; power < kNumAlignments; ++power) {
      size_t alignment = Pow(2, power);
      size_t size = alignment * mult;
      void* ptr = allocator->allocate(size, alignment);
      EXPECT_EQ(0, reinterpret_cast<size_t>(ptr) % alignment);
      if (size > 0) {
        EXPECT_NE((void*)nullptr, ptr);
        memset(ptr, 0, size);
      }
      ptrs.push_back(ptr);
      sizes.push_back(size);
      alignments.push_back(alignment);
    }
  }

  // Free them
  for (size_t i = 0; i < ptrs.size(); ++i) {
    allocator->deallocate(ptrs[i], sizes[i], alignments[i]);
  }
}

TEST(Allocator, NewDelete) {
  auto* allocator = GetDefaultAllocator();

  // Default construct an object
  auto* str = New<std::string>(allocator);
  *str = "my string";
  EXPECT_STREQ("my string", str->c_str());
  Delete(allocator, str);

  // Pass arguments to the constructor of an object
  str = New<std::string>(allocator, "this works too");
  EXPECT_STREQ("this works too", str->c_str());
  Delete(allocator, str);
}
