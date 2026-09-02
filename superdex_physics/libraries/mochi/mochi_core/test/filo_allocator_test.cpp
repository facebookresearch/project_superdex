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
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/dynamic_array.h>

#include <algorithm>
#include <cstddef>
#include <vector>

using namespace mochi;

static void TestFiloAllocator_SingleByte(FiloAllocator* alloc) {
  alloc->Shrink();
  char* a = static_cast<char*>(alloc->allocate(1, 1));
  char* b = static_cast<char*>(alloc->allocate(1, 1));
  EXPECT_NE(a, (char*)nullptr);
  EXPECT_NE(a, (char*)nullptr);
  EXPECT_NE(a, b);
  *a = 'a';
  *b = 'b';
  EXPECT_EQ('a', *a);
  EXPECT_EQ('b', *b);
  alloc->deallocate(b, 1, 1);
  alloc->deallocate(a, 1, 1);
  char* a2 = static_cast<char*>(alloc->allocate(1, 1));
  char* b2 = static_cast<char*>(alloc->allocate(1, 1));
  EXPECT_EQ(a, a2); // Expect address reuse
  EXPECT_EQ(b, b2); // Expect address reuse
  alloc->deallocate(b, 1, 1);
  alloc->deallocate(a, 1, 1);
}

static void TestFiloAllocator_VariousSizes(FiloAllocator* alloc) {
  alloc->Shrink();
  constexpr int kMaxSize = 25;
  std::vector<void*> ptrs;
  ptrs.reserve(1 + kMaxSize);
  for (int i0 = 0; i0 <= kMaxSize; ++i0) {
    for (int i = 0; i <= kMaxSize; ++i) {
      ptrs.push_back(alloc->allocate((i0 + i) % kMaxSize));
      EXPECT_NE((void*)nullptr, ptrs.back());
    }
    // Expect all addresses to be unique
    auto sortedPtrs = ptrs;
    std::sort(sortedPtrs.begin(), sortedPtrs.end());
    for (int i = 1; i < isize(ptrs); ++i) {
      EXPECT_NE(sortedPtrs[i - 1], sortedPtrs[i]);
    }
    for (int i = kMaxSize; i >= 0; --i) {
      alloc->deallocate(ptrs.back(), (i0 + i) % kMaxSize);
      ptrs.pop_back();
    }
  }
}

static void TestFiloAllocator_VariousAlignments(FiloAllocator* alloc) {
  alloc->Shrink();
  char* a = static_cast<char*>(alloc->allocate(1, 2));
  char* b = static_cast<char*>(alloc->allocate(1, 4));
  char* c = static_cast<char*>(alloc->allocate(1, 8));
  char* d = static_cast<char*>(alloc->allocate(1, 16));
  char* e = static_cast<char*>(alloc->allocate(1, 32));
  char* f = static_cast<char*>(alloc->allocate(1, 64));
  char* g = static_cast<char*>(alloc->allocate(1, 128));
  char* h = static_cast<char*>(alloc->allocate(1, 256));
  EXPECT_EQ(0, reinterpret_cast<size_t>(a) % 2);
  EXPECT_EQ(0, reinterpret_cast<size_t>(b) % 4);
  EXPECT_EQ(0, reinterpret_cast<size_t>(c) % 8);
  EXPECT_EQ(0, reinterpret_cast<size_t>(d) % 16);
  EXPECT_EQ(0, reinterpret_cast<size_t>(e) % 32);
  EXPECT_EQ(0, reinterpret_cast<size_t>(f) % 64);
  EXPECT_EQ(0, reinterpret_cast<size_t>(g) % 128);
  EXPECT_EQ(0, reinterpret_cast<size_t>(h) % 256);
  alloc->deallocate(h, 1, 256);
  alloc->deallocate(g, 1, 128);
  alloc->deallocate(f, 1, 64);
  alloc->deallocate(e, 1, 32);
  alloc->deallocate(d, 1, 16);
  alloc->deallocate(c, 1, 8);
  alloc->deallocate(b, 1, 4);
  alloc->deallocate(a, 1, 2);
}

static void TestFiloAllocator_PageOverflow(FiloAllocator* alloc) {
  alloc->Shrink();
  // Make many small allocations with various alignment.
  std::vector<void*> ptrs;
  for (int align = 1; align <= FiloAllocator::kMaxAlignment; align <<= 1) {
    // Repeat enough times to overflow the page size multiple times
    int count = 2 * (int)alloc->GetBytesRemainingInPage() / (8 * align);
    ptrs.reserve(count * 2);
    for (int i = 0; i < count; ++i) {
      ptrs.push_back(alloc->allocate(1, 1));
      ptrs.push_back(alloc->allocate(8 * align, align));
      EXPECT_EQ(0, reinterpret_cast<size_t>(ptrs.back()) % align);
    }
    while (!ptrs.empty()) {
      alloc->deallocate(ptrs.back(), 8 * align, align);
      ptrs.pop_back();
      alloc->deallocate(ptrs.back(), 1, 1);
      ptrs.pop_back();
    }
    ptrs.clear();
  }
}

static void TestFiloAllocator_FreePages(FiloAllocator* alloc) {
  alloc->Shrink();
  // First small allocation ensures that a valid page will exist
  auto* a = alloc->allocate(1, 1);

  // Fill the rest of the page
  auto bytesRemaining = alloc->GetBytesRemainingInPage();
  EXPECT_LT(0, bytesRemaining);
  auto bSize = bytesRemaining & ~(FiloAllocator::kMinAlignment - 1); // Fill page once aligned
  EXPECT_LT(0, bSize);
  auto* b = alloc->allocate(bSize, 1);
  EXPECT_NE(a, b);
  EXPECT_EQ(0, alloc->GetBytesRemainingInPage());

  // The next small allocation should create a new page
  auto* c = alloc->allocate(1, 1);
  EXPECT_NE(a, c);
  EXPECT_NE(b, c);
  EXPECT_LT(0, alloc->GetBytesRemainingInPage());

  // When we release 'c', the page should be empty again
  alloc->deallocate(c, 1, 1);

  // When we release again, a page should go onto the free list
  alloc->deallocate(b, bSize, 1);

  // Allocate again to pull it from the free list
  b = alloc->allocate(bSize, 1);
  EXPECT_EQ(0, alloc->GetBytesRemainingInPage());
  c = alloc->allocate(1, 1);
  EXPECT_LT(0, alloc->GetBytesRemainingInPage());

  // Make an allocation that will require a larger-than-usual page
  auto bigSize = std::max(alloc->GetNextPageSize() * 2, alloc->GetBytesRemainingInPage() * 2);
  auto* d = alloc->allocate(bigSize, 1);
  EXPECT_NE(a, d);
  EXPECT_NE(b, d);
  EXPECT_NE(c, d);

  // Repeat
  auto* e = alloc->allocate(bigSize, 1);
  EXPECT_NE(d, e);

  // Release all
  alloc->deallocate(e, bigSize, 1);
  alloc->deallocate(d, bigSize, 1);
  alloc->deallocate(c, 1, 1);
  alloc->deallocate(b, bSize, 1);
  alloc->deallocate(a, 1, 1);

  // Immediately make a large allocation. The head of the free list won't be large enough, so it
  // will pull one from the middle of the list.
  a = alloc->allocate(bigSize, 1);
  memset(a, 0xAA, bigSize);

  // Repeat to pull a page from the end of the free list
  b = alloc->allocate(bigSize, 1);
  memset(b, 0xBB, bigSize);

  // Do several more allocations to empty the free list and start allocating new pages.
  std::vector<void*> ptrs;
  for (int i = 0; i < 1000; ++i) {
    auto size = (i + 100) % 255;
    ptrs.push_back(alloc->allocate(size, 1));
    memset(ptrs.back(), size, size);
  }

  // Make sure none of the memory was corrupted.
  for (int i = 0; i < bigSize; ++i) {
    EXPECT_EQ(0xAA, static_cast<uint8_t*>(a)[i]);
    EXPECT_EQ(0xBB, static_cast<uint8_t*>(b)[i]);
  }
  for (int i = 0; i < 1000; ++i) {
    auto size = (i + 100) % 255;
    for (int j = 0; j < size; ++j) {
      EXPECT_EQ(size, static_cast<uint8_t*>(ptrs[i])[j]);
    }
  }

  // Cleanpu
  for (int i = 1000 - 1; i >= 0; --i) {
    auto size = (i + 100) % 255;
    alloc->deallocate(ptrs.back(), size, 1);
    ptrs.pop_back();
  }
  EXPECT_EQ(0, ptrs.size());
  alloc->deallocate(b, bigSize, 1);
  alloc->deallocate(a, bigSize, 1);
}

static void TestFiloAllocator(FiloAllocator* alloc) {
  TestFiloAllocator_SingleByte(alloc);
  TestFiloAllocator_FreePages(alloc);
  TestFiloAllocator_VariousSizes(alloc);
  TestFiloAllocator_VariousAlignments(alloc);
  TestFiloAllocator_PageOverflow(alloc);
}

TEST(FiloAllocator, Default) {
  // Default construct and destroy without allocating anything
  {
    FiloAllocator alloc;
    EXPECT_EQ(0, alloc.GetBytesRemainingInPage()); // Does not allocate initially
    EXPECT_EQ(FiloAllocator::kDefaultPageSize, alloc.GetNextPageSize());
  }

  // Default construct and run tests
  {
    FiloAllocator a;
    TestFiloAllocator(&a);
  }
}

TEST(FiloAllocator, PageSizes) {
  // Various dynamic page sizes
  constexpr size_t kPageSizes[] = {63, 64, 65, 128, 500, 999};
  for (auto sz : kPageSizes) {
    FiloAllocator alloc;
    alloc.SetNextPageSize(sz);
    EXPECT_LE(sz, alloc.GetNextPageSize());
    TestFiloAllocator(&alloc);
  }
}

TEST(FiloAllocator, UserMemory) {
  // Various sizes of user-provided memory for the first page.
  // For this constructor, the size must e a multiple of kMinAlignment.
  constexpr size_t kPageSizes[] = {48, 64, 128, 400, 800};
  for (auto sz : kPageSizes) {
    std::vector<std::byte> memory(sz);
    FiloAllocator alloc(memory.data(), memory.size());
    TestFiloAllocator(&alloc);
  }
}

// clang-format really wants to mess up the formatting of this function for some reason.
// clang-format off
TEST(FiloAllocator, StackMemory) {
  // Various sizes of stack memory to use via MOCHI_FILO_STACK_ALLOCATOR.
  // Will be rounded to a multiple of kMinAlignment, so any size value is OK.
  {
    MOCHI_FILO_STACK_ALLOCATOR(alloc, 64);
    TestFiloAllocator(&alloc);
  }
  {
    MOCHI_FILO_STACK_ALLOCATOR(alloc, 1000);
    TestFiloAllocator(&alloc);
  }
  {
    MOCHI_FILO_STACK_ALLOCATOR(alloc, 5555);
    TestFiloAllocator(&alloc);
  }

  // Verify that we can allocate the exact amount requested without needing heap memory.
  {
    constexpr auto kAllocSize = 123; // Arbitrary
    constexpr auto kAlignment = FiloAllocator::kMinAlignment;
    MOCHI_FILO_STACK_ALLOCATOR(alloc, kAllocSize);

    // Allocate memory with an explicit alignment. The default alignment argument for
    // Allocator::allocate is sizeof(std::max_align_t) which is 8 bytes for almost
    // all build configuration. Unfortuantely, it is 16 bytes for Linux TSAN. That would fail the
    // test because 16 byte alignment requires a bit of extra book keeping in the FiloAllocator
    // implementation.
    auto* ptr = static_cast<std::byte*>(alloc.allocate(kAllocSize, kAlignment));

    // Verify that the memory came from within the buffer (on the stack)
    auto const* memoryBegin =
        stack_memory_for_alloc; // Buffer declared by MOCHI_FILO_STACK_ALLOCATOR
    auto const* memoryEnd = memoryBegin + sizeof(stack_memory_for_alloc);
    EXPECT_LE(memoryBegin, ptr);
    EXPECT_GE(memoryEnd, ptr + kAllocSize);

    // Cleanup
    alloc.deallocate(ptr, kAllocSize, kAlignment);

    // GCC is afraid that ~FiloAllocator() might free an address that didn't come from the heap.
    // GCC is wrong in this case.
    MOCHI_WARNING_PUSH()
    MOCHI_WARNING_IGNORE_GCC_CLANG(GCC diagnostic ignored "-Wfree-nonheap-object")
  }
  MOCHI_WARNING_POP()
}
// clang-format on

#if MOCHI_ALLOCATOR_DEBUG
TEST(FiloAllocator, GetMaxBytesUsed) {
  DynamicArray<std::byte> mem(32);
  FiloAllocator alloc(mem.data(), mem.size());
  EXPECT_EQ(0, alloc.GetMaxBytesUsed());

  // Allocate part of the initial buffer
  auto* p1 = alloc.allocate(17, 1);
  size_t expectedValue = RoundUp((size_t)17, FiloAllocator::kMinAlignment);
  EXPECT_EQ(expectedValue, alloc.GetMaxBytesUsed());

  // Allocate a little more
  auto* p2 = alloc.allocate(4, 4);
  expectedValue += RoundUp((size_t)4, FiloAllocator::kMinAlignment);
  EXPECT_EQ(expectedValue, alloc.GetMaxBytesUsed());

  // Free p2
  alloc.deallocate(p2, 4, 4);
  EXPECT_EQ(expectedValue, alloc.GetMaxBytesUsed()); // no change

  // Reallocate p2
  p2 = alloc.allocate(4, 4);
  EXPECT_EQ(expectedValue, alloc.GetMaxBytesUsed()); // same high water mark

  // Allocate something that will require a new page
  auto* p3 = alloc.allocate(32, 1);
  expectedValue += RoundUp((size_t)32, FiloAllocator::kMinAlignment);
  EXPECT_EQ(expectedValue, alloc.GetMaxBytesUsed()); // Does not include page header

  // Cleanup
  alloc.deallocate(p3, 32, 1);
  alloc.deallocate(p2, 4, 4);
  alloc.deallocate(p1, 17, 1);
  EXPECT_EQ(expectedValue, alloc.GetMaxBytesUsed()); // no change
}
#endif // MOCHI_ALLOCATOR_DEBUG
