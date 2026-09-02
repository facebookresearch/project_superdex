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

#pragma once

#include <mochi_core/memory/allocator.h>
#include <mochi_core/utils/debug.h>

#include <cstddef>
#include <cstring>

#if MOCHI_ALLOCATOR_DEBUG
#include <vector>
#endif // MOCHI_ALLOCATOR_DEBUG

namespace mochi {

/*********************************************************************************************
  FiloAllocator
*/

/**
 * @brief Fast FIRST-IN-LAST-OUT allocator intened for temporary memory allocations.
 *
 * @note Allocations MUST be freed in reverse order (e.g. at the end of the current scope).
 * @note Memory usage may grow to the high water mark unless you call Shrink().
 */
class FiloAllocator final : public Allocator {
  MOCHI_DECLARE_NO_COPY(FiloAllocator);

  struct Header {
    Header* prev; // Linked list of page headers
    size_t offset; // Next allocation comes from address (Data() + offset)
    size_t size; // Usable page size (bytes). Page alocation size is (size + sizeof(Header)).
    std::byte* Data() {
      // Page data always comes immediately after the header
      return reinterpret_cast<std::byte*>(this) + sizeof(Header);
    }
  };

 public:
  static constexpr size_t kMinAlignment = 8;
  static constexpr size_t kMaxAlignment = 256;
  static constexpr size_t kMinPageSize = kMinAlignment;
  static constexpr size_t kHeaderSize = sizeof(Header);
  static constexpr size_t kMinBufferSize = kMinAlignment + kHeaderSize;
  static constexpr size_t kDefaultPageSize = 4096 - sizeof(Header); // 4 KiB with header

  /**
   * @brief Construct a new FiloAllocator object with the default page size.
   */
  FiloAllocator() = default;

  /**
   * @brief Construct a new FiloAllocator object using an existing memory buffer for the first page.
   *
   * The actual number of bytes available for allocation may be less due to booking overhead and
   * alignment. If additional pages are required, then they will be allocated from the heap.
   *
   * @param userMem Address of the caller's memory buffer.
   * @param userMemSize Size of the caller's memory buffer in bytes.
   *
   * @note The memory buffer must outlive this FiloAllocator
   * @note The memory buffer address must be aligned to at least kMinAlignment.
   * @note The memory buffer size must be a non-zero multiple of kMinAlignment and >=
   * kMinBufferSize.
   *
   * @see MOCHI_FILO_STACK_ALLOCATOR
   */
  FiloAllocator(void* userMem, size_t userMemSize);

  /**
   * @brief Destroy the FiloAllocator object and free any memory it allocated from the heap.
   */
  ~FiloAllocator() override;

  /**
   * @brief When new heap memory is needed, a new page will be allocated. Set the minimum size of
   * that page. Value will be rounded up to a multiple of kMinAlignment. The actual size may be
   * larger if the user requests a large contiguous allocation.
   *
   * @param newPageSize Size in bytes
   */
  void SetNextPageSize(size_t newPageSize);

  /**
   * @brief Get the minimum size of the next page.
   *
   * @return Size in bytes
   *
   * @see SetNextPageSize
   */
  [[nodiscard]] size_t GetNextPageSize() const noexcept;

  /**
   * @brief Release any unused pages that were allocated from the heap.
   */
  void Shrink();

  /**
   * @brief Return the number of bytes still available in the current page (for debugging)
   *
   * @return Size in ytes
   */
  [[nodiscard]] size_t GetBytesRemainingInPage() const;

#if MOCHI_ALLOCATOR_DEBUG
  /**
   * @brief Return the maximum number of bytes used to satisfy allocation requests.
   *
   * @remarks If you constructed the FiloAllocator with a pre-allocated buffer of at least this
   * size, then no additional heap memory would be required. The value may be larger than the sum of
   * allocation requests because of alignment and overhead. It may be less than the total memory
   * allocated from the system because some capacity may have remained unused.
   *
   * @return Size in bytes
   */
  [[nodiscard]] size_t GetMaxBytesUsed() const;
#endif

 private:
  static_assert(
      sizeof(Header) % kMinAlignment == 0,
      "Header size must be a multiple of the minimum guaranteed alignment.");
  static constexpr size_t kAlignmentSuffixSize = 3; // bytes

  template <typename T>
  [[nodiscard]] T AlignUp(T val, size_t alignment);
  void PushPage(size_t size);
  void PopPage();
  static void FreePage(Header* page);

  // Allocator overrides:
  void* do_allocate(std::size_t sizeInBytes, std::size_t alignment) final;
  void do_deallocate(void* ptr, std::size_t sizeInBytes, std::size_t alignment) final;
  bool do_is_equal(Allocator const& other) const noexcept final;

  inline static constexpr Header s_empty = {};
  size_t _nextPageSize = kDefaultPageSize; // Usable size of next page (not including header size)
  Header* _current = const_cast<Header*>(&s_empty); // Linked list of current & previous pages
  Header* _free = nullptr; // Linked list of free pages that can be reused

#if MOCHI_ALLOCATOR_DEBUG
  // Extra diagnostics for debug builds
  struct DebugInfo {
    void* ptr = nullptr;
    size_t size = 0;
    size_t alignment = 0;
    size_t prevOffset = 0;
  };
  // Using std::vector here, not DynamicArray, so it won't trigger breakpoints in our DynamicArray
  // or DebugAllocator code.
  std::vector<DebugInfo> _debug;
  size_t _currentBytesUsed = 0;
  size_t _maxBytesUsed = 0;
#endif // MOCHI_ALLOCATOR_DEBUG
};

/*********************************************************************************************
  MOCHI_FILO_STACK_ALLOCATOR
*/

/**
 * @brief Declare a FiloAllocator as a local variable on the stack using the specified number of
 * bytes (must be constexpr) of stack memory for the first page. You will then be able to allocate
 * up to that number of bytes without requiring any heap allocation.
 *
 * Exception: If your allocations will require greater than default alignment (e.g. for SIMD types),
 * then you will need to increase the sizeInBytes parameter to account for alignment and some extra
 * book keeping.
 *
 * @param variableName The name of the local variable to declare on the stack.
 * @param sizeInBytes The number of bytes to allocate on the stack for the first page.
 *
 * @note The actual amount of stack memory will be a few bytes larger than requested to include room
 * for a header and to round up to a multiple of the minimum required alignment.
 * @note We do this using a macro instead of a template class because the attribute MOCHI_NO_INIT is
 * not allowed on member variables. For performance, it is important that the compiler never try to
 * zero-initialize the buffer.
 *
 * @code{.cpp}
 *  MOCHI_FILO_STACK_ALLOCATOR(myAlloc, 4096);
 *  DynamicArray<int> myArray(&myAlloc);
 *  myArray.push_back(42);
 * @endcode
 */
#define MOCHI_FILO_STACK_ALLOCATOR(variableName, sizeInBytes)                     \
  alignas(FiloAllocator::kMinAlignment) std::byte stack_memory_for_##variableName \
      [((static_cast<size_t>(sizeInBytes) + (FiloAllocator::kMinAlignment - 1)) & \
        ~(FiloAllocator::kMinAlignment - 1)) +                                    \
       FiloAllocator::kHeaderSize] MOCHI_NO_INIT;                                 \
  FiloAllocator variableName(                                                     \
      stack_memory_for_##variableName, sizeof(stack_memory_for_##variableName))

} // namespace mochi

#include "filo_allocator_inl.h"
