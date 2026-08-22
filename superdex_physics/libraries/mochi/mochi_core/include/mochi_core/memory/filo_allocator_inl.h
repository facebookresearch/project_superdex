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

// Reverse include for Intellisense
#include "filo_allocator.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace mochi {

inline FiloAllocator::FiloAllocator(void* userMem, size_t userMemSize)
    : _nextPageSize(std::max(userMemSize - sizeof(Header), kDefaultPageSize)),
      _current(reinterpret_cast<Header*>(userMem)) {
  MOCHI_ALLOCATOR_ASSERT(userMem != nullptr, "Null pointer");
  MOCHI_ALLOCATOR_ASSERT(
      (reinterpret_cast<size_t>(userMem) % kMinAlignment) == 0,
      "Memory buffer must have the required alignment.");
  MOCHI_ALLOCATOR_ASSERT(
      (userMemSize % kMinAlignment) == 0,
      "Memory buffer size must be a multiple of the kMinAlignment.");
  MOCHI_ALLOCATOR_ASSERT(userMemSize >= kMinBufferSize, "Buffer size too small");
  _current->prev = nullptr;
  _current->offset = 0;
  _current->size = userMemSize - sizeof(Header);
}

inline FiloAllocator::~FiloAllocator() {
  // This should be true if all memory was correctly deallocated (see above)
  MOCHI_ALLOCATOR_ASSERT(
      _debug.empty() && (_current->offset == 0) &&
          (_current->prev == nullptr ||
           (_current->prev->prev == nullptr && _current->prev->offset == 0)),
      "All memory allocated via this FiloAllocator must be freed before the allocator can safely be destroyed");

  // Release all free pages
  while (_free) {
    auto* page = _free;
    _free = page->prev;
    FreePage(page);
  }

  // We never free the oldest page because it is either the s_empty page, or it points to
  // user-provided memory.
  while (_current->prev) {
    auto* page = _current;
    _current = page->prev;
    FreePage(page);
  }

#if MOCHI_ALLOCATOR_DEBUG
  _current = nullptr;
  _free = nullptr;
#endif // MOCHI_ALLOCATOR_DEBUG
}

template <typename T>
inline T FiloAllocator::AlignUp(T val, size_t alignment) {
  auto mask = alignment - 1;
  if constexpr (std::is_pointer_v<T>) {
    auto address = reinterpret_cast<std::uintptr_t>(val);
    MOCHI_ALLOCATOR_ASSERT(
        address <= std::numeric_limits<std::uintptr_t>::max() - mask,
        "Address can't be rounded up for alignment");
    auto result = (address + mask) & ~mask;
    return reinterpret_cast<T>(result); // NOLINT(performance-no-int-to-ptr)
  } else {
    auto value = static_cast<size_t>(val);
    MOCHI_ALLOCATOR_ASSERT(
        value <= std::numeric_limits<size_t>::max() - mask,
        "Value is too large. This indicates a problem in the calling code.");
    auto result = (value + mask) & ~mask;
    return static_cast<T>(result);
  }
}

inline void FiloAllocator::SetNextPageSize(size_t newPageSize) {
  // Must be  at least kMinPageSize and rounded up to a multiple of kMinAlignment
  _nextPageSize = AlignUp(std::max(kMinPageSize, newPageSize), kMinAlignment);
}

inline size_t FiloAllocator::GetNextPageSize() const noexcept {
  return _nextPageSize;
}

inline void FiloAllocator::PopPage() {
  MOCHI_ALLOCATOR_ASSERT(
      _current != nullptr && _current->offset == 0,
      "Cannot pop a page with outstanding allocations");
  MOCHI_ALLOCATOR_ASSERT(_current->prev != nullptr, "Never pop the oldest page");

  // Move current page to the free list
  auto* page = _current;
  _current = _current->prev;
  page->prev = _free;
  _free = page;
}

inline void FiloAllocator::FreePage(Header* page) {
#if MOCHI_ALLOCATOR_DEBUG
  MOCHI_ALLOCATOR_ASSERT(
      (page->offset == 0) && (page->size > 0),
      "Attempting to free a page with outstanding allocations");
  std::memset(page, 0xFE, sizeof(Header)); // Freed mem pattern
#endif // MOCHI_ALLOCATOR_DEBUG
  free(page);
}

inline void FiloAllocator::Shrink() {
  // If the current page is empty and we own it, then move it to the free list
  if ((_current->offset == 0) && (_current->prev != nullptr)) {
    PopPage();
  }
  // Cleanup the free page list
  while (_free) {
    auto* page = _free;
    _free = page->prev;
    FreePage(page);
  }
}

inline size_t FiloAllocator::GetBytesRemainingInPage() const {
  return _current->size - _current->offset;
}

#if MOCHI_ALLOCATOR_DEBUG
inline size_t FiloAllocator::GetMaxBytesUsed() const {
  return _maxBytesUsed;
}
#endif

inline void FiloAllocator::PushPage(size_t size) {
  auto allocSize = size + sizeof(Header);

  // If the current page is empty and it is not the first page (e.g. it was allocated from the
  // heap), then move it to the free list. This can happen when the user requests a single
  // allocation that is larger than the current page.
  if (_current->offset == 0 && _current->prev != nullptr) {
    auto* page = _current;
    _current = _current->prev;
    page->prev = _free;
    _free = page;
  }

  // First, check the free list
  Header* next = nullptr;
  for (auto* page = _free; page != nullptr; next = page, page = page->prev) {
    if (page->size >= size) {
      if (page == _free) {
        _free = page->prev;
      }
      if (next) {
        next->prev = page->prev;
      }
      page->prev = _current;
      _current = page;
      return;
    }
  }

  // Round up to a multiple of the normal allocation size
  auto nextAllocSize = _nextPageSize + sizeof(Header);
  if (allocSize > nextAllocSize) {
    allocSize = ((allocSize + nextAllocSize - 1) / nextAllocSize) * nextAllocSize;
    nextAllocSize = allocSize;
    size = allocSize - sizeof(Header);
  }

  // For performance, we want to have a small number of large pages.
  // Therefore, we increase the page size over time (with an upper bound).
  constexpr size_t kMaxSizeForAutomaticGrowth = 4 * 1024 * 1024; // 4 MiB
  nextAllocSize = std::min(kMaxSizeForAutomaticGrowth, nextAllocSize * 2);
  _nextPageSize = nextAllocSize - sizeof(Header);

  // Allocate a new page from the heap
  auto* page = static_cast<Header*>(malloc(allocSize));
  if (page == nullptr) {
    throw std::bad_alloc();
  }
  MOCHI_ALLOCATOR_ASSERT(
      reinterpret_cast<size_t>(page) % kMinAlignment == 0, "Insufficient alignment");
  page->prev = _current;
  page->offset = 0;
  page->size = size;
  _current = page;
}

inline void* FiloAllocator::do_allocate(std::size_t sizeInBytes, std::size_t alignment) {
  MOCHI_ALLOCATOR_ASSERT(alignment <= kMaxAlignment, "Unsupported alignment");
  // Standard requires 0-byte allocation to return a non-null pointer that can be successfully
  // deallocated. Treat it as a 1 byte allocation instead.
  sizeInBytes = std::max(sizeInBytes, (size_t)1);
  std::byte* ptr; // NOLINT
#if MOCHI_ALLOCATOR_DEBUG
  size_t prevOffset; // NOLINT
#endif
  if (alignment <= kMinAlignment) {
    // We always guarantee the minimum alignment, even if the user asks for something smaller.
    // In this case, we round the allocation size up to the next multiple. No additional book
    // keeping is required.
    auto alignedSize = AlignUp(sizeInBytes, kMinAlignment);
    if (_current->offset + alignedSize > _current->size) {
      PushPage(std::max(alignedSize, _nextPageSize)); // Also guarantees kMinAlignment
    }
    ptr = _current->Data() + _current->offset;
#if MOCHI_ALLOCATOR_DEBUG
    prevOffset = _current->offset;
#endif // MOCHI_ALLOCATOR_DEBUGF
    _current->offset += alignedSize;
  } else {
    // If the user requests a larger alignment value (e.g. 16 or 32 sizeInBytes), then we may need
    // to add some padding. We will store the amount of padding in the suffix after the allocated
    // block so that we can correctly decrement the offset when the memory is deallocated. The
    // suffix size is a multiple of kMinAlignment to ensure the _current->offset is always a
    // multiple as well.
    auto* base = _current->Data() + _current->offset;
    ptr = AlignUp(base, alignment);
    auto alignedSizeWithSuffix = AlignUp(sizeInBytes + kAlignmentSuffixSize, kMinAlignment);
    if (ptr + alignedSizeWithSuffix > _current->Data() + _current->size) {
      PushPage(std::max(alignedSizeWithSuffix + alignment, _nextPageSize));
      base = _current->Data();
      ptr = AlignUp(base, alignment);
    }
    MOCHI_ALLOCATOR_ASSERT(
        ptr + alignedSizeWithSuffix <= base + _current->size,
        "Internal Error. Allocation should fit in the current page.");
    auto padding = static_cast<uint8_t>(ptr - base);
    auto* iptr = reinterpret_cast<uint8_t*>(ptr);
    iptr[sizeInBytes + 1] = padding; // write padding (middle byte of suffix)
#if MOCHI_ALLOCATOR_DEBUG
    // Surround the padding byte in sentinel values to help detect corruption
    iptr[sizeInBytes + 0] = 0xEA;
    iptr[sizeInBytes + 2] = 0xAE;
    prevOffset = _current->offset;
#endif // MOCHI_ALLOCATOR_DEBUG
    _current->offset += padding + alignedSizeWithSuffix;
  }

#if MOCHI_ALLOCATOR_DEBUG
  _debug.emplace_back(DebugInfo{ptr, sizeInBytes, alignment, prevOffset});
  std::memset(ptr, 0xAB, sizeInBytes); // Uninitialized mem pattern
  MOCHI_ALLOCATOR_ASSERT(
      (reinterpret_cast<size_t>(ptr) % kMinAlignment) == 0,
      "All allocations are expected to satisfy at least the minimum alignment,");
  _currentBytesUsed += _current->offset - prevOffset;
  _maxBytesUsed = Max(_maxBytesUsed, _currentBytesUsed);
#endif // MOCHI_ALLOCATOR_DEBUG

  return ptr;
}

inline void
FiloAllocator::do_deallocate(void* ptr, std::size_t sizeInBytes, std::size_t alignment) {
  // Standard requires 0-byte allocation to return a non-null pointer that can be successfully
  // deallocated. Treat it as a 1 byte allocation instead.
  sizeInBytes = std::max(sizeInBytes, (size_t)1);

  // Check all the things
  MOCHI_ALLOCATOR_ASSERT(
      !_debug.empty(), "Attempting to deallocate memory, but none was allocated");
  MOCHI_ALLOCATOR_ASSERT(
      _debug.back().ptr == ptr,
      "Address does not match the most recent allocation. FiloAllocator requires you to deallocate in reverse order of allocation.");
  MOCHI_ALLOCATOR_ASSERT(
      _debug.back().size == sizeInBytes,
      "Deallocation size is incorrect. Must match allocation size.");
  MOCHI_ALLOCATOR_ASSERT(
      _debug.back().alignment == alignment,
      "Deallocation alignment is incorrect. Must match allocation alignment.");
  MOCHI_ALLOCATOR_ASSERT(
      (alignment <= kMinAlignment) ||
          ((static_cast<std::byte const*>(ptr)[sizeInBytes + 0] == static_cast<std::byte>(0xEA)) &&
           (static_cast<std::byte const*>(ptr)[sizeInBytes + 2] == static_cast<std::byte>(0xAE))),
      "Memory corruption detected in sentinel values");
  // If the current page is empty (presumably from a previous call to do_deallocate), then move it
  // to the free list.
  if (_current->offset == 0) {
    PopPage();
  }

  // Reverse the effects of do_allocate (see above). This may leave the current page empty.
  if (alignment <= kMinAlignment) {
    auto alignedSize = AlignUp(sizeInBytes, kMinAlignment);
    MOCHI_ALLOCATOR_ASSERT(_current->offset >= alignedSize, "Invalid page offset");
    _current->offset -= alignedSize;
#if MOCHI_ALLOCATOR_DEBUG
    MOCHI_ALLOCATOR_ASSERT(
        _currentBytesUsed >= alignedSize, "Tracking of total bytes used is out-of-sync");
    _currentBytesUsed -= alignedSize;
#endif
  } else {
    auto alignedSizeWithSuffix = AlignUp(sizeInBytes + kAlignmentSuffixSize, kMinAlignment);
    auto padding = static_cast<uint8_t const*>(ptr)[sizeInBytes + 1]; // middle byte of suffix
    auto sizeToFree = alignedSizeWithSuffix + padding;
    MOCHI_ALLOCATOR_ASSERT(_current->offset >= sizeToFree, "Invalid page offset");
    _current->offset -= sizeToFree;
#if MOCHI_ALLOCATOR_DEBUG
    MOCHI_ALLOCATOR_ASSERT(
        _currentBytesUsed >= sizeToFree, "Tracking of total bytes used is out-of-sync");
    _currentBytesUsed -= sizeToFree;
#endif
  }

#if MOCHI_ALLOCATOR_DEBUG
  MOCHI_ALLOCATOR_ASSERT(
      _current->offset == _debug.back().prevOffset,
      "Offset after deallocation is incorrect. This is either an internal error, or the padding value may have been corrupted.");
  _debug.pop_back();
  std::memset(ptr, 0xFE, sizeInBytes); // Freed mem pattern
#endif // MOCHI_ALLOCATOR_DEBUG
}

inline bool FiloAllocator::do_is_equal(Allocator const& other) const noexcept {
  return &other == this;
}

} // namespace mochi
