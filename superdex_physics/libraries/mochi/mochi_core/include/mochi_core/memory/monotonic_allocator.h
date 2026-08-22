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
#include <mochi_core/utils/basic_utils.h>

#include <cstddef>
#include <new>

namespace mochi {

// Fast memory allocator for temporary storage. Memory is allocated from an existing buffer by
// simply moving the offset forward. Memory is not released until the allocator is destroyed or
// explicitly reset. Similar to std::pmr::monotonic_buffer_resource.
class MonotonicAllocator final : public Allocator {
 public:
  MOCHI_DECLARE_MOVE(MonotonicAllocator);
  MOCHI_DECLARE_NO_COPY(MonotonicAllocator);

  MonotonicAllocator() = delete;
  ~MonotonicAllocator() override = default;

  // Construct a MonotonicAllocator using an existing block of memory.
  //
  // TODO: In the future, we could allow the user to specify another Allocator* to use when the
  // initial buffer is exhausted, similar to FiloAllocator. Currently it will throw std::bad_alloc
  // if out of memory.
  MonotonicAllocator(void* buffer, size_t size) noexcept
      : _begin(static_cast<std::byte*>(buffer)), _end(_begin + size), _pos(_begin) {}

  // Reset to zero allocated memory as if all allocations had been released.
  MOCHI_FORCE_INLINE void Reset() {
    _pos = _begin;
  }

 private:
  void* do_allocate(std::size_t sizeInBytes, std::size_t alignment) override {
    // Standard requires 0-byte allocation to return a non-null pointer that can be successfully
    // deleted. Treat it as a 1 byte allocation instead.
    sizeInBytes = Max(sizeInBytes, (size_t)1);

    auto pos = reinterpret_cast<uintptr_t>(_pos);
    auto aligned = (pos + alignment - 1) & ~(alignment - 1);
    auto* next =
        reinterpret_cast<std::byte*>(aligned) + sizeInBytes; // NOLINT(performance-no-int-to-ptr)
    if (next > _end) {
      throw std::bad_alloc();
    } else {
      _pos = next;
      return reinterpret_cast<void*>(aligned); // NOLINT(performance-no-int-to-ptr)
    }
  }

  void do_deallocate(
      [[maybe_unused]] void* ptr,
      [[maybe_unused]] std::size_t sizeInBytes,
      [[maybe_unused]] std::size_t alignment) override {
    MOCHI_ALLOCATOR_ASSERT(
        (ptr >= _begin) && (static_cast<std::byte*>(ptr) + Max(sizeInBytes, size_t(1)) <= _end),
        "Attempting to deallocate memory which did not come from this allocator");
  }

  bool do_is_equal(Allocator const& other) const noexcept override {
    // Not compatible with other allocators.
    return &other == this;
  }

  std::byte* _begin = nullptr;
  std::byte* _end = nullptr;
  std::byte* _pos = nullptr;
};

} // namespace mochi
