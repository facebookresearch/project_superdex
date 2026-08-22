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
#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/basic_utils.h>

#include <cstddef>

namespace mochi {

/*********************************************************************************************
  AlignedAllocator
*/

/**
 * @brief Allocator adapter that rounds up all allocation sizes and alignments to a minimum of
 * kMinAlignment bytes. Delegates to an inner allocator for the actual memory operations.
 *
 * @tparam kMinAlignment Minimum alignment in bytes. Must be a power of two.
 */
template <size_t kMinAlignment>
class AlignedAllocator final : public Allocator {
  static_assert(
      kMinAlignment >= 1 && (kMinAlignment & (kMinAlignment - 1)) == 0,
      "kMinAlignment must be a power of two");

 public:
  /**
   * @brief Construct an AlignedAllocator using the default allocator.
   */
  AlignedAllocator() : _allocator(GetDefaultAllocator()) {}

  /**
   * @brief Construct an AlignedAllocator with the specified inner allocator.
   *
   * @param[in] allocator Pointer to the inner allocator. Must not be nullptr.
   *
   * @note The inner allocator must outlive this AlignedAllocator.
   */
  explicit AlignedAllocator(Allocator* allocator) : _allocator(allocator) {
    MOCHI_ALLOCATOR_ASSERT(allocator != nullptr, "Inner allocator must not be nullptr");
  }

 private:
  static constexpr std::size_t RoundUpToAlign(std::size_t value) {
    return (value + kMinAlignment - 1) & ~(kMinAlignment - 1);
  }

  void* do_allocate(std::size_t sizeInBytes, std::size_t alignment) override {
    return _allocator->allocate(RoundUpToAlign(sizeInBytes), Max(alignment, kMinAlignment));
  }

  void do_deallocate(void* ptr, std::size_t sizeInBytes, std::size_t alignment) override {
    _allocator->deallocate(ptr, RoundUpToAlign(sizeInBytes), Max(alignment, kMinAlignment));
  }

  bool do_is_equal(Allocator const& other) const noexcept override {
    auto const* otherAligned = dynamic_cast<AlignedAllocator<kMinAlignment> const*>(&other);
    return otherAligned != nullptr && _allocator->is_equal(*otherAligned->_allocator);
  }

  Allocator* _allocator = nullptr;
};

/*********************************************************************************************
  CacheAlignedAllocator
*/

/**
 * @brief An @ref AlignedAllocator that aligns to the CPU cache line size.
 */
using CacheAlignedAllocator = AlignedAllocator<MOCHI_CACHE_LINE_SIZE>;

} // namespace mochi
