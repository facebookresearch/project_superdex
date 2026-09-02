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

namespace mochi {

/*********************************************************************************************
  AlignedAllocator
*/

/**
 * @brief Allocator adapter that rounds up all allocation sizes and alignments to a specified
 * minimum value. Delegates to an inner allocator for the actual memory operations.
 */
class AlignedAllocator final : public Allocator {
 public:
  /**
   * @brief Construct an AlignedAllocator with the specified minimum alignment.
   *
   * @param minAlignment All allocation sizes and alignments will be rounded up to this value. Must
   * be a power of two.
   * @param innerAllocator Allocator used for backing storage. Defaults to @ref
   * GetDefaultAllocator().
   *
   * @note The inner allocator must outlive this AlignedAllocator.

   */
  explicit AlignedAllocator(size_t minAlignment, Allocator* innerAllocator = GetDefaultAllocator())
      : _minAlignment(minAlignment), _allocator(innerAllocator) {
    MOCHI_ASSERT(IsPowerOfTwo(minAlignment), "Minimum alignment must be a power of two");
    MOCHI_ASSERT(innerAllocator != nullptr, "Inner allocator must not be nullptr");
  }

  /**
   * @brief Return this allocator's minimum alignment value
   */
  [[nodiscard]] size_t GetMinimumAlignment() const {
    return _minAlignment;
  }

 private:
  constexpr std::size_t RoundUpToAlign(std::size_t value) const {
    return (value + _minAlignment - 1) & ~(_minAlignment - 1);
  }

  void* do_allocate(std::size_t sizeInBytes, std::size_t alignment) override {
    return _allocator->allocate(RoundUpToAlign(sizeInBytes), Max(alignment, _minAlignment));
  }

  void do_deallocate(void* ptr, std::size_t sizeInBytes, std::size_t alignment) override {
    _allocator->deallocate(ptr, RoundUpToAlign(sizeInBytes), Max(alignment, _minAlignment));
  }

  bool do_is_equal(Allocator const& other) const noexcept override {
    auto const* otherAligned = dynamic_cast<AlignedAllocator const*>(&other);
    return (otherAligned != nullptr) && (otherAligned->_minAlignment == _minAlignment) &&
        _allocator->is_equal(*otherAligned->_allocator);
  }

  size_t _minAlignment = 1;
  Allocator* _allocator = nullptr;
};

/*********************************************************************************************
  GetCacheAlignedAllocator
*/

/**
 * @brief Return a pointer to a @ref AlignedAllocator that will place every allocation at the start
 * of a data cache line.
 *
 * @note This function returns a pointer to a static object. Do not attempt to delete it.
 */
[[nodiscard]] AlignedAllocator* GetCacheAlignedAllocator();

} // namespace mochi
