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

#include <cstddef>

namespace mochi::test {

// Allocator with static counters to find out if memory was handled properly
class TestAllocator final : public mochi::Allocator {
 public:
  inline static int s_allocate = 0;
  inline static int s_deallocate = 0;
  inline static int s_bytes = 0;
  inline static int s_lastAllocAlignment = 0;
  inline static int s_lastDeallocAlignment = 0;
  inline static bool s_compatibleWithOtherInstances = true;

  static void ResetCounters() {
    s_allocate = s_deallocate = s_bytes = s_lastAllocAlignment = s_lastDeallocAlignment = 0;
  }

  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    ++s_allocate;
    s_bytes += static_cast<int>(bytes);
    s_lastAllocAlignment = alignment;
    return _realAllocator->allocate(bytes, alignment);
  }

  void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override {
    ++s_deallocate;
    s_bytes -= static_cast<int>(bytes);
    s_lastDeallocAlignment = alignment;
    _realAllocator->deallocate(p, bytes, alignment);
  }

  bool do_is_equal(Allocator const& other) const noexcept override {
    return (&other == this) || s_compatibleWithOtherInstances;
  }

 private:
  Allocator* _realAllocator = GetDefaultAllocator();
};

} // namespace mochi::test
