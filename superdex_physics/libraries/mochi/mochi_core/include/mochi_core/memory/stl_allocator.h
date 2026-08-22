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

namespace mochi {

// An STL-compatible allocator template using a polymorphic mochi::Allocator pointer.
template <typename T>
class StlAllocator {
 public:
  using value_type = T;

  StlAllocator() = default;

  // This constructor is not "explicit" so that you can pass a pointer to a mochi::Allocator to the
  // STL container's constructor (not a pointer to a mochi::StlAllocator wrapper).
  StlAllocator(Allocator* alloc) : _allocator(alloc) {}

  template <typename U>
  StlAllocator(StlAllocator<U> const& other) : _allocator(other.get_allocator()) {}

  T* allocate(std::size_t n) {
    return static_cast<T*>(_allocator->allocate(n * sizeof(T), alignof(T)));
  }

  void deallocate(T* ptr, std::size_t n) {
    _allocator->deallocate(ptr, n * sizeof(T), alignof(T));
  }

  Allocator* get_allocator() const {
    return _allocator;
  }

  friend bool operator==(StlAllocator const& a, StlAllocator const& b) {
    return a._allocator->is_equal(*b._allocator);
  }

  friend bool operator!=(StlAllocator const& a, StlAllocator const& b) {
    return !(a == b);
  }

 private:
  Allocator* _allocator = GetDefaultAllocator();
};

} // namespace mochi
