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

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/debug.h>

#include "allocator.h" // Reverse include for Intellisense

// Supported standard library features
#define MOCHI_HAS_MEMORY_RESOURCE 1
#define MOCHI_HAS_EXPERIMENTAL_MEMORY_RESOURCE 0
#define MOCHI_HAS_PMR_NEW_DELETE_RESOURCE 1

// Exception for Apple platforms with Clang versions older than 15.0.0.
#if defined(__APPLE__) && defined(__clang_major__) && !defined(__cpp_lib_memory_resource)
#if __clang_major__ < 15
#undef MOCHI_HAS_MEMORY_RESOURCE
#define MOCHI_HAS_MEMORY_RESOURCE 0
#undef MOCHI_HAS_PMR_NEW_DELETE_RESOURCE
#define MOCHI_HAS_PMR_NEW_DELETE_RESOURCE 0
#undef MOCHI_HAS_EXPERIMENTAL_MEMORY_RESOURCE
#define MOCHI_HAS_EXPERIMENTAL_MEMORY_RESOURCE 1
#endif
#endif

// Exception for Unreal Engine Linux builds. UE5's bundled libc++ lacks <memory_resource>.
#ifdef UNREALIOS_LINUX
#if UNREALIOS_LINUX
#undef MOCHI_HAS_MEMORY_RESOURCE
#define MOCHI_HAS_MEMORY_RESOURCE 0
#undef MOCHI_HAS_PMR_NEW_DELETE_RESOURCE
#define MOCHI_HAS_PMR_NEW_DELETE_RESOURCE 0
#undef MOCHI_HAS_EXPERIMENTAL_MEMORY_RESOURCE
#define MOCHI_HAS_EXPERIMENTAL_MEMORY_RESOURCE 0
#endif
#endif

// Exception for Android. Necessary because the Unreal Engine Android compiler is old and doesn't
// support <memory_resource> at all.
#if MOCHI_PLATFORM_ANDROID
#undef MOCHI_HAS_MEMORY_RESOURCE
#define MOCHI_HAS_MEMORY_RESOURCE 0
#undef MOCHI_HAS_PMR_NEW_DELETE_RESOURCE
#define MOCHI_HAS_PMR_NEW_DELETE_RESOURCE 0
#undef MOCHI_HAS_EXPERIMENTAL_MEMORY_RESOURCE
#define MOCHI_HAS_EXPERIMENTAL_MEMORY_RESOURCE 0
#endif

#if MOCHI_HAS_MEMORY_RESOURCE
#include <memory_resource>
#elif MOCHI_HAS_EXPERIMENTAL_MEMORY_RESOURCE
#include <experimental/memory_resource>
#endif

namespace mochi {

/*********************************************************************************************
  Macros
*/

#if MOCHI_ALLOCATOR_DEBUG
#if MOCHI_HAS_VA_OPT
#define MOCHI_ALLOCATOR_ASSERT(condition_without_side_effects, ...) \
  MOCHI_ASSERT(condition_without_side_effects __VA_OPT__(, )##__VA_ARGS__)
#else
// Strict C++17 compatibility requires __VA_ARGS__ to be non-empty, to prevent a trailing comma.
// Therefore, MOCHI_ALLOCATOR_ASSERT must always have a message string in C++17 limited headers,
// even if the information in that string is redundant.
#define MOCHI_ALLOCATOR_ASSERT(condition_without_side_effects, ...) \
  MOCHI_ASSERT(condition_without_side_effects, ##__VA_ARGS__)
#endif
#else
#define MOCHI_ALLOCATOR_ASSERT(...)
#endif

/*********************************************************************************************
  Allocator
*/

inline void* Allocator::allocate(std::size_t sizeInBytes, std::size_t alignment) {
  MOCHI_ALLOCATOR_ASSERT(IsPowerOfTwo(alignment), "Alignment must be a power of two");
  void* ptr = do_allocate(sizeInBytes, alignment);
  MOCHI_ALLOCATOR_ASSERT(
      ptr != nullptr || sizeInBytes == 0, "Allocation failed without throwing an exception");
  MOCHI_ALLOCATOR_ASSERT(
      !(reinterpret_cast<intptr_t>(ptr) & (alignment - 1)),
      "Address does not have the requested alignment of %zu bytes",
      alignment);
  return ptr;
}

inline void Allocator::deallocate(void* ptr, std::size_t sizeInBytes, std::size_t alignment) {
  MOCHI_ALLOCATOR_ASSERT(IsPowerOfTwo(alignment), "Alignment must be a power of two");
  MOCHI_ALLOCATOR_ASSERT(
      !(reinterpret_cast<intptr_t>(ptr) & (alignment - 1)),
      "Address does not have the stated alignment");
  return do_deallocate(ptr, sizeInBytes, alignment);
}

inline bool Allocator::is_equal(Allocator const& other) const noexcept {
  return do_is_equal(other);
}

/*********************************************************************************************
  Default Allocator
*/

inline void* DefaultAllocator::do_allocate(size_t sizeInBytes, size_t alignment) {
#if MOCHI_PMR_USES_JEMALLOC
  // Some fbcode build modes use jemalloc to implement operators new and delete, and thus
  // std::prm::new_delete_resource. The problem is that jemalloc always rounds up to a minimum size
  // of 16 bytes. We have to do the same or the following can happen:
  //   1) void* ptr = myAllocator.allocate(8, 8); // Request 8 bytes
  //   2) Behind the scene, jemalloc actually allocates 16 bytes
  //   3) myAllocator.deallocate(ptr, 8, 8); // Matches the call to allocate
  //   4) jemalloc reports a fatal error because 16 bytes were allocated, but only 8 bytes are being
  //   deallocated.
  //
  // Our work-around is to round up to 16 bytes all the time.
  //
  sizeInBytes = Max(sizeInBytes, size_t(16));
#endif

#if MOCHI_HAS_MEMORY_RESOURCE && MOCHI_HAS_PMR_NEW_DELETE_RESOURCE
  return std::pmr::new_delete_resource()->allocate(sizeInBytes, alignment);
#elif MOCHI_HAS_EXPERIMENTAL_MEMORY_RESOURCE && MOCHI_HAS_PMR_NEW_DELETE_RESOURCE
  return std::experimental::pmr::new_delete_resource()->allocate(sizeInBytes, alignment);
#elif defined(__cpp_lib_aligned_alloc)
  // Use std::aligned_alloc. Confirmed to work for Apple Clang 14.0.3
  if (alignment <= 16) {
    return std::malloc(sizeInBytes);
  } else {
    return std::aligned_alloc(alignment, sizeInBytes);
  }
#else
  // Use posix_memalign for larger allocations on Android.
  if (alignment <= 8) {
    return malloc(sizeInBytes);
  } else {
    void* ptr = nullptr;
    posix_memalign(&ptr, alignment, sizeInBytes);
    return ptr;
  }
#endif
}

inline void DefaultAllocator::do_deallocate(
    void* ptr,
    [[maybe_unused]] size_t sizeInBytes,
    [[maybe_unused]] size_t alignment) {
#if MOCHI_PMR_USES_JEMALLOC
  // See comment in do_allocate.
  sizeInBytes = Max(sizeInBytes, size_t(16));
#endif

#if MOCHI_HAS_MEMORY_RESOURCE && MOCHI_HAS_PMR_NEW_DELETE_RESOURCE
  return std::pmr::new_delete_resource()->deallocate(ptr, sizeInBytes, alignment);
#elif MOCHI_HAS_EXPERIMENTAL_MEMORY_RESOURCE && MOCHI_HAS_PMR_NEW_DELETE_RESOURCE
  return std::experimental::pmr::new_delete_resource()->deallocate(ptr, sizeInBytes, alignment);
#elif defined(__cpp_lib_aligned_alloc)
  std::free(ptr); // Free memory from std::alloc_aligned
#else
  free(ptr); // Free memory from poxis_memalign
#endif
}

inline bool DefaultAllocator::do_is_equal(Allocator const& other) const noexcept {
  // We can't assume that two DefaultAllocators with different addresses are compatible, beause they
  // might be compiled into different dynamic libraries with incompatible system allocators. Mochi
  // code always uses GetDefaultAllocator() which always returns a consistent address within
  // statically linked libraries (e.g. all Mochi libraries compiled into one DLL). Therefore Mochi
  // containers using DefaultAllocator can move their allocations efficiently within Mochi code, but
  // may need to perform a deep copy when crossing DLL-boundaries.
  return (&other == this);
}

/*********************************************************************************************
  Utilities
*/

[[nodiscard]] MOCHI_FORCE_INLINE Allocator* GetDefaultAllocator() {
  static DefaultAllocator s_allocator;
  return &s_allocator;
}

} // namespace mochi
