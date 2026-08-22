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

#include <mochi_core/mochi_config.h>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace mochi {

/*********************************************************************************************
  Macros
*/

// Define MOCHI_ALLOCATOR_DEBUG to 1 to enable safety checks. Enabled by default in debug builds.
#ifndef MOCHI_ALLOCATOR_DEBUG
#define MOCHI_ALLOCATOR_DEBUG MOCHI_DEBUG
#endif

/*********************************************************************************************
  Allocator
*/

/**
 * @brief Virtual interface class for memory allocation and deallocation.
 *
 * This class provides a polymorphic interface for memory allocation operations, similar to
 * std::pmr::memory_resource, which is not used because of compatibility issues with various
 * platform/compiler combinations. Derived classes must implement the protected members.
 */
class Allocator {
 public:
  virtual ~Allocator() = default;

  /**
   * @brief Allocate memory of the specified size and alignment.
   *
   * @param sizeInBytes
   * @param alignment Must be a power of 2.
   * @return void*
   * @throws std::bad_alloc
   *
   * @note The caller is responsible for deallocating the memory by calling the deallocate method on
   * the same Allocator instance, or a compatible Allocator instance.
   *
   * @see deallocate
   * @see is_equal
   */
  void* allocate(std::size_t sizeInBytes, std::size_t alignment = alignof(std::max_align_t));

  /**
   * @brief Deallocate a block of memory that was previously allocated by this Allocator instance or
   * by a compatible Allocator instance.
   *
   * @param ptr Address of the memory block to deallocate.
   * @param sizeInBytes Must match the size that was passed to the allocate method.
   * @param alignment Must match the alignment that was passed to the allocate method.
   *
   * @see allocate
   * @see is_equal
   */
  void
  deallocate(void* ptr, std::size_t sizeInBytes, std::size_t alignment = alignof(std::max_align_t));

  /**
   * @brief Return true if memory allocated by this Allocator instance can be deallocated by the
   * other Allocator instance, and visa versa.
   *
   * @param other
   * @return bool
   */
  bool is_equal(Allocator const& other) const noexcept;

 protected:
  virtual void* do_allocate(std::size_t sizeInBytes, std::size_t alignment) = 0;
  virtual void do_deallocate(void* ptr, std::size_t sizeInBytes, std::size_t alignment) = 0;
  virtual bool do_is_equal(Allocator const& other) const noexcept = 0;
};

/*********************************************************************************************
  Default Allocator
*/

/**
 * @brief Mochi's default implementation of the Allocator interface.
 *
 * This class allows you to allocate memory on any thread and then deallocate it on any thread,
 * similar to std::pmr::new_delete_resource. All instance of this class are considered to be "equal"
 * so that memory allocated from one instance can be deallocated by another instance.
 */
class DefaultAllocator final : public Allocator {
  void* do_allocate(size_t sizeInBytes, size_t alignment) override;
  void do_deallocate(void* ptr, size_t sizeInBytes, size_t alignment) override;
  bool do_is_equal(Allocator const& other) const noexcept override;
};

/**
 * @brief Get an instance of the DefaultAllocator class. Usable in any context.
 *
 * @return Allocator*
 */
[[nodiscard]] MOCHI_FORCE_INLINE Allocator* GetDefaultAllocator();

/*********************************************************************************************
  Utilities
*/

/**
 * @brief Create a new object of type T using memory allocated by a polymorphic Allocator.
 *
 * @tparam T Type of object to create
 * @tparam A Type of Allocator
 * @tparam Args Optional types forwarded to the constructor
 * @param allocator Pointer to an Allocator instance
 * @param args Optional arguments forwarded to the constructor for type T
 * @return T* Address of the new object
 *
 * @note The caller is responsible for calling Delete with the same Allocator instance, or a
 * compatible Allocator instance (see Allocator::is_equal).
 */
template <class T, class A, class... Args>
[[nodiscard]] inline T* New(A* allocator, Args&&... args) {
  static_assert(std::is_base_of_v<Allocator, A>, "Expected an allocator pointer");
  T* ptr = static_cast<T*>(allocator->allocate(sizeof(T), alignof(T)));
  new (ptr) T(std::forward<Args>(args)...);
  return ptr;
}

/**
 * @brief Destroy and deallocate an object that was created via New<T>.
 *
 * @tparam T Type of object to destroy
 * @tparam A Type of Allocator
 * @param allocator The Allocator instance that was previously passed to New<T>
 * @param ptr Address of the object to destroy
 */
template <class T, class A>
inline void Delete(A* allocator, T* ptr) {
  static_assert(std::is_base_of_v<Allocator, A>, "Expected an allocator pointer");
  ptr->~T();
  allocator->deallocate(ptr, sizeof(T), alignof(T));
}

} // namespace mochi

#include "allocator_inl.h"
