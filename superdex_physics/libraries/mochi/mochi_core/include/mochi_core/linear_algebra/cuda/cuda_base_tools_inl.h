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

#include <mochi_core/linear_algebra/base_tools.h>
#include <mochi_core/linear_algebra/cuda/cuda_api.h>

//
// This header file is not self-contained.
// It describes a specialization of the class `BaseStorage`
// for objects stored on the GPU via CUDA.
// It should not be included by itself.
//

namespace mochi::krylov::details {

template <typename Scalar>
class BaseStorage<Scalar, krylov::kDynamic, Ownership::Cuda> {
 public:
  BaseStorage() = default;

  explicit BaseStorage(size_t n)
      : v{reinterpret_cast<Scalar*>(mochi::details::CudaMalloc(n * sizeof(Scalar)))} {
    //--- MOCHI_ASSERT_VERBOSE is not the right way to throw an error here.
    //--- Waiting on decision for right treatment of error
    MOCHI_ASSERT_VERBOSE(
        v != nullptr || (n * sizeof(Scalar) == 0), "Not enough CUDA-memory available.");
  }

  BaseStorage(BaseStorage const& rhs) = delete;

  BaseStorage& operator=(BaseStorage const& rhs) = delete;

  /// @brief Return raw pointer to the memory (overload for compatibility with std containers)
  MOCHI_ANY Scalar* data() {
    return v;
  }

  /// @brief Return raw pointer to the memory (overload for compatibility with std containers)
  MOCHI_ANY Scalar const* data() const {
    return v;
  }

  /// @brief Return raw pointer to the memory.
  MOCHI_ANY Scalar* Data() {
    return v;
  }

  /// @brief Return raw pointer to the memory.
  MOCHI_ANY Scalar const* Data() const {
    return v;
  }

  /// @brief Returns a similar storage space
  ///
  /// @param[in] n Size of the storage space
  /// @returns BaseStorage object of the specific size
  ///
  /// @note Here the storage will be on the GPU
  ///
  BaseStorage GetSimilar(size_t n) const {
    return BaseStorage(n);
  }

  void Resize(size_t n) {
    this->~BaseStorage();
    new (this) BaseStorage(n);
  }

  /// @brief Destructor
  ~BaseStorage() {
    if (v) {
      mochi::details::CudaFree(v);
    }
  }

 protected:
  Scalar* v = nullptr;
};

/// @brief BaseStorage for fixed dimension on the GPU device
///
/// @note It is not clear how to create the fixed-size storage on Cuda.
/// Until a permanent solution is found, we will use dynamic storage on Cuda.
template <typename Scalar, int kSize>
class BaseStorage<Scalar, kSize, Ownership::Cuda> {
 public:
  BaseStorage() {
    static_assert(kSize > 0, "Incompatible parameters");
    v = reinterpret_cast<Scalar*>(mochi::details::CudaMalloc(kSize * sizeof(Scalar)));
  }

  BaseStorage(BaseStorage const& rhs) = delete;

  BaseStorage& operator=(BaseStorage const& rhs) = delete;

  /// @brief Return raw pointer to the memory (overload for compatibility with std containers)
  MOCHI_ANY Scalar* data() {
    return v;
  }

  /// @brief Return raw pointer to the memory (overload for compatibility with std containers)
  MOCHI_ANY Scalar const* data() const {
    return v;
  }

  /// @brief Return raw pointer to the memory.
  MOCHI_ANY Scalar* Data() {
    return v;
  }

  /// @brief Return raw pointer to the memory.
  MOCHI_ANY Scalar const* Data() const {
    return v;
  }

  /// @brief Returns a similar storage space
  ///
  /// @param[in] n Size of the storage space
  /// @returns BaseStorage object of the specific size
  ///
  /// @note Here the storage will be on the GPU
  ///
  BaseStorage GetSimilar([[maybe_unused]] size_t n) const {
    MOCHI_ASSERT_VERBOSE(n == kSize, "Inconsistent dimensions");
    return BaseStorage{};
  }

  /// @brief Destructor
  ~BaseStorage() {
    if (v) {
      mochi::details::CudaFree(v);
    }
  }

 protected:
  Scalar* v = nullptr;
};

template <typename Scalar, int kSize>
struct BaseStorage<Scalar, kSize, Ownership::CudaView> {
  static_assert((kSize == krylov::kDynamic) || (kSize > 0), "Incompatible size parameter");

  explicit BaseStorage(Scalar* v_in = nullptr) : v(v_in) {}

  BaseStorage(BaseStorage const& rhs) = delete;

  BaseStorage& operator=(BaseStorage const& rhs) = delete;

  /// @brief Return raw pointer to the memory (overload for compatibility with std containers)
  MOCHI_ANY Scalar* data() {
    return v;
  }

  /// @brief Return raw pointer to the memory (overload for compatibility with std containers)
  MOCHI_ANY Scalar const* data() const {
    return v;
  }

  /// @brief Return raw pointer to the memory.
  MOCHI_ANY Scalar* Data() {
    return v;
  }

  /// @brief Return raw pointer to the memory.
  MOCHI_ANY Scalar const* Data() const {
    return v;
  }

  /// @brief Returns a storage space of similar type with a specific size
  ///
  /// @param[in] n Size of the storage space
  /// @returns BaseStorage object of the specific size
  ///
  /// @note When the ownership is View, the variable `n` is unused
  /// as the dimension is not stored in the class.
  ///
  BaseStorage GetSimilar([[maybe_unused]] size_t n) const {
    return BaseStorage{v};
  }

  ~BaseStorage() = default;

 protected:
  Scalar* v = {};
};

} // namespace mochi::krylov::details
