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
#include <mochi_core/mochi_platform.h>

#include <cstddef>
#include <type_traits>

namespace mochi::details {

/// @brief Routine to allocate memory on the device
///
/// @note Returns nullptr when the allocation fails
void* CudaMalloc(std::size_t lenInBytes);

/// @brief Routine to delete memory on the device
void CudaFree(void* p);

/// @brief Routine to set to 0 a chunk of memory allocated via CudaMalloc
///
/// @param[in,out] ptr Pointer to the memory location on the device
/// @param[in] lenInBytes Length of the memory in bytes
void CudaMemSetZero(void* ptr, std::size_t lenInBytes);

void CudaMemCopy(void* dest, void const* src, std::size_t lenInBytes);

template <typename U>
void CudaCopy(U* dest, U const* src, std::size_t len) {
  CudaMemCopy((void*)dest, (void const*)src, sizeof(U) * len);
}

/// @brief Copy the vector `src` into the vector `dest`
/// @param[in] dest Device pointer
/// @param[in] src Device pointer
/// @param[in] len Number of elements in the vector
void CudaDeviceCopy(float* dest, float const* src, std::size_t len);

/// @brief Copy the vector `src` into the vector `dest`
/// @param[in] dest Device pointer
/// @param[in] src Device pointer
/// @param[in] len Number of elements in the vector
void CudaDeviceCopy(double* dest, double const* src, std::size_t len);

/// @brief Copy data from an array to another
///
/// @param[in] dest Destination memory address
/// @param[in] destLdInBytes Pitch(= stride) of destination memory (in bytes)
/// @param[in] src Source memory address
/// @param[in] srcLdInBytes Pitch(= stride) of source memory (in bytes)
/// @param[in] cudaWidthInBytes Width of matrix transfer (columns in bytes)
/// @param[in] cudaHeight Height of matrix transfer (rows)
///
/// Copies a matrix (cudaHeight rows of cudaWidthInBytes bytes each)
/// from the memory area pointed to by src to the memory area pointed to by dest.
/// cudaWidthInBytes must not exceed either destLdInBytes or srcLdInBytes (no check is performed).
///
/// @note The matrix is treated as row-major.
void CudaMemCopy2D(
    void* dest,
    std::size_t destLdInBytes,
    void const* src,
    std::size_t srcLdInBytes,
    std::size_t cudaWidthInBytes,
    std::size_t cudaHeight);

/// @brief Copy data from an array to another
///
/// @param[in] dest Destination memory address
/// @param[in] destLd Pitch(= stride, Leading Dimension) of destination memory
/// @param[in] src Source memory address
/// @param[in] srcLd Pitch(= stride, Leading Dimension) of source memory
/// @param[in] cudaWidth Width of matrix transfer (columns)
/// @param[in] cudaHeight Height of matrix transfer (rows)
///
/// Copies a matrix (cudaHeight rows of cudaWidth each)
/// from the memory area pointed to by src to the memory area pointed to by dest.
/// cudaWidth must not exceed either destLD or srcLD (no check is performed).
///
/// @note The matrix is treated as row-major.
template <typename U>
void CudaCopy2D(
    U* dest,
    std::size_t destLd,
    U const* src,
    std::size_t srcLd,
    std::size_t cudaWidth,
    std::size_t cudaHeight) {
  static_assert((MOCHI_USE_CUDA) && std::is_trivially_copyable_v<U>);
  CudaMemCopy2D(
      dest, sizeof(U) * destLd, src, sizeof(U) * srcLd, sizeof(U) * cudaWidth, cudaHeight);
}

/// @brief Parameter to describe an upper limit on the number of terms in an expression
/// This upper limit is due to the fact that CUDA requires to know the length of that expression.
constexpr int kMaxNumTermsInExpression = 16;

/// @brief Dot product between two vectors stored on the device
///
/// @param[in] n Length of the vector
/// @param[in] x Pointer to the memory location for the vector 'x'
/// @param[in] y Pointer to the memory location for the vector 'y'
/// @returns Result of the dot product 'x^T y'
/// @note The returned result is on the host.
template <typename Scalar>
Scalar CudaDot(int n, Scalar const* x, Scalar const* y);

} // namespace mochi::details
