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

#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/simd.h>

namespace mochi {

/**
 * @brief Types for batched processing.
 *
 * @tparam kBatchSize Number of work items per batch.
 */
template <int kBatchSize>
struct BatchTypes {
  static_assert(kBatchSize > 0, "Invalid batch size");

  /** @brief Batched scalar type. */
  using Real = Simd<real, RoundUp(kBatchSize, Simd<real>::kSize)>;

  /** @brief Batched double-precision type. */
  using Double = Simd<double, Real::kSize>;

  /** @brief Batched integer type. */
  using Int = Simd<int, Real::kSize>;

  /** @brief Batched 3-vector. */
  using Real3 = NdArray<Real, 3>;

  /** @brief Batched 6-vector. */
  using Real6 = NdArray<Real, 6>;

  /** @brief Batched 9-vector. */
  using Real9 = NdArray<Real, 9>;

  /** @brief Batched 2x2 row-major matrix. */
  using Real2x2 = NdArray<Real, 2, 2>;

  /** @brief Batched 3x3 row-major matrix. */
  using Real3x3 = NdArray<Real, 3, 3>;

  /** @brief Batched symmetric 2x2 matrix stored as [00, 01, 11]. */
  using SymMatrix2x2 = Real3;

  /** @brief Batched symmetric 3x3 matrix stored as [00, 11, 22, 01, 02, 12]. */
  using SymMatrix3x3 = Real6;
};

template <int kBatchSize>
using BatchReal = typename BatchTypes<kBatchSize>::Real;

template <int kBatchSize>
using BatchDouble = typename BatchTypes<kBatchSize>::Double;

template <int kBatchSize>
using BatchInt = typename BatchTypes<kBatchSize>::Int;

template <int kBatchSize>
using BatchReal3 = typename BatchTypes<kBatchSize>::Real3;

template <int kBatchSize>
using BatchReal6 = typename BatchTypes<kBatchSize>::Real6;

template <int kBatchSize>
using BatchReal9 = typename BatchTypes<kBatchSize>::Real9;

template <int kBatchSize>
using BatchReal2x2 = typename BatchTypes<kBatchSize>::Real2x2;

template <int kBatchSize>
using BatchReal3x3 = typename BatchTypes<kBatchSize>::Real3x3;

template <int kBatchSize>
using BatchSymMatrix2x2 = typename BatchTypes<kBatchSize>::SymMatrix2x2;

template <int kBatchSize>
using BatchSymMatrix3x3 = typename BatchTypes<kBatchSize>::SymMatrix3x3;

} // namespace mochi
