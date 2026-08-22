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

#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/vmatrix.h>

#include <cstddef>
#include <utility>

/**
 * @brief Helpers for setting up and inspecting batched test inputs / benchmark fixtures.
 *
 * @details This header hosts non-gtest helpers shared by tests and benchmarks that bridge between
 * scalar memory layout (e.g., `Span<Matrix3x3r>`) and lane-transposed batched (SIMD) layout.
 */

namespace mochi::test {

inline constexpr int kBatchTestMaxSize = 16;

template <int kBatchSize>
[[nodiscard]] BatchReal6<kBatchSize> LoadBatchSymMatrix3x3(Span<Matrix3x3r const> mats) {
  MOCHI_ASSERT_VERBOSE(mats.size() >= kBatchSize, "Insufficient matrices.");
  BatchReal6<kBatchSize> result = {};
  for (int i = 0; i < kBatchSize; ++i) {
    result[0] = Set(result[0], i, mats[i][0][0]);
    result[1] = Set(result[1], i, mats[i][1][1]);
    result[2] = Set(result[2], i, mats[i][2][2]);
    result[3] = Set(result[3], i, mats[i][0][1]);
    result[4] = Set(result[4], i, mats[i][0][2]);
    result[5] = Set(result[5], i, mats[i][1][2]);
  }
  return result;
}

template <int kBatchSize>
[[nodiscard]] BatchReal3x3<kBatchSize> LoadBatchMatrix3x3(Span<Matrix3x3r const> mats) {
  MOCHI_ASSERT_VERBOSE(mats.size() >= kBatchSize, "Insufficient matrices.");
  BatchReal3x3<kBatchSize> result = {};
  for (int i = 0; i < kBatchSize; ++i) {
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        result[r][c] = Set(result[r][c], i, mats[i][r][c]);
      }
    }
  }
  return result;
}

template <int kBatchSize>
void StoreBatchReal3(BatchReal3<kBatchSize> const& vals, Span<Real3> out) {
  StoreTransposed<kBatchSize>(&out[0][0], vals[0], vals[1], vals[2]);
}

template <int kBatchSize>
void StoreBatchMatrix3x3(BatchReal3x3<kBatchSize> const& mat, Span<Matrix3x3r> out) {
  for (int i = 0; i < kBatchSize; ++i) {
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        out[i][r][c] = mat[r][c][i];
      }
    }
  }
}

template <typename V>
auto GetLane(V const& v, int lane) {
  return v[lane];
}

template <typename T>
struct GetLaneResult {
  using type = decltype(std::declval<T const&>()[0]);
};

template <typename T, size_t D0, size_t... Dims>
struct GetLaneResult<NdArray<T, D0, Dims...>> {
  using type = NdArray<typename GetLaneResult<T>::type, D0, Dims...>;
};

template <typename V, size_t D0, size_t... Dims>
auto GetLane(NdArray<V, D0, Dims...> const& a, int lane) {
  typename GetLaneResult<NdArray<V, D0, Dims...>>::type result{};
  for (size_t i = 0; i < D0; ++i) {
    result[i] = GetLane(a[i], lane);
  }
  return result;
}

} // namespace mochi::test

#define MOCHI_BATCH_TEST(Suite, Name, TestFn) \
  TEST(Suite, Name) {                         \
    TestFn<1>();                              \
    TestFn<2>();                              \
    TestFn<3>();                              \
    TestFn<4>();                              \
    TestFn<5>();                              \
    TestFn<6>();                              \
    TestFn<7>();                              \
    TestFn<8>();                              \
    TestFn<9>();                              \
    TestFn<test::kBatchTestMaxSize>();        \
  }
