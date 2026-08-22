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

#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/memory/monotonic_allocator.h>
#include <mochi_core/test/mochi_test_helpers.h>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>

using namespace mochi;
using namespace mochi::krylov;

TEST(MatrixMemory, NewDelete) {
  auto* mgr = GetDefaultAllocator();

  krylov::details::BaseStorage<real, krylov::kDynamic, krylov::Ownership::Owner> data(15, mgr);
  EXPECT_EQ(sizeof(data), sizeof(std::size_t) + sizeof(real*) + sizeof(Allocator*));

  Matrix<real> mat(5, 3, mgr);
  for (int i = 0; i < mat.Rows(); ++i) {
    for (int j = 0; j < mat.Cols(); ++j) {
      mat(i, j) = real(i + 0.1 * j + 0.3);
    }
  }

  Matrix<real> g(5, 3);
  for (int i = 0; i < g.Rows(); ++i) {
    for (int j = 0; j < g.Cols(); ++j) {
      g(i, j) = real(i + 0.1 * j + 0.3);
    }
  }

  EXPECT_TRUE(test::NearEqualMatrices(mat, g));
}

TEST(MatrixMemory, MonotonicBuffer) {
  std::array<real, 512> pool{};
  auto mgr = MonotonicAllocator(pool.data(), pool.size() * sizeof(real));

  krylov::details::BaseStorage<real, krylov::kDynamic, krylov::Ownership::Owner> data(13, &mgr);
  EXPECT_EQ(sizeof(data), sizeof(std::size_t) + sizeof(real*) + sizeof(Allocator*));

  Matrix<real> mat(5, 3, &mgr);
  EXPECT_TRUE((mat.Data() >= pool.data()) && (mat.Data() + 5 * 3 <= pool.data() + pool.size()));
  EXPECT_GE(mat.Data(), pool.data() + 13);
  EXPECT_LE(mat.Data(), pool.data() + 128);

  for (int i = 0; i < mat.Rows(); ++i) {
    for (int j = 0; j < mat.Cols(); ++j) {
      mat(i, j) = real(i + 0.1 * j + 0.3);
    }
  }

  ColumnVector<real> v(256, &mgr);
  EXPECT_TRUE((v.Data() >= pool.data()) && (v.Data() + 256 <= pool.data() + pool.size()));

  Matrix<real> g(5, 3);
  for (int i = 0; i < g.Rows(); ++i) {
    for (int j = 0; j < g.Cols(); ++j) {
      g(i, j) = real(i + 0.1 * j + 0.3);
    }
  }

  EXPECT_TRUE(test::NearEqualMatrices(mat, g));
}
