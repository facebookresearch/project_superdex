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

#include <mochi_physics/src/mochi_soft_rom_polynomial_crom_systems.h>
#include "mochi_physics_test_fixture.h"

#include <gtest/gtest.h>

#include <array>
#include <unordered_set>

using namespace mochi;

// Hash function needed for unordered_set
// because we are using Int3 as value type
struct HashFunction {
  size_t operator()(Int3 const& x) const {
    return x[0] ^ x[1] ^ x[2];
  }
};

static auto ExpectedMultiIndex(int order) {
  std::unordered_set<Int3, HashFunction> s;
  if (order >= 0) {
    s.insert({0, 0, 0});
  }

  if (order >= 1) {
    s.insert({1, 0, 0});
    s.insert({0, 0, 1});
    s.insert({0, 1, 0});
  }

  if (order >= 2) {
    s.insert({2, 0, 0});
    s.insert({0, 2, 0});
    s.insert({0, 0, 2});
    s.insert({1, 1, 0});
    s.insert({1, 0, 1});
    s.insert({0, 1, 1});
  }

  if (order >= 3) {
    s.insert({3, 0, 0});
    s.insert({2, 1, 0});
    s.insert({1, 2, 0});
    s.insert({0, 3, 0});
    s.insert({2, 0, 1});
    s.insert({1, 1, 1});
    s.insert({0, 2, 1});
    s.insert({1, 0, 2});
    s.insert({0, 1, 2});
    s.insert({0, 0, 3});
  }

  if (order >= 4) {
    s.insert({3, 1, 0});
    s.insert({0, 0, 4});
    s.insert({0, 1, 3});
    s.insert({1, 0, 3});
    s.insert({0, 2, 2});
    s.insert({1, 1, 2});
    s.insert({4, 0, 0});
    s.insert({2, 0, 2});
    s.insert({0, 4, 0});
    s.insert({0, 3, 1});
    s.insert({1, 2, 1});
    s.insert({2, 1, 1});
    s.insert({2, 2, 0});
    s.insert({3, 0, 1});
    s.insert({1, 3, 0});
  }

  return s;
}

constexpr real kTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-12 : 1e-4;
constexpr std::array<int, 5> kTestOrders = {0, 1, 2, 3, 4};

TEST(MochiRomPolynomialCrom, MultiIndex) {
  for (int order : kTestOrders) {
    auto const mi = rom::polynomial_crom::ComputeMultiIndexSortedByTotalOrder(order);
    auto expected = ExpectedMultiIndex(order);
    for (auto it : mi) {
      EXPECT_TRUE(expected.contains(it));
      expected.erase(it);
    }

    //  at the end, should have an empty set
    EXPECT_TRUE(expected.empty());
  }
}

static ColumnVector<Real3> CreatePositions() {
  ColumnVector<Real3> pos(2);
  pos[0] = Real3{0.1_r, 0.2_r, -0.3_r};
  pos[1] = Real3{0.4_r, 1.1_r, 0.6_r};
  return pos;
}

static real ExpectedValue(Int3 const& inds, Real3 const& p) {
  real result = 1_r;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < inds[i]; ++j) {
      result *= p[i];
    }
  }
  return result;
}

TEST(MochiRomPolynomialCrom, MatrixEvaluation) {
  std::array<int, 5> expectedCounts = {3, 12, 30, 60, 105};
  auto const pos = CreatePositions();
  for (int order : kTestOrders) {
    auto const mi = rom::polynomial_crom::ComputeMultiIndexSortedByTotalOrder(order);
    auto M = rom::polynomial_crom::CreateBasisMatrix(order, pos);
    EXPECT_TRUE(M.Rows() == 6);
    EXPECT_TRUE(M.Cols() == expectedCounts[order]);

    for (int i = 0; i < pos.size(); ++i) {
      int const dofRow = i * 3;
      for (int j = 0; j < isize(mi); ++j) {
        real const expectedValue = ExpectedValue(mi[j], pos[i]);
        EXPECT_NEAR(M(dofRow, j * 3), expectedValue, kTol);
        EXPECT_NEAR(M(dofRow + 1, j * 3 + 1), expectedValue, kTol);
        EXPECT_NEAR(M(dofRow + 2, j * 3 + 2), expectedValue, kTol);
      }
    }
  }
}
