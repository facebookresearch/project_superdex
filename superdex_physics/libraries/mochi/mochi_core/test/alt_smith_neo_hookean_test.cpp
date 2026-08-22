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

#include "material_test_helpers.h"

#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/materials/alt_smith_neo_hookean.h>
#include <mochi_core/materials/batched_smith_neo_hookean.h>
#include <mochi_core/test/batch_helpers.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/nd_array.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

using namespace mochi;

static auto ToStridedMatrix(NdArray<real, 3, 3> const& A) {
  StridedMatrix<real, 3, 3> res{};
  for (auto i = 0; i < 3; ++i) {
    for (auto j = 0; j < 3; ++j) {
      res(i, j) = A[i][j];
    }
  }
  return res;
}

static auto ToStridedMatrix(NdArray<real, 3, 3, 3, 3> const& C) {
  StridedMatrix<real, 9, 9> res{};
  for (auto i = 0; i < 3; ++i) {
    for (auto j = 0; j < 3; ++j) {
      for (auto k = 0; k < 3; ++k) {
        for (auto l = 0; l < 3; ++l) {
          res(3 * j + i, 3 * l + k) = C[i][j][k][l];
        }
      }
    }
  }
  return res;
}

template <typename BatchMatrix>
static Matrix3x3r GetBatchMatrixLane(BatchMatrix const& batch, int lane) {
  Matrix3x3r out{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out[i][j] = batch[i][j][lane];
    }
  }
  return out;
}

template <typename BatchTensor>
static Tensor3x3x3x3r GetBatchTensorLane(BatchTensor const& batch, int lane) {
  Tensor3x3x3x3r out{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          out[i][j][k][l] = batch[i][j][k][l][lane];
        }
      }
    }
  }
  return out;
}

TEST(AltSmithNeoHookeanMaterial, MatchesBatchedResponse) {
  SmithNeoHookeanMaterialParams params;
  params.youngsModulus = 1.0_r;
  params.poissonRatio = 0.4_r;
  params.psdStrategy = MaterialPsdStrategy::None;

  // Parameterization of Lame parameters for linear materials.
  auto lambda = params.youngsModulus * params.poissonRatio /
      ((1_r + params.poissonRatio) * (1_r - 2_r * params.poissonRatio));
  auto mu = params.youngsModulus / (2_r * (1_r + params.poissonRatio));
  // Reparameterization to be consistent with linear elasticity. This reparameterization
  // is mentioned in Stable Neo-Hookean Flesh Simulation [Smith et al. 2018, Sec. 3.4].
  lambda += mu * (5.0_r / 6.0_r);
  mu *= 4.0_r / 3.0_r;
  auto const alpha = 1_r + mu / lambda - mu / (lambda * 4_r);
  auto const lame = mochi::materials::BuildBatchParams<1>(params);
  auto sum = []<std::size_t... I>(auto const* v, std::index_sequence<I...>) {
    return (... + (v[I] * v[I]));
  };

  for (auto const& F : GenerateTestSet(/*randomCases*/ 100, /*maxRandomStretch*/ 1_r)) {
    std::array<Matrix3x3r, 1> const Fs = {F};
    auto const batchedF = mochi::test::LoadBatchMatrix3x3<1>(Fs);
    BatchDouble<1> energy MOCHI_NO_INIT;
    BatchReal3x3<1> pk1 MOCHI_NO_INIT;
    NdArray<BatchReal3x3<1>, 3, 3> tangent MOCHI_NO_INIT;
    BatchedSmithNeoHookeanConstitutiveResponse<1>(
        lame,
        batchedF,
        &energy,
        &pk1,
        &tangent,
        /*projectPsd*/ false,
        mochi::materials::MaterialPsdOracle::None);

    auto const stridedF = ToStridedMatrix(F);
    auto const C = mochi::materials::AltSmithNeoHookean(mu, lambda, alpha, stridedF);
    auto const state =
        mochi::materials::AltSmithNeoHookeanState<true, true, true>(mu, lambda, alpha, stridedF);

    auto ref = ToStridedMatrix(GetBatchTensorLane(tangent, 0));
    auto const refSqr = sum(ref.Data(), std::make_index_sequence<81>{});
    ref -= C;
    auto diffSqr = sum(ref.Data(), std::make_index_sequence<81>{});
    EXPECT_NEAR(diffSqr, 0_r, 1e-5_r * refSqr);

    ref = ToStridedMatrix(GetBatchTensorLane(tangent, 0));
    ref -= state.hessian;
    diffSqr = sum(ref.Data(), std::make_index_sequence<81>{});
    EXPECT_NEAR(diffSqr, 0_r, 1e-5_r * refSqr);

    EXPECT_NEAR(
        state.energy - energy[0],
        0_r,
        std::max(1e-5_r * Abs(static_cast<real>(energy[0])), 1e-5_r));

    auto refStress = ToStridedMatrix(GetBatchMatrixLane(pk1, 0));
    refStress -= state.stress;
    diffSqr = sum(refStress.Data(), std::make_index_sequence<9>{});
    EXPECT_NEAR_EQ(diffSqr, 0_r);
  }
}
