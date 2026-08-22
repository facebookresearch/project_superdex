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

// Batched material invariant checks that require tet geometry: rest states, rigid motion,
// minimum-energy configurations, shape targets, linear-elastic limits.

#include "batched_material_test_helpers.h"

#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_core/utils/array.h>
#include <mochi_core/utils/matrix_utils.h>

#include <array>

namespace mochi::materials::test {

using Vector3r = ColumnVector<real, 3>;
using Matrix3r = Matrix<real, 3, 3>;

struct RegularTetFixture {
  Array<Vector3r, 4> restVertices;
  Matrix3r dmInv;
};

[[nodiscard]] inline RegularTetFixture MakeRegularTetFixture() {
  RegularTetFixture fixture{
      .restVertices = {
          Vector3r{1_r, -1_r, -1_r},
          Vector3r{1_r, 1_r, 1_r},
          Vector3r{-1_r, 1_r, -1_r},
          Vector3r{-1_r, -1_r, 1_r}}};
  fixture.dmInv = ComputeDmInv(
      fixture.restVertices[0],
      fixture.restVertices[1],
      fixture.restVertices[2],
      fixture.restVertices[3]);
  return fixture;
}

[[nodiscard]] inline Array<Vector3r, 4> MakeDeformedTetVertices(
    RegularTetFixture const& fixture,
    Matrix3x3r const& F) {
  Matrix3r dm{};
  dm.Col(0) = fixture.restVertices[1] - fixture.restVertices[0];
  dm.Col(1) = fixture.restVertices[2] - fixture.restVertices[0];
  dm.Col(2) = fixture.restVertices[3] - fixture.restVertices[0];
  Matrix3r const ds = AsView<real, 3, 3>(F) * dm;
  return {
      fixture.restVertices[0],
      ds.Col(0) + fixture.restVertices[0],
      ds.Col(1) + fixture.restVertices[0],
      ds.Col(2) + fixture.restVertices[0]};
}

template <typename Spec, int kBS>
void PrepareInvariantParams(std::array<typename Spec::Params, kBS>& params) {
  if constexpr (requires(typename Spec::Params& param) { Spec::PrepareInvariantParam(param); }) {
    for (auto& param : params) {
      Spec::PrepareInvariantParam(param);
    }
  }
}

template <typename Spec>
[[nodiscard]] Array<Vector3r, 4> MakeZeroEnergyTetVertices(
    RegularTetFixture const& fixture,
    typename Spec::Params const& params) {
  if constexpr (Spec::kHasMatchingTarget) {
    return MakeDeformedTetVertices(fixture, Spec::MakeTargetDeformation(params));
  } else {
    return fixture.restVertices;
  }
}

[[nodiscard]] inline Matrix3x3r ComputeFNdArray(
    Vector3r const& v0,
    Vector3r const& v1,
    Vector3r const& v2,
    Vector3r const& v3,
    Matrix3r const& dmInv) {
  return ToNdArray3x3(ToSimdMatrix(ComputeF(v0, v1, v2, v3, dmInv)));
}

template <typename Spec>
[[nodiscard]] Matrix3x3r MakeZeroEnergyF(
    RegularTetFixture const& fixture,
    typename Spec::Params const& params) {
  auto const vertices = MakeZeroEnergyTetVertices<Spec>(fixture, params);
  return ComputeFNdArray(vertices[0], vertices[1], vertices[2], vertices[3], fixture.dmInv);
}

[[nodiscard]] inline Tensor3x3x3x3r MakeIsotropicLinearElasticTangent(
    real youngsModulus,
    real poissonRatio) {
  auto const [lambda, mu] = utils::ComputeLameConstants(youngsModulus, poissonRatio);
  Tensor3x3x3x3r tangent{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          tangent[i][j][k][l] =
              lambda * (i == j) * (k == l) + mu * ((i == k) * (j == l) + (i == l) * (j == k));
        }
      }
    }
  }
  return tangent;
}

template <typename Spec>
[[nodiscard]] double ExpectedRestEnergy(typename Spec::Params const& params) {
  if constexpr (requires { Spec::ExpectedRestEnergy(params); }) {
    return Spec::ExpectedRestEnergy(params);
  } else {
    return 0.0;
  }
}

template <typename Spec, int kBS, BatchedParamLayout kLayout>
void VerifyBatchedDefaultUndeformedTet() {
  auto const indices = MakePermutedElementIndices<kBS>();
  std::array<Matrix3x3r, kBS> Fs{};
  for (int lane = 0; lane < kBS; ++lane) {
    Fs[lane] = Eye<3, real>();
  }

  std::array<typename Spec::Params, kBS> params{};
  for (bool projectPsd : {false, true}) {
    auto const out = EvalBatchedFull<Spec, kBS>(params, indices, Fs, kLayout, projectPsd);
    for (int lane = 0; lane < kBS; ++lane) {
      auto const& laneParams = GetLaneParams<Spec, kBS>(params, indices, lane, kLayout);
      Matrix3x3r const zero{};
      ExpectEnergyNear(0.0, out.energy[lane], GetEnergyAbsTol(laneParams));
      ExpectMatrixNear(zero, GetPk1Lane(out.pk1, lane), GetPk1AbsTol<Spec>(laneParams));
    }
  }
}

template <typename Spec, int kBS, BatchedParamLayout kLayout>
void VerifyBatchedUndeformedTet() {
  auto const fixture = MakeRegularTetFixture();
  auto const indices = MakePermutedElementIndices<kBS>();
  std::array<Matrix3x3r, kBS> Fs{};
  for (int lane = 0; lane < kBS; ++lane) {
    Fs[lane] = ComputeFNdArray(
        fixture.restVertices[0],
        fixture.restVertices[1],
        fixture.restVertices[2],
        fixture.restVertices[3],
        fixture.dmInv);
  }

  auto params =
      MakeElementParams<Spec, kBS>(/*seed*/ 123, MaterialPsdStrategy::MaterialDefault, kLayout);
  PrepareInvariantParams<Spec, kBS>(params);
  for (bool projectPsd : {false, true}) {
    auto const out = EvalBatchedFull<Spec, kBS>(params, indices, Fs, kLayout, projectPsd);
    for (int lane = 0; lane < kBS; ++lane) {
      auto const& laneParams = GetLaneParams<Spec, kBS>(params, indices, lane, kLayout);
      ExpectEnergyNear(
          ExpectedRestEnergy<Spec>(laneParams), out.energy[lane], GetEnergyAbsTol(laneParams));
      Matrix3x3r const zero{};
      ExpectMatrixNear(zero, GetPk1Lane(out.pk1, lane), GetPk1AbsTol<Spec>(laneParams));
    }
  }
}

template <typename Spec, int kBS, BatchedParamLayout kLayout>
void VerifyBatchedRotatedTet() {
  auto const fixture = MakeRegularTetFixture();
  auto const indices = MakePermutedElementIndices<kBS>();

  for (int testIdx = 0; testIdx < kBatchedMaterialTestCases; ++testIdx) {
    auto params =
        MakeElementParams<Spec, kBS>(testIdx, MaterialPsdStrategy::MaterialDefault, kLayout);
    PrepareInvariantParams<Spec, kBS>(params);
    std::array<Matrix3x3r, kBS> restFs{};
    std::array<Matrix3x3r, kBS> rotatedFs{};
    for (int lane = 0; lane < kBS; ++lane) {
      auto const& laneParams = GetLaneParams<Spec, kBS>(params, indices, lane, kLayout);
      auto const vertices = MakeZeroEnergyTetVertices<Spec>(fixture, laneParams);
      restFs[lane] = MakeZeroEnergyF<Spec>(fixture, laneParams);
      auto generator = RandomGenerator(testIdx * kBS + lane);
      Matrix3r const R = GetRandomRotationMatrix(generator);
      rotatedFs[lane] = ComputeFNdArray(
          R * vertices[0], R * vertices[1], R * vertices[2], R * vertices[3], fixture.dmInv);
    }
    for (bool projectPsd : {false, true}) {
      auto const rest = EvalBatchedFull<Spec, kBS>(params, indices, restFs, kLayout, projectPsd);
      auto const rotated =
          EvalBatchedFull<Spec, kBS>(params, indices, rotatedFs, kLayout, projectPsd);
      for (int lane = 0; lane < kBS; ++lane) {
        auto const& laneParams = GetLaneParams<Spec, kBS>(params, indices, lane, kLayout);
        ExpectEnergyNear(rest.energy[lane], rotated.energy[lane], GetEnergyAbsTol(laneParams));
        Matrix3x3r const zero{};
        ExpectMatrixNear(zero, GetPk1Lane(rotated.pk1, lane), GetPk1AbsTol<Spec>(laneParams));
      }
    }
  }
}

template <typename Spec, int kBS, BatchedParamLayout kLayout>
void VerifyBatchedMinimumEnergy() {
  auto const fixture = MakeRegularTetFixture();
  auto const indices = MakePermutedElementIndices<kBS>();
  auto params =
      MakeElementParams<Spec, kBS>(/*seed*/ 234, MaterialPsdStrategy::MaterialDefault, kLayout);
  PrepareInvariantParams<Spec, kBS>(params);

  std::array<Matrix3x3r, kBS> restFs{};
  for (int lane = 0; lane < kBS; ++lane) {
    auto const& laneParams = GetLaneParams<Spec, kBS>(params, indices, lane, kLayout);
    restFs[lane] = MakeZeroEnergyF<Spec>(fixture, laneParams);
  }
  for (bool projectPsd : {false, true}) {
    auto const rest = EvalBatchedFull<Spec, kBS>(params, indices, restFs, kLayout, projectPsd);

    constexpr int kTetVertexCount = 4;
    for (int vertIndex = 0; vertIndex < kTetVertexCount; ++vertIndex) {
      for (int testIdx = 0; testIdx < kBatchedMaterialTestCases; ++testIdx) {
        std::array<Matrix3x3r, kBS> perturbedFs{};
        for (int lane = 0; lane < kBS; ++lane) {
          auto const& laneParams = GetLaneParams<Spec, kBS>(params, indices, lane, kLayout);
          auto vertices = MakeZeroEnergyTetVertices<Spec>(fixture, laneParams);
          unsigned int const seed = (vertIndex * kBatchedMaterialTestCases + testIdx) * kBS + lane;
          vertices[vertIndex] += Vector3r::Random(seed, -0.25_r, 0.25_r);
          perturbedFs[lane] =
              ComputeFNdArray(vertices[0], vertices[1], vertices[2], vertices[3], fixture.dmInv);
        }
        auto const perturbed =
            EvalBatchedFull<Spec, kBS>(params, indices, perturbedFs, kLayout, projectPsd);
        for (int lane = 0; lane < kBS; ++lane) {
          auto const& laneParams = GetLaneParams<Spec, kBS>(params, indices, lane, kLayout);
          EXPECT_GE(perturbed.energy[lane] + GetEnergyAbsTol(laneParams), rest.energy[lane]);
        }
      }
    }
  }
}

template <typename Spec, int kBS, BatchedParamLayout kLayout>
void VerifyBatchedMatchingShapeTarget() {
  auto const indices = MakePermutedElementIndices<kBS>();
  auto const params =
      MakeElementParams<Spec, kBS>(/*seed*/ 456, MaterialPsdStrategy::MaterialDefault, kLayout);
  std::array<Matrix3x3r, kBS> Fs{};
  for (int lane = 0; lane < kBS; ++lane) {
    Fs[lane] =
        Spec::MakeTargetDeformation(GetLaneParams<Spec, kBS>(params, indices, lane, kLayout));
  }
  for (bool projectPsd : {false, true}) {
    auto const out = EvalBatchedFull<Spec, kBS>(params, indices, Fs, kLayout, projectPsd);
    for (int lane = 0; lane < kBS; ++lane) {
      auto const& laneParams = GetLaneParams<Spec, kBS>(params, indices, lane, kLayout);
      Matrix3x3r const zero{};
      ExpectEnergyNear(0.0, out.energy[lane], GetEnergyAbsTol(laneParams));
      ExpectMatrixNear(zero, GetPk1Lane(out.pk1, lane), GetPk1AbsTol<Spec>(laneParams));
    }
  }
}

template <typename Spec, int kBS, BatchedParamLayout kLayout>
void VerifyBatchedLinearElasticLimit() {
  auto const indices = MakePermutedElementIndices<kBS>();
  for (int testIdx = 0; testIdx < kBatchedMaterialTestCases; ++testIdx) {
    auto const params = MakeElementParams<Spec, kBS>(testIdx, MaterialPsdStrategy::None, kLayout);
    std::array<Matrix3x3r, kBS> Fs{};
    for (int lane = 0; lane < kBS; ++lane) {
      Fs[lane] = Eye<3, real>();
    }
    auto const out = EvalBatchedFull<Spec, kBS>(params, indices, Fs, kLayout, false);
    for (int lane = 0; lane < kBS; ++lane) {
      auto const& laneParams = GetLaneParams<Spec, kBS>(params, indices, lane, kLayout);
      auto const expected =
          MakeIsotropicLinearElasticTangent(laneParams.youngsModulus, laneParams.poissonRatio);
      real const tol = laneParams.youngsModulus * (MOCHI_USE_DOUBLE_PRECISION ? 1e-12_r : 1e-5_r);
      ExpectTensorNear(expected, GetTangentLane(out.tangent, lane), tol);
    }
  }
}

} // namespace mochi::materials::test
