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

// Generic batched material-test harness helpers for lane consistency, output masks, finite
// differences, PSD behavior, per-element parameter layouts.

#include "material_model_test_specs.h"

#include <mochi_core/test/batch_helpers.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/decomposition_utils.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/nd_array.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <type_traits>

namespace mochi::materials::test {

enum class BatchedParamLayout {
  // Broadcasts a single params object through the direct batched material overload.
  Homogeneous,
  // Builds one per-element params entry and evaluates all lanes against element index 0.
  PerElementHomogeneous,
  // Builds distinct per-element params entries and gathers lanes through explicit indices.
  PerElementHeterogeneous,
};

template <int kBS>
struct BatchedMaterialOutputs {
  BatchDouble<kBS> energy;
  BatchReal3x3<kBS> pk1;
  NdArray<BatchReal3x3<kBS>, 3, 3> tangent;
};

template <int kBS>
[[nodiscard]] NdArray<int, kBS> MakePermutedElementIndices() {
  NdArray<int, kBS> indices{};
  for (int i = 0; i < kBS; ++i) {
    indices[i] = (3 * i + 1) % kBS;
  }
  return indices;
}

template <int kBS>
[[nodiscard]] NdArray<int, kBS> MakeRepeatedElementIndices() {
  NdArray<int, kBS> indices{};
  for (int i = 0; i < kBS; ++i) {
    indices[i] = (i % 2 == 0) ? 0 : (kBS - 1);
  }
  return indices;
}

template <typename V>
[[nodiscard]] Matrix3x3r GetPk1Lane(V const& pk1, int lane) {
  Matrix3x3r out{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out[i][j] = pk1[i][j][lane];
    }
  }
  return out;
}

template <typename V>
[[nodiscard]] Tensor3x3x3x3r GetTangentLane(V const& tangent, int lane) {
  Tensor3x3x3x3r out{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          out[i][j][k][l] = tangent[i][j][k][l][lane];
        }
      }
    }
  }
  return out;
}

inline void ExpectFiniteDifferenceNear(real expected, real actual, real absTol, real relTol) {
  EXPECT_NEAR(expected, actual, Max(absTol, relTol * Max(Abs(expected), Abs(actual))));
}

template <typename Spec, int kBS>
[[nodiscard]] std::array<typename Spec::Params, kBS>
MakeElementParams(int seed, MaterialPsdStrategy psdStrategy, BatchedParamLayout layout) {
  std::array<typename Spec::Params, kBS> params{};
  for (int i = 0; i < kBS; ++i) {
    int const laneSeed =
        layout == BatchedParamLayout::PerElementHeterogeneous ? seed * kBS + i : seed;
    params[i] = Spec::RandomParams(laneSeed, psdStrategy);
  }
  return params;
}

template <int kBS>
[[nodiscard]] std::array<Matrix3x3r, kBS> MakeBatchFs(int seed) {
  static auto const testSet = GenerateTestSet(
      /*randomCases*/ kBatchedMaterialTestCases, MOCHI_USE_DOUBLE_PRECISION ? 1_r : 0.75_r);
  std::array<Matrix3x3r, kBS> Fs{};
  for (int i = 0; i < kBS; ++i) {
    Fs[i] = testSet[(seed * kBS + i) % isize(testSet)];
  }
  return Fs;
}

template <typename Spec, int kBS>
[[nodiscard]] typename Spec::Params const& GetLaneParams(
    std::array<typename Spec::Params, kBS> const& params,
    NdArray<int, kBS> const& indices,
    int lane,
    BatchedParamLayout layout) {
  if (layout == BatchedParamLayout::PerElementHeterogeneous) {
    return params[indices[lane]];
  }
  return params[0];
}

template <typename Spec, int kBS>
void EvalBatched(
    std::array<typename Spec::Params, kBS> const& params,
    NdArray<int, kBS> const& indices,
    std::array<Matrix3x3r, kBS> const& Fs,
    BatchedParamLayout layout,
    BatchDouble<kBS>* energy,
    BatchReal3x3<kBS>* pk1,
    NdArray<BatchReal3x3<kBS>, 3, 3>* tangent,
    bool projectPsd) {
  auto const batchedF = mochi::test::LoadBatchMatrix3x3<kBS>(Fs);
  if (layout == BatchedParamLayout::Homogeneous) {
    auto const batchParams = BuildBatchParams<kBS>(params[0]);
    Spec::template Eval<kBS>(batchParams, batchedF, energy, pk1, tangent, projectPsd);
  } else {
    auto const perElem = layout == BatchedParamLayout::PerElementHomogeneous
        ? BuildPerElementParams(params[0])
        : ConcatPerElementParams(params);
    auto const fn = Spec::template MakeBatchedConstitutiveResponse<kBS>(perElem);
    fn(indices, batchedF, energy, pk1, tangent, projectPsd);
  }
}

template <typename Spec, int kBS>
[[nodiscard]] BatchedMaterialOutputs<kBS> EvalBatchedFull(
    std::array<typename Spec::Params, kBS> const& params,
    NdArray<int, kBS> const& indices,
    std::array<Matrix3x3r, kBS> const& Fs,
    BatchedParamLayout layout,
    bool projectPsd) {
  BatchDouble<kBS> energy MOCHI_NO_INIT;
  BatchReal3x3<kBS> pk1 MOCHI_NO_INIT;
  NdArray<BatchReal3x3<kBS>, 3, 3> tangent MOCHI_NO_INIT;
  EvalBatched<Spec, kBS>(params, indices, Fs, layout, &energy, &pk1, &tangent, projectPsd);
  return {.energy = energy, .pk1 = pk1, .tangent = tangent};
}

template <typename Spec>
[[nodiscard]] real GetPk1AbsTol(typename Spec::Params const& params) {
  if constexpr (requires { Spec::GetPk1AbsTol(params); }) {
    return Spec::GetPk1AbsTol(params);
  }
  constexpr real kAbsTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-8_r : 1e-5_r;
  return Max(
      kAbsTol, 25_r * std::numeric_limits<real>::epsilon() * GetMaterialStiffnessScale(params));
}

template <typename ParamsT>
[[nodiscard]] real GetFdAbsTol(ParamsT const& params) {
  return Max(
      kBatchedMaterialFdAbsFloor,
      GetMaterialStiffnessScale(params) * kBatchedMaterialFdAbsStiffnessScale);
}

inline constexpr real kBatchedMaterialEnergyAbsTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-10_r : 1e-6_r;

inline void
ExpectEnergyNear(double expected, double actual, double absTol = kBatchedMaterialEnergyAbsTol) {
  EXPECT_NEAR(expected, actual, Max(kBatchedMaterialRelTol * Abs(expected), absTol));
}

template <typename Spec>
[[nodiscard]] constexpr bool NeedsSeparatedRandomSingularValues() {
  if constexpr (requires { Spec::kNeedsSeparatedRandomSingularValues; }) {
    return Spec::kNeedsSeparatedRandomSingularValues;
  } else {
    return false;
  }
}

template <typename Spec>
[[nodiscard]] bool PassesFiniteDifferenceConditioningChecks(
    Matrix3x3r const& F,
    typename Spec::Params const& params) {
  if constexpr (NeedsSeparatedRandomSingularValues<Spec>()) {
    // Check conditioning on the effective deformation the material's analytic SVD actually sees:
    // F * target for shape-targeting specs, or F itself when there is no MakeTargetDeformation.
    auto const svdInput = [&]() {
      if constexpr (requires { Spec::MakeTargetDeformation(params); }) {
        return Dot(F, Spec::MakeTargetDeformation(params));
      } else {
        return F;
      }
    }();
    Real3 sigma{};
    Matrix3x3r U{};
    Matrix3x3r VT{};
    RotationVariantSvd(svdInput, U, sigma, VT);
    return ::mochi::details::AreRandomDeformationStretchesSeparated(Abs(sigma));
  } else {
    return true;
  }
}

template <typename Spec>
[[nodiscard]] Matrix3x3r MakeDeterministicFiniteDifferenceF([[maybe_unused]]
                                                            typename Spec::Params const& params) {
  // Well-conditioned fallback when random sampling keeps failing the conditioning checks. The
  // material's analytic SVD sees the effective deformation F * target, so we return
  // F = mat * target^-1, making that effective deformation equal `mat`. The diagonal entries are
  // nonzero and spaced well beyond kRandomDeformationMinAbsStretchSeparation, so the conditioning
  // checks always pass.
  Matrix3x3r const mat{Real3{0.2_r, 0_r, 0_r}, Real3{0_r, 0.45_r, 0_r}, Real3{0_r, 0_r, 0.7_r}};
  if constexpr (requires { Spec::MakeTargetDeformation(params); }) {
    return Dot(mat, Invert(Spec::MakeTargetDeformation(params)));
  } else {
    return mat;
  }
}

template <typename Spec, int kBS>
[[nodiscard]] std::array<Matrix3x3r, kBS> MakeBatchFiniteDifferenceFs(
    int seed,
    std::array<typename Spec::Params, kBS> const& params,
    NdArray<int, kBS> const& indices,
    BatchedParamLayout layout) {
  auto generator = RandomGenerator(seed);
  constexpr int kBatchedMaterialMaxFiniteDifferenceFAttempts = 64;
  constexpr real kMaxAbsStretch = MOCHI_USE_DOUBLE_PRECISION ? 1_r : 0.75_r;
  static_assert(kMaxAbsStretch > ::mochi::details::kRandomDeformationMinAbsStretch);
  std::array<Matrix3x3r, kBS> Fs{};
  for (int lane = 0; lane < kBS; ++lane) {
    auto const& laneParams = GetLaneParams<Spec, kBS>(params, indices, lane, layout);
    bool accepted = false;
    for (int attempt = 0; attempt < kBatchedMaterialMaxFiniteDifferenceFAttempts; ++attempt) {
      Fs[lane] = ::mochi::details::MakeRandomDeformationGradient(
          generator, kMaxAbsStretch, NeedsSeparatedRandomSingularValues<Spec>());
      if (PassesFiniteDifferenceConditioningChecks<Spec>(Fs[lane], laneParams)) {
        accepted = true;
        break;
      }
    }
    if (!accepted) {
      // Preserve random coverage in the common case, but avoid making test success depend on
      // unbounded rejection sampling.
      MOCHI_LOG_WARNING(
          "Failed to sample a finite-difference deformation gradient satisfying conditioning "
          "checks after %d attempts for lane %d. Using deterministic fallback.",
          kBatchedMaterialMaxFiniteDifferenceFAttempts,
          lane);
      Fs[lane] = MakeDeterministicFiniteDifferenceF<Spec>(laneParams);
    }
  }
  return Fs;
}

template <typename ParamsT>
[[nodiscard]] real GetEnergyAbsTol(ParamsT const& params) {
  constexpr real kEps = MOCHI_USE_DOUBLE_PRECISION ? 1e-10_r : 1e-6_r;
  return kEps * Max(1_r, GetMaterialStiffnessScale(params));
}

inline void ExpectMatrixNear(
    Matrix3x3r const& expected,
    Matrix3x3r const& actual,
    real absTol = kBatchedMaterialAbsTol) {
  auto const& expectedFlat = reinterpret_cast<NdArray<real, 9> const&>(expected);
  auto const& actualFlat = reinterpret_cast<NdArray<real, 9> const&>(actual);
  EXPECT_LE(
      Norm(expectedFlat - actualFlat), Max(kBatchedMaterialRelTol * Norm(expectedFlat), absTol));
}

inline void ExpectTensorNear(
    Tensor3x3x3x3r const& expected,
    Tensor3x3x3x3r const& actual,
    real absTol = kBatchedMaterialAbsTol) {
  auto const& expectedFlat = reinterpret_cast<NdArray<real, 81> const&>(expected);
  auto const& actualFlat = reinterpret_cast<NdArray<real, 81> const&>(actual);
  EXPECT_LE(
      Norm(expectedFlat - actualFlat), Max(kBatchedMaterialRelTol * Norm(expectedFlat), absTol));
}

template <typename Spec, int kBS, BatchedParamLayout kLayout>
void VerifyBatchedMatchesSingleLaneBatched(NdArray<int, kBS> const& indices) {
  for (auto psdStrategy : Spec::kSupportedPsdStrategies) {
    for (bool projectPsd : {false, true}) {
      for (int testIdx = 0; testIdx < kBatchedMaterialTestCases; ++testIdx) {
        auto const params = MakeElementParams<Spec, kBS>(testIdx, psdStrategy, kLayout);
        auto const Fs = MakeBatchFs<kBS>(testIdx);
        auto const actual = EvalBatchedFull<Spec, kBS>(params, indices, Fs, kLayout, projectPsd);

        for (int lane = 0; lane < kBS; ++lane) {
          auto const& laneParams = GetLaneParams<Spec, kBS>(params, indices, lane, kLayout);
          std::array<typename Spec::Params, 1> expectedParams{laneParams};
          std::array<Matrix3x3r, 1> expectedFs{Fs[lane]};
          NdArray<int, 1> expectedIndices{};
          auto const expected = EvalBatchedFull<Spec, 1>(
              expectedParams,
              expectedIndices,
              expectedFs,
              BatchedParamLayout::Homogeneous,
              projectPsd);

          ExpectEnergyNear(expected.energy[0], actual.energy[lane], GetEnergyAbsTol(laneParams));
          ExpectMatrixNear(
              GetPk1Lane(expected.pk1, 0),
              GetPk1Lane(actual.pk1, lane),
              GetPk1AbsTol<Spec>(laneParams));
          ExpectTensorNear(
              GetTangentLane(expected.tangent, 0), GetTangentLane(actual.tangent, lane));
        }
      }
    }
  }
}

template <typename Spec>
[[nodiscard]] constexpr bool ShouldRunBatchedFiniteDifference() {
#if MOCHI_USE_DOUBLE_PRECISION
  return true;
#else
  if constexpr (requires { Spec::kRunFiniteDifferenceInSinglePrecision; }) {
    return Spec::kRunFiniteDifferenceInSinglePrecision;
  } else {
    return false;
  }
#endif
}

template <typename Spec>
[[nodiscard]] constexpr bool ShouldRunProjectedFiniteDifference() {
  if constexpr (requires { Spec::kRunProjectedFiniteDifference; }) {
    return Spec::kRunProjectedFiniteDifference;
  } else {
    return false;
  }
}

template <typename Spec, int kBS>
[[nodiscard]] constexpr bool ShouldSkipPsdTangentCheck(
    std::array<typename Spec::Params, kBS> const& params,
    NdArray<int, kBS> const& indices,
    int lane,
    BatchedParamLayout layout,
    MaterialPsdStrategy psdStrategy) {
  if constexpr (std::is_same_v<typename Spec::Params, StVenantKirchhoffMaterialParams>) {
    // StVK Fast projection does not guarantee PSD tangents for auxetic materials.
    return psdStrategy == MaterialPsdStrategy::Fast &&
        GetLaneParams<Spec, kBS>(params, indices, lane, layout).poissonRatio < 0_r;
  } else {
    return false;
  }
}

template <typename Spec, int kBS, BatchedParamLayout kLayout>
void VerifyBatchedOutputMasks() {
  auto const indices = MakePermutedElementIndices<kBS>();
  for (auto psdStrategy : Spec::kSupportedPsdStrategies) {
    for (bool projectPsd : {false, true}) {
      for (int testIdx = 0; testIdx < kBatchedMaterialTestCases; ++testIdx) {
        auto const params = MakeElementParams<Spec, kBS>(testIdx, psdStrategy, kLayout);
        auto const Fs = MakeBatchFs<kBS>(testIdx);
        auto const baseline = EvalBatchedFull<Spec, kBS>(params, indices, Fs, kLayout, projectPsd);

        for (int mask = 0; mask < 8; ++mask) {
          bool const evalEnergy = (mask & 1) != 0;
          bool const evalPK1 = (mask & 2) != 0;
          bool const evalTangent = (mask & 4) != 0;

          BatchDouble<kBS> energy MOCHI_NO_INIT;
          BatchReal3x3<kBS> pk1 MOCHI_NO_INIT;
          NdArray<BatchReal3x3<kBS>, 3, 3> tangent MOCHI_NO_INIT;
          EvalBatched<Spec, kBS>(
              params,
              indices,
              Fs,
              kLayout,
              evalEnergy ? &energy : nullptr,
              evalPK1 ? &pk1 : nullptr,
              evalTangent ? &tangent : nullptr,
              projectPsd);

          for (int lane = 0; lane < kBS; ++lane) {
            auto const& laneParams = GetLaneParams<Spec, kBS>(params, indices, lane, kLayout);
            if (evalEnergy) {
              ExpectEnergyNear(baseline.energy[lane], energy[lane]);
            }
            if (evalPK1) {
              ExpectMatrixNear(
                  GetPk1Lane(baseline.pk1, lane),
                  GetPk1Lane(pk1, lane),
                  GetPk1AbsTol<Spec>(laneParams));
            }
            if (evalTangent) {
              ExpectTensorNear(
                  GetTangentLane(baseline.tangent, lane), GetTangentLane(tangent, lane));
            }
          }
        }
      }
    }
  }
}

template <typename Spec, int kBS, BatchedParamLayout kLayout>
void VerifyBatchedPk1FiniteDifference() {
  auto const indices = MakePermutedElementIndices<kBS>();
  for (auto psdStrategy : Spec::kSupportedPsdStrategies) {
    for (bool projectPsd : {false, true}) {
      if (projectPsd && !ShouldRunProjectedFiniteDifference<Spec>()) {
        continue;
      }
      for (int testIdx = 0; testIdx < kBatchedMaterialTestCases; ++testIdx) {
        auto const params = MakeElementParams<Spec, kBS>(testIdx, psdStrategy, kLayout);
        auto Fs = MakeBatchFiniteDifferenceFs<Spec, kBS>(testIdx, params, indices, kLayout);
        auto const base = EvalBatchedFull<Spec, kBS>(params, indices, Fs, kLayout, projectPsd);
        real const offset = std::cbrt(std::numeric_limits<real>::epsilon());

        // Perturb one lane at a time while evaluating the full batch to catch lane cross-talk in
        // the production kBS path.
        for (int lane = 0; lane < kBS; ++lane) {
          for (int row : {0, 1, 2}) {
            for (int col : {0, 1, 2}) {
              auto forwardFs = Fs;
              forwardFs[lane][row][col] += offset;
              auto const forward =
                  EvalBatchedFull<Spec, kBS>(params, indices, forwardFs, kLayout, projectPsd);

              auto backwardFs = Fs;
              backwardFs[lane][row][col] -= offset;
              auto const backward =
                  EvalBatchedFull<Spec, kBS>(params, indices, backwardFs, kLayout, projectPsd);

              real const fdPk1 = (forward.energy[lane] - backward.energy[lane]) / (2_r * offset);
              auto const basePk1 = GetPk1Lane(base.pk1, lane);
              auto const& laneParams = GetLaneParams<Spec, kBS>(params, indices, lane, kLayout);
              ExpectFiniteDifferenceNear(
                  basePk1[row][col], fdPk1, GetFdAbsTol(laneParams), kBatchedMaterialFdRelTol);
            }
          }
        }
      }
    }
  }
}

template <typename Spec, int kBS, BatchedParamLayout kLayout>
void VerifyBatchedTangentFiniteDifference() {
  auto const indices = MakePermutedElementIndices<kBS>();
  for (auto psdStrategy : Spec::kSupportedPsdStrategies) {
    for (bool projectPsd : {false, true}) {
      if (projectPsd && !ShouldRunProjectedFiniteDifference<Spec>()) {
        continue;
      }
      for (int testIdx = 0; testIdx < kBatchedMaterialTestCases; ++testIdx) {
        auto const params = MakeElementParams<Spec, kBS>(testIdx, psdStrategy, kLayout);
        auto Fs = MakeBatchFiniteDifferenceFs<Spec, kBS>(testIdx, params, indices, kLayout);
        auto const base = EvalBatchedFull<Spec, kBS>(params, indices, Fs, kLayout, projectPsd);
        real const offset = std::cbrt(std::numeric_limits<real>::epsilon());

        // Perturb one lane at a time while evaluating the full batch to catch lane cross-talk in
        // the production kBS path.
        for (int lane = 0; lane < kBS; ++lane) {
          for (int row : {0, 1, 2}) {
            for (int col : {0, 1, 2}) {
              auto forwardFs = Fs;
              forwardFs[lane][row][col] += offset;
              auto const forward =
                  EvalBatchedFull<Spec, kBS>(params, indices, forwardFs, kLayout, projectPsd);

              auto backwardFs = Fs;
              backwardFs[lane][row][col] -= offset;
              auto const backward =
                  EvalBatchedFull<Spec, kBS>(params, indices, backwardFs, kLayout, projectPsd);

              auto const tangent = GetTangentLane(base.tangent, lane);
              auto const forwardPk1 = GetPk1Lane(forward.pk1, lane);
              auto const backwardPk1 = GetPk1Lane(backward.pk1, lane);
              auto const& laneParams = GetLaneParams<Spec, kBS>(params, indices, lane, kLayout);
              real const absTol = GetFdAbsTol(laneParams);
              for (int i : {0, 1, 2}) {
                for (int j : {0, 1, 2}) {
                  real const fdPk1 = (forwardPk1[i][j] - backwardPk1[i][j]) / (2_r * offset);
                  ExpectFiniteDifferenceNear(
                      tangent[i][j][row][col], fdPk1, absTol, kBatchedMaterialFdRelTol);
                }
              }
            }
          }
        }
      }
    }
  }
}

template <typename Spec, int kBS, BatchedParamLayout kLayout>
void VerifyBatchedPsdStrategyDoesNotAffectEnergyOrPk1() {
  auto const indices = MakePermutedElementIndices<kBS>();
  for (int testIdx = 0; testIdx < kBatchedMaterialTestCases; ++testIdx) {
    auto const baseParams =
        MakeElementParams<Spec, kBS>(testIdx, MaterialPsdStrategy::None, kLayout);
    auto const Fs = MakeBatchFs<kBS>(testIdx);
    auto const base = EvalBatchedFull<Spec, kBS>(baseParams, indices, Fs, kLayout, false);

    for (auto psdStrategy : Spec::kSupportedPsdStrategies) {
      for (bool projectPsd : {false, true}) {
        auto const params = MakeElementParams<Spec, kBS>(testIdx, psdStrategy, kLayout);
        auto const out = EvalBatchedFull<Spec, kBS>(params, indices, Fs, kLayout, projectPsd);
        for (int lane = 0; lane < kBS; ++lane) {
          auto const& laneParams = GetLaneParams<Spec, kBS>(params, indices, lane, kLayout);
          ExpectEnergyNear(base.energy[lane], out.energy[lane], GetEnergyAbsTol(laneParams));
          ExpectMatrixNear(
              GetPk1Lane(base.pk1, lane),
              GetPk1Lane(out.pk1, lane),
              GetPk1AbsTol<Spec>(laneParams));
        }
      }
    }
  }
}

#if MOCHI_USE_EIGEN
template <typename Spec, int kBS, BatchedParamLayout kLayout>
void VerifyBatchedPsdProjectionProducesPsdTangent() {
  auto const indices = MakePermutedElementIndices<kBS>();
  auto const testSet = GenerateTestSet(kBatchedMaterialTestCases, 1_r);
  for (auto psdStrategy : Spec::kSupportedPsdStrategies) {
    if (psdStrategy == MaterialPsdStrategy::None) {
      continue;
    }
    for (int testIdx = 0; testIdx < isize(testSet); ++testIdx) {
      auto const params = MakeElementParams<Spec, kBS>(testIdx, psdStrategy, kLayout);
      std::array<Matrix3x3r, kBS> Fs{};
      for (int lane = 0; lane < kBS; ++lane) {
        Fs[lane] = testSet[(testIdx + lane) % isize(testSet)];
      }
      auto const out = EvalBatchedFull<Spec, kBS>(params, indices, Fs, kLayout, true);
      for (int lane = 0; lane < kBS; ++lane) {
        if (ShouldSkipPsdTangentCheck<Spec, kBS>(params, indices, lane, kLayout, psdStrategy)) {
          continue;
        }
        auto tangent = GetTangentLane(out.tangent, lane);
        using EMatType = Eigen::Matrix<real, 9, 9, Eigen::RowMajor | Eigen::DontAlign>;
        auto eigenMat = Eigen::Map<EMatType>(&tangent[0][0][0][0]);
        auto eigenvals = Eigen::SelfAdjointEigenSolver<EMatType>(eigenMat).eigenvalues();
        real const tol = GetPsdProjectionEigenvalueTol(eigenMat);
        EXPECT_GE(eigenvals.minCoeff(), -tol);
      }
    }
  }
}

template <typename Spec, int kBS, BatchedParamLayout kLayout>
void VerifyBatchedAnalyticPsdProjection(real relTol) {
  auto const indices = MakePermutedElementIndices<kBS>();
  auto const testSet =
      GenerateTestSet(kBatchedMaterialTestCases, 1_r, NeedsSeparatedRandomSingularValues<Spec>());
  for (auto psdStrategy : kAnalyticPsdProjectionStrategies) {
    if (!IsPsdStrategySupported<typename Spec::Params>(psdStrategy)) {
      continue;
    }
    for (int testIdx = 0; testIdx < isize(testSet); ++testIdx) {
      auto const params = MakeElementParams<Spec, kBS>(testIdx, psdStrategy, kLayout);
      std::array<Matrix3x3r, kBS> Fs{};
      for (int lane = 0; lane < kBS; ++lane) {
        Fs[lane] = testSet[(testIdx + lane) % isize(testSet)];
      }
      auto const unprojected = EvalBatchedFull<Spec, kBS>(params, indices, Fs, kLayout, false);
      auto const projected = EvalBatchedFull<Spec, kBS>(params, indices, Fs, kLayout, true);
      for (int lane = 0; lane < kBS; ++lane) {
        auto ref = GetTangentLane(unprojected.tangent, lane);
        EigenProjectPsd(ref, psdStrategy);
        auto const actual = GetTangentLane(projected.tangent, lane);
        auto const& refFlat = reinterpret_cast<NdArray<real, 81> const&>(ref);
        auto const& actualFlat = reinterpret_cast<NdArray<real, 81> const&>(actual);
        EXPECT_LE(Norm(refFlat - actualFlat), relTol * Norm(refFlat));
      }
    }
  }
}
#endif // MOCHI_USE_EIGEN

} // namespace mochi::materials::test
