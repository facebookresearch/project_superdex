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

#include "batched_material_invariant_test_helpers.h"

#include "data/material_test_data.h"

#include <mochi_core/materials/batched_materials.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/container_utils.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/span.h>

#include <gtest/gtest.h>

#include <array>

using namespace mochi;
using namespace mochi::materials;
using namespace mochi::materials::test;

template <typename Spec, int kBS>
static void RunCommonForBatchSize() {
  auto const permutedIndices = MakePermutedElementIndices<kBS>();
  VerifyBatchedMatchesSingleLaneBatched<Spec, kBS, BatchedParamLayout::Homogeneous>(
      permutedIndices);
  VerifyBatchedMatchesSingleLaneBatched<Spec, kBS, BatchedParamLayout::PerElementHomogeneous>(
      permutedIndices);
  VerifyBatchedMatchesSingleLaneBatched<Spec, kBS, BatchedParamLayout::PerElementHeterogeneous>(
      permutedIndices);

  VerifyBatchedOutputMasks<Spec, kBS, BatchedParamLayout::Homogeneous>();
  VerifyBatchedOutputMasks<Spec, kBS, BatchedParamLayout::PerElementHomogeneous>();
  VerifyBatchedOutputMasks<Spec, kBS, BatchedParamLayout::PerElementHeterogeneous>();

  if constexpr (ShouldRunBatchedFiniteDifference<Spec>()) {
    VerifyBatchedPk1FiniteDifference<Spec, kBS, BatchedParamLayout::Homogeneous>();
    VerifyBatchedTangentFiniteDifference<Spec, kBS, BatchedParamLayout::Homogeneous>();
    VerifyBatchedPk1FiniteDifference<Spec, kBS, BatchedParamLayout::PerElementHeterogeneous>();
    VerifyBatchedTangentFiniteDifference<Spec, kBS, BatchedParamLayout::PerElementHeterogeneous>();
  }

  VerifyBatchedPsdStrategyDoesNotAffectEnergyOrPk1<Spec, kBS, BatchedParamLayout::Homogeneous>();
  VerifyBatchedPsdStrategyDoesNotAffectEnergyOrPk1<
      Spec,
      kBS,
      BatchedParamLayout::PerElementHomogeneous>();
  VerifyBatchedPsdStrategyDoesNotAffectEnergyOrPk1<
      Spec,
      kBS,
      BatchedParamLayout::PerElementHeterogeneous>();

#if MOCHI_USE_EIGEN
  VerifyBatchedPsdProjectionProducesPsdTangent<Spec, kBS, BatchedParamLayout::Homogeneous>();
  VerifyBatchedPsdProjectionProducesPsdTangent<
      Spec,
      kBS,
      BatchedParamLayout::PerElementHomogeneous>();
  VerifyBatchedPsdProjectionProducesPsdTangent<
      Spec,
      kBS,
      BatchedParamLayout::PerElementHeterogeneous>();
  if constexpr (Spec::kHasAnalyticPsdProjectionTests) {
    VerifyBatchedAnalyticPsdProjection<Spec, kBS, BatchedParamLayout::Homogeneous>(
        Spec::kAnalyticPsdProjectionRelTol);
  }
#endif
}

template <typename Spec>
static void RunCommonMaterialTests() {
  RunCommonForBatchSize<Spec, 1>();
  RunCommonForBatchSize<Spec, 4>();
  RunCommonForBatchSize<Spec, 8>();
}

template <typename Spec, int kBS>
static void RunInvariantsForBatchSize() {
  if constexpr (requires { Spec::kHasDefaultFrameInvariantRestState; }) {
    if constexpr (Spec::kHasDefaultFrameInvariantRestState) {
      VerifyBatchedDefaultUndeformedTet<Spec, kBS, BatchedParamLayout::Homogeneous>();
      VerifyBatchedDefaultUndeformedTet<Spec, kBS, BatchedParamLayout::PerElementHeterogeneous>();
    }
  }
  if constexpr (Spec::kHasFrameInvariantRestState) {
    VerifyBatchedUndeformedTet<Spec, kBS, BatchedParamLayout::Homogeneous>();
    VerifyBatchedUndeformedTet<Spec, kBS, BatchedParamLayout::PerElementHeterogeneous>();
  }
  if constexpr (Spec::kHasRotationInvariantRestState) {
    VerifyBatchedRotatedTet<Spec, kBS, BatchedParamLayout::Homogeneous>();
    VerifyBatchedRotatedTet<Spec, kBS, BatchedParamLayout::PerElementHeterogeneous>();
  }
  if constexpr (Spec::kHasMinimumEnergy) {
    VerifyBatchedMinimumEnergy<Spec, kBS, BatchedParamLayout::Homogeneous>();
    VerifyBatchedMinimumEnergy<Spec, kBS, BatchedParamLayout::PerElementHeterogeneous>();
  }
  if constexpr (Spec::kHasMatchingTarget) {
    VerifyBatchedMatchingShapeTarget<Spec, kBS, BatchedParamLayout::Homogeneous>();
    VerifyBatchedMatchingShapeTarget<Spec, kBS, BatchedParamLayout::PerElementHeterogeneous>();
  }
  if constexpr (Spec::kHasLinearElasticLimit) {
    VerifyBatchedLinearElasticLimit<Spec, kBS, BatchedParamLayout::Homogeneous>();
    VerifyBatchedLinearElasticLimit<Spec, kBS, BatchedParamLayout::PerElementHeterogeneous>();
  }
}

template <typename Spec>
static void RunInvariantMaterialTests() {
  RunInvariantsForBatchSize<Spec, 1>();
  RunInvariantsForBatchSize<Spec, 4>();
  RunInvariantsForBatchSize<Spec, 8>();
}

template <typename Spec>
static void RunMaterialTests() {
  RunCommonMaterialTests<Spec>();
  RunInvariantMaterialTests<Spec>();
}

static void RunSmithNeoHookeanOracleMaterialTests(MaterialPsdOracle oracle) {
  static_assert(
      static_cast<int>(MaterialPsdOracle::Count) == 3,
      "Please update switch statement below if MaterialPsdOracle enum changes");
  switch (oracle) {
    case MaterialPsdOracle::None:
      RunMaterialTests<SmithNeoHookeanOracleSpec<MaterialPsdOracle::None>>();
      break;
    case MaterialPsdOracle::Correct:
      RunMaterialTests<SmithNeoHookeanOracleSpec<MaterialPsdOracle::Correct>>();
      break;
    case MaterialPsdOracle::Conservative:
      RunMaterialTests<SmithNeoHookeanOracleSpec<MaterialPsdOracle::Conservative>>();
      break;
    case MaterialPsdOracle::Count:
      MOCHI_ASSERT(false, "Unexpected PSD oracle.");
      break;
  }
}

static constexpr real kGoldenDataDenominatorEpsilon = 1e-8_r;
static constexpr real kGoldenDataRelTol = 1e-4_r;

static void ExpectGoldenDataTensorNear(
    Tensor3x3x3x3r const& expected,
    Tensor3x3x3x3r const& actual) {
  auto const& expectedFlat = reinterpret_cast<NdArray<real, 81> const&>(expected);
  auto const& actualFlat = reinterpret_cast<NdArray<real, 81> const&>(actual);
  EXPECT_NEAR(
      0_r,
      Norm(expectedFlat - actualFlat),
      kGoldenDataRelTol * (Norm(expectedFlat) + kGoldenDataDenominatorEpsilon));
}

template <int kBS, typename GoldenData>
static void ExpectGoldenDataMatch(
    GoldenData const& expected,
    BatchDouble<kBS> const& energy,
    BatchReal3x3<kBS> const& pk1,
    NdArray<BatchReal3x3<kBS>, 3, 3> const& tangent,
    int lane) {
  EXPECT_NEAR(
      Abs(expected.strainEnergy - energy[lane]) /
          (Abs(expected.strainEnergy) + kGoldenDataDenominatorEpsilon),
      0_r,
      kGoldenDataRelTol);
  auto const pk1Lane = GetPk1Lane(pk1, lane);
  EXPECT_NEAR(
      Norm3x3(ToSimdMatrix(pk1Lane - expected.piolaKirchhoffStress)) /
          (Norm3x3(ToSimdMatrix(expected.piolaKirchhoffStress)) + kGoldenDataDenominatorEpsilon),
      0_r,
      kGoldenDataRelTol);
  auto const tangentLane = GetTangentLane(tangent, lane);
  ExpectGoldenDataTensorNear(expected.tangent, tangentLane);
}

template <int kBS, typename EvalFn>
[[nodiscard]] static BatchedMaterialOutputs<kBS> EvalBatchedMaterialFull(EvalFn eval) {
  BatchDouble<kBS> energy MOCHI_NO_INIT;
  BatchReal3x3<kBS> pk1 MOCHI_NO_INIT;
  NdArray<BatchReal3x3<kBS>, 3, 3> tangent MOCHI_NO_INIT;
  eval(&energy, &pk1, &tangent);
  return {.energy = energy, .pk1 = pk1, .tangent = tangent};
}

template <int kBS>
struct ActiveNeoHookeanComponentOutputs {
  BatchedMaterialOutputs<kBS> composite;
  BatchedMaterialOutputs<kBS> smith;
  BatchedMaterialOutputs<kBS> aniso;
};

template <int kBS>
[[nodiscard]] static ActiveNeoHookeanComponentOutputs<kBS> EvalActiveNeoHookeanComponents(
    ActiveNeoHookeanMaterialParams const& params,
    BatchReal3x3<kBS> const& batchedF,
    bool projectPsd) {
  return {
      .composite = EvalBatchedMaterialFull<kBS>([&](auto* energy, auto* pk1, auto* tangent) {
        BatchedActiveNeoHookeanConstitutiveResponse<kBS>(
            BuildBatchParams<kBS>(params), batchedF, energy, pk1, tangent, projectPsd);
      }),
      .smith = EvalBatchedMaterialFull<kBS>([&](auto* energy, auto* pk1, auto* tangent) {
        BatchedSmithNeoHookeanConstitutiveResponse<kBS>(
            BuildBatchParams<kBS>(params.passiveIsotropic),
            batchedF,
            energy,
            pk1,
            tangent,
            projectPsd);
      }),
      .aniso = EvalBatchedMaterialFull<kBS>([&](auto* energy, auto* pk1, auto* tangent) {
        BatchedActiveAnisoArapConstitutiveResponse<kBS>(
            BuildBatchParams<kBS>(params.activeAnisotropic),
            batchedF,
            energy,
            pk1,
            tangent,
            projectPsd);
      }),
  };
}

template <int kBS>
static void VerifyStVenantKirchhoffGoldenData() {
  // The golden data compares against independently-generated first Piola-Kirchhoff stress and
  // tangent values for randomly generated deformation gradients.
  auto const testData = mochi::materials::test::st_venant_kirchhoff_test_data::kTestData;
  StVenantKirchhoffMaterialParams params;
  params.youngsModulus = 1.0_r;
  params.poissonRatio = 0.4_r;

  for (bool projectPsd : {false, true}) {
    for (auto psdStrategy :
         {MaterialPsdStrategy::None,
          MaterialPsdStrategy::Projection,
          MaterialPsdStrategy::AbsEigenProjection}) {
      params.psdStrategy = psdStrategy;
      auto const batchedParams = BuildBatchParams<kBS>(params);

      for (int start = 0; start < isize(testData); start += kBS) {
        std::array<Matrix3x3r, kBS> Fs{};
        for (int lane = 0; lane < kBS; ++lane) {
          Fs[lane] = testData[(start + lane) % isize(testData)].deformationGradient;
        }
        auto const batchedF = mochi::test::LoadBatchMatrix3x3<kBS>(Fs);
        BatchDouble<kBS> energy MOCHI_NO_INIT;
        BatchReal3x3<kBS> pk1 MOCHI_NO_INIT;
        NdArray<BatchReal3x3<kBS>, 3, 3> tangent MOCHI_NO_INIT;
        BatchedStVenantKirchhoffConstitutiveResponse<kBS>(
            batchedParams, batchedF, &energy, &pk1, &tangent, projectPsd);

        for (int lane = 0; lane < kBS; ++lane) {
          auto const& ref = testData[(start + lane) % isize(testData)];
          ExpectGoldenDataMatch<kBS>(ref, energy, pk1, tangent, lane);
        }
      }
    }
  }
}

template <int kBS>
static void VerifySmithNeoHookeanGoldenData(MaterialPsdOracle oracle) {
  // The golden data compares against independently-generated first Piola-Kirchhoff stress and
  // tangent values for randomly generated deformation gradients.
  auto const testData = mochi::materials::test::smith_neo_hookean_test_data::kTestData;
  SmithNeoHookeanMaterialParams params;
  params.youngsModulus = 1.0_r;
  params.poissonRatio = 0.4_r;
  params.psdStrategy = MaterialPsdStrategy::None;
  auto const batchedParams = BuildBatchParams<kBS>(params);

  for (int start = 0; start < isize(testData); start += kBS) {
    std::array<Matrix3x3r, kBS> Fs{};
    for (int lane = 0; lane < kBS; ++lane) {
      Fs[lane] = testData[(start + lane) % isize(testData)].deformationGradient;
    }
    auto const batchedF = mochi::test::LoadBatchMatrix3x3<kBS>(Fs);
    BatchDouble<kBS> energy MOCHI_NO_INIT;
    BatchReal3x3<kBS> pk1 MOCHI_NO_INIT;
    NdArray<BatchReal3x3<kBS>, 3, 3> tangent MOCHI_NO_INIT;
    BatchedSmithNeoHookeanConstitutiveResponse<kBS>(
        batchedParams, batchedF, &energy, &pk1, &tangent, false, oracle);

    for (int lane = 0; lane < kBS; ++lane) {
      auto const& ref = testData[(start + lane) % isize(testData)];
      ExpectGoldenDataMatch<kBS>(ref, energy, pk1, tangent, lane);
    }
  }
}

template <int kBS>
static void VerifyActiveNeoHookeanComposite() {
  for (int testIdx = 0; testIdx < kBatchedMaterialTestCases; ++testIdx) {
    auto const params = MakeRandomMaterialParams<ActiveNeoHookeanMaterialParams>(
        testIdx, MaterialPsdStrategy::None);

    std::array<Matrix3x3r, kBS> Fs = MakeBatchFs<kBS>(testIdx);
    auto const batchedF = mochi::test::LoadBatchMatrix3x3<kBS>(Fs);

    for (bool projectPsd : {false, true}) {
      auto const outputs = EvalActiveNeoHookeanComponents<kBS>(params, batchedF, projectPsd);

      for (int lane = 0; lane < kBS; ++lane) {
        ExpectEnergyNear(
            outputs.smith.energy[lane] + outputs.aniso.energy[lane],
            outputs.composite.energy[lane]);
        ExpectMatrixNear(
            GetPk1Lane(outputs.smith.pk1, lane) + GetPk1Lane(outputs.aniso.pk1, lane),
            GetPk1Lane(outputs.composite.pk1, lane));
        ExpectTensorNear(
            GetTangentLane(outputs.smith.tangent, lane) +
                GetTangentLane(outputs.aniso.tangent, lane),
            GetTangentLane(outputs.composite.tangent, lane));
      }
    }
  }
}

static void VerifyActiveNeoHookeanDefaultParamsComposite() {
  constexpr int kBS = 1;
  ActiveNeoHookeanMaterialParams params;
  EXPECT_GT(params.passiveIsotropic.youngsModulus, 0_r);
  EXPECT_GT(params.activeAnisotropic.alpha, 0_r);
  EXPECT_GT(params.activeAnisotropic.length, 0_r);

  for (int testIdx = 0; testIdx < kBatchedMaterialTestCases; ++testIdx) {
    std::array<Matrix3x3r, kBS> Fs{
        ToNdArray3x3(ToSimdMatrix(Matrix<real, 3, 3>::Random(testIdx, -1_r, 1_r)))};
    auto const batchedF = mochi::test::LoadBatchMatrix3x3<kBS>(Fs);

    for (bool projectPsd : {false, true}) {
      auto const outputs = EvalActiveNeoHookeanComponents<kBS>(params, batchedF, projectPsd);

      ExpectEnergyNear(
          outputs.smith.energy[0] + outputs.aniso.energy[0], outputs.composite.energy[0]);
      ExpectMatrixNear(
          GetPk1Lane(outputs.smith.pk1, 0) + GetPk1Lane(outputs.aniso.pk1, 0),
          GetPk1Lane(outputs.composite.pk1, 0));
      ExpectTensorNear(
          GetTangentLane(outputs.smith.tangent, 0) + GetTangentLane(outputs.aniso.tangent, 0),
          GetTangentLane(outputs.composite.tangent, 0));
    }
  }
}

// Runs generic batched checks for the linear elastic material.
TEST(BatchedMaterials, LinearElastic) {
  RunMaterialTests<LinearElasticSpec>();
}

// Runs generic batched checks for St. Venant-Kirchhoff, including PSD projections.
TEST(BatchedMaterials, StVenantKirchhoff) {
  RunMaterialTests<StVenantKirchhoffSpec>();
}

// Runs generic Smith neo-Hookean checks for the default oracle and each explicit oracle.
TEST(BatchedMaterials, SmithNeoHookean) {
  RunMaterialTests<SmithNeoHookeanSpec>();
  for (int i = 0; i < static_cast<int>(MaterialPsdOracle::Count); ++i) {
    RunSmithNeoHookeanOracleMaterialTests(static_cast<MaterialPsdOracle>(i));
  }
}

// Runs generic batched checks for Kim neo-Hookean, including rest, rigid-motion, and linear-limit
// invariants.
TEST(BatchedMaterials, KimNeoHookean) {
  RunMaterialTests<KimNeoHookeanSpec>();
}

// Runs generic batched checks for ARAP, including rigid-motion and minimum-energy invariants.
TEST(BatchedMaterials, Arap) {
  RunMaterialTests<ArapSpec>();
}

// Runs generic batched checks for active anisotropic ARAP, including rest, rigid-motion, and
// minimum-energy invariants.
TEST(BatchedMaterials, ActiveAnisoArap) {
  RunMaterialTests<ActiveAnisoArapSpec>();
}

// Runs generic batched checks for active shape-targeting ARAP, including matching-target states.
TEST(BatchedMaterials, ActiveShapeTargetingArap) {
  RunMaterialTests<ActiveShapeTargetingArapSpec>();
}

// Runs generic batched checks for active neo-Hookean, including rest, rigid-motion, and
// minimum-energy invariants.
TEST(BatchedMaterials, ActiveNeoHookean) {
  RunMaterialTests<ActiveNeoHookeanSpec>();
}

// Compares St. Venant-Kirchhoff batched outputs against golden data.
TEST(BatchedMaterials, StVenantKirchhoffGoldenData) {
  VerifyStVenantKirchhoffGoldenData<1>();
  VerifyStVenantKirchhoffGoldenData<4>();
  VerifyStVenantKirchhoffGoldenData<8>();
}

// Compares Smith neo-Hookean batched outputs against golden data for every oracle.
TEST(BatchedMaterials, SmithNeoHookeanGoldenData) {
  for (int i = 0; i < static_cast<int>(MaterialPsdOracle::Count); ++i) {
    auto const oracle = static_cast<MaterialPsdOracle>(i);
    VerifySmithNeoHookeanGoldenData<1>(oracle);
    VerifySmithNeoHookeanGoldenData<4>(oracle);
    VerifySmithNeoHookeanGoldenData<8>(oracle);
  }
}

// Verifies active neo-Hookean equals the sum of its Smith and active anisotropic ARAP parts.
TEST(BatchedMaterials, ActiveNeoHookeanComposite) {
  VerifyActiveNeoHookeanDefaultParamsComposite();
  VerifyActiveNeoHookeanComposite<1>();
  VerifyActiveNeoHookeanComposite<4>();
  VerifyActiveNeoHookeanComposite<8>();
}

// Verifies an active anisotropic ARAP degenerate lane returns zero while other lanes remain valid.
TEST(BatchedMaterials, ActiveAnisoArapDegenerateLane) {
  constexpr int kBS = 4;

  auto params =
      MakeRandomMaterialParams<ActiveAnisoArapMaterialParams>(0, MaterialPsdStrategy::Projection);
  // Use axis-aligned fiber direction to guarantee F * a = 0 exactly in floating point.
  params.anisoDir = {1_r, 0_r, 0_r};

  // F with zero first-column entries gives F * (1, 0, 0) = (0, 0, 0) exactly.
  Matrix3x3r const Fdeg = {Real3{0_r, 1_r, 0_r}, Real3{0_r, 0_r, 1_r}, Real3{0_r, 1_r, 1_r}};
  Matrix3x3r const Fid = Eye<3, real>();
  std::array<Matrix3x3r, kBS> Fs = {Fdeg, Fid, Fid, Fid};
  auto const batchedF = mochi::test::LoadBatchMatrix3x3<kBS>(Fs);

  for (bool projectPsd : {false, true}) {
    BatchDouble<kBS> energy MOCHI_NO_INIT;
    BatchReal3x3<kBS> pk1 MOCHI_NO_INIT;
    NdArray<BatchReal3x3<kBS>, 3, 3> tangent MOCHI_NO_INIT;
    auto const batchParams = BuildBatchParams<kBS>(params);
    BatchedActiveAnisoArapConstitutiveResponse<kBS>(
        batchParams, batchedF, &energy, &pk1, &tangent, projectPsd);

    std::array<Matrix3x3r, kBS> identityFs = {Fid, Fid, Fid, Fid};
    auto const identityF = mochi::test::LoadBatchMatrix3x3<kBS>(identityFs);
    BatchDouble<kBS> identityEnergy MOCHI_NO_INIT;
    BatchReal3x3<kBS> identityPk1 MOCHI_NO_INIT;
    NdArray<BatchReal3x3<kBS>, 3, 3> identityTangent MOCHI_NO_INIT;
    BatchedActiveAnisoArapConstitutiveResponse<kBS>(
        batchParams, identityF, &identityEnergy, &identityPk1, &identityTangent, projectPsd);

    EXPECT_EQ(0.0, energy[0]);
    Matrix3x3r const zero{};
    EXPECT_EQ(zero, GetPk1Lane(pk1, 0));
    EXPECT_EQ(Tensor3x3x3x3r{}, GetTangentLane(tangent, 0));
    for (int lane = 1; lane < kBS; ++lane) {
      ExpectEnergyNear(identityEnergy[lane], energy[lane]);
      ExpectMatrixNear(GetPk1Lane(identityPk1, lane), GetPk1Lane(pk1, lane));
      ExpectTensorNear(GetTangentLane(identityTangent, lane), GetTangentLane(tangent, lane));
    }
  }
}

// Verifies per-element Lame parameters built from E/nu arrays match direct conversion.
TEST(BatchedMaterials, BuildPerElementLameParamsFromArrays) {
  // Multi-element heterogeneous case: distinct E and nu per element, including negative nu.
  DynamicArray<real> E = {1e3_r, 1e4_r, 5e5_r, 2e6_r};
  DynamicArray<real> nu = {-0.4_r, 0_r, 0.25_r, 0.45_r};
  auto const peDefault = BuildPerElementLameParams(MakeConstSpan(E), MakeConstSpan(nu));
  ASSERT_EQ(isize(E), isize(peDefault.lambda));
  ASSERT_EQ(isize(E), isize(peDefault.mu));
  EXPECT_EQ(MaterialPsdStrategy::MaterialDefault, peDefault.psdStrategy);
  for (int i = 0; i < isize(E); ++i) {
    auto const [lambda, mu] = utils::ComputeLameConstants(E[i], nu[i]);
    EXPECT_NEAR_RTOL(lambda, peDefault.lambda[i], kBatchedMaterialRelTol);
    EXPECT_NEAR_RTOL(mu, peDefault.mu[i], kBatchedMaterialRelTol);
  }

  // Custom PSD strategy is propagated.
  auto const pePsd = BuildPerElementLameParams(
      MakeConstSpan(E), MakeConstSpan(nu), MaterialPsdStrategy::Projection);
  EXPECT_EQ(MaterialPsdStrategy::Projection, pePsd.psdStrategy);

  // Single-element case.
  DynamicArray<real> Eone = {1e5_r};
  DynamicArray<real> nuOne = {0.3_r};
  auto const peOne = BuildPerElementLameParams(MakeConstSpan(Eone), MakeConstSpan(nuOne));
  EXPECT_EQ(1, isize(peOne.lambda));
  EXPECT_EQ(1, isize(peOne.mu));
  auto const [lambdaOne, muOne] = utils::ComputeLameConstants(Eone[0], nuOne[0]);
  EXPECT_NEAR_RTOL(lambdaOne, peOne.lambda[0], kBatchedMaterialRelTol);
  EXPECT_NEAR_RTOL(muOne, peOne.mu[0], kBatchedMaterialRelTol);
}

// Verifies heterogeneous per-element gather honors permuted and repeated element indices.
TEST(BatchedMaterials, PerElementGatherUsesPermutedAndRepeatedIndices) {
  constexpr int kBS = 4;
  VerifyBatchedMatchesSingleLaneBatched<
      KimNeoHookeanSpec,
      kBS,
      BatchedParamLayout::PerElementHeterogeneous>(MakePermutedElementIndices<kBS>());
  VerifyBatchedMatchesSingleLaneBatched<
      KimNeoHookeanSpec,
      kBS,
      BatchedParamLayout::PerElementHeterogeneous>(MakeRepeatedElementIndices<kBS>());
}
