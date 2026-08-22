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

// Per-material test specifications for the batched material harness: parameter factories,
// tolerances, supported invariants, and calls into each batched constitutive response.

#include "material_test_helpers.h"

#include <mochi_core/materials/batched_materials.h>
#include <mochi_core/materials/material_utils.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/container_utils.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/rand_utils.h>

#include <array>
#include <cmath>
#include <limits>

namespace mochi::materials::test {

inline constexpr int kBatchedMaterialTestCases = MOCHI_DEBUG ? 20 : 200;
inline constexpr real kBatchedMaterialRelTol = MOCHI_USE_DOUBLE_PRECISION ? 2e-8_r : 2e-2_r;
inline constexpr real kBatchedMaterialAbsTol = 1e-8_r;
inline constexpr real kBatchedMaterialFdAbsFloor = MOCHI_USE_DOUBLE_PRECISION ? 1e-8_r : 1e-5_r;
inline constexpr real kBatchedMaterialFdAbsStiffnessScale =
    MOCHI_USE_DOUBLE_PRECISION ? 1e-7_r : 5e-4_r;
inline constexpr real kBatchedMaterialFdRelTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-7_r : 5e-4_r;
inline constexpr real kMinTestYoungsModulus = 1_r;
inline constexpr real kMaxTestYoungsModulus = 1e7_r;
inline constexpr real kMinTestPoissonRatio = -0.99_r;
inline constexpr real kMaxTestPoissonRatio = 0.49_r;
inline constexpr real kMinTestArapStiffness = 0.1_r;
inline constexpr real kMaxTestArapStiffness = 1e6_r;
inline constexpr real kMinTestAnisoAlpha = 0.1_r;
inline constexpr real kMaxTestAnisoAlpha = 1e6_r;

template <typename ParamsT>
[[nodiscard]] ParamsT MakeRandomMaterialParams(unsigned int seed, MaterialPsdStrategy strategy);

template <typename ParamsT>
[[nodiscard]] ParamsT MakeRandomYoungsPoissonParams(
    unsigned int seed,
    MaterialPsdStrategy strategy) {
  auto generator = RandomGenerator(seed);
  return {
      .youngsModulus = RandomUniformValue(generator, kMinTestYoungsModulus, kMaxTestYoungsModulus),
      .poissonRatio = RandomUniformValue(generator, kMinTestPoissonRatio, kMaxTestPoissonRatio),
      .psdStrategy = strategy};
}

template <>
[[nodiscard]] inline LinearElasticMaterialParams MakeRandomMaterialParams(
    unsigned int seed,
    MaterialPsdStrategy /*strategy*/) {
  auto generator = RandomGenerator(seed);
  return {
      .youngsModulus = RandomUniformValue(generator, kMinTestYoungsModulus, kMaxTestYoungsModulus),
      .poissonRatio = RandomUniformValue(generator, kMinTestPoissonRatio, kMaxTestPoissonRatio)};
}

template <>
[[nodiscard]] inline StVenantKirchhoffMaterialParams MakeRandomMaterialParams(
    unsigned int seed,
    MaterialPsdStrategy strategy) {
  return MakeRandomYoungsPoissonParams<StVenantKirchhoffMaterialParams>(seed, strategy);
}

template <>
[[nodiscard]] inline SmithNeoHookeanMaterialParams MakeRandomMaterialParams(
    unsigned int seed,
    MaterialPsdStrategy strategy) {
  return MakeRandomYoungsPoissonParams<SmithNeoHookeanMaterialParams>(seed, strategy);
}

template <>
[[nodiscard]] inline KimNeoHookeanMaterialParams MakeRandomMaterialParams(
    unsigned int seed,
    MaterialPsdStrategy strategy) {
  return MakeRandomYoungsPoissonParams<KimNeoHookeanMaterialParams>(seed, strategy);
}

template <>
[[nodiscard]] inline ArapMaterialParams MakeRandomMaterialParams(
    unsigned int seed,
    MaterialPsdStrategy strategy) {
  auto generator = RandomGenerator(seed);
  return {
      .stiffness = RandomUniformValue(generator, kMinTestArapStiffness, kMaxTestArapStiffness),
      .psdStrategy = strategy};
}

template <>
[[nodiscard]] inline ActiveAnisoArapMaterialParams MakeRandomMaterialParams(
    unsigned int seed,
    MaterialPsdStrategy strategy) {
  auto generator = RandomGenerator(seed);
  ActiveAnisoArapMaterialParams params;
  params.alpha = RandomUniformValue(generator, kMinTestAnisoAlpha, kMaxTestAnisoAlpha);
  params.length = RandomUniformValue(generator, 0_r, 2_r);
  params.anisoDir = ActiveAnisoArapMaterialParams::ComputeFiberDirection(
      kPI * RandomUniformValue(generator, 0_r, 2_r), kPI * RandomUniformValue(generator, 0_r, 2_r));
  params.psdStrategy = strategy;
  return params;
}

template <>
[[nodiscard]] inline ActiveShapeTargetingArapMaterialParams MakeRandomMaterialParams(
    unsigned int seed,
    MaterialPsdStrategy strategy) {
  auto generator = RandomGenerator(seed);
  ActiveShapeTargetingArapMaterialParams params;
  params.stiffness = RandomUniformValue(generator, kMinTestArapStiffness, kMaxTestArapStiffness);
  // The distribution below ensures the target deformation I + shapeTargetTensor is invertible.
  // Required to avoid ill-conditioning in finite-difference tests.
  params.shapeTargetTensor = {
      RandomUniformValue(generator, -0.1_r, 0.45_r),
      RandomUniformValue(generator, -0.1_r, 0.45_r),
      RandomUniformValue(generator, -0.1_r, 0.45_r),
      RandomUniformValue(generator, -0.1_r, 0.45_r),
      RandomUniformValue(generator, -0.1_r, 0.45_r),
      RandomUniformValue(generator, -0.1_r, 0.45_r)};
  params.psdStrategy = strategy;
  return params;
}

template <>
[[nodiscard]] inline ActiveNeoHookeanMaterialParams MakeRandomMaterialParams(
    unsigned int seed,
    MaterialPsdStrategy strategy) {
  ActiveNeoHookeanMaterialParams params;
  params.passiveIsotropic = MakeRandomMaterialParams<SmithNeoHookeanMaterialParams>(seed, strategy);
  params.activeAnisotropic =
      MakeRandomMaterialParams<ActiveAnisoArapMaterialParams>(seed, strategy);
  return params;
}

template <typename ParamsT>
[[nodiscard]] real GetMaterialStiffnessScale(ParamsT const& params) {
  if constexpr (requires {
                  params.youngsModulus;
                  params.poissonRatio;
                }) {
    auto const [lambda, mu] =
        utils::ComputeLameConstants(params.youngsModulus, params.poissonRatio);
    return Max(Abs(lambda), Abs(mu));
  } else if constexpr (requires { params.stiffness; }) {
    return params.stiffness;
  } else if constexpr (requires { params.alpha; }) {
    return params.alpha;
  } else if constexpr (requires { params.passiveIsotropic; }) {
    return Max(
        GetMaterialStiffnessScale(params.passiveIsotropic),
        GetMaterialStiffnessScale(params.activeAnisotropic));
  } else {
    return 1_r;
  }
}

inline void AppendPerElementParams(PerElementLameParams& dst, PerElementLameParams const& src) {
  Append(dst.lambda, src.lambda);
  Append(dst.mu, src.mu);
}

inline void AppendPerElementParams(PerElementArapParams& dst, PerElementArapParams const& src) {
  Append(dst.stiffness, src.stiffness);
}

inline void AppendPerElementParams(
    PerElementActiveAnisoArapParams& dst,
    PerElementActiveAnisoArapParams const& src) {
  Append(dst.alpha, src.alpha);
  Append(dst.length, src.length);
  Append(dst.anisoDir, src.anisoDir);
}

inline void AppendPerElementParams(
    PerElementActiveShapeTargetingArapParams& dst,
    PerElementActiveShapeTargetingArapParams const& src) {
  Append(dst.stiffness, src.stiffness);
  Append(dst.shapeTargetTensor, src.shapeTargetTensor);
}

inline void AppendPerElementParams(
    PerElementActiveNeoHookeanParams& dst,
    PerElementActiveNeoHookeanParams const& src) {
  AppendPerElementParams(dst.lame, src.lame);
  AppendPerElementParams(dst.aniso, src.aniso);
}

// Build a PerElement*Params containing N entries by concatenating per-element data from each
// scalar params object; each entry contributes one element through BuildPerElementParams.
template <typename ParamsT, size_t N>
[[nodiscard]] auto ConcatPerElementParams(std::array<ParamsT, N> const& params) {
  static_assert(N > 0);
  auto out = BuildPerElementParams(params[0]);
  for (size_t i = 1; i < N; ++i) {
    auto next = BuildPerElementParams(params[i]);
    AppendPerElementParams(out, next);
  }
  return out;
}

struct LinearElasticSpec {
  using Params = LinearElasticMaterialParams;

  static constexpr auto kSupportedPsdStrategies = GetSupportedPsdStrategies<Params>();
  static constexpr bool kHasFrameInvariantRestState = true;
  static constexpr bool kHasRotationInvariantRestState = false;
  static constexpr bool kHasMinimumEnergy = false;
  static constexpr bool kHasMatchingTarget = false;
  static constexpr bool kHasLinearElasticLimit = false;
  static constexpr bool kHasAnalyticPsdProjectionTests = false;
  static constexpr bool kRunFiniteDifferenceInSinglePrecision = true;
  static constexpr bool kRunProjectedFiniteDifference = true;

  [[nodiscard]] static Params RandomParams(unsigned int seed, MaterialPsdStrategy strategy) {
    return MakeRandomMaterialParams<Params>(seed, strategy);
  }

  template <int kBS>
  static void Eval(
      BatchLameParams<kBS> const& params,
      BatchReal3x3<kBS> const& F,
      BatchDouble<kBS>* energy,
      BatchReal3x3<kBS>* pk1,
      NdArray<BatchReal3x3<kBS>, 3, 3>* tangent,
      bool projectPsd) {
    BatchedLinearElasticConstitutiveResponse<kBS>(params, F, energy, pk1, tangent, projectPsd);
  }

  template <int kBS>
  [[nodiscard]] static auto MakeBatchedConstitutiveResponse(PerElementLameParams const& perElem) {
    return mochi::materials::MakeBatchedConstitutiveResponse<Params, kBS>(perElem);
  }
};

struct StVenantKirchhoffSpec {
  using Params = StVenantKirchhoffMaterialParams;

  static constexpr auto kSupportedPsdStrategies = GetSupportedPsdStrategies<Params>();
  static constexpr bool kHasFrameInvariantRestState = true;
  static constexpr bool kHasRotationInvariantRestState = false;
  static constexpr bool kHasMinimumEnergy = false;
  static constexpr bool kHasMatchingTarget = false;
  static constexpr bool kHasLinearElasticLimit = false;
  static constexpr bool kHasAnalyticPsdProjectionTests = true;
  static constexpr real kAnalyticPsdProjectionRelTol = MOCHI_USE_DOUBLE_PRECISION ? 2e-8_r : 5e-3_r;
  static constexpr bool kRunFiniteDifferenceInSinglePrecision = true;

  [[nodiscard]] static Params RandomParams(unsigned int seed, MaterialPsdStrategy strategy) {
    return MakeRandomMaterialParams<Params>(seed, strategy);
  }

  template <int kBS>
  static void Eval(
      BatchLameParams<kBS> const& params,
      BatchReal3x3<kBS> const& F,
      BatchDouble<kBS>* energy,
      BatchReal3x3<kBS>* pk1,
      NdArray<BatchReal3x3<kBS>, 3, 3>* tangent,
      bool projectPsd) {
    BatchedStVenantKirchhoffConstitutiveResponse<kBS>(params, F, energy, pk1, tangent, projectPsd);
  }

  template <int kBS>
  [[nodiscard]] static auto MakeBatchedConstitutiveResponse(PerElementLameParams const& perElem) {
    return mochi::materials::MakeBatchedConstitutiveResponse<Params, kBS>(perElem);
  }
};

struct SmithNeoHookeanSpec {
  using Params = SmithNeoHookeanMaterialParams;

  static constexpr auto kSupportedPsdStrategies = GetSupportedPsdStrategies<Params>();
  static constexpr bool kHasFrameInvariantRestState = false;
  static constexpr bool kHasRotationInvariantRestState = false;
  static constexpr bool kHasMinimumEnergy = false;
  static constexpr bool kHasMatchingTarget = false;
  static constexpr bool kHasLinearElasticLimit = true;
  static constexpr bool kHasAnalyticPsdProjectionTests = true;
  static constexpr real kAnalyticPsdProjectionRelTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-8_r : 5e-3_r;
  static constexpr bool kRunFiniteDifferenceInSinglePrecision = true;

  [[nodiscard]] static Params RandomParams(unsigned int seed, MaterialPsdStrategy strategy) {
    return MakeRandomMaterialParams<Params>(seed, strategy);
  }

  template <int kBS>
  static void Eval(
      BatchLameParams<kBS> const& params,
      BatchReal3x3<kBS> const& F,
      BatchDouble<kBS>* energy,
      BatchReal3x3<kBS>* pk1,
      NdArray<BatchReal3x3<kBS>, 3, 3>* tangent,
      bool projectPsd) {
    BatchedSmithNeoHookeanConstitutiveResponse<kBS>(params, F, energy, pk1, tangent, projectPsd);
  }

  template <int kBS>
  [[nodiscard]] static auto MakeBatchedConstitutiveResponse(PerElementLameParams const& perElem) {
    return mochi::materials::MakeBatchedConstitutiveResponse<Params, kBS>(perElem);
  }
};

template <MaterialPsdOracle kOracle>
struct SmithNeoHookeanOracleSpec : SmithNeoHookeanSpec {
  template <int kBS>
  static void Eval(
      BatchLameParams<kBS> const& params,
      BatchReal3x3<kBS> const& F,
      BatchDouble<kBS>* energy,
      BatchReal3x3<kBS>* pk1,
      NdArray<BatchReal3x3<kBS>, 3, 3>* tangent,
      bool projectPsd) {
    BatchedSmithNeoHookeanConstitutiveResponse<kBS>(
        params, F, energy, pk1, tangent, projectPsd, kOracle);
  }

  template <int kBS>
  [[nodiscard]] static auto MakeBatchedConstitutiveResponse(PerElementLameParams const& perElem) {
    return mochi::materials::MakeBatchedConstitutiveResponse<Params, kBS>(perElem, kOracle);
  }
};

struct KimNeoHookeanSpec {
  using Params = KimNeoHookeanMaterialParams;

  static constexpr auto kSupportedPsdStrategies = GetSupportedPsdStrategies<Params>();
  static constexpr bool kHasFrameInvariantRestState = true;
  static constexpr bool kHasRotationInvariantRestState = true;
  static constexpr bool kHasMinimumEnergy = true;
  static constexpr bool kHasMatchingTarget = false;
  static constexpr bool kHasLinearElasticLimit = true;
  static constexpr bool kHasAnalyticPsdProjectionTests = true;
  static constexpr real kAnalyticPsdProjectionRelTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-8_r : 5e-3_r;

  [[nodiscard]] static Params RandomParams(unsigned int seed, MaterialPsdStrategy strategy) {
    return MakeRandomMaterialParams<Params>(seed, strategy);
  }

  template <int kBS>
  static void Eval(
      BatchLameParams<kBS> const& params,
      BatchReal3x3<kBS> const& F,
      BatchDouble<kBS>* energy,
      BatchReal3x3<kBS>* pk1,
      NdArray<BatchReal3x3<kBS>, 3, 3>* tangent,
      bool projectPsd) {
    BatchedKimNeoHookeanConstitutiveResponse<kBS>(params, F, energy, pk1, tangent, projectPsd);
  }

  template <int kBS>
  [[nodiscard]] static auto MakeBatchedConstitutiveResponse(PerElementLameParams const& perElem) {
    return mochi::materials::MakeBatchedConstitutiveResponse<Params, kBS>(perElem);
  }
};

struct ArapSpec {
  using Params = ArapMaterialParams;

  static constexpr auto kSupportedPsdStrategies = GetSupportedPsdStrategies<Params>();
  static constexpr bool kHasFrameInvariantRestState = true;
  static constexpr bool kHasRotationInvariantRestState = true;
  static constexpr bool kHasMinimumEnergy = true;
  static constexpr bool kHasMatchingTarget = false;
  static constexpr bool kHasLinearElasticLimit = false;
  static constexpr bool kHasAnalyticPsdProjectionTests = true;
  static constexpr bool kNeedsSeparatedRandomSingularValues = true;
  static constexpr real kAnalyticPsdProjectionRelTol =
      MOCHI_USE_DOUBLE_PRECISION ? 5e-14_r : 1e-5_r;

  [[nodiscard]] static Params RandomParams(unsigned int seed, MaterialPsdStrategy strategy) {
    return MakeRandomMaterialParams<Params>(seed, strategy);
  }

  template <int kBS>
  static void Eval(
      BatchArapParams<kBS> const& params,
      BatchReal3x3<kBS> const& F,
      BatchDouble<kBS>* energy,
      BatchReal3x3<kBS>* pk1,
      NdArray<BatchReal3x3<kBS>, 3, 3>* tangent,
      bool projectPsd) {
    BatchedArapConstitutiveResponse<kBS>(params, F, energy, pk1, tangent, projectPsd);
  }

  template <int kBS>
  [[nodiscard]] static auto MakeBatchedConstitutiveResponse(PerElementArapParams const& perElem) {
    return mochi::materials::MakeBatchedConstitutiveResponse<Params, kBS>(perElem);
  }
};

struct ActiveAnisoArapSpec {
  using Params = ActiveAnisoArapMaterialParams;

  static constexpr auto kSupportedPsdStrategies = GetSupportedPsdStrategies<Params>();
  static constexpr bool kHasFrameInvariantRestState = true;
  static constexpr bool kHasRotationInvariantRestState = true;
  static constexpr bool kHasMinimumEnergy = true;
  static constexpr bool kHasMatchingTarget = false;
  static constexpr bool kHasLinearElasticLimit = false;
  static constexpr bool kHasAnalyticPsdProjectionTests = true;
  static constexpr real kAnalyticPsdProjectionRelTol =
      MOCHI_USE_DOUBLE_PRECISION ? 1e-13_r : 1e-4_r;

  [[nodiscard]] static Params RandomParams(unsigned int seed, MaterialPsdStrategy strategy) {
    return MakeRandomMaterialParams<Params>(seed, strategy);
  }

  static void PrepareInvariantParam(Params& params) {
    params.length = 1_r;
  }

  template <int kBS>
  static void Eval(
      BatchActiveAnisoArapParams<kBS> const& params,
      BatchReal3x3<kBS> const& F,
      BatchDouble<kBS>* energy,
      BatchReal3x3<kBS>* pk1,
      NdArray<BatchReal3x3<kBS>, 3, 3>* tangent,
      bool projectPsd) {
    BatchedActiveAnisoArapConstitutiveResponse<kBS>(params, F, energy, pk1, tangent, projectPsd);
  }

  template <int kBS>
  [[nodiscard]] static auto MakeBatchedConstitutiveResponse(
      PerElementActiveAnisoArapParams const& perElem) {
    return mochi::materials::MakeBatchedConstitutiveResponse<Params, kBS>(perElem);
  }
};

struct ActiveShapeTargetingArapSpec {
  using Params = ActiveShapeTargetingArapMaterialParams;

  static constexpr auto kSupportedPsdStrategies = GetSupportedPsdStrategies<Params>();
  static constexpr bool kHasFrameInvariantRestState = false;
  static constexpr bool kHasRotationInvariantRestState = true;
  static constexpr bool kHasMinimumEnergy = true;
  static constexpr bool kHasMatchingTarget = true;
  static constexpr bool kHasLinearElasticLimit = false;
  static constexpr bool kHasAnalyticPsdProjectionTests = true;
  static constexpr bool kNeedsSeparatedRandomSingularValues = true;
  static constexpr bool kHasDefaultFrameInvariantRestState = true;
  static constexpr real kPk1AbsFloor = MOCHI_USE_DOUBLE_PRECISION ? 1e-7_r : 1.1e-5_r;
  static constexpr real kAnalyticPsdProjectionRelTol =
      MOCHI_USE_DOUBLE_PRECISION ? 1e-13_r : 2e-5_r;

  [[nodiscard]] static real GetPk1AbsTol(Params const& params) {
    real constexpr kRelTol =
        MOCHI_USE_DOUBLE_PRECISION ? 20_r * std::numeric_limits<real>::epsilon() : 5e-5_r;
    return Max(kPk1AbsFloor, params.stiffness * kRelTol);
  }

  [[nodiscard]] static Params RandomParams(unsigned int seed, MaterialPsdStrategy strategy) {
    return MakeRandomMaterialParams<Params>(seed, strategy);
  }

  [[nodiscard]] static Matrix3x3r MakeTargetDeformation(Params const& params) {
    auto const& s = params.shapeTargetTensor;
    return {
        Real3{s[0] + 1_r, s[1], s[2]},
        Real3{s[1], s[3] + 1_r, s[4]},
        Real3{s[2], s[4], s[5] + 1_r}};
  }

  template <int kBS>
  static void Eval(
      BatchActiveShapeTargetingArapParams<kBS> const& params,
      BatchReal3x3<kBS> const& F,
      BatchDouble<kBS>* energy,
      BatchReal3x3<kBS>* pk1,
      NdArray<BatchReal3x3<kBS>, 3, 3>* tangent,
      bool projectPsd) {
    BatchedActiveShapeTargetingArapConstitutiveResponse<kBS>(
        params, F, energy, pk1, tangent, projectPsd);
  }

  template <int kBS>
  [[nodiscard]] static auto MakeBatchedConstitutiveResponse(
      PerElementActiveShapeTargetingArapParams const& perElem) {
    return mochi::materials::MakeBatchedConstitutiveResponse<Params, kBS>(perElem);
  }
};

struct ActiveNeoHookeanSpec {
  using Params = ActiveNeoHookeanMaterialParams;

  static constexpr auto kSupportedPsdStrategies = GetSupportedPsdStrategies<Params>();
  static constexpr bool kHasFrameInvariantRestState = true;
  static constexpr bool kHasRotationInvariantRestState = true;
  static constexpr bool kHasMinimumEnergy = true;
  static constexpr bool kHasMatchingTarget = false;
  static constexpr bool kHasLinearElasticLimit = false;
  static constexpr bool kHasAnalyticPsdProjectionTests = false;

  [[nodiscard]] static Params RandomParams(unsigned int seed, MaterialPsdStrategy strategy) {
    return MakeRandomMaterialParams<Params>(seed, strategy);
  }

  [[nodiscard]] static double ExpectedRestEnergy(Params const& params) {
    // Active aniso ARAP energy is zero at F = I with default length = 1. Smith neo-Hookean has a
    // non-zero rest-state energy from its log-barrier and shifted volume penalty; this replicates
    // the Lame reparameterization from Smith et al. 2018, Sec. 3.4 and evaluates
    // Psi(I) = 1/2 * (lambdaHat * (1 - alpha)^2 - muHat * log(4)).
    auto const& passive = params.passiveIsotropic;
    real const mu = passive.youngsModulus / (2_r * (1_r + passive.poissonRatio));
    real const lambda = passive.youngsModulus * passive.poissonRatio /
        ((1_r + passive.poissonRatio) * (1_r - 2_r * passive.poissonRatio));
    real const lambdaHat = lambda + mu * (5_r / 6_r);
    real const muHat = mu * (4_r / 3_r);
    real const alpha = 1_r + muHat / lambdaHat - muHat / (lambdaHat * 4_r);
    return 0.5 * (lambdaHat * Sqr(1_r - alpha) - muHat * std::log(4.0));
  }

  static void PrepareInvariantParam(Params& params) {
    params.activeAnisotropic.length = 1_r;
  }

  template <int kBS>
  static void Eval(
      BatchActiveNeoHookeanParams<kBS> const& params,
      BatchReal3x3<kBS> const& F,
      BatchDouble<kBS>* energy,
      BatchReal3x3<kBS>* pk1,
      NdArray<BatchReal3x3<kBS>, 3, 3>* tangent,
      bool projectPsd) {
    BatchedActiveNeoHookeanConstitutiveResponse<kBS>(params, F, energy, pk1, tangent, projectPsd);
  }

  template <int kBS>
  [[nodiscard]] static auto MakeBatchedConstitutiveResponse(
      PerElementActiveNeoHookeanParams const& perElem) {
    return mochi::materials::MakeBatchedConstitutiveResponse<Params, kBS>(perElem);
  }
};

} // namespace mochi::materials::test
