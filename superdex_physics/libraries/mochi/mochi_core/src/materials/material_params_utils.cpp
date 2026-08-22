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

#include <mochi_core/materials/material_params_utils.h>

#include <mochi_core/materials/batched_materials.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/constants.h>

using namespace mochi;

static void ValidateSoftMaterialType(SoftMaterialType type, Error& error) {
  MOCHI_ERROR_IF_NOT(
      (static_cast<int>(type) >= 0) &&
          (static_cast<int>(type) < static_cast<int>(SoftMaterialType::Count)),
      error,
      "Unsupported soft material type.");
}

static void
ValidatePsdStrategy(SoftMaterialType type, MaterialPsdStrategy psdStrategy, Error& error) {
  static_assert(
      static_cast<int>(SoftMaterialType::Count) == 6,
      "Please update the following switch statement if the SoftMaterialType enum changes");
  bool supported = false;
  switch (type) {
    case SoftMaterialType::NeoHookean:
      supported = materials::IsPsdStrategySupported<NeoHookeanMaterialParams>(psdStrategy);
      break;
    case SoftMaterialType::StVenantKirchhoff:
      supported = materials::IsPsdStrategySupported<StVenantKirchhoffMaterialParams>(psdStrategy);
      break;
    case SoftMaterialType::LinearElastic:
      supported = materials::IsPsdStrategySupported<LinearElasticMaterialParams>(psdStrategy);
      break;
    case SoftMaterialType::ActiveNeoHookean:
      supported = materials::IsPsdStrategySupported<ActiveNeoHookeanMaterialParams>(psdStrategy);
      break;
    case SoftMaterialType::ActiveShapeTargetingArap:
      supported =
          materials::IsPsdStrategySupported<ActiveShapeTargetingArapMaterialParams>(psdStrategy);
      break;
    case SoftMaterialType::Arap:
      supported = materials::IsPsdStrategySupported<ArapMaterialParams>(psdStrategy);
      break;
    default:
      MOCHI_ERROR_SET(error, "Unsupported soft material type.");
      break;
  }
  MOCHI_ERROR_IF_NOT(supported, error, "PSD strategy not supported for this material type.");
}

template <typename MaterialParamsType>
static void ValidatePsdStrategy(MaterialParamsType const& params, Error& error) {
  MOCHI_ERROR_IF_NOT(
      materials::IsPsdStrategySupported(params),
      error,
      "PSD strategy not supported for this material type.");
}

template <typename MaterialParamsType>
static void ValidateYoungsModulusAndPoissonRatio(MaterialParamsType const& params, Error& error) {
  MOCHI_ERROR_IF_NOT(
      IsFinite(params.youngsModulus) && (params.youngsModulus > 0),
      error,
      "Invalid material params: Young's modulus must be positive.");
  MOCHI_ERROR_IF_NOT(
      IsFinite(params.poissonRatio) && (params.poissonRatio > -1_r) &&
          (params.poissonRatio < 0.5_r),
      error,
      "Invalid material params: Poisson ratio must be in (-1, 0.5).");
}

template <typename MaterialParamsType>
static void ValidateArapStiffness(MaterialParamsType const& params, Error& error) {
  MOCHI_ERROR_IF_NOT(
      IsFinite(params.stiffness) && (params.stiffness > 0),
      error,
      "Invalid material params: ARAP stiffness must be finite and greater than zero.");
}

void mochi::ValidateSoftMaterialParams(SoftMaterialParams const& params, Error& error) {
  ValidateSoftMaterialType(params.type, error);

  MOCHI_ERROR_IF_NOT(
      IsFinite(params.density) && (params.density > 0),
      error,
      "Invalid material params: Density must be positive and finite.");
  MOCHI_ERROR_IF_NOT(
      IsFinite(params.massDampingCoefficient) && (params.massDampingCoefficient >= 0_r),
      error,
      "Invalid material params: massDampingCoefficient must be finite and non-negative.");
  MOCHI_ERROR_IF_NOT(
      IsFinite(params.stiffnessDampingCoefficient) && (params.stiffnessDampingCoefficient >= 0_r),
      error,
      "Invalid material params: stiffnessDampingCoefficient must be finite and non-negative.");
  MOCHI_ERROR_RETURN(error);

  static_assert(
      static_cast<int>(SoftMaterialType::Count) == 6,
      "Please update the switch statement below if SoftMaterialType enum changes");
  switch (params.type) {
    case SoftMaterialType::NeoHookean: {
      ValidatePsdStrategy(params.neoHookean, error);
      ValidateYoungsModulusAndPoissonRatio(params.neoHookean, error);
      break;
    }
    case SoftMaterialType::StVenantKirchhoff: {
      ValidatePsdStrategy(params.stVenantKirchhoff, error);
      ValidateYoungsModulusAndPoissonRatio(params.stVenantKirchhoff, error);
      if (params.stVenantKirchhoff.poissonRatio < 0_r &&
          params.stVenantKirchhoff.psdStrategy == MaterialPsdStrategy::Fast) {
        MOCHI_LOG_WARNING(
            "StVenantKirchhoff material with negative Poisson ratio (%.4g) and MaterialPsdStrategy::Fast: "
            "the Fast PSD projection does not guarantee a positive semi-definite Hessian in the auxetic regime. "
            "Consider using MaterialPsdStrategy::Projection or MaterialPsdStrategy::AbsEigenProjection instead.",
            params.stVenantKirchhoff.poissonRatio);
      }
      break;
    }
    case SoftMaterialType::LinearElastic: {
      ValidateYoungsModulusAndPoissonRatio(params.linearElastic, error);
      break;
    }
    case SoftMaterialType::ActiveNeoHookean: {
      auto const& p = params.activeNeoHookean;
      ValidatePsdStrategy(p.passiveIsotropic, error);
      ValidatePsdStrategy(p.activeAnisotropic, error);
      ValidateYoungsModulusAndPoissonRatio(p.passiveIsotropic, error);
      MOCHI_ERROR_IF_NOT(
          IsFinite(p.activeAnisotropic.alpha) && (p.activeAnisotropic.alpha >= 0),
          error,
          "Invalid material params: Aniso alpha must be finite and not negative.");
      MOCHI_ERROR_IF_NOT(
          IsFinite(p.activeAnisotropic.length) && (p.activeAnisotropic.length >= 0),
          error,
          "Invalid material params: Aniso length must be finite and not negative.");
      break;
    }
    case SoftMaterialType::ActiveShapeTargetingArap: {
      ValidatePsdStrategy(params.activeShapeTargetingArap, error);
      ValidateArapStiffness(params.activeShapeTargetingArap, error);
      MOCHI_ERROR_IF_NOT(
          IsFinite(params.activeShapeTargetingArap.shapeTargetTensor),
          error,
          "Invalid material params: Shape target tensor must be finite.");
      break;
    }
    case SoftMaterialType::Arap: {
      ValidatePsdStrategy(params.arap, error);
      ValidateArapStiffness(params.arap, error);
      break;
    }
    default: {
      MOCHI_ERROR_SET(error, "Unsupported soft material type.");
      break;
    }
  }
}

void mochi::ValidateSoftMaterialParams(
    PerElementSoftMaterialDataView const& data,
    int numElements,
    Error& error) {
  ValidateSoftMaterialType(data.type, error);
  ValidatePsdStrategy(data.type, data.psdStrategy, error);
  MOCHI_ERROR_RETURN(error);

  auto checkArray = [numElements](auto const& array, real min, real max, bool minMaxInclusive) {
    if (isize(array) != numElements) {
      return false;
    }
    if (!IsFinite(MakeConstSpan(array))) {
      return false;
    }
    if (IsFinite(min) || IsFinite(max)) {
      auto [minVal, maxVal] = MinMax(MakeConstSpan(array));
      if (minMaxInclusive) {
        return (minVal >= min) && (maxVal <= max);
      } else {
        return (minVal > min) && (maxVal < max);
      }
    }
    return true;
  };

  switch (data.type) {
    case SoftMaterialType::NeoHookean:
    case SoftMaterialType::StVenantKirchhoff:
    case SoftMaterialType::LinearElastic:
    case SoftMaterialType::ActiveNeoHookean:
      MOCHI_ERROR_IF(
          !checkArray(data.youngsModulus, 0_r, kInf, false),
          error,
          "Invalid material: youngs modulus array size must match the number of elements. Each value must be finite and greater than zero.");

      MOCHI_ERROR_IF(
          !checkArray(data.poissonRatio, -1_r, 0.5_r, false),
          error,
          "Invalid material: Poisson ratio array size must match the number of elements. Each value must be in the range (-1, 0.5).");
      break;
    default:
      if (!data.youngsModulus.empty()) {
        MOCHI_LOG_WARNING(
            "Material data has unused youngs modulus parameters. Did you mean to select a different material type?");
      }
      if (!data.poissonRatio.empty()) {
        MOCHI_LOG_WARNING(
            "Material data has unused poisson ratio parameters. Did you mean to select a different material type?");
      }
      break;
  }

  if (data.type == SoftMaterialType::ActiveNeoHookean) {
    MOCHI_ERROR_IF(
        !checkArray(data.anisoAlpha, 0_r, kInf, true),
        error,
        "Invalid material: Aniso alpha array size must match the number of elements. Each value must also be finite and not negative.");
    MOCHI_ERROR_IF(
        !checkArray(data.anisoLength, 0_r, kInf, true),
        error,
        "Invalid material: Aniso length array size must match the number of elements. Each value must also be finite and not negative.");
    MOCHI_ERROR_IF(
        !checkArray(data.anisoTheta, -kInf, kInf, false),
        error,
        "Invalid material: Aniso theta array must match the number of elements. Each value must also be finite.");
    MOCHI_ERROR_IF(
        !checkArray(data.anisoPhi, -kInf, kInf, false),
        error,
        "Invalid material: Aniso phi array must match the number of elements. Each value must also be finite.");
  } else if (
      !data.anisoAlpha.empty() || !data.anisoLength.empty() || !data.anisoTheta.empty() ||
      !data.anisoPhi.empty()) {
    MOCHI_LOG_WARNING(
        "Material data has unused anisotropic parameters. Did you mean to select a different material type?");
  }

  if ((data.type == SoftMaterialType::ActiveShapeTargetingArap) ||
      (data.type == SoftMaterialType::Arap)) {
    MOCHI_ERROR_IF(
        !checkArray(data.arapStiffness, 0, kInf, false),
        error,
        "Invalid material: ARAP stiffness array must match the number of elements. Each value must also be finite and greater than zero.");
  } else if (!data.arapStiffness.empty()) {
    MOCHI_LOG_WARNING(
        "Material data has unused ARAP stiffness values. Did you mean to select a different material type?");
  }

  if (data.type == SoftMaterialType::ActiveShapeTargetingArap) {
    MOCHI_ERROR_IF_NOT(
        isize(data.shapeTargetTensor) == numElements * 6,
        error,
        "Invalid material: Shape target tensor array should have 6 values per element.");
    MOCHI_ERROR_IF_NOT(
        IsFinite(MakeConstSpan(data.shapeTargetTensor)),
        error,
        "Invalid material: Shape target tensor array contains non-finite values.");
    MOCHI_ERROR_RETURN(error);
  } else if (!data.shapeTargetTensor.empty()) {
    MOCHI_LOG_WARNING(
        "Material data has unused shape target tensor values. Did you mean to select a different material type?");
  }
}
