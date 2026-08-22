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

// PLEASE DO NOT ADD OTHER INCLUDES HERE. This header is included in the mochi_physics public API.
#include <mochi_core/materials/material_types.h>
#include <mochi_core/utils/reflection.h>

namespace mochi {

/**
 * @brief Parameters for the Linear Elastic (Hookean) material model.
 *
 * @details Simplest material model with linear stress-strain relationship. Computationally fast
 * but only accurate for small deformations.
 *
 * @note Not suitable for large deformations.
 * @note No PSD enforcement is needed. The Hessian matrix is positive definite regardless of the
 * PSD strategy.
 *
 * @see SoftMaterialType::LinearElastic
 */
struct LinearElasticMaterialParams {
  /** @brief Young's modulus [Pa]. Must be positive. */
  real youngsModulus = kDefaultYoungsModulus;

  /**
   * @brief Poisson's ratio (dimensionless).
   *
   * @note Must be in (-1, 0.5). Near 0.5 = nearly incompressible.
   */
  real poissonRatio = kDefaultPoissonRatio;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(LinearElasticMaterialParams const&) const = default;
#endif

  MOCHI_STRUCT_BEGIN(mochi::LinearElasticMaterialParams)
  MOCHI_FIELD(youngsModulus) MOCHI_ATTRIBUTE(Units("Pa"));
  MOCHI_FIELD(poissonRatio)
  MOCHI_STRUCT_END()
};

namespace materials {
// Linear elastic has an isotropic reference tangent by construction.
template <>
inline constexpr bool kIsotropicReferenceStiffness<LinearElasticMaterialParams> = true;
} // namespace materials

} // namespace mochi
