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
 * @brief Parameters for the Saint Venant-Kirchhoff (StVK) hyperelastic material model.
 *
 * @details Classical non-linear hyperelastic material. Simpler than neo-Hookean but may exhibit
 * instabilities under large deformations, particularly under large compression.
 *
 * @see SoftMaterialType::StVenantKirchhoff
 */
struct StVenantKirchhoffMaterialParams {
  /** @brief Young's modulus [Pa]. Must be positive. */
  real youngsModulus = kDefaultYoungsModulus;

  /**
   * @brief Poisson's ratio (dimensionless).
   *
   * @note Must be in (-1, 0.5). Near 0.5 = nearly incompressible.
   */
  real poissonRatio = kDefaultPoissonRatio;

  /**
   * @brief Strategy for ensuring positive semi-definite Hessian matrices.
   *
   * @see MaterialPsdStrategy
   */
  MaterialPsdStrategy psdStrategy = MaterialPsdStrategy::Projection;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(StVenantKirchhoffMaterialParams const&) const = default;
#endif

  MOCHI_STRUCT_BEGIN(mochi::StVenantKirchhoffMaterialParams)
  MOCHI_FIELD(youngsModulus) MOCHI_ATTRIBUTE(Units("Pa"));
  MOCHI_FIELD(poissonRatio)
  MOCHI_FIELD(psdStrategy)
  MOCHI_STRUCT_END()
};

namespace materials {
// St. Venant-Kirchhoff has an isotropic reference tangent by construction.
template <>
inline constexpr bool kIsotropicReferenceStiffness<StVenantKirchhoffMaterialParams> = true;
} // namespace materials

} // namespace mochi
