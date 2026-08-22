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
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/reflection.h>

namespace mochi {

/**
 * @brief Parameters for the Active Anisotropic ARAP material model.
 *
 * @details Anisotropic energy term that models directional fiber contraction. The fiber direction
 * is specified in spherical coordinates (θ, φ) and internally stored as a unit direction vector.
 *
 * @see [Anisotropic Elasticity for Inversion-Safety and Element Rehabilitation (Kim et al.,
 * 2019)](http://tkim.graphics/ANISOTROPY/AnisotropyAndRehab.pdf)
 */
struct ActiveAnisoArapMaterialParams {
  /**
   * @brief Anisotropic stiffness (α) along fiber direction [Pa].
   *
   * @note Must not be negative.
   */
  real alpha = kDefaultAnisoAlpha;

  /**
   * @brief Anisotropic reference length (dimensionless).
   *
   * @note Must not be negative.
   */
  real length = kDefaultAnisoLength;

  /**
   * @brief Fiber direction as a unit vector: (cos(φ)cos(θ), sin(φ), cos(φ)sin(θ)).
   *
   * @see ComputeFiberDirection, GetTheta, GetPhi
   */
  Real3 anisoDir = {1_r, 0_r, 0_r};

  /**
   * @brief Strategy for ensuring positive semi-definite Hessian matrices.
   *
   * @see MaterialPsdStrategy
   */
  MaterialPsdStrategy psdStrategy = MaterialPsdStrategy::Projection;

  /**
   * @brief Returns the azimuthal angle (θ) [rad] from the stored fiber direction.
   *
   * @see ComputeFiberDirection, GetPhi
   */
  [[nodiscard]] real GetTheta() const;

  /**
   * @brief Returns the elevation angle (φ) [rad] from the stored fiber direction.
   *
   * @see ComputeFiberDirection, GetTheta
   */
  [[nodiscard]] real GetPhi() const;

  /**
   * @brief Computes the fiber direction unit vector from spherical angles.
   *
   * @param[in] theta Azimuthal angle (θ) [rad].
   * @param[in] phi Elevation angle (φ) [rad].
   * @return Unit vector: (cos(φ)cos(θ), sin(φ), cos(φ)sin(θ)).
   */
  [[nodiscard]] static constexpr Real3 ComputeFiberDirection(real theta, real phi);

#if MOCHI_LANGUAGE_CPP20
  bool operator==(ActiveAnisoArapMaterialParams const&) const = default;
#endif

  MOCHI_STRUCT_BEGIN(mochi::ActiveAnisoArapMaterialParams)
  MOCHI_FIELD(alpha) MOCHI_ATTRIBUTE(Units("Pa"));
  MOCHI_FIELD(length)
  MOCHI_FIELD(anisoDir)
  MOCHI_FIELD(psdStrategy)
  MOCHI_STRUCT_END()
};

} // namespace mochi

#include "active_aniso_arap_params_inl.h"
