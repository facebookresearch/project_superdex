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
 * @brief Parameters for the Active Shape Targeting ARAP material model.
 *
 * @details Active material model based on the concept of "shape targeting". Unlike passive models
 * that only react to external loads, it introduces an internal actuation mechanism.
 *
 * The actuation is defined by the shape target tensor, S_t, which specifies the desired local
 * shape of the object. The model then resists deformation from this target shape while allowing
 * rigid rotation.
 *
 * This approach is effective for simulating biological phenomena like muscle contraction.
 *
 * @note The "activeness" is controlled externally by setting the shape target tensor S_t. The
 * model itself does not have an intrinsic, self-actuating mechanism without this input.
 *
 * @see [Shape Targeting: A Versatile Active Elasticity Constitutive Model (Klár et al.,
 * 2020)](https://par.nsf.gov/servlets/purl/10230451)
 * @see SoftMaterialType::ActiveShapeTargetingArap
 */
struct ActiveShapeTargetingArapMaterialParams {
  /**
   * @brief ARAP stiffness parameter (μ) [Pa].
   *
   * @note Must be greater than zero.
   */
  real stiffness = kDefaultArapStiffness;

  /**
   * @brief Shape target tensor (6 values).
   *
   * @details Defines the shape target tensor as a symmetric 3x3 matrix added to the identity: S_t =
   * I + [[s0, s1, s2], [s1, s3, s4], [s2, s4, s5]]. The shape target tensor specifies the desired
   * local shape of the object. The model then resists deformation from this target shape while
   * allowing rigid rotation.
   */
  Real6 shapeTargetTensor{0_r, 0_r, 0_r, 0_r, 0_r, 0_r};

  /**
   * @brief Strategy for ensuring positive semi-definite Hessian matrices.
   *
   * @see MaterialPsdStrategy
   */
  MaterialPsdStrategy psdStrategy = MaterialPsdStrategy::Projection;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(ActiveShapeTargetingArapMaterialParams const&) const = default;
#endif

  MOCHI_STRUCT_BEGIN(mochi::ActiveShapeTargetingArapMaterialParams)
  MOCHI_FIELD(stiffness) MOCHI_ATTRIBUTE(Units("Pa"));
  MOCHI_FIELD(shapeTargetTensor)
  MOCHI_FIELD(psdStrategy)
  MOCHI_STRUCT_END()
};

} // namespace mochi
