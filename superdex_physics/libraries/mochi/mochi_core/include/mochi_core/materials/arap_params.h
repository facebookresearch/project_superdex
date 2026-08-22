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
 * @brief Parameters for the As-Rigid-As-Possible (ARAP) material model.
 *
 * @details ARAP's primary goal is to deform the object such that each local region undergoes a
 * transformation that is as close to a rigid rotation as possible.
 *
 * Because it preserves local shape details well and produces intuitive, natural-looking results,
 * ARAP is widely used in computer graphics for applications like interactive mesh editing and
 * character posing.
 *
 * @see [Dynamic Deformables (Kim and Eberle,
 * 2022)](https://www.tkim.graphics/DYNAMIC_DEFORMABLES/DynamicDeformables.pdf)
 * @see SoftMaterialType::Arap
 */
struct ArapMaterialParams {
  /**
   * @brief ARAP stiffness parameter (μ) [Pa].
   *
   * @note Must be greater than zero.
   */
  real stiffness = kDefaultArapStiffness;

  /**
   * @brief Strategy for ensuring positive semi-definite Hessian matrices.
   *
   * @see MaterialPsdStrategy
   */
  MaterialPsdStrategy psdStrategy = MaterialPsdStrategy::Projection;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(ArapMaterialParams const&) const = default;
#endif

  MOCHI_STRUCT_BEGIN(mochi::ArapMaterialParams)
  MOCHI_FIELD(stiffness) MOCHI_ATTRIBUTE(Units("Pa"));
  MOCHI_FIELD(psdStrategy)
  MOCHI_STRUCT_END()
};

namespace materials {
// ARAP has an isotropic reference tangent by construction.
template <>
inline constexpr bool kIsotropicReferenceStiffness<ArapMaterialParams> = true;
} // namespace materials

} // namespace mochi
