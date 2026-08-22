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

#include <mochi_core/materials/material_types.h>
#include <mochi_core/utils/reflection.h>

namespace mochi {

/**
 * @brief Parameters for the stable Neo-Hookean hyperelastic material model based on Kim and Eberle
 * (2022).
 *
 * @see [Dynamic Deformables, Eq. (6.9) (Kim and Eberle,
 * 2022)](https://www.tkim.graphics/DYNAMIC_DEFORMABLES/DynamicDeformables.pdf)
 */
struct KimNeoHookeanMaterialParams {
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
  bool operator==(KimNeoHookeanMaterialParams const&) const = default;
#endif

  MOCHI_STRUCT_BEGIN(mochi::KimNeoHookeanMaterialParams)
  MOCHI_FIELD(youngsModulus) MOCHI_ATTRIBUTE(Units("Pa"));
  MOCHI_FIELD(poissonRatio)
  MOCHI_FIELD(psdStrategy)
  MOCHI_STRUCT_END()
};

} // namespace mochi
