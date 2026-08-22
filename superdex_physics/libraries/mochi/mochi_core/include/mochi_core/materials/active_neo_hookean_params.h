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
#include <mochi_core/materials/active_aniso_arap_params.h>
#include <mochi_core/materials/material_types.h>
#include <mochi_core/materials/smith_neo_hookean_params.h>
#include <mochi_core/utils/reflection.h>

namespace mochi {

/**
 * @brief Parameters for the Active Neo-Hookean composite material model.
 *
 * @details Combines a passive isotropic hyperelastic component (Neo-Hookean) with an active
 * anisotropic fiber contraction component (Aniso ARAP). Used to model materials with directional
 * actuation like muscle tissue.
 *
 * @see [Stable Neo-Hookean Flesh Simulation (Smith et al.,
 * 2018)](https://www.tkim.graphics/NEO/StableNeoHookean2018.pdf)
 * @see [Anisotropic Elasticity for Inversion-Safety and Element Rehabilitation (Kim et al.,
 * 2019)](http://tkim.graphics/ANISOTROPY/AnisotropyAndRehab.pdf)
 * @see SoftMaterialType::ActiveNeoHookean, SmithNeoHookeanMaterialParams,
 * ActiveAnisoArapMaterialParams
 */
struct ActiveNeoHookeanMaterialParams {
  /**
   * @brief Passive isotropic hyperelastic component parameters (Neo-Hookean).
   *
   * @see SmithNeoHookeanMaterialParams
   */
  SmithNeoHookeanMaterialParams passiveIsotropic = {};

  /**
   * @brief Active anisotropic fiber contraction component parameters (Active Aniso ARAP).
   *
   * @see ActiveAnisoArapMaterialParams
   */
  ActiveAnisoArapMaterialParams activeAnisotropic = {};

#if MOCHI_LANGUAGE_CPP20
  bool operator==(ActiveNeoHookeanMaterialParams const&) const = default;
#endif

  MOCHI_STRUCT_BEGIN(mochi::ActiveNeoHookeanMaterialParams)
  MOCHI_FIELD(passiveIsotropic)
  MOCHI_FIELD(activeAnisotropic)
  MOCHI_STRUCT_END()
};

} // namespace mochi
