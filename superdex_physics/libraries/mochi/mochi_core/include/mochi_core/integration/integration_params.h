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
#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/reflection.h>

namespace mochi {
/** @brief Time integration methods. */
enum struct IntegrationMethod {
  /**
   * @brief Backward Euler: 1 step, 1 stage, 1st order, L-stable. Also known as BDF1 and DIRK11.
   */
  BackwardEuler,

  /** @brief BDF2: 2 steps, 1 stage, 2nd order, A-stable. */
  BDF2,

  /** @brief BDF3: 3 steps, 1 stage, 3rd order. */
  BDF3,

  /** @brief DIRK(2,2): 1 step, 2 stages, 2nd order, L-stable. */
  DIRK22,

  /**
   * @brief [Experimental] DIRK(2,3): 1 step, 2 stages, 3rd order.
   *
   * @warning This is an experimental feature. It may be changed or removed in the future. Use at
   * your own risk.
   *
   * @note Its step does not end at its last stage, so its steps use velocity-regularized friction
   * for all contacts (see @ref ContactParams::preferElastoPlasticFriction). Queries refreshed
   * without a step approximate it with one BackwardEuler stage, which may use elasto-plastic
   * friction.
   */
  DIRK23,

  /** @brief DIRK(3,3): 1 step, 3 stages, 3rd order, L-stable. */
  DIRK33,

  /**
   * @brief [Experimental] Symplectic DIRK(1,2): 1 step, 1 stage, 2nd order, A-stable. Also known as
   * Gauss and implicit midpoint.
   *
   * @warning This is an experimental feature. It may be changed or removed in the future. Use at
   * your own risk.
   *
   * @note Its step does not end at its last stage, so its steps use velocity-regularized friction
   * for all contacts (see @ref ContactParams::preferElastoPlasticFriction). Queries refreshed
   * without a step approximate it with one BackwardEuler stage, which may use elasto-plastic
   * friction.
   */
  SymplecticDIRK12,

  /**
   * @brief [Experimental] Symplectic DIRK(2,2): 1 step, 2 stages, 2nd order, A-stable.
   *
   * @warning This is an experimental feature. It may be changed or removed in the future. Use at
   * your own risk.
   *
   * @note Its step does not end at its last stage, so its steps use velocity-regularized friction
   * for all contacts (see @ref ContactParams::preferElastoPlasticFriction). Queries refreshed
   * without a step approximate it with one BackwardEuler stage, which may use elasto-plastic
   * friction.
   */
  SymplecticDIRK22,

  /** @brief Number of time integration method enum values. */
  Count,

  /** @brief Default time integration method. */
  Default = BackwardEuler,
};
} // namespace mochi

MOCHI_ENUM_BEGIN(mochi::IntegrationMethod)
MOCHI_ENUM_ITEM(BackwardEuler)
MOCHI_ENUM_ITEM(BDF2)
MOCHI_ENUM_ITEM(BDF3)
MOCHI_ENUM_ITEM(DIRK22)
MOCHI_ENUM_ITEM(DIRK23)
MOCHI_ENUM_ITEM(DIRK33)
MOCHI_ENUM_ITEM(SymplecticDIRK12)
MOCHI_ENUM_ITEM(SymplecticDIRK22)
MOCHI_ENUM_COUNT(Count)
MOCHI_ENUM_END()
