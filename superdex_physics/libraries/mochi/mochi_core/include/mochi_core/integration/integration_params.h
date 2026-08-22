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
  BackwardEuler, ///< Backward Euler: 1 step, 1 stage, 1st order, L-stable. Also known as BDF1 and
                 ///< DIRK11.
  BDF2, ///< BDF2: 2 steps, 1 stage, 2nd order, A-stable.
  BDF3, ///< BDF3: 3 steps, 1 stage, 3rd order.
  DIRK22, ///< DIRK(2,2): 1 step, 2 stages, 2nd order, L-stable.
  DIRK23, ///< DIRK(2,3): 1 step, 2 stages, 3rd order.
  DIRK33, ///< DIRK(3,3): 1 step, 3 stages, 3rd order, L-stable.
  SymplecticDIRK12, ///< Symplectic DIRK(1,2): 1 step, 1 stage, 2nd order, A-stable. Also known as
                    ///< Gauss and implicit midpoint.
  SymplecticDIRK22, ///< Symplectic DIRK(2,2): 1 step, 2 stages, 2nd order, A-stable.
  Count, ///< Number of time integration method enum values.
  Default = BackwardEuler, ///< Default time integration method.
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
