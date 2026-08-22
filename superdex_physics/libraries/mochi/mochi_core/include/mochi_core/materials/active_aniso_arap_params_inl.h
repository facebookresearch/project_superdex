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
#include "active_aniso_arap_params.h" // Reverse include for intellisense

#include <mochi_core/utils/basic_utils.h>

namespace mochi {

inline real ActiveAnisoArapMaterialParams::GetTheta() const {
  return ATan2(anisoDir[2], anisoDir[0]);
}

inline real ActiveAnisoArapMaterialParams::GetPhi() const {
  return ASin(anisoDir[1]);
}

constexpr Real3 ActiveAnisoArapMaterialParams::ComputeFiberDirection(real theta, real phi) {
  real const cosPhi = Cos(phi);
  return {cosPhi * Cos(theta), Sin(phi), cosPhi * Sin(theta)};
}

} // namespace mochi
