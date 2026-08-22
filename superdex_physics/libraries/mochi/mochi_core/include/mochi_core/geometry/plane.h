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
#include <mochi_core/mochi_config.h>
#include <mochi_core/utils/concepts.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/reflection.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/vmatrix.h>

#include <type_traits>

namespace mochi {

/**************************************************************************************************
  Plane
*/
class Plane final {
 public:
  Real3 normal = {0_r, 1_r, 0_r}; ///< Outward-facing plane normal (unit length).
  real distance = {0_r}; ///< Signed distance from the origin along the normal [m].

  MOCHI_FORCE_INLINE Plane() = default;
  MOCHI_FORCE_INLINE Plane(Vec4r normal, real distFromOrigin)
      : normal(ToReal3(normal)), distance(distFromOrigin) {}
  MOCHI_FORCE_INLINE Plane(Real3 const& normal, real distFromOrigin)
      : normal(normal), distance(distFromOrigin) {}

  [[nodiscard]] MOCHI_FORCE_INLINE Real3 GetNormal() const {
    return normal;
  }

  [[nodiscard]] MOCHI_FORCE_INLINE real GetDistanceFromOrigin() const {
    return distance;
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Vec4r VGetPacked() const {
    return ToSimd(normal, distance);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Vec4r VGetNormal() const {
    return ToSimd(normal, 0_r);
  }

#if MOCHI_LANGUAGE_CPP20
  bool operator==(Plane const& other) const = default;
  bool operator!=(Plane const& other) const = default;
#endif

  MOCHI_STRUCT_BEGIN(mochi::Plane)
  MOCHI_FIELD(normal)
  MOCHI_FIELD(distance) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_STRUCT_END()
};

namespace details {
template <>
inline constexpr bool IsPrimitiveShapeDef<Plane> = true;
} // namespace details

} // namespace mochi
