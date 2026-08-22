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
  Sphere
*/

class Sphere final {
 public:
  Real3 center{}; ///< Center of the sphere [m].
  real radius{}; ///< Radius of the sphere [m].

  MOCHI_FORCE_INLINE Sphere() = default;
  MOCHI_FORCE_INLINE Sphere(Vec4r center, real radius) : center(ToReal3(center)), radius(radius) {}
  MOCHI_FORCE_INLINE Sphere(Real3 const& center, real radius) : center(center), radius(radius) {}
  MOCHI_FORCE_INLINE Sphere(Vec4r packedCenterAndRadius)
      : center(ToReal3(packedCenterAndRadius)), radius(Get<3>(packedCenterAndRadius)) {}

  [[nodiscard]] MOCHI_FORCE_INLINE Real3 GetCenter() const {
    return center;
  }

  [[nodiscard]] MOCHI_FORCE_INLINE real GetRadius() const {
    return radius;
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Vec4r VGetPacked() const {
    return ToSimd(center, radius);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Vec4r VGetCenter() const {
    return ToSimd(center, 1_r);
  }

#if MOCHI_LANGUAGE_CPP20
  bool operator==(Sphere const& other) const = default;
  bool operator!=(Sphere const& other) const = default;
#endif

  MOCHI_STRUCT_BEGIN(mochi::Sphere)
  MOCHI_FIELD(center) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_FIELD(radius) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_STRUCT_END()
};

namespace details {
template <>
inline constexpr bool IsPrimitiveShapeDef<Sphere> = true;
} // namespace details

} // namespace mochi
