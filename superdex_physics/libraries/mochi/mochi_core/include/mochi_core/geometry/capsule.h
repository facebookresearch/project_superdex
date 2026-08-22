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

#include <mochi_core/mochi_config.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/vmatrix.h>

namespace mochi {

/**************************************************************************************************
  Capsule
*/
class Capsule final {
 public:
  MOCHI_FORCE_INLINE Capsule() = default;

  // Create a Capsule from two points and a radius
  MOCHI_FORCE_INLINE static Capsule FromPoints(Vec4r posA, Vec4r posB, real radius) {
    return Capsule{Blend<0, 0, 0, 1>(posA, Vec4r{radius}), ToSimdDirection(posB - posA)};
  }
  MOCHI_FORCE_INLINE static Capsule FromPoints(Real3 posA, Real3 posB, real radius) {
    return FromPoints(ToSimd(posA), ToSimd(posB), radius);
  }

  // Create a Capsule from a point, and scaled direction vector, and a radius
  MOCHI_FORCE_INLINE static Capsule FromPointAndVector(Vec4r posA, Vec4r vecAToB, real radius) {
    return Capsule{Blend<0, 0, 0, 1>(posA, Vec4r{radius}), ToSimdDirection(vecAToB)};
  }
  MOCHI_FORCE_INLINE static Capsule FromPointAndVector(Real3 posA, Real3 vecAToB, real radius) {
    return FromPointAndVector(ToSimd(posA), ToSimd(vecAToB), radius);
  }

  // clang-format off

  // SIMD Accessors
  MOCHI_FORCE_INLINE Vec4r VGetA() const { return ToSimdPoint(_posA_radius); }
  MOCHI_FORCE_INLINE Vec4r VGetB() const { return ToSimdPoint(_posA_radius + _vecAToB); }
  MOCHI_FORCE_INLINE Vec4r VGetAB() const { return _vecAToB; }
  MOCHI_FORCE_INLINE Vec4r VGetRadius() const { return Broadcast<3>(_posA_radius); }
  MOCHI_FORCE_INLINE Vec4r VGetLengthAB() const { return VNorm<3>(_vecAToB); } // distance from A to B
  MOCHI_FORCE_INLINE Vec4r VGetPackedARadius() const { return _posA_radius; } // (x, y, z, r)

  // Scalar Accessors
  MOCHI_FORCE_INLINE Real3 GetA() const { return ToReal3(VGetA()); }
  MOCHI_FORCE_INLINE Real3 GetB() const { return ToReal3(VGetB()); }
  MOCHI_FORCE_INLINE Real3 GetAB() const { return ToReal3(_vecAToB); }
  MOCHI_FORCE_INLINE real GetRadius() const { return Get<3>(_posA_radius); }
  MOCHI_FORCE_INLINE real GetLengthAB() const { return Get<0>(VGetLengthAB()); }  // distance from A to B

  // clang-format on

 private:
  // Called by static functions, which are more explicitly named
  MOCHI_FORCE_INLINE Capsule(Vec4r posAandRadius, Vec4r vecAB)
      : _posA_radius(posAandRadius), _vecAToB(vecAB) {}

  Vec4r _posA_radius = SimdZero(); // (x, y, z, r)
  Vec4r _vecAToB = SimdZero(); // (dx, dy, dz, 0)
};

} // namespace mochi
