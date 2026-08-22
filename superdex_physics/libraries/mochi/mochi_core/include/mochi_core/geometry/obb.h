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
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/matrix_transform_rt.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/reflection.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/vmatrix.h>

#include <type_traits>

namespace mochi {

/**************************************************************************************************
  Obb - Oriented Bounding Box
*/
class Obb final {
 public:
  MOCHI_FORCE_INLINE Obb() = default;

  // All other constructors forward to this one
  MOCHI_FORCE_INLINE Obb(MatrixTransformRT const& transform, Vec4r halfExtents)
      : _transform(transform), _halfExtents(ToSimdDirection(halfExtents)) {
    MOCHI_ASSERT_VERBOSE(
        (Get<0>(_halfExtents) >= 0_r) && (Get<1>(_halfExtents) >= 0_r) &&
            (Get<2>(_halfExtents) >= 0_r),
        "Extents must >= 0");
  }

  // clang-format off

  MOCHI_FORCE_INLINE Obb(TransformRT const& transform, Vec4r halfExtents) : Obb(ToMatrixTransformRT(transform), halfExtents) {}
  MOCHI_FORCE_INLINE Obb(TransformRT const& transform, Real3 const& halfExtents) : Obb(ToMatrixTransformRT(transform), ToSimd(halfExtents)) {}
  MOCHI_FORCE_INLINE Obb(MatrixTransformRT const& transform, Real3 const& halfExtents) : Obb(transform, ToSimd(halfExtents)) {}

  // SIMD accessors (note that the 4th component is always 1 by convention)
  [[nodiscard]] MOCHI_FORCE_INLINE Vec4r VGetCenter() const { return _transform.VGetTranslation(); }
  [[nodiscard]] MOCHI_FORCE_INLINE VMatrix3x3r VGetRotation() const { return _transform.VGetRotation(); }
  [[nodiscard]] MOCHI_FORCE_INLINE Vec4r VGetHalfExtents() const { return _halfExtents; }
  [[nodiscard]] MOCHI_FORCE_INLINE Vec4r VGetSize() const { return 2.0_r * _halfExtents; }

  // Scalar accessors (for convenience)
  [[nodiscard]] MOCHI_FORCE_INLINE Real3 GetCenter() const { return ToReal3(VGetCenter()); }
  [[nodiscard]] MOCHI_FORCE_INLINE Matrix3x3r GetRotation() const { return _transform.GetRotation(); }
  [[nodiscard]] MOCHI_FORCE_INLINE Real3 GetHalfExtents() const { return ToReal3(VGetHalfExtents()); }
  [[nodiscard]] MOCHI_FORCE_INLINE Real3 GetSize() const { return ToReal3(VGetSize()); }

  // Retrieves orientation and center as translation.
  [[nodiscard]] MOCHI_FORCE_INLINE MatrixTransformRT const& GetTransform() const { return _transform; }

  // Retrieves the Obb's corners.
  template <int i> [[nodiscard]] MOCHI_FORCE_INLINE Vec4r VGetCorner() const {
    //    3 +--------+ 7
    //     /|       /|    y  z
    //    / |      / |    | /
    // 2 +--------+ 6|    |/
    //   |  |     |  |    *-- x
    //   |1 +-----|--+ 5
    //   | /      | /
    //   |/       |/
    // 0 +--------+ 4
    bool constexpr negX = (i / 4) % 2 == 0;
    bool constexpr negY = (i / 2) % 2 == 0;
    bool constexpr negZ = i % 2 == 0;
    return _transform.TransformPoint(Neg<negX, negY, negZ, false>(_halfExtents));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE NdArray<Vec4r, 8> VGetCorners() const {
    return { VGetCorner<0>(), VGetCorner<1>(), VGetCorner<2>(), VGetCorner<3>(),
             VGetCorner<4>(), VGetCorner<5>(), VGetCorner<6>(), VGetCorner<7>() };
  }

  template <int i> [[nodiscard]] MOCHI_FORCE_INLINE Real3 GetCorner() const { return ToReal3(VGetCorner<i>()); }

  [[nodiscard]] MOCHI_FORCE_INLINE NdArray<real, 8, 3> GetCorners() const {
    return { GetCorner<0>(), GetCorner<1>(), GetCorner<2>(), GetCorner<3>(),
             GetCorner<4>(), GetCorner<5>(), GetCorner<6>(), GetCorner<7>() };
  }

  // clang-format on

 private:
  // TODO: Look into the slow performance of Obb geometry utilities.
  // TODO: Remove SIMD padding from this class, or at least remove it from reflection serialization.
  MatrixTransformRT _transform;
  Vec4r _halfExtents = SimdZero();

 public:
  MOCHI_STRUCT_BEGIN(mochi::Obb)
  MOCHI_FIELD_NAME(_transform, "transform")
  MOCHI_FIELD_NAME(_halfExtents, "halfExtents")
  MOCHI_STRUCT_END()
};

namespace details {
template <>
inline constexpr bool IsPrimitiveShapeDef<Obb> = true;
} // namespace details

} // namespace mochi
