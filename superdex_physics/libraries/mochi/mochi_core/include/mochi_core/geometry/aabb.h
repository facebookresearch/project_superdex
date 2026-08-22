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
  Aabb - Axis-Aligned Bounding Box
*/
class Aabb final {
 public:
  // clang-format off

  MOCHI_FORCE_INLINE Aabb() = default;
  MOCHI_FORCE_INLINE Aabb(Real3 const& min, Real3 const& max) : _min(ToSimd(min, 1_r)), _max(ToSimd(max, 1_r)) {}
  MOCHI_FORCE_INLINE Aabb(Vec4r min, Vec4r max) : _min(ToSimdPoint(min)), _max(ToSimdPoint(max)) {}

  // SIMD accessors (note that the 4th component is always 1 by convention)
  [[nodiscard]] MOCHI_FORCE_INLINE Vec4r VGetMin() const { return _min; }
  [[nodiscard]] MOCHI_FORCE_INLINE Vec4r VGetMax() const { return _max; }
  [[nodiscard]] MOCHI_FORCE_INLINE Vec4r VGetHalfExtents() const { return 0.5_r * VGetSize(); }
  [[nodiscard]] MOCHI_FORCE_INLINE Vec4r VGetCenter() const { return 0.5_r * (_max + _min); }
  [[nodiscard]] MOCHI_FORCE_INLINE Vec4r VGetSize() const { return _max - _min; }

  // Scalar accessors
  [[nodiscard]] MOCHI_FORCE_INLINE Real3 GetMin() const { return ToReal3(_min); }
  MOCHI_FORCE_INLINE void SetMin(Real3 const& min) { _min = ToSimd(min, 1_r); }
  [[nodiscard]] MOCHI_FORCE_INLINE Real3 GetMax() const { return ToReal3(_max); }
  MOCHI_FORCE_INLINE void SetMax(Real3 const& max) { _max = ToSimd(max, 1_r); }
  [[nodiscard]] MOCHI_FORCE_INLINE Real3 GetHalfExtents() const { return ToReal3(VGetHalfExtents()); }
  [[nodiscard]] MOCHI_FORCE_INLINE Real3 GetCenter() const { return ToReal3(VGetCenter()); }
  [[nodiscard]] MOCHI_FORCE_INLINE Real3 GetSize() const { return ToReal3(VGetSize()); }

  // Exact equality
  [[nodiscard]] MOCHI_FORCE_INLINE bool operator==(Aabb const& rhs) const { return AllTrue<3>(VEqual(_min, rhs._min) & VEqual(_max, rhs._max)); }
  [[nodiscard]] MOCHI_FORCE_INLINE bool operator!=(Aabb const& rhs) const { return !(*this == rhs); }

  // clang-format on
 private:
  Vec4r _min = ToSimdPoint(SimdZero());
  Vec4r _max = ToSimdPoint(SimdZero());

#if MOCHI_USE_REFLECTION
  template <typename T>
  friend struct ::SReflectTypeTraits;
#endif // MOCHI_USE_REFLECTION
};

namespace details {
template <>
inline constexpr bool IsPrimitiveShapeDef<Aabb> = true;
} // namespace details

} // namespace mochi

/************************************************************************************
  Reflection support for Aabb.
  Serializes like struct with two fields of type Real3.
*/
#if MOCHI_USE_REFLECTION
template <>
struct SReflectTypeTraits<mochi::Aabb> {
  static constexpr SReflect::CoreType coreType = SReflect::CoreType::CT_struct;
  static SReflect::StructTypeInfo const& GetTypeInfo() {
    static auto const* s_typeInfo = []() {
      auto* ti = SReflect::MakeStructTypeInfo<mochi::Aabb>("mochi::Aabb");
      SReflect::detail::AddField(
          *ti, "min", offsetof(mochi::Aabb, _min), SReflect::GetTypeInfo<mochi::Real3>());
      SReflect::detail::AddField(
          *ti, "max", offsetof(mochi::Aabb, _max), SReflect::GetTypeInfo<mochi::Real3>());
      return ti;
    }();
    return *s_typeInfo;
  }
};
#endif // MOCHI_USE_REFLECTION
