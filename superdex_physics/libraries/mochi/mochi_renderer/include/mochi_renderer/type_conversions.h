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

#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/quaternion.h>

#include <math/mat4.h>
#include <math/quat.h>
#include <math/vec3.h>
#include <math/vec4.h>

#include <type_traits>

// Representation adapters between Mochi and Filament math types.
// These convert storage only; no coordinate-space transformations.

namespace mochi_renderer {

/**
 * @brief Convert a Mochi three-vector to Filament storage.
 *
 * @tparam ToScalar Output scalar type. Defaults to @p FromScalar (no change).
 * @tparam FromScalar Input scalar type.
 * @param value Mochi value input.
 * @return Output value in Filament format.
 */
template <typename ToScalar = void, typename FromScalar>
[[nodiscard]] inline auto ToFilament(mochi::NdArray<FromScalar, 3> const& value) {
  using To = std::conditional_t<std::is_void_v<ToScalar>, FromScalar, ToScalar>;
  return filament::math::vec3<To>{
      static_cast<To>(value[0]), static_cast<To>(value[1]), static_cast<To>(value[2])};
}

/**
 * @brief Convert a Filament three-vector to Mochi storage.
 *
 * @tparam ToScalar Output scalar type. Defaults to @p FromScalar (no change).
 * @tparam FromScalar Input scalar type.
 * @param value Filament input value.
 * @return Output value in Mochi format.
 */
template <typename ToScalar = void, typename FromScalar>
[[nodiscard]] inline auto ToMochi(filament::math::vec3<FromScalar> const& value) {
  using To = std::conditional_t<std::is_void_v<ToScalar>, FromScalar, ToScalar>;
  return mochi::NdArray<To, 3>{
      static_cast<To>(value.x), static_cast<To>(value.y), static_cast<To>(value.z)};
}

/**
 * @brief Convert a Mochi four-vector to Filament storage.
 *
 * @tparam ToScalar Output scalar type. Defaults to @p FromScalar (no change).
 * @tparam FromScalar Input scalar type.
 * @param value Mochi value input.
 * @return Output value in Filament format.
 */
template <typename ToScalar = void, typename FromScalar>
[[nodiscard]] inline auto ToFilament(mochi::NdArray<FromScalar, 4> const& value) {
  using To = std::conditional_t<std::is_void_v<ToScalar>, FromScalar, ToScalar>;
  return filament::math::vec4<To>{
      static_cast<To>(value[0]),
      static_cast<To>(value[1]),
      static_cast<To>(value[2]),
      static_cast<To>(value[3])};
}

/**
 * @brief Convert a Filament four-vector to Mochi storage.
 *
 * @tparam ToScalar Output scalar type. Defaults to @p FromScalar (no change).
 * @tparam FromScalar Input scalar type.
 * @param value Filament input value.
 * @return Output value in Mochi format.
 */
template <typename ToScalar = void, typename FromScalar>
[[nodiscard]] inline auto ToMochi(filament::math::vec4<FromScalar> const& value) {
  using To = std::conditional_t<std::is_void_v<ToScalar>, FromScalar, ToScalar>;
  return mochi::NdArray<To, 4>{
      static_cast<To>(value.x),
      static_cast<To>(value.y),
      static_cast<To>(value.z),
      static_cast<To>(value.w)};
}

/**
 * @brief Convert a @ref mochi::Quaternion to Filament storage.
 *
 * @tparam ToScalar Output scalar type. Defaults to @ref mochi::real (no change).
 * @param value Mochi value input.
 * @return Output value in Filament format.
 */
template <typename ToScalar = mochi::real>
[[nodiscard]] inline auto ToFilament(mochi::Quaternion const& value) {
  mochi::Real4 const q = value.ToReal4(); // Mochi stores XYZW.
  return filament::math::details::TQuaternion<ToScalar>{
      static_cast<ToScalar>(q[3]),
      static_cast<ToScalar>(q[0]),
      static_cast<ToScalar>(q[1]),
      static_cast<ToScalar>(q[2])}; // Filament takes (w, x, y, z).
}

/**
 * @brief Convert a Filament quaternion to a @ref mochi::Quaternion.
 *
 * @tparam ToScalar Output scalar type must be mochi::real because that's all that mochi::Quaternion
 * supports. This parameter is only here for consistency with the other overloads.
 * @tparam FromScalar Input scalar type.
 * @param value Filament input value.
 * @return Output value in Mochi format.
 */
template <typename ToScalar = mochi::real, typename FromScalar>
[[nodiscard]] inline mochi::Quaternion ToMochi(
    filament::math::details::TQuaternion<FromScalar> const& value) {
  static_assert(
      std::is_same_v<ToScalar, mochi::real>,
      "mochi::Quaternion only supports scalar type mochi::real");
  return mochi::Quaternion{
      static_cast<mochi::real>(value.x),
      static_cast<mochi::real>(value.y),
      static_cast<mochi::real>(value.z),
      static_cast<mochi::real>(value.w)};
}

/**
 * @brief Convert a Mochi 4x4 matrix (row-major) to Filament storage (column-major).
 *
 * @tparam ToScalar Output scalar type. Defaults to @p FromScalar (no change).
 * @tparam FromScalar Input scalar type.
 * @param value Mochi value input.
 * @return Output value in Filament format.
 */
template <typename ToScalar = void, typename FromScalar>
[[nodiscard]] inline auto ToFilament(mochi::NdArray<FromScalar, 4, 4> const& value) {
  using To = std::conditional_t<std::is_void_v<ToScalar>, FromScalar, ToScalar>;
  filament::math::details::TMat44<To> result;
  for (size_t row = 0; row < 4; ++row) {
    for (size_t col = 0; col < 4; ++col) {
      result[col][row] = static_cast<To>(value[row][col]);
    }
  }
  return result;
}

/**
 * @brief Convert a Filament 4x4 matrix (column-major) to Mochi storage (row-major).
 *
 * @tparam ToScalar Output scalar type. Defaults to @p FromScalar (no change).
 * @tparam FromScalar Input scalar type.
 * @param value Filament input value.
 * @return Output value in Mochi format.
 */
template <typename ToScalar = void, typename FromScalar>
[[nodiscard]] inline auto ToMochi(filament::math::details::TMat44<FromScalar> const& value) {
  using To = std::conditional_t<std::is_void_v<ToScalar>, FromScalar, ToScalar>;
  mochi::NdArray<To, 4, 4> result{};
  for (size_t row = 0; row < 4; ++row) {
    for (size_t col = 0; col < 4; ++col) {
      result[row][col] = static_cast<To>(value[col][row]);
    }
  }
  return result;
}

} // namespace mochi_renderer
