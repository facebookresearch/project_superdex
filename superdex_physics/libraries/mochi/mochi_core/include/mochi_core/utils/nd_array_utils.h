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

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/span.h>

#include <iterator>
#include <type_traits>

namespace mochi {

// Memberwise static cast
template <typename ToNdArray, typename FromScalar, size_t D0, size_t... DIMS>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr ToNdArray StaticCast(
    NdArray<FromScalar, D0, DIMS...> const& x) {
  static_assert(kIsNdArray<ToNdArray>, "Invalid cast");
  static_assert(D0 == ToNdArray::dims[0], "Size mismatch");
  if constexpr (std::is_same_v<ToNdArray, NdArray<FromScalar, D0, DIMS...>>) {
    return x;
  } else {
    using ToValueType = typename ToNdArray::value_type;
    ToNdArray result{};
    for (size_t i = 0; i < D0; ++i) {
      result[i] = StaticCast<ToValueType>(x[i]);
    }
    return result;
  }
}

// result[i] = Fn(a[i])
#define MOCHI_UNROLL_FN_ARRAY(result, Fn, a) \
  if constexpr (D0 <= 4) {                   \
    (result)[0] = Fn(a[0]);                  \
    if constexpr (D0 > 1) {                  \
      (result)[1] = Fn(a[1]);                \
    }                                        \
    if constexpr (D0 > 2) {                  \
      (result)[2] = Fn(a[2]);                \
    }                                        \
    if constexpr (D0 > 3) {                  \
      (result)[3] = Fn(a[3]);                \
    }                                        \
  } else {                                   \
    for (size_t i = 0; i < D0; ++i) {        \
      (result)[i] = Fn(a[i]);                \
    }                                        \
  }

// Memberwise absolute value
template <typename T, size_t D0, size_t... DIMS>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr auto Abs(NdArray<T, D0, DIMS...> const& a) {
  NdArray<T, D0, DIMS...> result = {};
  MOCHI_UNROLL_FN_ARRAY(result, Abs, a);
  return result;
}

// Memberwise reciprocal
template <typename T, size_t D0, size_t... DIMS>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr auto Rcp(NdArray<T, D0, DIMS...> const& a) {
  NdArray<T, D0, DIMS...> result = {};
  MOCHI_UNROLL_FN_ARRAY(result, Rcp, a);
  return result;
}

// Memberwise Sign (1 or -1 for each element)
template <typename T, size_t D0, size_t... DIMS>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr auto Sign(NdArray<T, D0, DIMS...> const& a) {
  NdArray<T, D0, DIMS...> result = {};
  MOCHI_UNROLL_FN_ARRAY(result, Sign, a);
  return result;
}

#undef MOCHI_UNROLL_FN_ARRAY

/**************************************************************************************************
Flatten

Takes an NdArray or a Span of NdArrays and reinterprets it as a Span of the flattened value
type. For example: Span<Int3> span; Span<int> flat_span = Flatten(span);
*/
template <typename T, size_t D0, size_t... DIMS>
[[nodiscard]] MOCHI_FORCE_INLINE Span<T> Flatten(NdArray<T, D0, DIMS...>& x) {
  return {reinterpret_cast<T*>(x.data()), NdArray<T, D0, DIMS...>::flattened_size};
}
template <typename T, size_t D0, size_t... DIMS>
[[nodiscard]] MOCHI_FORCE_INLINE Span<T const> Flatten(NdArray<T, D0, DIMS...> const& x) {
  return {reinterpret_cast<T const*>(x.data()), NdArray<T, D0, DIMS...>::flattened_size};
}
template <typename T, size_t D0, size_t... DIMS>
[[nodiscard]] MOCHI_FORCE_INLINE Span<T> Flatten(Span<NdArray<T, D0, DIMS...>> const& x) {
  return Span<T>{
      reinterpret_cast<T*>(x.data()), x.size() * NdArray<T, D0, DIMS...>::flattened_size};
}
template <typename T, size_t D0, size_t... DIMS>
[[nodiscard]] MOCHI_FORCE_INLINE Span<T const> Flatten(
    Span<NdArray<T, D0, DIMS...> const> const& x) {
  return Span<T const>{
      reinterpret_cast<T const*>(x.data()), x.size() * NdArray<T, D0, DIMS...>::flattened_size};
}

/**************************************************************************************************
  Unflatten

  Takes Span<T> for some arithmetic type T, and reinterprets it as a Span<NdArray<T, ...>>. The
  size of the input span must be a multiple of the size of the NdArray type being returned.
*/
template <typename NdArrayT, typename ArrayT>
[[nodiscard]] MOCHI_FORCE_INLINE Span<NdArrayT> Unflatten(ArrayT&& flat) {
  constexpr size_t elemSize = NdArrayT::flattened_size;
  size_t const elemCount = std::size(flat) / elemSize;
  MOCHI_ASSERT_VERBOSE(
      flat.size() % elemSize == 0,
      "Flattened array size must be a multiple of the unflattened array size");
  return Span<NdArrayT>{reinterpret_cast<NdArrayT*>(std::data(flat)), elemCount};
}

} // namespace mochi
