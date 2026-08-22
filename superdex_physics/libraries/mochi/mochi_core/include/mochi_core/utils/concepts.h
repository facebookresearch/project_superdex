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

#if MOCHI_LANGUAGE_CPP20
#include <concepts>
#endif // MOCHI_LANGUAGE_CPP20
#include <type_traits>

namespace mochi {

namespace details {
template <typename T>
inline constexpr bool IsPrimitiveShapeDef = false;

template <typename T>
inline constexpr bool IsSdfBvDef = false;

// Identity specialization for arithmetic types. Fires a compile error for non-arithmetic types that
// are missing a specialization.
template <class T, class = void>
struct ScalarTypeDef {
  static_assert(
      std::is_arithmetic_v<T>,
      "ScalarType<T>: T is not a recognized scalar type and has no ScalarTypeDef specialization.");
  using type = std::remove_cv_t<T>;
};

// Specialization for types exposing ::Scalar (e.g., Simd, Matrix, SparseMatrix).
template <class T>
struct ScalarTypeDef<T, std::void_t<typename T::Scalar>> {
  using type = typename ScalarTypeDef<std::decay_t<typename T::Scalar>>::type;
};

} // namespace details

#if MOCHI_LANGUAGE_CPP20
template <typename T>
concept IsConst = std::is_const_v<T>;

template <typename T>
concept IsNonConst = !IsConst<T>;

template <typename T>
concept IsArithmetic = std::is_arithmetic_v<T>;

/// @brief Empty object when no value is needed.
struct Void {};

template <typename T>
concept IsNotVoidObject = !std::is_same_v<Void, std::decay_t<T>>;

template <typename T>
concept IsPrimitiveShape = details::IsPrimitiveShapeDef<std::decay_t<T>>;

template <typename T>
concept IsSdfBv = details::IsSdfBvDef<std::decay_t<T>>;
#else
template <typename T>
inline constexpr bool IsPrimitiveShape = details::IsPrimitiveShapeDef<std::decay_t<T>>;

template <typename T>
inline constexpr bool IsSdfBv = details::IsSdfBvDef<std::decay_t<T>>;

#endif // MOCHI_LANGUAGE_CPP20

// Underlying scalar type (type itself for scalar types, recursive unwrap for compound types such as
// Simd, Matrix, NdArray, or Span).
template <class T>
using ScalarType = typename details::ScalarTypeDef<std::decay_t<T>>::type;

} // namespace mochi
