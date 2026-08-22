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
#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/concepts.h>
#include <mochi_core/utils/simd.h>

#include <cstdint>
#include <cstring>

#if MOCHI_COMPILER_CUDA
#include <cuda_fp16.h>
#elif MOCHI_ARCH_X64_AVX2 && !defined(__FLT16_MAX__)
#include <immintrin.h>
#endif

// MOCHI_HAS_SIMD_HALF is true (1) when Simd<Half, N> is supported for some values of N.
#if MOCHI_USE_SIMD && (MOCHI_ARCH_ARM_NEON || MOCHI_ARCH_X64_AVX2)
#define MOCHI_HAS_SIMD_HALF 1
#else
#define MOCHI_HAS_SIMD_HALF 0
#endif

namespace mochi {

/**
 * @brief 16-bit IEEE 754 half-precision floating-point storage type.
 *
 * @details Half is a storage-only type. Arithmetic operations are intentionally not provided
 * because they are not portable across all supported compilers. Half converts implicitly to
 * `float` and `double` for arithmetic, and supports explicit construction from any arithmetic type
 * via `Half(someNumber)` or `StaticCast<Half>(someNumber)`.
 *
 * The internal storage type is chosen per platform:
 * - `__half` (CUDA's native half type) when compiling under NVCC.
 * - `_Float16` (C23 standard type) identified via `__FLT16_MAX__` (C++23 introduced
 * `std::float16_t`, but `_Float16` is more portable due to a longer support history).
 * - `uint16_t` (raw IEEE 754 bit pattern) otherwise.
 */
struct Half {
  /// Half does not default initialize on the stack, similar to other arithmetic types.
  /// However, Half{} is guaranteed to be zero-initialized.
  Half() = default;

  /// Explicit conversion from float.
  MOCHI_ANY explicit Half(float f);

  /// Explicit conversion from any other arithmetic type (via float).
  template <typename T>
    requires(std::is_arithmetic_v<T> && !std::is_same_v<T, float>)
  MOCHI_ANY MOCHI_FORCE_INLINE explicit Half(T value) : Half(static_cast<float>(value)) {}

  /// Implicit conversion to float
  [[nodiscard]] MOCHI_ANY operator float() const;

  /// Equality follows IEEE 754 rules for -0.0 and special values. NaNs are never equal.
  [[nodiscard]] MOCHI_ANY bool operator==(Half const& rhs) const;

#if MOCHI_COMPILER_CUDA
  __half data;
#elif defined(__FLT16_MAX__)
  _Float16 data;
#else
  uint16_t data;
#endif
};

// Specialization of IsHalfDef and ScalarTypeDef for Half.
namespace details {
template <>
inline constexpr bool IsHalfDef<Half> = true;

template <>
struct ScalarTypeDef<Half, void> {
  using type = Half;
};
} // namespace details

/** @brief Convert Half to any arithmetic type via StaticCast. */
template <typename To>
[[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE To StaticCast(Half a) {
  return static_cast<To>(static_cast<float>(a));
}

/** @brief Convert any arithmetic type to Half via StaticCast. */
template <typename To, typename From>
  requires IsHalf<To>
[[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE Half StaticCast(From a) {
  return Half(static_cast<float>(a));
}

/** @brief Copy bits from Half to any 2-byte trivially copyable type.  */
template <typename To>
[[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE To ReinterpretCast(Half a) {
  static_assert(sizeof(To) == sizeof(Half));
  static_assert(std::is_trivially_copyable_v<To>);
  To result MOCHI_NO_INIT;
  memcpy(&result, &a, sizeof(result)); // std::bit_cast not used here for CUDA compatibility
  return result;
}

/** @brief Copy bits from any 2-byte trivially copyable type to Half. */
template <typename To, typename From>
  requires IsHalf<To>
[[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE Half ReinterpretCast(From a) {
  static_assert(sizeof(From) == sizeof(Half));
  static_assert(std::is_trivially_copyable_v<From>);
  Half result MOCHI_NO_INIT;
  memcpy(&result, &a, sizeof(result)); // std::bit_cast not used here for CUDA compatibility
  return result;
}

/// @brief Smallest positive normal value of type Half.
inline Half const kHalfMin = ReinterpretCast<Half>(uint16_t{0x0400});

/// @brief Largest finite value of type Half.
inline Half const kHalfMax = ReinterpretCast<Half>(uint16_t{0x7BFF});

/**
 * @brief Reference implementation to convert a 16-bit half-precision bit pattern to a 32-bit float.
 *
 * @details Implements the IEEE 754 half-to-float conversion via bit manipulation.
 * Handles normals, denormals, ±zero, ±infinity, and NaN (preserves payload bits).
 *
 * @param h The 16-bit half-precision bit pattern.
 * @return The corresponding 32-bit float value.
 */
[[nodiscard]] inline float HalfBitsToFloat(uint16_t h);

/**
 * @brief Reference implementation to convert a 32-bit float to a 16-bit half-precision bit pattern.
 *
 * @details Implements the IEEE 754 float-to-half conversion using round-to-nearest-even
 * (IEEE 754 default rounding mode). Handles normals, denormals, ±zero, ±infinity, NaN,
 * overflow (→ ±inf), and underflow (→ ±zero or denormal).
 *
 * @param f The 32-bit float value to convert.
 * @return The corresponding 16-bit half-precision bit pattern.
 */
[[nodiscard]] inline uint16_t FloatToHalfBits(float f);

} // namespace mochi

#include "half_inl.h"

// Support for Simd<Half> on ARM NEON and X64 AVX2 (not currently supported via SIMD emulation).
#if MOCHI_USE_SIMD
#if MOCHI_ARCH_ARM_NEON
#include "simd/arm/arm_simd_half_inl.h"
#elif MOCHI_ARCH_X64_AVX2
#include "simd/x64/x64_simd_half_inl.h"
#endif
#endif

namespace mochi {

// Aliases for Simd<Half> types.
using Vec8h = Simd<Half, 8>;
using Vec16h = Simd<Half, 16>;

} // namespace mochi
