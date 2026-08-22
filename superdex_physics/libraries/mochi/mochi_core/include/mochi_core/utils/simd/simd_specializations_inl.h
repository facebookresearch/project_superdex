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

#include "../debug.h"
#include "../simd.h" // for Intelisense

namespace mochi {

/***********************************************************************************************
  Constrains a templated scalar parameter U to types convertible to scalar type S, but
  explicitly not bool. Rejecting bool prevents silent promotion of comparison results into
  broadcast values (e.g., `Simd<float> v = (a == b)` would otherwise broadcast 0.0f or 1.0f
  instead of producing a mask).
*/
#define MOCHI_REQUIRES_NON_BOOL_SCALAR(U, S) \
  MOCHI_CONCEPT((std::is_convertible_v<U, S> && !std::is_same_v<U, bool>))

/***********************************************************************************************
  Boilerplate code shared by all Simd<T, N> specializations
*/
#define MOCHI_NATIVE_SIMD_IMPL_BOILERPLATE(T, N, NativeT)           \
  NativeT raw;                                                      \
  static constexpr int kSize = N;                                   \
  static constexpr bool kIsSupported = true;                        \
  static constexpr bool kIsComposite = false;                       \
  static constexpr bool kIsEmulated = false;                        \
  using NativeType = NativeT;                                       \
  using Scalar = T;                                                 \
  MOCHI_FORCE_INLINE Simd() = default;                              \
  MOCHI_FORCE_INLINE ~Simd() = default;                             \
  MOCHI_FORCE_INLINE Simd(Simd const& rhs) = default;               \
  MOCHI_FORCE_INLINE Simd(Simd&& rhs) = default;                    \
  MOCHI_FORCE_INLINE Simd(NativeType rhs) : raw(rhs) {};            \
  MOCHI_FORCE_INLINE static constexpr size_t size() {               \
    return kSize;                                                   \
  }                                                                 \
  MOCHI_FORCE_INLINE Simd& operator=(Simd const& rhs) = default;    \
  MOCHI_FORCE_INLINE Simd& operator=(Simd&& rhs) = default;         \
  template <class U, MOCHI_REQUIRES_NON_BOOL_SCALAR(U, Scalar)>     \
  MOCHI_FORCE_INLINE Simd& operator=(U rhs) {                       \
    raw = Simd{rhs}.raw;                                            \
    return *this;                                                   \
  }                                                                 \
  [[nodiscard]] MOCHI_FORCE_INLINE Scalar operator[](int i) const { \
    return Get(*this, i); /* return by value */                     \
  }

} // namespace mochi

/**
  Architecture dependent Simd specializations
*/
#if MOCHI_USE_SIMD
#if MOCHI_ARCH_ARM
#include "arm/arm_simd_inl.h"
#elif MOCHI_ARCH_X64
#include "x64/x64_simd_inl.h"
#else
#error No supported SIMD architecture was detected. MOCHI_USE_SIMD should be zero.
#endif
// Partial specialization supporting Simd<T, N> for larger values of N
#include "simd_composite_inl.h"
#else
// Use scalar fallback when !MOCHI_USE_SIMD
#include "simd_arch_emulator_inl.h"
#endif
