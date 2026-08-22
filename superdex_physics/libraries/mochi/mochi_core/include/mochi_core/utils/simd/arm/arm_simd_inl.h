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

/**
  This file contains Simd specializations for ARM CPUs with NEON support.
*/

#include "../../simd.h" // for IntelliSense

#if MOCHI_USE_SIMD && MOCHI_ARCH_ARM_NEON

#include <arm_neon.h>

/***********************************************************************************************
  Simd Specializations for ARM Architecture
*/

// NOTE: Order of these headers matters in some cases. Do not sort alphabetically.

#include "arm_simd_tables_inl.h"

#include "arm_simd_int_4_inl.h"

#include "arm_simd_int64_2_inl.h"

#include "arm_simd_double_2_inl.h"

#include "arm_simd_float_4_inl.h"

/***********************************************************************************************
  Simd Utilities for ARM Architecture
*/

namespace mochi {

// clang-format off
template<> MOCHI_FORCE_INLINE Simd<double,  2> StaticCast<Simd<double,  2>, Simd<int64_t, 2>>(Simd<int64_t, 2> const& a) { return vcvtq_f64_s64(a.raw); }
template<> MOCHI_FORCE_INLINE Simd<int64_t, 2> StaticCast<Simd<int64_t, 2>, Simd<double,  2>>(Simd<double,  2> const& a) { return vcvtq_s64_f64(a.raw); }
template<> MOCHI_FORCE_INLINE Simd<float,   4> StaticCast<Simd<float,   4>, Simd<int,     4>>(Simd<int,     4> const& a) { return vcvtq_f32_s32(a.raw); }
template<> MOCHI_FORCE_INLINE Simd<int,     4> StaticCast<Simd<int,     4>, Simd<float,   4>>(Simd<float,   4> const& a) { return vcvtq_s32_f32(a.raw); }

template<> MOCHI_FORCE_INLINE Simd<double,  2> ReinterpretCast<Simd<double,  2>, Simd<int64_t, 2>>(Simd<int64_t, 2> const& a) { return vreinterpretq_f64_s64(a.raw); }
template<> MOCHI_FORCE_INLINE Simd<double,  2> ReinterpretCast<Simd<double,  2>, Simd<float,   4>>(Simd<float,   4> const& a) { return vreinterpretq_f64_f32(a.raw); }
template<> MOCHI_FORCE_INLINE Simd<double,  2> ReinterpretCast<Simd<double,  2>, Simd<int,     4>>(Simd<int,     4> const& a) { return vreinterpretq_f64_s32(a.raw); }
template<> MOCHI_FORCE_INLINE Simd<int64_t, 2> ReinterpretCast<Simd<int64_t, 2>, Simd<double,  2>>(Simd<double,  2> const& a) { return vreinterpretq_s64_f64(a.raw); }
template<> MOCHI_FORCE_INLINE Simd<int64_t, 2> ReinterpretCast<Simd<int64_t, 2>, Simd<float,   4>>(Simd<float,   4> const& a) { return vreinterpretq_s64_f32(a.raw); }
template<> MOCHI_FORCE_INLINE Simd<int64_t, 2> ReinterpretCast<Simd<int64_t, 2>, Simd<int,     4>>(Simd<int,     4> const& a) { return vreinterpretq_s64_s32(a.raw); }
template<> MOCHI_FORCE_INLINE Simd<float,   4> ReinterpretCast<Simd<float,   4>, Simd<double,  2>>(Simd<double,  2> const& a) { return vreinterpretq_f32_f64(a.raw); }
template<> MOCHI_FORCE_INLINE Simd<float,   4> ReinterpretCast<Simd<float,   4>, Simd<int64_t, 2>>(Simd<int64_t, 2> const& a) { return vreinterpretq_f32_s64(a.raw); }
template<> MOCHI_FORCE_INLINE Simd<float,   4> ReinterpretCast<Simd<float,   4>, Simd<int,     4>>(Simd<int,     4> const& a) { return vreinterpretq_f32_s32(a.raw); }
template<> MOCHI_FORCE_INLINE Simd<int,     4> ReinterpretCast<Simd<int,     4>, Simd<double,  2>>(Simd<double,  2> const& a) { return vreinterpretq_s32_f64(a.raw); }
template<> MOCHI_FORCE_INLINE Simd<int,     4> ReinterpretCast<Simd<int,     4>, Simd<int64_t, 2>>(Simd<int64_t, 2> const& a) { return vreinterpretq_s32_s64(a.raw); }
template<> MOCHI_FORCE_INLINE Simd<int,     4> ReinterpretCast<Simd<int,     4>, Simd<float,   4>>(Simd<float,   4> const& a) { return vreinterpretq_s32_f32(a.raw); }
// clang-format on

} // namespace mochi

#endif // MOCHI_USE_SIMD && MOCHI_ARCH_ARM_NEON
