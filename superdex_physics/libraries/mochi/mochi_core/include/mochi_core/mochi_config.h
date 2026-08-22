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

/**************************************************************************************************
  This header declares macros which enable/disable Mochi features.
  These macros are always defined to 1 or 0 so you can use #if instead of #ifdef.
  These macros are externally customizable (e.g. via BUCK of Visual Studio configuration).
  If they have not already been defined by the user, then we will define the defaults here.
*/

// no #pragma once for pure C inclusion
#ifndef MOCHI_CONFIG_H
#define MOCHI_CONFIG_H

#include <mochi_core/mochi_platform.h>

#include <cstddef>

/**************************************************************************************************
  Debugging
    - MOCHI_DEBUG                   Enables additional debug checks (slower).
    - MOCHI_OPTIMIZED               Defined to 1 if this is a build in which optimizations are
                                    enabled. Else defined to 0. Usually equivalent to !MOCHI_DEBUG.
    - MOCHI_WARN_IF_NOT_OPTIMIZED   Causes the compiler to print a warning message if
  !MOCHI_OPTIMIZED.
    - MOCHI_ASSERT_ENABLED          Enables MOCHI_ASSERT macros. Else they compile out.
    - MOCHI_ASSERT_VERBOSE_ENABLED  Enables MOCHI_ASSERT_VERBOSE macros. Else they compile out.
    - MOCHI_LOG_ENABLED             Enables the MOCHI_LOG family of macros. Else they compile out.
    - MOCHI_PROFILE_ENABLE          Enables the MOCHI_PROFILE family of macros. Else they compile
  out.
*/

#ifndef MOCHI_DEBUG
#ifdef NDEBUG
#define MOCHI_DEBUG 0
#else
#define MOCHI_DEBUG 1
#endif
#endif

#ifndef MOCHI_OPTIMIZED
#ifdef NDEBUG
#define MOCHI_OPTIMIZED 1
#else
#define MOCHI_OPTIMIZED 0
#endif
#endif

#ifndef MOCHI_WARN_IF_NOT_OPTIMIZED
#define MOCHI_WARN_IF_NOT_OPTIMIZED 0
#endif

#ifndef MOCHI_ASSERT_ENABLED
#define MOCHI_ASSERT_ENABLED 1
#endif

#ifndef MOCHI_ASSERT_VERBOSE_ENABLED
#define MOCHI_ASSERT_VERBOSE_ENABLED (MOCHI_DEBUG && MOCHI_ASSERT_ENABLED)
#endif

#ifndef MOCHI_LOG_ENABLED
#define MOCHI_LOG_ENABLED 1
#endif

#ifndef MOCHI_PROFILE_ENABLE
#define MOCHI_PROFILE_ENABLE 0
#endif

/**************************************************************************************************
  If MOCHI_INTERNAL is defined to 1, then the build will include samples and tests that depend on
  assets from "assets/internal". Do this only if you have access to the internal assets.
*/

#ifndef MOCHI_INTERNAL
#define MOCHI_INTERNAL 0
#endif

/**************************************************************************************************
  MOCHI_USE_MPC
    Gates an optional controller integration at compile time. Defaults to MOCHI_INTERNAL. Define
    MOCHI_USE_MPC explicitly to override the default.
*/

#ifndef MOCHI_USE_MPC
#define MOCHI_USE_MPC MOCHI_INTERNAL
#endif

/**************************************************************************************************
  Floating Point
    - MOCHI_USE_DOUBLE_PRECISION    If enabled type mochi::real will be double-precision
*/

#ifndef MOCHI_USE_DOUBLE_PRECISION
#define MOCHI_USE_DOUBLE_PRECISION 0
#endif

#if MOCHI_LANGUAGE_CPP
namespace mochi {
#if MOCHI_USE_DOUBLE_PRECISION
using real = double;
#else
using real = float;
#endif

inline constexpr real operator""_r(long double val) {
  return static_cast<real>(val);
}
inline constexpr real operator""_r(unsigned long long int val) {
  return static_cast<real>(val);
}

/**************************************************************************************************
  Integer literals for size_t and ptrdiff_t. NOTE: These have been recently accepted for the C++23
  standard. http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p0330r8.html
*/
inline constexpr ptrdiff_t operator""_z(unsigned long long int val) {
  return static_cast<ptrdiff_t>(val);
}
inline constexpr size_t operator""_uz(unsigned long long int val) {
  return static_cast<size_t>(val);
}

} // namespace mochi
#endif

/**************************************************************************************************
  MOCHI_USE_SIMD
    By default, Mochi makes use of SIMD vector instructions when supported by the CPU architecture.
    Define MOCHI_USE_SIMD=0 to disable SIMD and enable scalar emulation instead.
*/
#ifndef MOCHI_USE_SIMD
#define MOCHI_USE_SIMD 1
#endif

#if MOCHI_USE_SIMD && !MOCHI_ARCH_ARM_NEON && !MOCHI_ARCH_X64_AVX2
// SIMD not supported for this architecture
#undef MOCHI_USE_SIMD
#define MOCHI_USE_SIMD 0
#endif

#if !MOCHI_USE_SIMD
// Emulate 32 16-byte SIMD registers. A common use case for SIMD emulation is NVIDIA GPUs, where
// each thread has access to 255 32-bit registers. 32 SIMD vectors of 16 bytes will take 128 of
// them.
#undef MOCHI_SIMD_REGISTER_COUNT
#undef MOCHI_SIMD_REGISTER_SIZE_BYTES
#define MOCHI_SIMD_REGISTER_COUNT 32
#define MOCHI_SIMD_REGISTER_SIZE_BYTES 16
#endif

/**************************************************************************************************
  Optional Third Party Libraries
    - MOCHI_USE_CUDA                 Enable CUDA
    - MOCHI_USE_CUDSS                Enable cudss (used for some experimental features with CUDA)
    - MOCHI_USE_EIGEN                Enable Eigen (used for some experimental features)
    - MOCHI_USE_HDF5                 Enable HDF5 (used for binary file IO)
    - MOCHI_USE_TORCH                Enable Torch (used for neural network evaluation)
    - MOCHI_USE_TRACY                Enable Tracy (used for profiling)
    - MOCHI_USE_MATH_ACCELERATION    Enable math acceleration libraries (e.g. Accelerate on macOS)
*/

#ifndef MOCHI_USE_CUDA
#define MOCHI_USE_CUDA 0
#endif

#ifndef MOCHI_USE_CUDSS
#define MOCHI_USE_CUDSS 0
#endif

#if MOCHI_USE_CUDA == 0 && MOCHI_USE_CUDSS != 0
#undef MOCHI_USE_CUDSS
#define MOCHI_USE_CUDSS 0
#endif

#ifndef MOCHI_USE_EIGEN
#define MOCHI_USE_EIGEN 0
#endif

#ifndef MOCHI_USE_HDF5
#define MOCHI_USE_HDF5 0
#endif

/**************************************************************************************************
  MOCHI_USE_OSC
    Gates the optional Operational Space Controller (OSC) at compile time. Defaults to
    MOCHI_INTERNAL. Defined separately so OSC can later be toggled independently of MOCHI_INTERNAL.
*/
#ifndef MOCHI_USE_OSC
#define MOCHI_USE_OSC MOCHI_INTERNAL
#endif

#ifndef MOCHI_USE_TORCH
#define MOCHI_USE_TORCH 0
#endif

#ifndef MOCHI_USE_TRACY
#define MOCHI_USE_TRACY 0
#endif
#if MOCHI_USE_TRACY && !defined(TRACY_ENABLE)
#define TRACY_ENABLE 1
#endif

#ifndef MOCHI_USE_MATH_ACCELERATION
#define MOCHI_USE_MATH_ACCELERATION 0
#endif

/**************************************************************************************************
  Linking
*/

// MOCHI_CORE_STATIC_LINKING can only be defined to 1 when building MochiCore, or when building
// projects that will statically link with MochiCore. Projects that take a header-only dependency
// on MochiCore should leave the default of 0.
#ifndef MOCHI_CORE_STATIC_LINKING
#define MOCHI_CORE_STATIC_LINKING 0
#endif

// If MOCHI_USE_EXTERN_TEMPLATE is defined to 1, then extern templates will be used in headers
// to reduce code bloat. However, extern template declarations require static linking with the
// corresponding definitions. Therefore extern templates are not declared in projects that take
// a header-only dependency on MochiCore.
#ifndef MOCHI_USE_EXTERN_TEMPLATE
#if MOCHI_CORE_STATIC_LINKING
#define MOCHI_USE_EXTERN_TEMPLATE 1
#else
#define MOCHI_USE_EXTERN_TEMPLATE 0
#endif
#endif

// Some types have built-in features for Pybind. These are normally compiled out except in select
// cpp files which redefine MOCHI_INCLUDE_PYBIND_SUPPORT to 1.
#ifndef MOCHI_INCLUDE_PYBIND_SUPPORT
#define MOCHI_INCLUDE_PYBIND_SUPPORT 0
#endif

// Define MOCHI_PMR_USES_JEMALLOC to 1 if jemalloc is used to implement
// std::pmr::new_delete_resource. A work-around is required in this case.
#ifndef MOCHI_PMR_USES_JEMALLOC
#define MOCHI_PMR_USES_JEMALLOC 0
#endif

#endif // MOCHI_CONFIG_H
