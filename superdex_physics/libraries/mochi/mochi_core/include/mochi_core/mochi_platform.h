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
  This header declares macros which depend on the current platform, compiler, or CPU
  features. See mochi_config.h for additional macros which enable/disable optional features.
*/

// no #pragma once for pure C inclusion
#ifndef MOCHI_PLATFORM_H
#define MOCHI_PLATFORM_H

#if defined(__cplusplus)
#include <type_traits>
#endif

/**************************************************************************************************
  Language:
    - MOCHI_LANGUAGE_CPP    Compiling with a C++ compiler (any version)
    - MOCHI_LANGUAGE_CPP17  Compiling with a C++ compiler that supports C++17 (or newer)
    - MOCHI_LANGUAGE_CPP20  Compiling with a C++ compiler that supports C++20 (or newer)
    - MOCHI_LANGUAGE_C      Compiling with a C compiler
*/
#if defined(__cplusplus)
#define MOCHI_LANGUAGE_CPP 1
#define MOCHI_LANGUAGE_CPP17 (__cplusplus >= 201703L)
#ifndef MOCHI_LANGUAGE_CPP20 // Can be defined to 0 in build script to test C++17 fallbacks.
#define MOCHI_LANGUAGE_CPP20 (__cplusplus >= 202002L)
#endif
#define MOCHI_LANGUAGE_C 0
#else
#define MOCHI_LANGUAGE_CPP 0
#define MOCHI_LANGUAGE_CPP17 0
#ifndef MOCHI_LANGUAGE_CPP20
#define MOCHI_LANGUAGE_CPP20 0
#endif
#define MOCHI_LANGUAGE_C 1
#endif

/**************************************************************************************************
  Architecture:
    - MOCHI_ARCH_CPU        Compiling for CPU (regular CPU code or CUDA CPU host)
    - MOCHI_ARCH_GPU        Compiling for GPU (CUDA GPU device code)
    - MOCHI_ARCH_ARM        Compiling for an ARM CPU (64-bit ARMv8 or newer)
    - MOCHI_ARCH_ARM_NEON   Compiling for an ARM CPU with NEON (mandatory for ARMv8 or newer)
    - MOCHI_ARCH_ARM_NEON_FP16_ARITHMETIC
                            Compiling with the ARM FP16 vector arithmetic feature enabled
    - MOCHI_ARCH_ARM_SVE    Compiling for an ARM CPU with SVE (mandatory for ARMv9 or newer)
    - MOCHI_ARCH_ARM_SME    Compiling for an ARM CPU with SME (scalable matrix extension)
    - MOCHI_ARCH_X64        Compiling for an x64 CPU (64-bit x86, also called x86_64)
    - MOCHI_ARCH_X64_AVX2   Compiling for an x64 CPU with AVX2 vector extension
    - MOCHI_ARCH_X64_FMA    Compiling for an x64 CPU with FMA (fused multiply add) extension
    - MOCHI_ARCH_X64_SVML   Compiling for an x64 CPU with SVML (small vector math library) extension
*/
#if defined(__CUDA_ARCH__)
#define MOCHI_ARCH_CPU 0
#define MOCHI_ARCH_GPU 1
#else
#define MOCHI_ARCH_CPU 1
#define MOCHI_ARCH_GPU 0
#endif

// NOTE: The Unreal Engine build system has its own macros.
#if defined(PLATFORM_ENABLE_VECTORINTRINSICS_NEON) && PLATFORM_ENABLE_VECTORINTRINSICS_NEON
#define MOCHI_UNREAL_ARCH_NEON 1
#else
#define MOCHI_UNREAL_ARCH_NEON 0
#endif

#if defined(__aarch64__) || MOCHI_UNREAL_ARCH_NEON
#define MOCHI_ARCH_ARM 1
#else
#define MOCHI_ARCH_ARM 0
#endif

#if MOCHI_ARCH_CPU && MOCHI_ARCH_ARM && (defined(__ARM_NEON) || MOCHI_UNREAL_ARCH_NEON)
#define MOCHI_ARCH_ARM_NEON 1
#else
#define MOCHI_ARCH_ARM_NEON 0
#endif

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#define MOCHI_ARCH_ARM_NEON_FP16_ARITHMETIC 1
#else
#define MOCHI_ARCH_ARM_NEON_FP16_ARITHMETIC 0
#endif

// NOTE: The Unreal Engine build system has its own macros.
#define MOCHI_UNREAL_ARCH_AVX2 0

#if defined(PLATFORM_CPU_X86_FAMILY)
#if PLATFORM_CPU_X86_FAMILY
#undef MOCHI_UNREAL_ARCH_AVX2
#define MOCHI_UNREAL_ARCH_AVX2 1 // Assume AVX2 for Unreal Engine builds
#endif
#endif

// UNREALIOS_LINUX_EDITOR disables AVX2 for the Linux editor build.
// The editor target shares its PCH with UnrealEditor, which is compiled without -mavx2.
// Adding -mavx2 per-module causes a PCH/AST feature mismatch.
// Game target does use AVX2.
#if defined(UNREALIOS_LINUX_EDITOR)
#if UNREALIOS_LINUX_EDITOR
#undef MOCHI_UNREAL_ARCH_AVX2
#define MOCHI_UNREAL_ARCH_AVX2 0
#endif
#endif

#if defined(_M_X64) || defined(__x86_64) || MOCHI_UNREAL_ARCH_AVX2
#define MOCHI_ARCH_X64 1
#else
#define MOCHI_ARCH_X64 0
#endif

#if MOCHI_ARCH_CPU && MOCHI_ARCH_X64 && (defined(__AVX2__) || MOCHI_UNREAL_ARCH_AVX2)
#define MOCHI_ARCH_X64_AVX2 1
#else
#define MOCHI_ARCH_X64_AVX2 0
#endif

#if MOCHI_ARCH_CPU && MOCHI_ARCH_X64_AVX2 && \
    (defined(__FMA__) || (defined(_MSC_VER) && !defined(__clang__)))
#define MOCHI_ARCH_X64_FMA 1
#else
#define MOCHI_ARCH_X64_FMA 0
#endif

#if MOCHI_ARCH_CPU && MOCHI_ARCH_X64_AVX2 && \
    (defined(__SVML__) || (defined(_MSC_VER) && !defined(__clang__)))
#define MOCHI_ARCH_X64_SVML 1
#else
#define MOCHI_ARCH_X64_SVML 0
#endif

// Error checking
#if !MOCHI_ARCH_ARM && (defined(__arm__) || defined(_M_ARM))
#error Mochi does not support older 32-bit versions of ARM
#endif
#if MOCHI_ARCH_ARM && MOCHI_ARCH_X64
#error Failed to detect correct architecture
#endif
#if MOCHI_UNREAL_ARCH_AVX2 && MOCHI_UNREAL_ARCH_NEON
#error Cannot have both AVX and NEON
#endif

/**************************************************************************************************
  Compiler:
    - MOCHI_COMPILER_MSVC       Microsoft Visual C++ (MSVC)
    - MOCHI_COMPILER_GCC        GNU C/C++ compiler
    - MOCHI_COMPILER_CLANG      CLANG C/C++ compiler
    - MOCHI_COMPILER_CUDA_CPU   CUDA compiler (NVCC) compiling a .cu file for the CPU.
    - MOCHI_COMPILER_CUDA_GPU   CUDA compiler (NVCC) compiling a .cu file for the GPU.
    - MOCHI_COMPILER_CUDA       CUDA compiler (NVCC) compiling a .cu file for CPU or GPU.

  NOTE:
    When compiling a .cu file, (MOCHI_COMPILER_CUDA && MOCHI_COMPILER_CLANG) can both be true
    because NVCC uses the host compiler (e.g. clang) to preprocess the file. If the host compiler
    was GCC or MSVC, then MOCHI_COMPILER_GCC or MOCHI_COMPILER_MSVC would be true while compiling
    that .cu file.
*/

#if defined(_MSC_VER) && !defined(__clang__)
// NOTE: Both _MSC_VER and __clang__ are defined when the Clang frontend targets the MSVC
//       backend (clang-cl). The code which uses MOCHI_COMPILER_MSVC only cares about the frontend
//       (e.g. C++ warnings, minor differences in xmmintrin.h, etc...)
#define MOCHI_COMPILER_MSVC 1
#else
#define MOCHI_COMPILER_MSVC 0
#endif

// MOCHI_MSVC_TRADITIONAL indicates that MSVC is doing its old non-standard compliant behavior.
// You can make it more standard compliant (less "traditional"), but it is not the default.
#if defined(_MSVC_TRADITIONAL)
#define MOCHI_MSVC_TRADITIONAL _MSVC_TRADITIONAL
#else
#define MOCHI_MSVC_TRADITIONAL 0
#endif

#if defined(__GNUC__) && !defined(__clang__)
#define MOCHI_COMPILER_GCC 1
#else
#define MOCHI_COMPILER_GCC 0
#endif
#if defined(__clang__)
#define MOCHI_COMPILER_CLANG 1
#if (__clang_major__ < 19)
#define MOCHI_CLANG_AWAIT_SUSPEND_BUG 1
#else
#define MOCHI_CLANG_AWAIT_SUSPEND_BUG 0
#endif
#else
#define MOCHI_COMPILER_CLANG 0
#define MOCHI_CLANG_AWAIT_SUSPEND_BUG 0
#endif

#if defined(__CUDA_ARCH__) // CUDA GPU
#define MOCHI_COMPILER_CUDA_CPU 0
#define MOCHI_COMPILER_CUDA_GPU 1
#define MOCHI_COMPILER_CUDA 1
#elif defined(__CUDACC__) // CUDA CPU
#define MOCHI_COMPILER_CUDA_CPU 1
#define MOCHI_COMPILER_CUDA_GPU 0
#define MOCHI_COMPILER_CUDA 1
#else // Neither
#define MOCHI_COMPILER_CUDA_CPU 0
#define MOCHI_COMPILER_CUDA_GPU 0
#define MOCHI_COMPILER_CUDA 0
#endif

// Sanitizers
#if defined(__SANITIZE_ADDRESS__) && (MOCHI_COMPILER_CLANG || MOCHI_COMPILER_GCC)
#define MOCHI_COMPILER_ASAN 1
#else
#define MOCHI_COMPILER_ASAN 0
#endif
#if defined(__SANITIZE_THREAD__) && (MOCHI_COMPILER_CLANG || MOCHI_COMPILER_GCC)
#define MOCHI_COMPILER_TSAN 1
#else
#define MOCHI_COMPILER_TSAN 0
#endif
#if (MOCHI_COMPILER_CLANG || MOCHI_COMPILER_GCC)
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#undef MOCHI_COMPILER_ASAN
#define MOCHI_COMPILER_ASAN 1
#endif
#endif
#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#undef MOCHI_COMPILER_TSAN
#define MOCHI_COMPILER_TSAN 1
#endif
#endif
#endif

#if MOCHI_COMPILER_CLANG
#if __has_feature(cxx_rtti)
#define MOCHI_COMPILER_HAS_RTTI 1
#else
#define MOCHI_COMPILER_HAS_RTTI 0
#endif
#elif MOCHI_COMPILER_GCC
#ifdef __GXX_RTTI
#define MOCHI_COMPILER_HAS_RTTI 1
#else
#define MOCHI_COMPILER_HAS_RTTI 0
#endif
#elif MOCHI_COMPILER_MSVC
#ifdef _CPPRTTI
#define MOCHI_COMPILER_HAS_RTTI 1
#else
#define MOCHI_COMPILER_HAS_RTTI 0
#endif
#else
// RTTI is assumed to be enabled on other compilers.
// If not, then add a check here.
#define MOCHI_COMPILER_HAS_RTTI 1
#endif

/**************************************************************************************************
  MOCHI_HAS_VA_OPT indicates whether or not __VA_OPT__ is supported.
*/
#define MOCHI_HAS_VA_OPT (MOCHI_LANGUAGE_CPP20 && !MOCHI_MSVC_TRADITIONAL)

/**************************************************************************************************
  Platform:
    - MOCHI_PLATFORM_ANDROID    Compiling for Android.
    - MOCHI_PLATFORM_LINUX      Compiling for Linux (not including Android)
    - MOCHI_PLATFORM_MACOS      Compiling for MacOS.
    - MOCHI_PLATFORM_WINDOWS    Compiling for Windows.

*/
#if defined(__ANDROID__)
#define MOCHI_PLATFORM_ANDROID 1
#else
#define MOCHI_PLATFORM_ANDROID 0
#endif

#if defined(__linux__) && !defined(__ANDROID__)
#define MOCHI_PLATFORM_LINUX 1
#else
#define MOCHI_PLATFORM_LINUX 0
#endif

#if defined(__APPLE__)
#define MOCHI_PLATFORM_MACOS 1
#else
#define MOCHI_PLATFORM_MACOS 0
#endif

#if defined(_WIN32)
#define MOCHI_PLATFORM_WINDOWS 1
#else
#define MOCHI_PLATFORM_WINDOWS 0
#endif

/**************************************************************************************************
  CUDA Function Attributes:
    - MOCHI_ANY         Function can run on host or device
    - MOCHI_CPU         Function can only run on CPU host
    - MOCHI_GPU         Function can only run on GPU device
    - MOCHI_GPU_KERNEL  Function is a "global" GPU kernel
*/
#if MOCHI_COMPILER_CUDA
#define MOCHI_ANY __host__ __device__
#define MOCHI_CPU __host__
#define MOCHI_GPU __device__
#define MOCHI_GPU_KERNEL __global__
#else
#define MOCHI_ANY
#define MOCHI_CPU
#define MOCHI_GPU
#define MOCHI_GPU_KERNEL
#endif

/**************************************************************************************************
  MOCHI_FORCE_INLINE
    A stronger version of 'inline'. Ignored in Debug configuration.
    WARNING: Only use this on small functions to avoid code bloat.

  Example:
    MOCHI_FORCE_INLINE void Foo() {}
*/
#ifdef NDEBUG
// NOTE: Order matters because MOCHI_COMPILER_* macros may not be mutually exclusive. Check CUDA GPU
// first.
#if MOCHI_COMPILER_CUDA_GPU
#define MOCHI_FORCE_INLINE __forceinline__
#elif MOCHI_COMPILER_MSVC
#define MOCHI_FORCE_INLINE __forceinline
#else
#define MOCHI_FORCE_INLINE __attribute__((always_inline)) inline
#endif
#else
#define MOCHI_FORCE_INLINE inline
#endif

/**************************************************************************************************
  MOCHI_FORCE_INLINE_LAMBDA
    A stronger version of 'inline' for a lambda's call operator (operator()): directs the compiler
    to inline the lambda body into its call sites.

    WARNING: Only use on small lambdas in hot paths (avoids code bloat).

    NOTE: Only effective for *direct* calls through the concrete closure type (e.g. the lambda held
    by 'auto' or passed as a template parameter). If the lambda is type-erased into an std::function
    (or called via function pointer / virtual dispatch), the call is indirect and the body is NOT
    inlined at that boundary.

  Example:
    auto foo = [](int x) MOCHI_FORCE_INLINE_LAMBDA { return x + 1; };
*/
#ifdef NDEBUG
// NOTE: Order matters because MOCHI_COMPILER_* macros may not be mutually exclusive. Check CUDA GPU
// first.
#if MOCHI_COMPILER_CUDA_GPU
#define MOCHI_FORCE_INLINE_LAMBDA
#elif (MOCHI_COMPILER_CLANG || MOCHI_COMPILER_GCC)
#define MOCHI_FORCE_INLINE_LAMBDA __attribute__((always_inline))
#else
#if defined(__has_cpp_attribute)
#if __has_cpp_attribute(msvc::forceinline)
#define MOCHI_FORCE_INLINE_LAMBDA [[msvc::forceinline]]
#else
#define MOCHI_FORCE_INLINE_LAMBDA
#endif // #if __has_cpp_attribute(msvc::forceinline)
#else
#define MOCHI_FORCE_INLINE_LAMBDA
#endif // #if defined(__has_cpp_attribute)
#endif // #if MOCHI_COMPILER_CUDA_GPU
#else
#define MOCHI_FORCE_INLINE_LAMBDA
#endif // #ifdef NDEBUG

/**************************************************************************************************
  MOCHI_UNROLL_LOOP_N
    Requests that the compiler unroll the immediately following loop by N.

  NOTE: This is a best-effort hint. The compiler may ignore it. It expands to nothing on MSVC and on
  unrecognized compilers.

  Example:
    MOCHI_UNROLL_LOOP_N(4)
    for (int i = 0; i < 4; ++i) {}
*/
// NOTE: Order matters because MOCHI_COMPILER_* macros may not be mutually exclusive. Check CUDA GPU
// first.
#if MOCHI_COMPILER_CUDA_GPU
#define MOCHI_UNROLL_LOOP_N(N) _Pragma(MOCHI_PP_STRINGIFY(unroll N))
#elif MOCHI_COMPILER_MSVC
#define MOCHI_UNROLL_LOOP_N(N)
#elif MOCHI_COMPILER_GCC
#define MOCHI_UNROLL_LOOP_N(N) _Pragma(MOCHI_PP_STRINGIFY(GCC unroll N))
#elif MOCHI_COMPILER_CLANG
#define MOCHI_UNROLL_LOOP_N(N) _Pragma(MOCHI_PP_STRINGIFY(unroll N))
#else
#define MOCHI_UNROLL_LOOP_N(N)
#endif

/**************************************************************************************************
  MOCHI_NO_INLINE
    Directs the compiler NOT to inline a function.

  Example:
    MOCHI_NO_INLINE void Foo() {}
*/
#if MOCHI_COMPILER_CUDA_GPU
#define MOCHI_NO_INLINE __noinline__
#elif MOCHI_COMPILER_MSVC
#define MOCHI_NO_INLINE __declspec(noinline)
#else
#define MOCHI_NO_INLINE __attribute__((noinline))
#endif

/**************************************************************************************************
  MOCHI_RESTRICT
    Qualifies a pointer argument as restricted, promising to the compiler that
    the memory accessed via the pointer will not be aliased by any other pointer.
    This enables some compiler optimizations.

  Example:
    void MyCopy(float* MOCHI_RESTRICT dst, float const* MOCHI_RESTRICT src, size_t n) {
      memcpy(dst, src, n);
    }
*/
// NOTE: Order matters because MOCHI_COMPILER_* macros may not be mutually exclusive.
#if MOCHI_COMPILER_CUDA_GPU || MOCHI_COMPILER_GCC || MOCHI_COMPILER_CLANG
#define MOCHI_RESTRICT __restrict__
#elif MOCHI_COMPILER_MSVC
#define MOCHI_RESTRICT __restrict
#else
#define MOCHI_RESTRICT restrict
#endif

/**************************************************************************************************
 SIMD:
  MOCHI_SIMD_REGISTER_COUNT       Number of floating-point SIMD registers.
  MOCHI_SIMD_REGISTER_SIZE_BYTES  Size (in bytes) of each floating-point SIMD register.
*/
#if MOCHI_ARCH_X64_AVX2
#define MOCHI_SIMD_REGISTER_COUNT 16
#define MOCHI_SIMD_REGISTER_SIZE_BYTES 32
#elif MOCHI_ARCH_ARM_NEON
#define MOCHI_SIMD_REGISTER_COUNT 32
#define MOCHI_SIMD_REGISTER_SIZE_BYTES 16
#else
#define MOCHI_SIMD_REGISTER_COUNT Unsupported architecture
#define MOCHI_SIMD_REGISTER_SIZE_BYTES Unsupported architecture
#endif

/**************************************************************************************************
 Memory Cache
*/

/**
 * @brief Conservative upper limit for the size (and alignment) of a data cache line.
 *
 * @details Cache line size is a property of the CPU, so it is not known at compile time.
 * This conservative value can be used when compile-time alignment is needed. For a more
 * accurate runtime value use @ref GetCacheLineInfo.
 *
 * @see MOCHI_CONSERVATIVE_CACHE_ALIGN, GetCacheLineInfo
 */
#ifndef MOCHI_CONSERVATIVE_CACHE_LINE_SIZE
#define MOCHI_CONSERVATIVE_CACHE_LINE_SIZE 256
#endif

/**
 * @brief Aligns the memory of a variable so it starts at the beginning of a cache line.
 *
 * @details Cache line size is a property of the CPU, so it is not known at compile time. This
 * conservative value will waste some memory on most systems, but that is usually acceptable for
 * static variables and temporary variables.
 *
 * @see MOCHI_CONSERVATIVE_CACHE_LINE_SIZE, GetCacheLineInfo
 */
#define MOCHI_CONSERVATIVE_CACHE_ALIGN alignas(MOCHI_CONSERVATIVE_CACHE_LINE_SIZE)

/**************************************************************************************************
  MOCHI_NO_INIT
    Add this attribute to the end of a non-static local variable declaration to tell the compiler
    that it should NOT be initialized to zero. Without the attribute, clang/gcc will sometimes
    perform the initialization even though the programmer did not request it.

    Example:
      std::byte buffer0[4096] MOCHI_NO_INIT, buffer1[1024] MOCHI_NO_INIT;
*/
#if MOCHI_COMPILER_GCC || MOCHI_COMPILER_CLANG
// clang-format off
#define MOCHI_NO_INIT __attribute__((uninitialized)) /* NOLINT(cppcoreguidelines-init-variables) */
// clang-format on
#else
#define MOCHI_NO_INIT /* NOLINT(cppcoreguidelines-init-variables) */
#endif

/**************************************************************************************************
  MOCHI_DEBUG_BREAK()
    Trigger a debug breakpoint (fatal if no debugger is connected)
*/
// NOTE: Order matters because MOCHI_COMPILER_* macros may not be mutually exclusive. Check CUDA GPU
// first.
#if MOCHI_COMPILER_CUDA_GPU
#define MOCHI_DEBUG_BREAK() asm("brkpt;")
#elif MOCHI_PLATFORM_WINDOWS
#define MOCHI_DEBUG_BREAK() __debugbreak()
#else
#define MOCHI_DEBUG_BREAK() __builtin_trap()
#endif

/**************************************************************************************************
  Warning Suppression:
    Use these macros if you really need to suppress a specific compiler warning.
    Targeted suppression is better than global suppression.

  GCC and Clang:
    Clang is able to understand GCC's warning suppression syntax and they support many of the same
    warnings. Therefore, we often use the GCC warning suppression syntax for both via
    MOCHI_WARNING_IGNORE_GCC_CLANG. GCC and Clang both require you to specify the full argument
    string because _Pragma does not support string literal concatenation.

  Examples:
    MOCHI_WARNING_PUSH();
    MOCHI_WARNING_IGNORE_MSVC(4100);
    MOCHI_WARNING_IGNORE_GCC_CLANG(GCC diagnostic ignored "-Wunused-parameter");
    MOCHI_WARNING_IGNORE_CLANG(clang diagnostic ignored "-Wself-assign-overloaded");
    // YOUR CODE HERE
    MOCHI_WARNING_POP();
*/
// NOTE: Order matters because MOCHI_COMPILER_* macros may not be mutually exclusive. Check CUDA GPU
// first.
#if MOCHI_COMPILER_CUDA_GPU
#define MOCHI_WARNING_PUSH()
#define MOCHI_WARNING_PUSH_IGNORE_ALL()
#define MOCHI_WARNING_POP()
#elif MOCHI_COMPILER_MSVC
#define MOCHI_WARNING_PUSH() __pragma(warning(push))
#define MOCHI_WARNING_PUSH_IGNORE_ALL() __pragma(warning(push, 0))
#define MOCHI_WARNING_POP() __pragma(warning(pop))
#elif MOCHI_COMPILER_CLANG
#define MOCHI_WARNING_PUSH() _Pragma("GCC diagnostic push")
#define MOCHI_WARNING_PUSH_IGNORE_ALL() \
  _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Weverything\"")
#define MOCHI_WARNING_POP() _Pragma("GCC diagnostic pop")
#elif MOCHI_COMPILER_GCC
#define MOCHI_WARNING_PUSH() _Pragma("GCC diagnostic push")
#define MOCHI_WARNING_PUSH_IGNORE_ALL()                                      \
  _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Wall\"") \
      _Pragma("GCC diagnostic ignored \"-Wextra\"")
#define MOCHI_WARNING_POP() _Pragma("GCC diagnostic pop")
#else
#define MOCHI_WARNING_PUSH()
#define MOCHI_WARNING_PUSH_IGNORE_ALL()
#define MOCHI_WARNING_POP()
#endif

#if MOCHI_COMPILER_MSVC
#define MOCHI_WARNING_IGNORE_MSVC(X) __pragma(warning(disable : X))
#define MOCHI_WARNING_ENFORCE_MSVC(X) __pragma(warning(error : X))
#else
#define MOCHI_WARNING_IGNORE_MSVC(X)
#define MOCHI_WARNING_ENFORCE_MSVC(X)
#endif

#if MOCHI_COMPILER_GCC
#define MOCHI_WARNING_IGNORE_GCC(X) _Pragma(#X)
#define MOCHI_WARNING_ENFORCE_GCC(X) _Pragma(#X)
#else
#define MOCHI_WARNING_IGNORE_GCC(X)
#define MOCHI_WARNING_ENFORCE_GCC(X)
#endif

#if MOCHI_COMPILER_CLANG
#define MOCHI_WARNING_IGNORE_CLANG(X) _Pragma(#X)
#define MOCHI_WARNING_ENFORCE_CLANG(X) _Pragma(#X)
#else
#define MOCHI_WARNING_IGNORE_CLANG(X)
#define MOCHI_WARNING_ENFORCE_CLANG(X)
#endif

// Macros for warnings that are supported by both GCC and Clang
#if MOCHI_COMPILER_GCC || MOCHI_COMPILER_CLANG
#define MOCHI_WARNING_IGNORE_GCC_CLANG(X) _Pragma(#X)
#define MOCHI_WARNING_ENFORCE_GCC_CLANG(X) _Pragma(#X)
#else
#define MOCHI_WARNING_IGNORE_GCC_CLANG(X)
#define MOCHI_WARNING_ENFORCE_GCC_CLANG(X)
#endif

// Opt-ins for compiler specific warnings that are off by default.
// These warnings are enabled in every supported build.
/* Data member 'member1' will be initialized after data member 'member2' */
MOCHI_WARNING_ENFORCE_MSVC(5038)

// Opt-outs for other warnings.
/* function marked as __forceinline not inlined */
MOCHI_WARNING_IGNORE_MSVC(4714)

/**************************************************************************************************
  Warning Suppression for CUDA
*/

#if MOCHI_COMPILER_CUDA
#define MOCHI_WARNING_IGNORE_CUDA(X) __pragma(diag_suppress = X)
#define MOCHI_WARNING_ENFORCE_CUDA(X) __pragma(diag_error = X)
#else
#define MOCHI_WARNING_IGNORE_CUDA(X)
#define MOCHI_WARNING_ENFORCE_CUDA(X)
#endif

// Edge case: some templated utility functions may be instantiated using CPU-only
// functions, which will result in a warning from NVCC. In such cases, it is
// necessary to disable this check (until a better solution is found...)
#if MOCHI_COMPILER_CUDA_GPU
#if __CUDAVER__ >= 75000
#define MOCHI_DISABLE_CUDA_GPU_EXEC_CHECK() __pragma(nv_exec_check_disable)
#else
#define MOCHI_DISABLE_CUDA_GPU_EXEC_CHECK()
#endif
#else
#define MOCHI_DISABLE_CUDA_GPU_EXEC_CHECK()
#endif

#define MOCHI_TEMPLATE_FUNCTION MOCHI_DISABLE_CUDA_GPU_EXEC_CHECK()

// Opt-outs for compiler-specific warnings for third-party libraries or
// other exotic compilers (NVCC). Most of these are unavoidable, so we
// have to wrap inclusion locations with these directives.
#define MOCHI_WARNING_SUPPRESS_CUDA()                                    \
  /* Nonstandard extension used : nameless struct */                     \
  MOCHI_WARNING_IGNORE_MSVC(4201)                                        \
  /* '=' : conversion from 'OffsetT' to 'int', possible loss of data */  \
  MOCHI_WARNING_IGNORE_MSVC(4244)                                        \
  /* Structure was padded due to alignment specifier */                  \
  MOCHI_WARNING_IGNORE_MSVC(4324)                                        \
  /* Declaration of 'variable' hides class member */                     \
  MOCHI_WARNING_IGNORE_MSVC(4458)                                        \
  /* Unreferenced local function has been removed */                     \
  MOCHI_WARNING_IGNORE_MSVC(4505)                                        \
  /* Uninitalized local variable */                                      \
  MOCHI_WARNING_IGNORE_MSVC(4700)                                        \
  /* Assignment within conditional expression */                         \
  MOCHI_WARNING_IGNORE_MSVC(4706)                                        \
  /* Spurious "missing return statement at end of non-void function"     \
     in constexpr functions. Fixed in later versions of CUDA (>11.1). */ \
  MOCHI_WARNING_IGNORE_CUDA(implicit_return_from_non_void_function)

/**************************************************************************************************
  Warning Suppression for Eigen
*/

#define MOCHI_WARNING_SUPPRESS_EIGEN()                                                      \
  /* Conditional expression is constant */                                                  \
  MOCHI_WARNING_IGNORE_MSVC(4127)                                                           \
  /* Check operator precedence for possible error; use parentheses to clarify precedence */ \
  MOCHI_WARNING_IGNORE_MSVC(4554)                                                           \
  /* Conversion from '__int64' to 'uint64_t', signed/unsigned mismatch */                   \
  MOCHI_WARNING_IGNORE_MSVC(4245)                                                           \
  /* annotation ignored on a function that is explicitly defaulted */                       \
  MOCHI_WARNING_IGNORE_CUDA(esa_on_defaulted_function_ignored)

/**************************************************************************************************
  Runtime Checks Suppression
    Some third party headers (e.g. thrust) trigger MSVC's run-time checks when ran
    with Debug configuration. While these errors are mostly harmless, they are
    terribly cumbersome. These macros allow us to selectively suppress these checks.
*/
#if MOCHI_COMPILER_MSVC
#define MOCHI_DEBUG_RTCHECKS_ENABLE() __pragma(runtime_checks("sc", restore))
#define MOCHI_DEBUG_RTCHECKS_DISABLE() __pragma(runtime_checks("sc", off))
#else
#define MOCHI_DEBUG_RTCHECKS_ENABLE()
#define MOCHI_DEBUG_RTCHECKS_DISABLE()
#endif

/**************************************************************************************************
  No-Op
*/

#if MOCHI_COMPILER_MSVC
#include <intrin.h>
#define MOCHI_NOP() __nop();
#else
#define MOCHI_NOP() __asm__ __volatile__("nop");
#endif
// clang-format off
#define MOCHI_NOP_10()                                             \
  MOCHI_NOP(); MOCHI_NOP(); MOCHI_NOP(); MOCHI_NOP(); MOCHI_NOP(); \
  MOCHI_NOP(); MOCHI_NOP(); MOCHI_NOP(); MOCHI_NOP(); MOCHI_NOP();
#define MOCHI_NOP_50()                                             \
  MOCHI_NOP_10(); MOCHI_NOP_10(); MOCHI_NOP_10(); MOCHI_NOP_10(); MOCHI_NOP_10();
#define MOCHI_NOP_250()                                            \
  MOCHI_NOP_50(); MOCHI_NOP_50(); MOCHI_NOP_50(); MOCHI_NOP_50(); MOCHI_NOP_50();
// clang-format on

/**************************************************************************************************
  Pre-processor Helpers
*/

// Pre-processor concatenation, for use in other macros.
#define MOCHI_PP_CAT(a, b) MOCHI_PP_CAT_IMPL(a, b)
#define MOCHI_PP_CAT_IMPL(a, b) a##b

// Pre-processor stringization, for use in other macros (e.g. _Pragma operands).
#define MOCHI_PP_STRINGIFY(X) MOCHI_PP_STRINGIFY_IMPL(X)
#define MOCHI_PP_STRINGIFY_IMPL(X) #X

/**************************************************************************************************
  Move & Copy Semantics
*/

// Declares default move operations. Goes inside a class or struct declaration.
#define MOCHI_DECLARE_MOVE(Name)   \
  Name(Name&&) noexcept = default; \
  Name& operator=(Name&&) = default;

// Deletes the default move operations. Goes inside a class or struct declaration.
#define MOCHI_DECLARE_NO_MOVE(Name) \
  Name(Name&&) = delete;            \
  Name& operator=(Name&&) = delete;

// Declares default copy operations. Goes inside a class or struct declaration.
// Can be used in private scope to enable an explicit Copy() function.
#define MOCHI_DECLARE_COPY(Name) \
  Name(Name const&) = default;   \
  Name& operator=(Name const&) = default;

// Deletes the default copy operations. Goes inside a class or struct declaration.
#define MOCHI_DECLARE_NO_COPY(Name) \
  Name(Name const&) = delete;       \
  Name& operator=(Name const&) = delete;

// Deletes both the copy and move operations. Goes inside a class or struct declaration.
// Note that  "pinned" is a term borrowed from other languages. It means the address of the object
// is not allowed to change.
#define MOCHI_DECLARE_NO_COPY_NO_MOVE(Name) \
  MOCHI_DECLARE_NO_COPY(Name)               \
  MOCHI_DECLARE_NO_MOVE(Name)

// Declares unique ownership semantics (move but no copy). Goes inside a class or struct
// declaration.
#define MOCHI_DECLARE_MOVE_ONLY(Name) \
  MOCHI_DECLARE_MOVE(Name);           \
  MOCHI_DECLARE_NO_COPY(Name)

// Declares the default copy constructor and move constructor, but deletes assignment.
// Used for classes/structs containing view matrices, which cannot be copied via operator= (because
// that would copy values instead).
#define MOCHI_DECLARE_NO_ASSIGN(Name)    \
  Name& operator=(Name const&) = delete; \
  Name& operator=(Name&&) = delete;

// Declares of class with inheritance from an empty base to activate EBCO
// (Empty Base Class Optimization)
#if MOCHI_COMPILER_MSVC
#define MOCHI_EMPTY_BASE __declspec(empty_bases)
#else
#define MOCHI_EMPTY_BASE
#endif

/**************************************************************************************************
 Concepts
*/

#if MOCHI_LANGUAGE_CPP
// A unique type used in MOCHI_CONCEPT macros. Does not allow implicit conversions from other types.
namespace mochi {
enum class ConceptMatch { True };
} // namespace mochi
#endif // MOCHI_LANGUAGE_CPP

// Use this in function declarations that would 'requires' a type trait. Similar to C++20 concepts.
//
// Example:
//
//    // This function only matches overload resolution for arithmetic types
//    template <typename T, MOCHI_CONCEPT(std::is_arithmetic_v<T>)>
//    void Foo(T value);
//
#define MOCHI_CONCEPT(a) std::enable_if_t<a, mochi::ConceptMatch> = mochi::ConceptMatch::True

// Use this in function definitions where the declaration is elsewhere.
#define MOCHI_CONCEPT_DEF(a) std::enable_if_t<a, mochi::ConceptMatch>

/**************************************************************************************************
  Branch Prediction Hints

  Examples:
    if (condition) MOCHI_LIKELY {} // Hint: This branch will probably be taken
    if (condition) MOCHI_UNLIKELY {} // Hint: This branch will probably NOT be taken.
*/
#if MOCHI_LANGUAGE_CPP20
#define MOCHI_LIKELY [[likely]]
#define MOCHI_UNLIKELY [[unlikely]]
#else
#define MOCHI_LIKELY
#define MOCHI_UNLIKELY
#endif

/**************************************************************************************************
  Validation of the above macros
*/

#if !(MOCHI_COMPILER_CUDA || MOCHI_COMPILER_MSVC || MOCHI_COMPILER_GCC || MOCHI_COMPILER_CLANG)
#error "Unknown compiler"
#endif
#if !(                                                                        \
    MOCHI_PLATFORM_ANDROID || MOCHI_PLATFORM_LINUX || MOCHI_PLATFORM_MACOS || \
    MOCHI_PLATFORM_WINDOWS)
#error "Unknown platform"
#endif

#endif // MOCHI_PLATFORM_H
