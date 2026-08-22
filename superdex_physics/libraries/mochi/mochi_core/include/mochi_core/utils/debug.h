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
#include <mochi_core/utils/log.h>

#include <functional>

namespace mochi {

/* Useful diagnostics for the current translation unit. Compiler settings may differ between
   NVCC and the host compiler. Such settings will inevitably result in hard to debug crashes.
   These could help identify the problem.
*/
#if false
#define _STR(x) #x
#define STR(x) _STR(x)
#pragma message("[Translation Unit Diagnostics]")

#pragma message("PLATFORM(s):")
#pragma message("- Android: " STR(MOCHI_PLATFORM_ANDROID))
#pragma message("- Linux: " STR(MOCHI_PLATFORM_LINUX))
#pragma message("- MacOS: " STR(MOCHI_PLATFORM_MACOS))
#pragma message("- Windows: " STR(MOCHI_PLATFORM_WINDOWS))

#pragma message("COMPILER(s):")
#pragma message("- MSVC: " STR(MOCHI_COMPILER_MSVC))
#pragma message("- CLANG: " STR(MOCHI_COMPILER_CLANG))
#pragma message("- GCC: " STR(MOCHI_COMPILER_GCC))
#pragma message("- CUDA CPU: " STR(MOCHI_COMPILER_CUDA_CPU))
#pragma message("- CUDA GPU: " STR(MOCHI_COMPILER_CUDA_GPU))

#pragma message("Architecture:")
#pragma message("- CPU: " STR(MOCHI_ARCH_CPU))
#pragma message("- GPU: " STR(MOCHI_ARCH_GPU))
#pragma message("- ARM: " STR(MOCHI_ARCH_ARM))
#pragma message("- ARM NEON: " STR(MOCHI_ARCH_ARM_NEON))
#pragma message("- ARM SVE: " STR(MOCHI_ARCH_ARM_SVE))
#pragma message("- ARM SME: " STR(MOCHI_ARCH_ARM_SME))
#pragma message("- X64: " STR(MOCHI_ARCH_X64))
#pragma message("- X64 AVX2: " STR(MOCHI_ARCH_X64_AVX2))
#pragma message("- X64 FMA: " STR(MOCHI_ARCH_X64_FMA))
#pragma message("- X64 SVML: " STR(MOCHI_ARCH_X64_SVML))

#undef STR
#undef _STR
#endif

// MOCHI_ASSERT:
//   Used to catch mistakes made by programmers (e.g. bad parameters, bad internal state).
//   NOT used to report errors that the caller might handle (e.g. missing file, data not ready yet).
//   Enabled by default in all builds.
//
// WARNING:
//   MOCHI_ASSERT compiles out completely unless MOCHI_ASSERT_ENABLED.
//   Therefore, the condition should have NO SIDE EFFECTS.
//
// Examples:
//   MOCHI_ASSERT(ptr != nullptr, "File %s not found", filename);
//
// Terse Example (only legal in C++20 code, not in some public headers that are limited to C++17):
//   MOCHI_ASSERT(ptr != nullptr);
//
#if MOCHI_ASSERT_ENABLED
#if MOCHI_HAS_VA_OPT
#define MOCHI_ASSERT(condition_without_side_effects, ...) \
  MOCHI_ASSERT_IMPL(condition_without_side_effects __VA_OPT__(, )##__VA_ARGS__)
#else
#define MOCHI_ASSERT(condition_without_side_effects, ...) \
  MOCHI_ASSERT_IMPL(condition_without_side_effects, ##__VA_ARGS__)
#endif
#else
#define MOCHI_ASSERT(...)
#endif

// MOCHI_ASSERT_VERBOSE:
//   Similar to MOCHI_ASSERT (see above), except that it is only enabled in Debug builds by default.
//   Used to catch programmer mistakes in performance sensitive areas of the code. Can be enabled in
//   any build configuration by defining MOCHI_ASSERT_VERBOSE_ENABLED=1.
//
// WARNING:
//   MOCHI_ASSERT_VERBOSE compiles out completely unless MOCHI_ASSERT_VERBOSE_ENABLED.
//   Therefore, the condition should have NO SIDE EFFECTS.
//
#if MOCHI_ASSERT_VERBOSE_ENABLED
#if MOCHI_HAS_VA_OPT
#define MOCHI_ASSERT_VERBOSE(condition_without_side_effects, ...) \
  MOCHI_ASSERT_IMPL(condition_without_side_effects __VA_OPT__(, )##__VA_ARGS__)
#else
#define MOCHI_ASSERT_VERBOSE(condition_without_side_effects, ...) \
  MOCHI_ASSERT_IMPL(condition_without_side_effects, ##__VA_ARGS__)
#endif
#else
#define MOCHI_ASSERT_VERBOSE(...)
#endif

// Used to indicate not implemented functionalities.
#define MOCHI_NOT_IMPLEMENTED() MOCHI_ASSERT(false, "%s not implemented", __FUNCTION__)

// Used in code paths that require MOCHI_USE_EIGEN
#if MOCHI_USE_EIGEN
#define MOCHI_ASSERT_EIGEN()
#else
#define MOCHI_ASSERT_EIGEN() \
  MOCHI_ASSERT(              \
      MOCHI_USE_EIGEN,       \
      "This feature requires Eigen. To enable, add the Eigen dependency to your build configuration and define MOCHI_USE_EIGEN=1.")
#endif

// mochi::assert_cast has the same behavior of static_cast, except that it also uses MOCHI_ASSERT to
// report invalid casts. It works with pointers or references, but the types must be polymorphic
// (declaring at least one virtual function). If (!MOCHI_ASSERT_ENABLED), then assert_cast will
// compile out and only the static_cast will remain.
template <class DstPtrT, class SrcT>
MOCHI_FORCE_INLINE DstPtrT assert_cast(SrcT* ptr);
template <class DstRefT, class SrcT>
MOCHI_FORCE_INLINE DstRefT assert_cast(SrcT&& ref);

/**
 * @brief Callback invoked when an assertion fails.
 *
 * @details Receives the failed condition, formatted message, source file, and source line. Return
 * `true` to break execution at the failure site.
 */
using OnAssertFn =
    std::function<bool(char const* condition, char const* message, char const* file, int line)>;

/** @cond */

// Get the function to call when an assert fails
OnAssertFn GetAssertionFailureCallback();

// Set the function to call when an assert fails.
// Pass OnAssertFn{} to restore default behavior.
void SetAssertionFailureCallback(OnAssertFn fn);

// Call the current assertion failure callback.
// Return true if execution should stop at a break point.
bool OnAssertionFailure(char const* file, int line, char const* condition, char const* msg);

// Call the current assertion failure callback with a formatted message
// Return true if execution should stop at a break point.
template <typename... Args>
MOCHI_NO_INLINE bool OnAssertionFailure(
    char const* file,
    int line,
    char const* condition,
    char const* fmt,
    Args... args);

/** @endcond */

/**
 * @brief Return true if we can detect that a debugger is attached to this process. This may not be
 * supported on all platforms.
 *
 * @return bool
 */
[[nodiscard]] bool IsDebuggerAttached();

} // namespace mochi

#include "debug_inl.h"
