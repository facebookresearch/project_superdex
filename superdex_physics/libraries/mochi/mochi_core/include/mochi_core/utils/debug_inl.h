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
#include "debug.h" // Reverse include for Intellisense

#include <mutex>
#include <type_traits>

namespace mochi {

#if MOCHI_ARCH_CPU
// Assert reporting in CPU code
#if MOCHI_HAS_VA_OPT
#define MOCHI_ASSERT_ON_FAILURE(file, line, conditionStr, ...)                             \
  if (::mochi::OnAssertionFailure(file, line, conditionStr __VA_OPT__(, )##__VA_ARGS__)) { \
    MOCHI_DEBUG_BREAK();                                                                   \
  }
#else
#define MOCHI_ASSERT_ON_FAILURE(file, line, conditionStr, ...)                \
  if (::mochi::OnAssertionFailure(file, line, conditionStr, ##__VA_ARGS__)) { \
    MOCHI_DEBUG_BREAK();                                                      \
  }
#endif
#elif MOCHI_ARCH_GPU
// GPU paths cannot use the full logger functionality. Instead we have to rely on simpler
// mechanisms to get the point across. CUDA for example, supports printf.
//
// @TODO[Nate,Hector] Is ther an alternative for MOCHI_DEBUG_BREAK in CUDA code? The compiler
// currently reports errors any time it is used inside a constexpr function (of which we have many).
#define MOCHI_ASSERT_ON_FAILURE(file, line, conditionStr, ...)                    \
  printf("%s(%d): Assertion Failure! Expected (%s)\n", file, line, conditionStr); \
  printf("" __VA_ARGS__);                                                         \
  printf("\n");                                                                   \
  /*MOCHI_DEBUG_BREAK()*/
#else
#error Expected MOCHI_ARCH_CPU or MOCHI_ARCH_GPU
#define MOCHI_ASSERT_ON_FAILURE
#endif

// Assert macro implementation (does not compile out if used directly)
#if MOCHI_HAS_VA_OPT
#define MOCHI_ASSERT_IMPL(condition, ...)                                                  \
  if (condition) {                                                                         \
  } else                                                                                   \
    MOCHI_UNLIKELY {                                                                       \
      MOCHI_ASSERT_ON_FAILURE(__FILE__, __LINE__, #condition __VA_OPT__(, )##__VA_ARGS__); \
    }
#else
#define MOCHI_ASSERT_IMPL(condition, ...)                                     \
  if (condition) {                                                            \
  } else                                                                      \
    MOCHI_UNLIKELY {                                                          \
      MOCHI_ASSERT_ON_FAILURE(__FILE__, __LINE__, #condition, ##__VA_ARGS__); \
    }
#endif

// These implementation functions are inlined so that MOCHI_ASSERT can be used
// via header-only dependency. This does not require linking with MochiCore.
namespace assert_impl {

#if MOCHI_ARCH_CPU
inline std::recursive_mutex& GetAssertMutexRef() {
  static std::recursive_mutex s_assertMutex;
  return s_assertMutex;
}
inline OnAssertFn& GetOnAssertFnRef() {
  static OnAssertFn s_onAssertFn;
  return s_onAssertFn;
}
inline int& GetAssertCallDepthRef() {
  static int s_assertCallDepth = 0;
  return s_assertCallDepth;
}
#endif // MOCHI_ARCH_CPU

inline static bool
DefaultAssertHandler(char const* condition, char const* message, char const* file, int line) {
  MOCHI_LOG_ERROR(
      "\n"
      "*****************************************************************************\n"
      "MOCHI ASSERTION FAILURE:\n"
      "    Message:   %s\n"
      "    Expected:  (%s)\n"
      "    Location:  %s(%d)\n"
      "*****************************************************************************\n",
      message,
      condition,
      file,
      line);

  // Make sure the message is flushed before the breakpoint
  FlushLog();

  // TODO:
  // Show a dialog box with options like: "Abort", "Debug", and "Continue at your own risk".
  // For now, we will always stop at a breakpoint, or crash with a non-zero exit code if no debugger
  // is connected.
  bool const triggerDebugBreak = true;

  return triggerDebugBreak;
}

} // namespace assert_impl

template <class DstPtrT, class SrcT>
MOCHI_FORCE_INLINE DstPtrT assert_cast(SrcT* ptr) {
  static_assert(std::is_pointer_v<DstPtrT>, "Invalid cast. Expected a pointer type.");
  MOCHI_ASSERT(dynamic_cast<DstPtrT>(ptr) == static_cast<DstPtrT>(ptr), "Invalid cast");
  return static_cast<DstPtrT>(ptr);
}
template <class DstRefT, class SrcT>
MOCHI_FORCE_INLINE DstRefT assert_cast(SrcT&& ref) {
  static_assert(std::is_reference_v<DstRefT>, "Invalid cast. Expected a reference type.");
  MOCHI_ASSERT(
      dynamic_cast<std::remove_reference_t<DstRefT>*>(&ref) ==
          static_cast<std::remove_reference_t<DstRefT>*>(&ref),
      "Invalid cast");
  return static_cast<DstRefT>(ref);
}

/** @cond */

inline OnAssertFn GetAssertionFailureCallback() {
#if MOCHI_ARCH_CPU
  std::lock_guard<std::recursive_mutex> lock(assert_impl::GetAssertMutexRef());
  auto& fn = assert_impl::GetOnAssertFnRef();
  return fn ? fn : OnAssertFn{assert_impl::DefaultAssertHandler};
#else
  return OnAssertFn{assert_impl::DefaultAssertHandler};
#endif
}

inline void SetAssertionFailureCallback([[maybe_unused]] OnAssertFn fn) {
#if MOCHI_ARCH_CPU
  std::lock_guard<std::recursive_mutex> lock(assert_impl::GetAssertMutexRef());
  assert_impl::GetOnAssertFnRef() = fn ? fn : OnAssertFn{assert_impl::DefaultAssertHandler};
#endif
}

// Overload for asserts with WITHOUT formatted args. Never inline this function!
MOCHI_NO_INLINE inline bool
OnAssertionFailure(char const* file, int line, char const* condition, char const* msg) {
#if MOCHI_ARCH_CPU
  std::lock_guard<std::recursive_mutex> lock(assert_impl::GetAssertMutexRef());
  int& callDepth = assert_impl::GetAssertCallDepthRef();
  if (callDepth != 0) {
    // HELP! Assert failure within an assert handler function!
    MOCHI_DEBUG_BREAK();
  }
  ++callDepth;
  bool shouldBreak = GetAssertionFailureCallback()(condition, msg, file, line);
  --callDepth;
  return shouldBreak;
#else
  return GetAssertionFailureCallback()(condition, msg, file, line);
#endif
}

// Overload for asserts with a formatted message. Never inline this function!
template <typename... Args>
MOCHI_NO_INLINE bool OnAssertionFailure(
    char const* file,
    int line,
    char const* condition,
    char const* fmt,
    Args... args) {
  return OnAssertionFailure(file, line, condition, Format(fmt, args...).c_str());
}

// Overload for asserts with no message at all. Never inline this function!
MOCHI_NO_INLINE inline bool OnAssertionFailure(char const* file, int line, char const* condition) {
  return OnAssertionFailure(file, line, condition, "");
}

/** @endcond */

} // namespace mochi
